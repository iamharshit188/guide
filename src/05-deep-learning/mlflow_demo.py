"""
MLflow experiment tracking and model registry — local filesystem backend (no server needed).
pip install mlflow scikit-learn numpy
"""

import os
import time
import json
import tempfile
import shutil

import numpy as np

try:
    import mlflow
    import mlflow.sklearn
    from mlflow.tracking import MlflowClient
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("mlflow not installed. Run: pip install mlflow")

try:
    from sklearn.datasets import load_iris, load_wine
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score, train_test_split
    from sklearn.metrics import accuracy_score, f1_score, classification_report
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("scikit-learn not installed. Run: pip install scikit-learn")


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


# ── Dataset helpers ────────────────────────────────────────────

def load_dataset(name="iris"):
    if name == "iris":
        data = load_iris()
    else:
        data = load_wine()
    X_tr, X_te, y_tr, y_te = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42, stratify=data.target
    )
    return X_tr, X_te, y_tr, y_te, data.target_names.tolist()


# ── Experiment runner ──────────────────────────────────────────

def run_experiment(tracking_uri, experiment_name, model_configs, dataset_name="iris"):
    if not (MLFLOW_AVAILABLE and SKLEARN_AVAILABLE):
        return []

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    X_tr, X_te, y_tr, y_te, class_names = load_dataset(dataset_name)

    run_ids = []
    for config in model_configs:
        model_name = config["name"]
        model_cls  = config["model"]
        params     = config["params"]

        with mlflow.start_run(run_name=model_name) as run:
            # ── Log params ────────────────────────────────────
            mlflow.log_params(params)
            mlflow.log_param("dataset", dataset_name)
            mlflow.log_param("n_train", len(X_tr))
            mlflow.log_param("n_test",  len(X_te))
            mlflow.log_param("n_features", X_tr.shape[1])
            mlflow.log_param("n_classes", len(class_names))

            # ── Train ─────────────────────────────────────────
            pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("model",  model_cls(**params)),
            ])
            t0 = time.perf_counter()
            pipeline.fit(X_tr, y_tr)
            train_time_s = time.perf_counter() - t0

            # ── CV metrics ────────────────────────────────────
            cv_scores = cross_val_score(pipeline, X_tr, y_tr, cv=5,
                                        scoring="accuracy")
            cv_mean = float(cv_scores.mean())
            cv_std  = float(cv_scores.std())

            # ── Test metrics ──────────────────────────────────
            y_pred = pipeline.predict(X_te)
            test_acc = float(accuracy_score(y_te, y_pred))
            test_f1  = float(f1_score(y_te, y_pred, average="weighted"))

            # ── Log metrics ───────────────────────────────────
            mlflow.log_metric("cv_accuracy_mean", cv_mean)
            mlflow.log_metric("cv_accuracy_std",  cv_std)
            mlflow.log_metric("test_accuracy",    test_acc)
            mlflow.log_metric("test_f1_weighted", test_f1)
            mlflow.log_metric("train_time_s",     round(train_time_s, 4))

            # ── Simulate per-epoch metrics (for demo) ─────────
            for epoch in range(1, 11):
                # Fake training curve
                simulated_train_loss = 1.0 / (1 + epoch * 0.5) + np.random.default_rng(epoch).uniform(0, 0.02)
                mlflow.log_metric("train_loss", round(simulated_train_loss, 4), step=epoch)

            # ── Log artifact: classification report ───────────
            report_str = classification_report(y_te, y_pred,
                                               target_names=class_names)
            with tempfile.NamedTemporaryFile(mode="w", suffix=".txt",
                                             delete=False) as f:
                f.write(f"Model: {model_name}\n\n")
                f.write(report_str)
                tmp_path = f.name
            mlflow.log_artifact(tmp_path, artifact_path="reports")
            os.unlink(tmp_path)

            # ── Log the model ──────────────────────────────────
            signature = mlflow.models.infer_signature(
                X_tr, pipeline.predict(X_tr)
            )
            mlflow.sklearn.log_model(
                pipeline,
                artifact_path="model",
                signature=signature,
                registered_model_name=f"{experiment_name}_{model_name.replace(' ', '_')}",
            )

            run_ids.append({
                "run_id": run.info.run_id,
                "name": model_name,
                "test_accuracy": test_acc,
                "cv_mean": cv_mean,
                "cv_std": cv_std,
                "test_f1": test_f1,
                "train_time_s": round(train_time_s, 4),
            })

            print(f"  [{model_name}]  acc={test_acc:.4f}  "
                  f"cv={cv_mean:.4f}±{cv_std:.4f}  "
                  f"f1={test_f1:.4f}  "
                  f"time={train_time_s*1000:.1f}ms")

    return run_ids


# ── Model registry operations ──────────────────────────────────

def demo_model_registry(tracking_uri, experiment_name, run_results):
    if not MLFLOW_AVAILABLE or not run_results:
        return

    section("MODEL REGISTRY — VERSIONING + LIFECYCLE")
    client = MlflowClient(tracking_uri=tracking_uri)

    # Find best run by test accuracy
    best = max(run_results, key=lambda r: r["test_accuracy"])
    print(f"  Best run: {best['name']} (acc={best['test_accuracy']:.4f})")

    # List registered models
    print(f"\n  Registered models:")
    for rm in client.search_registered_models():
        latest = client.get_latest_versions(rm.name)
        for v in latest:
            print(f"    {rm.name} v{v.version}  "
                  f"stage={v.current_stage}  "
                  f"run_id={v.run_id[:8]}...")

    # Promote best model to Staging
    best_model_name = (
        f"{experiment_name}_{best['name'].replace(' ', '_')}"
    )
    try:
        versions = client.get_latest_versions(best_model_name)
        if versions:
            v = versions[0]
            client.transition_model_version_stage(
                name=best_model_name,
                version=v.version,
                stage="Staging",
                archive_existing_versions=False,
            )
            print(f"\n  Transitioned {best_model_name} v{v.version} → Staging")

            # Tag the version
            client.set_model_version_tag(
                name=best_model_name,
                version=v.version,
                key="approved_by",
                value="pipeline_auto",
            )
            client.set_model_version_tag(
                name=best_model_name,
                version=v.version,
                key="test_accuracy",
                value=str(round(best["test_accuracy"], 4)),
            )
    except Exception as e:
        print(f"  (Registry transition skipped: {e})")

    # Load model from registry by run_id (most reliable in local mode)
    best_run_id = best["run_id"]
    model_uri = f"runs:/{best_run_id}/model"
    print(f"\n  Loading model from run: {model_uri}")
    try:
        loaded_model = mlflow.sklearn.load_model(model_uri)
        _, X_te, _, y_te, _ = load_dataset()
        preds = loaded_model.predict(X_te)
        acc = accuracy_score(y_te, preds)
        print(f"  Loaded model accuracy: {acc:.4f} ✓")
    except Exception as e:
        print(f"  (Model load failed: {e})")


# ── Run comparison ─────────────────────────────────────────────

def demo_run_comparison(tracking_uri, experiment_name, run_results):
    if not MLFLOW_AVAILABLE or not run_results:
        return

    section("RUN COMPARISON TABLE")
    client = MlflowClient(tracking_uri=tracking_uri)

    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        return
    runs = mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["metrics.test_accuracy DESC"],
    )

    print(f"  {'Run name':25s}  {'test_acc':>9s}  {'cv_mean':>8s}  "
          f"{'cv_std':>7s}  {'f1':>7s}  {'time_ms':>8s}")
    print("  " + "-" * 72)

    for _, row in runs.iterrows():
        name = row.get("tags.mlflow.runName", "?")
        acc  = row.get("metrics.test_accuracy", float("nan"))
        cv_m = row.get("metrics.cv_accuracy_mean", float("nan"))
        cv_s = row.get("metrics.cv_accuracy_std", float("nan"))
        f1   = row.get("metrics.test_f1_weighted", float("nan"))
        t_s  = row.get("metrics.train_time_s", float("nan"))
        print(f"  {name:25s}  {acc:>9.4f}  {cv_m:>8.4f}  "
              f"{cv_s:>7.4f}  {f1:>7.4f}  {t_s*1000:>8.1f}")


# ── Autolog demo ───────────────────────────────────────────────

def demo_autolog(tracking_uri, experiment_name):
    if not (MLFLOW_AVAILABLE and SKLEARN_AVAILABLE):
        return

    section("AUTOLOG — ZERO BOILERPLATE TRACKING")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name + "_autolog")

    X_tr, X_te, y_tr, y_te, _ = load_dataset()

    # autolog hooks into sklearn.fit() automatically
    mlflow.sklearn.autolog(
        log_input_examples=False,
        log_model_signatures=True,
        log_models=True,
        silent=True,
    )

    with mlflow.start_run(run_name="autolog_rf"):
        rf = RandomForestClassifier(n_estimators=30, max_depth=5, random_state=42)
        rf.fit(X_tr, y_tr)
        test_acc = accuracy_score(y_te, rf.predict(X_te))
        mlflow.log_metric("test_accuracy", test_acc)
        print(f"  autolog run completed — test_acc={test_acc:.4f}")
        print(f"  All params and metrics logged automatically by mlflow.sklearn.autolog()")

    mlflow.sklearn.autolog(disable=True)


# ── Artifact logging ───────────────────────────────────────────

def demo_artifacts(tracking_uri, experiment_name):
    if not MLFLOW_AVAILABLE:
        return

    section("ARTIFACT LOGGING")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name + "_artifacts")

    with mlflow.start_run(run_name="artifact_demo"):
        with tempfile.TemporaryDirectory() as tmp:
            # Log a plain text file
            txt_path = os.path.join(tmp, "notes.txt")
            with open(txt_path, "w") as f:
                f.write("Training notes: iris classification experiment.\n")
                f.write("Features: sepal/petal length and width.\n")
            mlflow.log_artifact(txt_path, artifact_path="notes")

            # Log a JSON config
            cfg = {"model": "rf", "n_estimators": 50, "max_depth": 5,
                   "dataset": "iris", "seed": 42}
            cfg_path = os.path.join(tmp, "config.json")
            with open(cfg_path, "w") as f:
                json.dump(cfg, f, indent=2)
            mlflow.log_artifact(cfg_path, artifact_path="config")

            # Log multiple files from a directory
            subdir = os.path.join(tmp, "metrics")
            os.makedirs(subdir)
            for fold in range(3):
                with open(os.path.join(subdir, f"fold_{fold}.json"), "w") as f:
                    json.dump({"fold": fold, "accuracy": 0.95 + fold * 0.01}, f)
            mlflow.log_artifacts(subdir, artifact_path="cv_folds")

        # Log params + metrics inline
        mlflow.log_params(cfg)
        mlflow.log_metric("demo_metric", 0.99)

        run_id = mlflow.active_run().info.run_id
        print(f"  Artifacts logged to run {run_id[:8]}...")
        print(f"  Artifact types: text, JSON config, CV fold JSONs")
        print(f"  Retrieve: mlflow.artifacts.download_artifacts(run_id=...)")


# ── Tracking concepts ──────────────────────────────────────────

def explain_tracking_concepts():
    section("MLFLOW CONCEPTS REFERENCE")
    print("""
  Hierarchy:
    Tracking Server
      └── Experiments (named groups of runs)
            └── Runs (single execution)
                  ├── Params (logged once — hyperparams)
                  ├── Metrics (logged per step — loss, acc)
                  ├── Tags (key-value metadata)
                  └── Artifacts (files — models, reports, data)

  Backend stores:
    Tracking URI          | Stores metadata (params, metrics, tags)
    Artifact store        | Stores binary artifacts (models, files)
    ─────────────────────────────────────────────────────────────
    mlruns/ (default)     | SQLite + local filesystem
    sqlite:///mlflow.db   | SQLite + local filesystem
    postgresql://...      | PostgreSQL + S3/GCS
    databricks://         | Databricks hosted MLflow

  Model flavors (how a model is serialized):
    mlflow.sklearn        → pickle + conda env
    mlflow.pytorch        → state_dict + TorchScript
    mlflow.tensorflow     → SavedModel format
    mlflow.pyfunc         → generic Python callable (any framework)

  Registry stages:
    None → Staging → Production → Archived
    Use transitions to gate model promotion:
      None: just trained, not reviewed
      Staging: passed automated tests
      Production: serving live traffic
      Archived: superseded by newer version

  Key API patterns:
    mlflow.log_params(dict)               # batch param logging
    mlflow.log_metric("loss", v, step=t)  # per-step metric
    mlflow.log_artifact(path)             # upload a file
    mlflow.sklearn.log_model(model, "model", registered_model_name="iris")
    mlflow.sklearn.load_model("models:/iris/Production")
    MlflowClient().transition_model_version_stage(name, version, stage)
    """)


# ── Main ───────────────────────────────────────────────────────

def main():
    if not (MLFLOW_AVAILABLE and SKLEARN_AVAILABLE):
        explain_tracking_concepts()
        return

    # Use a temp directory so demo is self-contained (no leftover mlruns/)
    tmpdir = tempfile.mkdtemp(prefix="mlflow_demo_")
    tracking_uri = f"file://{tmpdir}/mlruns"
    experiment_name = "classification_benchmark"

    try:
        section("EXPERIMENT RUNS — LOGGING PARAMS, METRICS, MODELS")
        model_configs = [
            {
                "name": "LogisticRegression",
                "model": LogisticRegression,
                "params": {"max_iter": 200, "C": 1.0, "random_state": 42},
            },
            {
                "name": "RandomForest_50",
                "model": RandomForestClassifier,
                "params": {"n_estimators": 50, "max_depth": 5, "random_state": 42},
            },
            {
                "name": "RandomForest_100",
                "model": RandomForestClassifier,
                "params": {"n_estimators": 100, "max_depth": None, "random_state": 42},
            },
            {
                "name": "GradientBoosting",
                "model": GradientBoostingClassifier,
                "params": {"n_estimators": 50, "learning_rate": 0.1,
                           "max_depth": 3, "random_state": 42},
            },
        ]

        run_results = run_experiment(tracking_uri, experiment_name,
                                     model_configs, dataset_name="iris")

        demo_run_comparison(tracking_uri, experiment_name, run_results)
        demo_model_registry(tracking_uri, experiment_name, run_results)
        demo_autolog(tracking_uri, experiment_name)
        demo_artifacts(tracking_uri, experiment_name)
        explain_tracking_concepts()

        section("TRACKING DIRECTORY STRUCTURE")
        mlruns_dir = os.path.join(tmpdir, "mlruns")
        for root, dirs, files in os.walk(mlruns_dir):
            depth = root.replace(mlruns_dir, "").count(os.sep)
            if depth > 3:
                continue
            indent = "  " + "  " * depth
            rel = os.path.relpath(root, mlruns_dir)
            print(f"{indent}{rel}/")
            for f in files[:3]:
                print(f"{indent}  {f}")
            if len(files) > 3:
                print(f"{indent}  ... ({len(files)-3} more)")

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
