# Project 02 — Titanic Survival Predictor

> **Difficulty:** Beginner · **Module:** 02 — ML Basics to Advanced
> **Skills:** Feature engineering, K-fold CV, ROC/PR curves, model comparison

---

## What You'll Build

An end-to-end ML pipeline on the Titanic dataset: feature engineering → imputation/encoding → K-fold cross-validation comparing Logistic Regression, Random Forest, and XGBoost → ROC and PR curves from scratch → feature importance table. No sklearn metrics in the core evaluation — implement everything yourself.

---

## Skills Exercised

- Manual feature extraction (title from Name, FamilySize, binned Age)
- K-fold cross-validation (from scratch or via `sklearn.model_selection.KFold`)
- ROC-AUC: sorting predictions, computing TPR/FPR at each threshold
- PR curve: precision/recall at each threshold, average precision
- Gini impurity / MDI feature importance from Random Forest

---

## Approach

### Phase 1 — Load and engineer features
```
read titanic.csv (or download from seaborn: sns.load_dataset('titanic'))
create Title: extract from Name with regex (Mr, Mrs, Miss, Master, Rare)
create FamilySize: SibSp + Parch + 1
create IsAlone: FamilySize == 1
bin Age into 5 buckets: child(<12), teen(12-18), adult(18-35), middle(35-60), senior(60+)
drop: Name, Ticket, Cabin, PassengerId
```

### Phase 2 — Impute and encode
```
fill Age with median (by Pclass, Sex group)
fill Embarked with mode
encode Sex: male=0, female=1
one-hot encode Embarked, Title (drop first)
scale: StandardScaler on Age, Fare, FamilySize
```

### Phase 3 — K-fold CV comparison
```
models = [LogisticRegression(), RandomForestClassifier(n_estimators=100, random_state=42),
          GradientBoostingClassifier(n_estimators=100, random_state=42)]
for model in models:
    scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc')
    print model name, mean ± std
```

### Phase 4 — ROC and PR curves from scratch
```
train final model on full train set, predict_proba on held-out set
roc_curve_scratch(y_true, y_score):
    thresholds = sorted(unique y_score, descending)
    for t in thresholds:
        y_pred = y_score >= t
        tpr = TP / (TP + FN)
        fpr = FP / (FP + TN)
        append (fpr, tpr)
    auc = trapezoid area under curve
pr_curve_scratch(y_true, y_score): similar, precision vs recall
```

### Phase 5 — Feature importance table
```
fit RandomForest on full data
print: feature | MDI importance | rank
       + permutation importance (shuffle each feature, measure accuracy drop)
```

---

## Checkpoints

| Phase | What correct output looks like |
|-------|-------------------------------|
| 1 | `df.shape` after engineering ≈ (891, 12–15 cols); no Name/Ticket/Cabin |
| 2 | `df.isnull().sum()` → 0 for all columns used in model |
| 3 | LR AUC ≈ 0.84, RF ≈ 0.87, GBM ≈ 0.87–0.88 (±0.02) |
| 4 | ROC AUC from scratch matches `sklearn.metrics.roc_auc_score` within 0.001 |
| 5 | Sex, Pclass, Title_Mr among top 3 features by MDI |

---

## Extensions

1. **Calibration curve** — plot predicted probability bins vs. actual survival rate; use `sklearn.calibration.calibration_curve` as ground truth to verify your from-scratch version.
2. **SMOTE from scratch** — implement synthetic minority oversampling: for each minority sample, find k nearest neighbors, interpolate a new point along the line to a random neighbor.
3. **Threshold analysis** — sweep decision threshold 0.1–0.9; print F1, precision, recall, accuracy at each; find the threshold maximizing F-β at β=2.

---

## Hints

<details><summary>Hint 1 — Title extraction regex</summary>
<code>df['Name'].str.extract(r',\s*([A-Za-z]+)\.')</code>. Group rare titles (Dr, Rev, Col, etc.) into "Rare".
</details>

<details><summary>Hint 2 — Group-wise median imputation</summary>
<code>df['Age'] = df.groupby(['Pclass','Sex'])['Age'].transform(lambda x: x.fillna(x.median()))</code>
</details>

<details><summary>Hint 3 — Trapezoid AUC</summary>
Sort (fpr, tpr) points by fpr, then: <code>auc = sum((fpr[i+1]-fpr[i]) * (tpr[i]+tpr[i+1])/2)</code>
</details>

<details><summary>Hint 4 — Permutation importance</summary>
For each feature j: shuffle column j in X_test, re-predict, measure accuracy drop. Average over 5 shuffles. A large drop = high importance.
</details>

<details><summary>Hint 5 — Why FamilySize matters</summary>
Solo travelers and very large families (5+) had lower survival. FamilySize captures a U-shaped relationship that IsAlone alone misses.
</details>

---

*Back to [Module 02 — ML Basics to Advanced](../02-ml-basics.md)*
