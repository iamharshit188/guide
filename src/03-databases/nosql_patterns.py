"""
NoSQL patterns using a pure-Python in-memory document store.
Every operation is annotated with the equivalent pymongo/MongoDB call.
No external dependencies — runs immediately.

To run the same code against real MongoDB:
  pip install pymongo
  client = pymongo.MongoClient("mongodb://localhost:27017/")
  db = client["aiml_platform"]
  Replace: store.collection("X") → db["X"]
"""

import json
import copy
import time
import random
from datetime import datetime, timezone
from typing import Any


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


# ── Minimal In-Memory Document Store ─────────────────────────

class ObjectId:
    _counter = 0

    def __init__(self):
        ObjectId._counter += 1
        ts = int(time.time())
        self._id = f"{ts:08x}{ObjectId._counter:016x}"

    def __str__(self):
        return self._id

    def __repr__(self):
        return f"ObjectId('{self._id}')"


def _matches_filter(doc, filter_dict):
    """Recursive filter matching — subset of MongoDB query operators."""
    for key, condition in filter_dict.items():
        if key == "$and":
            if not all(_matches_filter(doc, sub) for sub in condition):
                return False
        elif key == "$or":
            if not any(_matches_filter(doc, sub) for sub in condition):
                return False
        elif key == "$not":
            if _matches_filter(doc, condition):
                return False
        else:
            # Dot-notation for nested fields: "metadata.source"
            val = _get_nested(doc, key)
            if isinstance(condition, dict):
                for op, operand in condition.items():
                    if op == "$eq"  and not (val == operand): return False
                    if op == "$ne"  and not (val != operand): return False
                    if op == "$gt"  and not (val >  operand): return False
                    if op == "$gte" and not (val >= operand): return False
                    if op == "$lt"  and not (val <  operand): return False
                    if op == "$lte" and not (val <= operand): return False
                    if op == "$in"  and val not in operand:   return False
                    if op == "$nin" and val in operand:       return False
                    if op == "$exists":
                        exists = val is not None
                        if operand != exists: return False
                    if op == "$regex" and not (isinstance(val, str) and operand in val):
                        return False
            else:
                if val != condition:
                    return False
    return True


def _get_nested(doc, key):
    """Traverse dot-notation key in dict."""
    parts = key.split(".")
    val = doc
    for p in parts:
        if isinstance(val, dict):
            val = val.get(p)
        else:
            return None
    return val


def _set_nested(doc, key, value):
    parts = key.split(".")
    d = doc
    for p in parts[:-1]:
        d = d.setdefault(p, {})
    d[parts[-1]] = value


def _apply_update(doc, update_dict):
    """Apply MongoDB update operators."""
    for op, fields in update_dict.items():
        for key, val in fields.items():
            if op == "$set":
                _set_nested(doc, key, val)
            elif op == "$unset":
                parts = key.split(".")
                d = doc
                for p in parts[:-1]:
                    d = d.get(p, {})
                d.pop(parts[-1], None)
            elif op == "$inc":
                cur = _get_nested(doc, key) or 0
                _set_nested(doc, key, cur + val)
            elif op == "$push":
                arr = _get_nested(doc, key) or []
                arr.append(val)
                _set_nested(doc, key, arr)
            elif op == "$pull":
                arr = _get_nested(doc, key) or []
                _set_nested(doc, key, [x for x in arr if x != val])
            elif op == "$addToSet":
                arr = _get_nested(doc, key) or []
                if val not in arr:
                    arr.append(val)
                _set_nested(doc, key, arr)


class Collection:
    def __init__(self, name):
        self.name = name
        self._docs = {}   # _id → doc
        self._indexes = {}  # field → {value → set of _ids}

    # ── CRUD ──

    def insert_one(self, document):
        """pymongo: collection.insert_one(document)"""
        doc = copy.deepcopy(document)
        if "_id" not in doc:
            doc["_id"] = str(ObjectId())
        self._docs[doc["_id"]] = doc
        self._update_indexes(doc)
        return doc["_id"]

    def insert_many(self, documents):
        """pymongo: collection.insert_many(documents)"""
        return [self.insert_one(d) for d in documents]

    def find_one(self, filter_dict=None):
        """pymongo: collection.find_one(filter)"""
        for doc in self._docs.values():
            if filter_dict is None or _matches_filter(doc, filter_dict):
                return copy.deepcopy(doc)
        return None

    def find(self, filter_dict=None, projection=None, sort=None, limit=0, skip=0):
        """pymongo: collection.find(filter, projection).sort(...).limit(n)"""
        results = [
            copy.deepcopy(doc) for doc in self._docs.values()
            if filter_dict is None or _matches_filter(doc, filter_dict)
        ]
        if sort:
            for key, direction in reversed(sort):
                results.sort(key=lambda d: (_get_nested(d, key) or 0),
                             reverse=(direction == -1))
        if skip:
            results = results[skip:]
        if limit:
            results = results[:limit]
        if projection:
            include = {k for k, v in projection.items() if v}
            exclude = {k for k, v in projection.items() if not v}
            filtered = []
            for doc in results:
                if include:
                    filtered.append({k: doc[k] for k in include if k in doc})
                else:
                    filtered.append({k: v for k, v in doc.items() if k not in exclude})
            return filtered
        return results

    def update_one(self, filter_dict, update_dict, upsert=False):
        """pymongo: collection.update_one(filter, update, upsert=False)"""
        for _id, doc in self._docs.items():
            if _matches_filter(doc, filter_dict):
                _apply_update(doc, update_dict)
                return {"matched": 1, "modified": 1}
        if upsert:
            new_doc = copy.deepcopy(filter_dict)
            _apply_update(new_doc, update_dict)
            self.insert_one(new_doc)
            return {"matched": 0, "upserted": new_doc.get("_id")}
        return {"matched": 0, "modified": 0}

    def update_many(self, filter_dict, update_dict):
        """pymongo: collection.update_many(filter, update)"""
        count = 0
        for doc in self._docs.values():
            if _matches_filter(doc, filter_dict):
                _apply_update(doc, update_dict)
                count += 1
        return {"matched": count, "modified": count}

    def delete_one(self, filter_dict):
        """pymongo: collection.delete_one(filter)"""
        for _id, doc in list(self._docs.items()):
            if _matches_filter(doc, filter_dict):
                del self._docs[_id]
                return {"deleted": 1}
        return {"deleted": 0}

    def delete_many(self, filter_dict):
        """pymongo: collection.delete_many(filter)"""
        to_del = [_id for _id, doc in self._docs.items()
                  if _matches_filter(doc, filter_dict)]
        for _id in to_del:
            del self._docs[_id]
        return {"deleted": len(to_del)}

    def count_documents(self, filter_dict=None):
        """pymongo: collection.count_documents(filter)"""
        return sum(1 for doc in self._docs.values()
                   if filter_dict is None or _matches_filter(doc, filter_dict))

    def replace_one(self, filter_dict, replacement):
        """pymongo: collection.replace_one(filter, replacement)"""
        for _id, doc in self._docs.items():
            if _matches_filter(doc, filter_dict):
                new_doc = copy.deepcopy(replacement)
                new_doc["_id"] = _id
                self._docs[_id] = new_doc
                return {"matched": 1, "modified": 1}
        return {"matched": 0, "modified": 0}

    # ── Aggregation pipeline (subset) ──

    def aggregate(self, pipeline):
        """
        pymongo: collection.aggregate(pipeline)
        Supported stages: $match, $group, $project, $sort, $limit, $skip, $unwind, $count
        """
        docs = [copy.deepcopy(d) for d in self._docs.values()]
        for stage in pipeline:
            op = list(stage.keys())[0]
            spec = stage[op]

            if op == "$match":
                docs = [d for d in docs if _matches_filter(d, spec)]

            elif op == "$group":
                groups = {}
                for doc in docs:
                    key_val = _get_nested(doc, spec["_id"].lstrip("$")) if spec["_id"] else None
                    gkey = json.dumps(key_val, default=str)
                    if gkey not in groups:
                        groups[gkey] = {"_id": key_val}
                    g = groups[gkey]
                    for field, expr in spec.items():
                        if field == "_id":
                            continue
                        if isinstance(expr, dict):
                            agg_op = list(expr.keys())[0]
                            src = list(expr.values())[0]
                            src_val = _get_nested(doc, src.lstrip("$")) if isinstance(src, str) else src
                            if agg_op == "$sum":
                                g[field] = g.get(field, 0) + (src_val or 0)
                            elif agg_op == "$avg":
                                g.setdefault(f"__{field}_sum", 0)
                                g.setdefault(f"__{field}_cnt", 0)
                                g[f"__{field}_sum"] += (src_val or 0)
                                g[f"__{field}_cnt"] += 1
                                g[field] = g[f"__{field}_sum"] / g[f"__{field}_cnt"]
                            elif agg_op == "$max":
                                g[field] = max(g.get(field, float("-inf")), src_val or float("-inf"))
                            elif agg_op == "$min":
                                g[field] = min(g.get(field, float("inf")), src_val or float("inf"))
                            elif agg_op == "$push":
                                g.setdefault(field, []).append(src_val)
                            elif agg_op == "$addToSet":
                                s = g.setdefault(field, [])
                                if src_val not in s:
                                    s.append(src_val)
                docs = [
                    {k: v for k, v in g.items() if not k.startswith("__")}
                    for g in groups.values()
                ]

            elif op == "$project":
                projected = []
                for doc in docs:
                    nd = {}
                    for k, v in spec.items():
                        if v == 1 or v is True:
                            nd[k] = _get_nested(doc, k)
                        elif v == 0 or v is False:
                            pass  # exclude
                        elif isinstance(v, str) and v.startswith("$"):
                            nd[k] = _get_nested(doc, v.lstrip("$"))
                        else:
                            nd[k] = v
                    projected.append(nd)
                docs = projected

            elif op == "$sort":
                for key, direction in reversed(list(spec.items())):
                    docs.sort(key=lambda d: (_get_nested(d, key) or 0),
                              reverse=(direction == -1))

            elif op == "$limit":
                docs = docs[:spec]

            elif op == "$skip":
                docs = docs[spec:]

            elif op == "$unwind":
                field = spec.lstrip("$")
                unwound = []
                for doc in docs:
                    arr = _get_nested(doc, field) or []
                    if isinstance(arr, list):
                        for item in arr:
                            nd = copy.deepcopy(doc)
                            _set_nested(nd, field, item)
                            unwound.append(nd)
                    else:
                        unwound.append(doc)
                docs = unwound

            elif op == "$count":
                docs = [{spec: len(docs)}]

        return docs

    def create_index(self, keys, unique=False, sparse=False):
        """pymongo: collection.create_index(keys, unique=False)"""
        for key in (keys if isinstance(keys, list) else [keys]):
            field = key[0] if isinstance(key, tuple) else key
            self._indexes[field] = {}
            for _id, doc in self._docs.items():
                val = _get_nested(doc, field)
                self._indexes[field].setdefault(val, set()).add(_id)
        return f"idx_{self.name}_{'_'.join(k[0] if isinstance(k,tuple) else k for k in ([keys] if not isinstance(keys,list) else keys))}"

    def _update_indexes(self, doc):
        for field, idx in self._indexes.items():
            val = _get_nested(doc, field)
            idx.setdefault(val, set()).add(doc["_id"])

    def distinct(self, field, filter_dict=None):
        """pymongo: collection.distinct(field, filter)"""
        docs = self.find(filter_dict)
        return list({_get_nested(d, field) for d in docs if _get_nested(d, field) is not None})


class Database:
    def __init__(self, name):
        self.name = name
        self._collections = {}

    def __getitem__(self, name):
        return self._collections.setdefault(name, Collection(name))

    def list_collection_names(self):
        return list(self._collections.keys())


# ── Main ──────────────────────────────────────────────────────

def main():
    rng = random.Random(42)
    db = Database("aiml_platform")

    section("1. DOCUMENT INSERT — SCHEMA FLEXIBILITY")
    docs_col = db["documents"]

    # pymongo equivalent: docs_col.insert_many([...])
    docs = [
        {
            "title": "Attention Is All You Need",
            "type": "paper",
            "metadata": {"year": 2017, "venue": "NeurIPS", "citations": 80000},
            "tags": ["transformer", "attention", "nlp"],
            "embedding_dim": 768,
            "vector": [rng.uniform(-1,1) for _ in range(8)],  # truncated for demo
        },
        {
            "title": "BERT: Pre-training of Deep Bidirectional Transformers",
            "type": "paper",
            "metadata": {"year": 2019, "venue": "NAACL", "citations": 45000},
            "tags": ["bert", "pretraining", "nlp"],
            "embedding_dim": 768,
            "vector": [rng.uniform(-1,1) for _ in range(8)],
        },
        {
            "title": "GPT-4 Technical Report",
            "type": "report",
            "metadata": {"year": 2023, "venue": "OpenAI", "citations": 5000},
            "tags": ["gpt", "llm", "multimodal"],
            "embedding_dim": 1536,
            "vector": [rng.uniform(-1,1) for _ in range(8)],
        },
        # Different schema — no "tags" field, has "author" (schema flexibility)
        {
            "title": "Python for Data Science",
            "type": "book",
            "author": "Wes McKinney",
            "metadata": {"year": 2017, "pages": 541},
            "embedding_dim": 384,
        },
    ]
    ids = docs_col.insert_many(docs)
    print(f"Inserted {len(ids)} documents (no schema enforcement)")
    print(f"  IDs: {ids[:2]}...")

    section("2. QUERY OPERATORS")
    # $gt, $lt, $in
    result = docs_col.find({"metadata.year": {"$gte": 2019}})
    print(f"Year >= 2019 ($gte): {[d['title'] for d in result]}")

    result = docs_col.find({"type": {"$in": ["paper", "report"]}})
    print(f"Type in [paper, report] ($in): {[d['title'] for d in result]}")

    result = docs_col.find({"metadata.citations": {"$gt": 10000}})
    print(f"Citations > 10000: {[d['title'] for d in result]}")

    # $and / $or
    result = docs_col.find({
        "$and": [
            {"metadata.year": {"$gte": 2017}},
            {"type": "paper"}
        ]
    })
    print(f"$and (year>=2017 AND type=paper): {[d['title'] for d in result]}")

    result = docs_col.find({
        "$or": [
            {"type": "book"},
            {"metadata.citations": {"$gt": 50000}}
        ]
    })
    print(f"$or (book OR citations>50k): {[d['title'] for d in result]}")

    section("3. UPDATE OPERATORS")
    col = db["ml_experiments"]
    exp_id = col.insert_one({
        "name": "bert-finetune-v1",
        "status": "running",
        "metrics": {"loss": 2.5, "accuracy": 0.0},
        "epochs_completed": 0,
        "tags": ["bert", "classification"],
        "log": [],
    })
    print(f"Experiment inserted: {exp_id}")

    # $set — update specific fields
    col.update_one({"name": "bert-finetune-v1"}, {"$set": {"status": "epoch_1_done"}})
    # $inc — increment counter
    col.update_one({"name": "bert-finetune-v1"}, {"$inc": {"epochs_completed": 1}})
    # $set nested field
    col.update_one({"name": "bert-finetune-v1"},
                   {"$set": {"metrics.loss": 1.8, "metrics.accuracy": 0.72}})
    # $push — append to array
    col.update_one({"name": "bert-finetune-v1"},
                   {"$push": {"log": {"epoch": 1, "loss": 1.8, "acc": 0.72}}})
    # $addToSet — add to set (no duplicates)
    col.update_one({"name": "bert-finetune-v1"},
                   {"$addToSet": {"tags": "finetuning"}})
    col.update_one({"name": "bert-finetune-v1"},
                   {"$addToSet": {"tags": "bert"}})  # duplicate — should not add

    doc = col.find_one({"name": "bert-finetune-v1"})
    print(f"\nAfter updates:")
    print(f"  status: {doc['status']}")
    print(f"  epochs_completed: {doc['epochs_completed']}")
    print(f"  metrics: {doc['metrics']}")
    print(f"  tags (addToSet deduplicated): {doc['tags']}")
    print(f"  log: {doc['log']}")

    section("4. AGGREGATION PIPELINE")
    # Seed experiment data
    exp_col = db["experiments"]
    models = ["bert", "gpt2", "roberta", "t5"]
    for i in range(40):
        exp_col.insert_one({
            "model": rng.choice(models),
            "lr": rng.choice([1e-5, 2e-5, 5e-5]),
            "batch_size": rng.choice([16, 32, 64]),
            "epochs": rng.randint(3, 10),
            "val_loss": round(rng.uniform(0.1, 2.0), 3),
            "val_acc": round(rng.uniform(0.6, 0.98), 3),
            "dataset": rng.choice(["imdb", "sst2", "yelp"]),
            "status": rng.choice(["done", "done", "done", "failed"]),
        })

    # Group by model: avg accuracy, count
    pipeline = [
        {"$match": {"status": "done"}},
        {"$group": {
            "_id": "$model",
            "n_runs": {"$sum": 1},
            "avg_acc": {"$avg": "$val_acc"},
            "max_acc": {"$max": "$val_acc"},
            "avg_loss": {"$avg": "$val_loss"},
        }},
        {"$sort": {"avg_acc": -1}},
    ]
    results = exp_col.aggregate(pipeline)
    print(f"Model performance summary (aggregation):")
    print(f"  {'Model':10s}  {'Runs':>6}  {'Avg Acc':>8}  {'Max Acc':>8}  {'Avg Loss':>9}")
    print("  " + "-" * 50)
    for r in results:
        print(f"  {r['_id']:10s}  {r['n_runs']:>6}  {r['avg_acc']:>8.4f}  "
              f"{r['max_acc']:>8.4f}  {r['avg_loss']:>9.4f}")

    section("5. INDEXING & DISTINCT")
    # Create index
    idx_name = exp_col.create_index([("model", 1), ("status", 1)])
    print(f"Created index: {idx_name}")

    # Distinct values
    models_found = exp_col.distinct("model")
    datasets = exp_col.distinct("dataset", {"status": "done"})
    print(f"Distinct models: {sorted(models_found)}")
    print(f"Distinct datasets (done runs): {sorted(datasets)}")

    section("6. ARRAY OPERATIONS — $UNWIND")
    tags_col = db["papers_tags"]
    tags_col.insert_many([
        {"title": "Paper A", "tags": ["nlp", "transformer", "bert"]},
        {"title": "Paper B", "tags": ["cv", "resnet", "classification"]},
        {"title": "Paper C", "tags": ["nlp", "rl", "gpt"]},
    ])

    # Count papers per tag
    tag_pipeline = [
        {"$unwind": "$tags"},
        {"$group": {"_id": "$tags", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
    ]
    tag_counts = tags_col.aggregate(tag_pipeline)
    print("Tag frequency (after $unwind):")
    for r in tag_counts:
        print(f"  {r['_id']:15s}: {r['count']}")

    section("7. UPSERT & REPLACE")
    state_col = db["model_state"]
    # Upsert: insert if not exists, update if exists
    for epoch in range(1, 4):
        state_col.update_one(
            {"model_name": "my_transformer"},
            {"$set": {"last_epoch": epoch, "checkpoint": f"ckpt_epoch_{epoch}.pt"},
             "$inc": {"total_epochs": 1}},
            upsert=True
        )
    doc = state_col.find_one({"model_name": "my_transformer"})
    print(f"Upserted model state: {doc}")

    section("8. CAP THEOREM COMPARISON")
    print("""
  Database Consistency vs Availability trade-offs:

  MongoDB (default w:1):
    • AP when network partitions occur
    • Reads from primary: strong consistency
    • Reads from secondary: eventual consistency
    • CP mode: writeConcern majority + readConcern majority

  Cassandra:
    • AP by design (always available)
    • Tunable consistency: ANY, ONE, QUORUM, ALL
    • QUORUM reads+writes ≈ CP (majority of replicas must agree)

  Redis Cluster:
    • CP when partitioned (some slots become unavailable)
    • Sub-millisecond latency for key-value ops
    • Used for caching, feature stores, session state in ML

  For ML workloads:
    • Feature store (read-heavy, low-latency): Redis + Cassandra
    • Experiment tracking (write-heavy, flexible schema): MongoDB
    • Vector search: ChromaDB / Pinecone / Weaviate (vector indexes)
    """)


if __name__ == "__main__":
    main()
