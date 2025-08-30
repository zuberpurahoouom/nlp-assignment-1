import pandas as pd
from tqdm import tqdm

def read_collection(path="collection.tsv", limit=None):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        total = sum(1 for _ in open(path, "r", encoding="utf-8")) if limit is None else limit
        for i, line in enumerate(tqdm(f, total=total, desc="Reading collection")):
            if limit is not None and i >= limit: break
            pid, passage = line.rstrip("\n").split("\t", 1)
            rows.append((pid, passage))
    df = pd.DataFrame(rows, columns=["pid", "text"])
    return df

def read_queries_dev(path="queries.dev.tsv", limit=None):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            qid, query = line.rstrip("\n").split("\t", 1)
            rows.append((qid, query))
    df = pd.DataFrame(rows, columns=["qid", "text"])
    return df