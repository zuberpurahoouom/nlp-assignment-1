import math
from index_bm25 import query_bm25
from tqdm.auto import tqdm
import pandas as pd

# ---------- Metrics (from scratch) ----------
def _dcg(rels):
    return sum((2**rel - 1) / math.log2(i + 2) for i, rel in enumerate(rels))

def ndcg_at_k(ranked_pids, rel_dict, k=10):
    rels_at_k = [rel_dict.get(pid, 0) for pid in ranked_pids[:k]]
    dcg = _dcg(rels_at_k)
    ideal = sorted(rel_dict.values(), reverse=True)[:k]
    idcg = _dcg(ideal)
    return 0.0 if idcg == 0 else dcg / idcg

def average_precision_at_k(ranked_pids, rel_set, k=10):
    """AP@k over binary relevance; queries with no relevant docs -> AP=0 (common convention)."""
    num_rel = 0
    ap = 0.0
    for i, pid in enumerate(ranked_pids[:k], start=1):
        if pid in rel_set:
            num_rel += 1
            ap += num_rel / i
    denom = min(len(rel_set), k)
    return 0.0 if denom == 0 else ap / denom

def recall_at_k(ranked_pids, rel_set, k=100):
    if len(rel_set) == 0:
        return 0.0
    hits = sum(1 for pid in ranked_pids[:k] if pid in rel_set)
    return hits / len(rel_set)

def qrels_df_to_dict(qrels_df: pd.DataFrame) -> dict[str, dict[str, int]]:
    # expects columns: qid, pid, rel (strings for qid/pid)
    g = qrels_df.groupby("qid", sort=False)
    return {
        qid: dict(zip(grp["pid"].astype(str), grp["rel"].astype(int)))
        for qid, grp in g
    }

# ---------- Evaluation loop (no TREC file needed) ----------
def evaluate_bm25_in_memory(queries_df, qrels, topk_run=1000, k_ndcg=10, k_map=10, k_rec=100, progress=True):
    """Runs retrieval and returns dict of mean metrics."""
    ndcgs, maps, recalls = [], [], []

    it = queries_df[["qid", "query"]].itertuples(index=False, name=None)
    if progress:
        it = tqdm(it, total=len(queries_df), desc="Evaluating", unit="q")

    for qid, query in it:
        hits = query_bm25(query, topk=topk_run)  # [(pid, score), ...]
        ranked_pids = [pid for pid, _ in hits]

        rel_dict = qrels.get(str(qid), {})
        rel_set  = {pid for pid, rel in rel_dict.items() if rel > 0}

        ndcgs.append(ndcg_at_k(ranked_pids, rel_dict, k=k_ndcg))
        maps.append(average_precision_at_k(ranked_pids, rel_set, k=k_map))
        recalls.append(recall_at_k(ranked_pids, rel_set, k=k_rec))

    return {
        f"ndcg@{k_ndcg}": float(sum(ndcgs) / len(ndcgs)) if ndcgs else 0.0,
        f"map@{k_map}":   float(sum(maps)  / len(maps))  if maps  else 0.0,
        f"recall@{k_rec}":float(sum(recalls)/ len(recalls)) if recalls else 0.0,
    }

