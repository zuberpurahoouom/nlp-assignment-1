import math

def _dcg(rels):
    return sum((2**rel - 1) / math.log2(i + 2) for i, rel in enumerate(rels))

def ndcg_at_k(ranked_pids, rel_dict, k=10):
    rels_at_k = [rel_dict.get(pid, 0) for pid in ranked_pids[:k]]
    dcg = _dcg(rels_at_k)
    ideal = sorted(rel_dict.values(), reverse=True)[:k]
    idcg = _dcg(ideal)
    return 0.0 if idcg == 0 else dcg / idcg

def average_precision_at_k(ranked_pids, rel_set, k=10):
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
