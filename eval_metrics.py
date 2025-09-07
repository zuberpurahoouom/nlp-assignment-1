import math

from tqdm.auto import tqdm
from whoosh import index
from whoosh.qparser import QueryParser, OrGroup
from whoosh.scoring import BM25F
from pre_process import tokenize


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

IDX_DIR = "indexes/whoosh"
K1, B = 1.2, 0.75

def evaluate_bm25(queries_df,
                             qrels_df,
                             topk_run=1000,
                             k_ndcg=10,
                             k_map=10,
                             k_rec=100):
    """Evaluate BM25 retrieval using Whoosh index.

    Parameters
    ----------
    queries_df : DataFrame with columns ['qid','query'].
    qrels_df : DataFrame with columns ['qid','pid','rel']; rel>0 indicates relevance.
    topk_run : number of documents to retrieve per query.
    k_ndcg, k_map, k_rec : cutoffs for metrics.
    """
    qr = qrels_df.astype({"qid":str,"pid":str,"rel":int})
    grouped = qr.groupby("qid", sort=False)
    qrels_dict = {
        qid: {pid: rel for pid, rel in zip(g["pid"], g["rel"]) if rel > 0}
        for qid, g in grouped
    }

    ix = index.open_dir(IDX_DIR)
    ndcgs, maps, recalls = [], [], []
    with ix.searcher(weighting=BM25F(k1=K1, b=B)) as searcher:
        qp = QueryParser("text", schema=ix.schema, group=OrGroup)
        it = queries_df[["qid","query"]].itertuples(index=False, name=None)
        it = tqdm(it, total=len(queries_df), desc="Evaluating", unit="q")

        for qid, query in it:
            q = qp.parse(" ".join(tokenize(query)))
            results = searcher.search(q, limit=topk_run)
            ranked_pids = [r["pid"] for r in results]

            rel_dict = qrels_dict.get(str(qid), {})
            rel_set  = {pid for pid, rel in rel_dict.items() if rel > 0}

            ndcgs.append(ndcg_at_k(ranked_pids, rel_dict, k=k_ndcg))
            maps.append(average_precision_at_k(ranked_pids, rel_set, k=k_map))
            recalls.append(recall_at_k(ranked_pids, rel_set, k=k_rec))

    return {
        f"ndcg@{k_ndcg}": float(sum(ndcgs)/len(ndcgs)) if ndcgs else 0.0,
        f"map@{k_map}":   float(sum(maps)/len(maps))   if maps  else 0.0,
        f"recall@{k_rec}":float(sum(recalls)/len(recalls)) if recalls else 0.0,
        "num_queries": queries_df.shape[0]
    }
