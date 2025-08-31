import pickle
from rank_bm25 import BM25Okapi
import os
import numpy as np
import concurrent.futures
from tqdm import tqdm
from pre_process import tokenize

idx_dir = "indexes/bm25"
# At module level
bm25 = None
pids = None

def build_bm25(df, out_dir="indexes/bm25"):
    os.makedirs(out_dir, exist_ok=True)
    tokenized_corpus = [tokenize(t) for t in tqdm(df["text"], desc="Tokenizing")]
    bm25 = BM25Okapi(tokenized_corpus, k1=1.2, b=0.75)
    with open(f"{out_dir}/bm25.pkl","wb") as f:
        pickle.dump(bm25, f)
    with open(f"{out_dir}/pids.pkl","wb") as f:
        pickle.dump(df["pid"].tolist(), f)

def process_query(args):
    global bm25, pids
    if bm25 is None:
        with open(f"{idx_dir}/bm25.pkl", "rb") as f:
            bm25 = pickle.load(f)
    if pids is None:
        with open(f"{idx_dir}/pids.pkl", "rb") as f:
            pids = pickle.load(f)
    qid, q, topk, tag = args
    hits = query_bm25(q, topk=topk, bm25=bm25, pids=pids)
    return [
        f"{qid}\tQ0\t{pid}\t{rank}\t{score:.6f}\t{tag}"
        for rank, (pid, score) in enumerate(hits, start=1)
    ]

def write_trec_run(queries, out_path, topk=1000, tag="bm25"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)  # Ensure directory exists
    args_list = [(qid, q, topk, tag) for qid, q, _ in queries.itertuples(index=False)]
    lines = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for result in tqdm(executor.map(process_query, args_list), total=len(args_list), desc="Processing queries"):
            lines.extend(result)
    with open(out_path, "w") as f:
        f.write("\n".join(lines))

def query_bm25(query, topk=10, bm25=None, pids=None):
    scores = bm25.get_scores(tokenize(query))
    top = np.argpartition(-scores, topk)[:topk]
    top = top[np.argsort(-scores[top])]
    return [(pids[i], float(scores[i])) for i in top]
