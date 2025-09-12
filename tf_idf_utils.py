from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Dict, Tuple, Optional,  Sequence
from tqdm import tqdm
from scipy import sparse
import numpy as np
import pandas as pd

def retrieve_topN_for_queries(
    vectorizer: TfidfVectorizer,
    doc_matrix: sparse.csr_matrix,
    doc_ids: Sequence[str],
    queries_df: pd.DataFrame,
    topN: int = 10
) -> Dict[str, List[Tuple[str, float]]]:
    """Return topN doc ids per query using cosine similarity (dot product on L2-normalized rows).

    Parameters
    ----------
    topN : int
        Number of documents to keep per query.
    """
    doc_ids = list(map(str, list(doc_ids)))
    qids = queries_df["qid"].astype(str).values
    qtexts = queries_df["query"].astype(str).values

    results: Dict[str, List[Tuple[str, float]]] = {}
    for qid, qtext in tqdm(zip(qids, qtexts), total=len(qids), desc="Retrieving"):
        qvec = vectorizer.transform([qtext])            # (1 x V)
        scores = (doc_matrix @ qvec.T).toarray().ravel() # (D,)

        if scores.size <= topN:
            idx = np.argsort(-scores)
        else:
            idx_part = np.argpartition(-scores, topN-1)[:topN]
            idx = idx_part[np.argsort(-scores[idx_part])]
        ranked = [(doc_ids[i], float(scores[i])) for i in idx]
        results[qid] = ranked[:topN]

    return results