import os

from tqdm import tqdm
from whoosh import index
from whoosh.analysis import RegexTokenizer, LowercaseFilter
from whoosh.fields import Schema, TEXT, ID
from whoosh.qparser import QueryParser, OrGroup
from whoosh.scoring import BM25F
from whoosh.writing import AsyncWriter

from pre_process import tokenize

# ---------------- cfg ----------------
IDX_DIR = "indexes/whoosh"   # was "indexes/bm25"
K1, B = 1.2, 0.75

SCHEMA = Schema(
    pid=ID(stored=True, unique=True),
    # keep analyzer simple; we pre-tokenize and join with spaces
    text=TEXT(stored=False, analyzer=RegexTokenizer() | LowercaseFilter()),
)

# --------------- build ---------------
def build_bm25(df, out_dir=IDX_DIR):
    os.makedirs(out_dir, exist_ok=True)

    ix = index.create_in(out_dir, SCHEMA)

    writer = AsyncWriter(ix)

    # Pre-tokenize to match your pipeline, then join tokens with spaces
    for pid, text in tqdm(
        zip(df["pid"].astype(str), df["text"].astype(str)),
        total=len(df),
        desc="Indexing (Whoosh BM25)"
    ):
        writer.add_document(pid=pid, text=" ".join(tokenize(text)))
    writer.commit()

# Convenience: direct top-k for interactive use
def query_bm25(query, topk=10):
    ix = index.open_dir(IDX_DIR)
    qp = QueryParser("text", schema=ix.schema, group=OrGroup)
    q = qp.parse(" ".join(tokenize(query)))
    with ix.searcher(weighting=BM25F(k1=K1, b=B)) as s:
        res = s.search(q, limit=topk)
        return [(hit["pid"], float(hit.score)) for hit in res]
