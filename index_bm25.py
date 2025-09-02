import os

from tqdm import tqdm
from whoosh import index
from whoosh.analysis import SpaceSeparatedTokenizer
from whoosh.fields import Schema, TEXT, ID
from whoosh.qparser import QueryParser, OrGroup
from whoosh.scoring import BM25F
from whoosh.writing import AsyncWriter

from pre_process import tokenize

IDX_DIR = "indexes/whoosh"
K1, B = 1.2, 0.75

SCHEMA = Schema(
    pid=ID(stored=True, unique=True),
    text=TEXT(stored=False, analyzer=SpaceSeparatedTokenizer()),
)

IDX_DIR = "indexes/whoosh"
K1, B = 1.2, 0.75

def build_bm25(df, out_dir=IDX_DIR, num_workers=None, limitmb=256, multisegment=True, use_pretok=True):
    """
    num_workers: number of worker processes (default: min(os.cpu_count(), len(df)))
    limitmb: memory per process for indexing buffers
    multisegment=True: faster commit (no big merge); you can ix.optimize() later if desired
    use_pretok=True: pass pre-tokenized text (" ".join(tokenize(...))) if True;
                     else pass raw text and let the schema's analyzer handle it.
    """
    os.makedirs(out_dir, exist_ok=True)
    ix = index.create_in(out_dir, SCHEMA)

    if num_workers is None:
        # conservative default; feel free to set to os.cpu_count()
        num_workers = max(1, os.cpu_count())

    try:
        # Multiprocess writer (Whoosh will spawn worker processes)
        writer = ix.writer(procs=num_workers, limitmb=limitmb, multisegment=multisegment)
    except TypeError:
        # older Whoosh: fall back to single-process + async commit
        writer = AsyncWriter(ix)

    it = zip(df["pid"].astype(str), df["text"].astype(str))
    for pid, text in tqdm(it, total=len(df), desc="Indexing (Whoosh BM25)"):
        if use_pretok:
            writer.add_document(pid=pid, text=" ".join(tokenize(text)))
        else:
            writer.add_document(pid=pid, text=text)

    writer.commit()
    # Optional: merge segments into one (slower; do only if needed)
    # ix.optimize()


def query_bm25(query, topk=10):
    ix = index.open_dir(IDX_DIR)
    qp = QueryParser("text", schema=ix.schema, group=OrGroup)
    q = qp.parse(" ".join(tokenize(query)))
    with ix.searcher(weighting=BM25F(k1=K1, b=B)) as s:
        res = s.search(q, limit=topk)
        return [(hit["pid"], float(hit.score)) for hit in res]
