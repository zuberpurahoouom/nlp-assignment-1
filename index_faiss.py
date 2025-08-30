import os, pickle, numpy as np, faiss, torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

MODEL = "multi-qa-MiniLM-L6-cos-v1"  # 384-dim

def _batch_encode(texts, model, batch=256):
    embs = []
    with torch.inference_mode():
        for i in tqdm(range(0, len(texts), batch), desc="Encoding"):
            enc = model.encode(texts[i:i+batch], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
            embs.append(enc)
    return np.vstack(embs)

def build_faiss(df, out_dir="indexes/faiss"):
    os.makedirs(out_dir, exist_ok=True)
    model = SentenceTransformer(MODEL)
    embs = _batch_encode(df["text"].tolist(), model)
    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine since normalized
    index.add(embs.astype("float32"))
    faiss.write_index(index, f"{out_dir}/index.faiss")
    with open(f"{out_dir}/pids.pkl","wb") as f:
        pickle.dump(df["pid"].tolist(), f)
    with open(f"{out_dir}/meta.pkl","wb") as f:
        pickle.dump({"model": MODEL, "dim": dim}, f)

def query_faiss(query, topk=10, idx_dir="indexes/faiss"):
    index = faiss.read_index(f"{idx_dir}/index.faiss")
    with open(f"{idx_dir}/pids.pkl","rb") as f:
        pids = pickle.load(f)
    with open(f"{idx_dir}/meta.pkl","rb") as f:
        meta = pickle.load(f)
    model = SentenceTransformer(meta["model"])
    q = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    sims, ids = index.search(q.astype("float32"), topk)
    return [(pids[i], float(s)) for i, s in zip(ids[0], sims[0])]

