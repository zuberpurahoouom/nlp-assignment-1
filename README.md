# Neural & Classical Information Retrieval Pipeline

## Overview
End-to-end Information Retrieval (IR) workflow combining classical lexical methods (TF-IDF, BM25) and neural dense / cross-encoder models (DPR + MiniLM cross-encoder) for passage retrieval and re-ranking. The project:
- Builds a reusable TSV-based corpus and query sets.
- Indexes the corpus with TF-IDF and Whoosh BM25.
- Trains / evaluates a Dual Encoder (DPR-style) to produce dense retrieval results.
- Trains a Cross-Encoder (MiniLM) to re-rank top passages from DPR.
- Computes standard IR metrics (nDCG@K, MAP@K, Recall@K).

## Repository Structure (Key Files)
| File / Dir | Purpose |
|------------|---------|
| `init-dataset.ipynb` | Generates / normalizes base TSV corpus & query files. |
| `tf-idf.ipynb` | Builds TF-IDF vectorizer + sparse matrix; retrieves top N per query. |
| `bm25.ipynb` | Builds Whoosh BM25 index and runs lexical retrieval. |
| `bm25_utils.py` | Helper functions for BM25 indexing & querying (Whoosh backend). |
| `tf_idf_utils.py` | TF-IDF retrieval utilities (vectorization + top-N scoring). |
| `dpr_training.ipynb` | Trains + evaluates DPR (dual encoder) using Hugging Face models; produces top candidate passages. |
| `cross_encoder.ipynb` | Trains MiniLM (cross-encoder) on query–passage pairs. |
| `rerank.ipynb` | Uses trained cross-encoder to re-rank DPR top-k outputs. |
| `eval_metrics.py` | nDCG, MAP, Recall metric implementations. |
| `qrels.train.tsv`, `qrels.dev.tsv`, `qrels_for_eval.tsv` | Relevance judgments. |
| `queries.*.tsv` | Query splits (train/dev/eval/test). |
| `qidpidtriples.*.tsv` | Training triples for DPR (query, pos pid, neg pid). |
| `indexes/` | Whoosh BM25 + (optionally) FAISS / dense indexes. |
| `cross-encoder-model/`, `dpr_context_encoder/`, `dpr_question_encoder/` | Saved model checkpoints. |
| `test-queries.ipynb` | Ad-hoc qualitative inspection of specific queries / corrections. |

## Data Initialization
Run first to materialize standardized corpus & query files.

Notebook: `init-dataset.ipynb`
Outputs typically include (depending on code inside the notebook):
- `collection.tsv` (passage id, text)
- Query splits: `queries.train.tsv`, `queries.dev.tsv`
- Relevance files: `qrels.train.tsv`, `qrels.dev.tsv`, `qrels_for_eval.tsv` (crafted by us)
- Training triples: `qidpidtriples.*.tsv`

Ensure resulting TSV columns follow expected schema (commonly: `pid\ttext`, `qid\tquery`, qrels: `qid\t0\tpid\trelevance`).

## Environment Setup
Example (adjust versions as needed):
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -U numpy scipy pandas scikit-learn tqdm whoosh faiss-cpu torch transformers sentence-transformers joblib
```
(Install CUDA-specific PyTorch if GPU is available.)

## Pipeline Execution Order
1. Dataset Initialization: `init-dataset.ipynb`
2. TF-IDF Retrieval: `tf-idf.ipynb`
3. BM25 Retrieval: `bm25.ipynb`
4. DPR Training & Dense Retrieval: `dpr_training.ipynb`
5. Cross-Encoder Training (MiniLM): `cross_encoder.ipynb`
6. Re-Ranking with Cross-Encoder: `rerank.ipynb`
7. (Optional) Exploratory tests: `test-queries.ipynb`

### 1. TF-IDF (`tf-idf.ipynb`)
Core steps normally covered in the notebook:
- Load `collection.tsv` into a DataFrame with columns (`pid`, `text`).
- Fit `TfidfVectorizer` (recommended: lowercase, token pattern matching alphanumerics, remove stopwords if desired).
- Persist artifacts: `tfidf_vectorizer.joblib`, `tfidf_doc_matrix.joblib`.
- For each query split (`queries.dev.tsv`, etc.), call `retrieve_topN_for_queries` (from `tf_idf_utils.py`) to get ranked lists.

`retrieve_topN_for_queries` details:
- Computes cosine similarity via dot product on L2-normalized TF-IDF rows.
- Uses partial argpartition for efficiency.
- Returns dict: `qid -> [(pid, score), ...]`.

### 2. BM25 (`bm25.ipynb`)
- Uses Whoosh schema defined in `bm25_utils.py`.
- Tokenization: lowercasing + regex `[a-z0-9]+'[a-z0-9]+|[a-z0-9]+`, stopword removal (`sklearn` English list), length > 1.
- Index path: `indexes/whoosh`.
- Retrieval via `query_bm25(query, topk=10)` returning list of `(pid, score)`.
- Parameters: `k1=1.2`, `b=0.75` (BM25F weighting object).

### 3. DPR Training & Retrieval (`dpr_training.ipynb`)
Typical flow:
- Load training triples (`qidpidtriples.*.tsv`).
- Initialize DPR question & context encoders from Hugging Face (checkpoints saved under `dpr_question_encoder/`, `dpr_context_encoder/`).
- Train with in-batch negatives.
- Encode corpus passages → store dense matrix / FAISS index (directory: `indexes/faiss`).
- Encode queries → retrieve top-k (e.g., top 10) passage ids for each query; save to a TSV (`dpr_results.tsv`).

### 4. Cross-Encoder Training (`cross_encoder.ipynb`)
- Model: MiniLM variant fine-tuned on (query, passage, relevance) pairs.
- Inputs were generated from DPR top-k plus relevance labels from qrels.
- Output: directory `cross-encoder-model/` with `config.json`, `model.safetensors`, tokenizer files.

### 5. Re-Ranking (`rerank.ipynb`)
- Load DPR retrieval results (top 10 per query).
- For each candidate pair (query, passage), score with cross-encoder.
- Sort by cross-encoder score → produce improved ranking, saved as `dpr_reranked_cross.tsv`.
- Evaluate using metrics below.

### 6. Test Queries (`test-queries.ipynb`)
- Manual probing: inspect whether cross-encoder corrects DPR ordering for difficult or ambiguous queries.
- Compare models accuracy on same query

## Metrics
Implemented in `eval_metrics.py`:
- nDCG@K: Discounted cumulative gain normalized by ideal DCG.
- MAP@K (mean average precision at K): Average precision truncated at K for each query, then calculate mean across queries.
- Recall@K: Fraction of relevant documents retrieved in top K.

Pseudocode usage example:
```python
from eval_metrics import ndcg_at_k, average_precision_at_k, recall_at_k

# ranked_pids: list of passage ids from a system
# rel_dict: {pid: graded_relevance}
# rel_set: set of relevant pids (binary)
ndcg10 = ndcg_at_k(ranked_pids, rel_dict, k=10)
map10 = average_precision_at_k(ranked_pids, rel_set, k=10)
recall50 = recall_at_k(ranked_pids, rel_set, k=50)
```

## Expected Artifacts & Intermediate Files
- Vectorization: `tfidf_vectorizer.joblib`, `tfidf_doc_matrix.joblib`
- Lexical index: `indexes/whoosh/*`
- Dense encoders: `dpr_question_encoder/`, `dpr_context_encoder/`
- Cross-encoder: `cross-encoder-model/`
- Retrieval outputs: `dpr_results.tsv`, `dpr_reranked_cross.tsv`
- Evaluation sets: `qrels_for_eval.tsv`, `queries.dev.tsv`

## Reproducibility Tips
- Fix random seeds (NumPy, PyTorch) inside notebooks where training occurs.
- Hugging Face checkpoints used (e.g., `facebook/dpr-question_encoder-single-nq-base`).
- Maintain consistent tokenization between training and inference for DPR.
