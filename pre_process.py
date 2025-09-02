import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

STOPWORDS = set(ENGLISH_STOP_WORDS)
TOKEN_RE = re.compile(r"[a-z0-9]+'[a-z0-9]+|[a-z0-9]+")

def tokenize(text: str):
    text = text.lower()
    tokens = TOKEN_RE.findall(text)
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]
