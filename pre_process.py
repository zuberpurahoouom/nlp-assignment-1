import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

STOPWORDS = set(ENGLISH_STOP_WORDS)
TOKEN_RE = re.compile(r"[a-z0-9]+'[a-z0-9]+|[a-z0-9]+")

def tokenize(text: str):
    text = text.lower()
    tokens = TOKEN_RE.findall(text)
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]

def normalize_for_vectorizers(text: str):
    # For TF-IDF we’ll pass a custom tokenizer; this helper is here if you want to pre-join tokens.
    return " ".join(tokenize(text))

# print(tokenize("This is a sample text, with punctuation! Let's see how it works."))  # Example usage
# print(normalize_for_vectorizers("This is a sample text, with punctuation! Let's see how it works."))  # Example usage