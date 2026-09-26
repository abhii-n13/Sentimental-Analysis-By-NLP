import string

import nltk
from nltk.corpus import stopwords

# Download required NLTK data (only downloads if not already present)
nltk.download("stopwords", quiet=True)

STOP_WORDS = set(stopwords.words("english"))


def remove_punc(txt: str) -> str:
    return txt.translate(str.maketrans("", "", string.punctuation))


def remove_numbers(txt: str) -> str:
    new = ""
    for ch in txt:
        if not ch.isdigit():
            new += ch
    return new


def remove_emojis(txt: str) -> str:
    new = ""
    for ch in txt:
        if ch.isascii():
            new += ch
    return new


def remove_stopwords(txt: str) -> str:
    words = txt.split()
    cleaned = [w for w in words if w not in STOP_WORDS]
    return " ".join(cleaned)


def clean_text(text: str) -> str:
    """Apply the full pipeline, in the same order as the notebook."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = remove_punc(text)
    text = remove_numbers(text)
    text = remove_emojis(text)
    text = remove_stopwords(text)
    return text