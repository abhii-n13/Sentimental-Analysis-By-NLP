"""
preprocessing.py
-----------------
Text cleaning that mirrors the steps used in nlp.ipynb BEFORE the text is
passed into the pipeline:
  1. lowercase
  2. remove punctuation
  3. remove numbers
  4. remove emojis / non-ASCII characters
  5. remove English stopwords (NLTK)

Note: emo_model.pkl is a full sklearn Pipeline (TfidfVectorizer + model),
but the pipeline does NOT include this custom cleaning step — that was
done on the dataframe with df['text'].apply(...) before fitting. So this
file must still run before calling pipeline.predict().
"""

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