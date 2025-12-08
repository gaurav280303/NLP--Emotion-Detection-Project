import streamlit as st
import numpy as np
import pandas as pd
import string
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# -----------------------------
# 1. DOWNLOAD NLTK RESOURCES
# -----------------------------
# These are needed for tokenization + stopwords
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)

stop_words = set(stopwords.words("english"))

# -----------------------------
# 2. TEXT PREPROCESSING FUNCTION
# -----------------------------
def preprocess_text(txt: str) -> str:
    """
    Cleans the input text:
    - lowercase
    - remove punctuation
    - remove numbers
    - remove non-ASCII (emojis etc.)
    - remove stopwords
    """
    if not isinstance(txt, str):
        txt = str(txt)

    # 1. lowercase
    txt = txt.lower()

    # 2. remove punctuation
    txt = txt.translate(str.maketrans("", "", string.punctuation))

    # 3. remove digits
    txt = "".join(ch for ch in txt if not ch.isdigit())

    # 4. remove non-ascii (emojis etc.)
    txt = "".join(ch for ch in txt if ch.isascii())

    # 5. remove stopwords
    tokens = word_tokenize(txt)
    tokens = [w for w in tokens if w not in stop_words]

    return " ".join(tokens)

# -----------------------------
# 3. LOAD DATA + ENCODE LABELS
# -----------------------------
@st.cache_data(show_spinner=True)
def load_and_prepare_data():
    # train.txt should be in the same folder as this file
    df = pd.read_csv("train.txt", sep=";", header=None, names=["text", "emotion"])

    # strip spaces from emotion labels
    df["emotion"] = df["emotion"].astype(str).str.strip()

    # create emotion -> number mapping
    unique_emotions = df["emotion"].unique()
    emotion_to_num = {emo: i for i, emo in e
