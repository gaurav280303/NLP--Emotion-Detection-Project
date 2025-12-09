# app.py
import streamlit as st
from pathlib import Path
import numpy as np
import pandas as pd
import nltk
import string
import joblib
import os

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# -----------------------------
# Resolve paths relative to this file (fixes FileNotFound issues)
# -----------------------------
BASE_DIR = Path(__file__).parent
TRAIN_PATH = BASE_DIR / "train.txt"
MODEL_PATH = BASE_DIR / "model.pkl"
VECT_PATH = BASE_DIR / "vectorizer.pkl"

# -----------------------------
# DOWNLOAD NLTK RESOURCES (resilient)
# -----------------------------
def ensure_nltk():
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords", quiet=True)

ensure_nltk()

try:
    stop_words = set(stopwords.words("english"))
except Exception:
    # fallback: empty set (not ideal but prevents crash)
    stop_words = set()

# -----------------------------
# TEXT PREPROCESSING FUNCTION
# -----------------------------
def preprocess_text(txt: str) -> str:
    """Lowercase, remove punctuation, numbers, non-ascii, stopwords."""
    if not isinstance(txt, str):
        txt = str(txt)

    txt = txt.lower()
    txt = txt.translate(str.maketrans("", "", string.punctuation))
    txt = "".join(ch for ch in txt if not ch.isdigit())
    txt = "".join(ch for ch in txt if ch.isascii())

    tokens = word_tokenize(txt)
    tokens = [w for w in tokens if w not in stop_words]

    return " ".join(tokens)

# -----------------------------
# LOAD & PREPARE DATA
# -----------------------------
@st.cache_data(show_spinner=True)
def load_and_prepare_data(train_path: Path):
    if not train_path.exists():
        raise FileNotFoundError(f"train file not found at {train_path}")
    df = pd.read_csv(train_path, sep=";", header=None, names=["text", "emotion"])
    df["emotion"] = df["emotion"].astype(str).str.strip()

    # create consistent mappings
    unique_emotions = df["emotion"].unique()
    emotion_to_num = {emo: i for i, emo in enumerate(unique_emotions)}
    num_to_emotion = {v: k for k, v in emotion_to_num.items()}

    df["emotion_num"] = df["emotion"].map(emotion_to_num)
    df["clean_text"] = df["text"].apply(preprocess_text)

    return df, emotion_to_num, num_to_emotion

# -----------------------------
# TRAIN THE MODEL (fallback)
# -----------------------------
@st.cache_resource(show_spinner=True)
def train_model_from_df(df):
    X = df["clean_text"]
    y = df["emotion_num"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_tfidf, y_train)

    accuracy = model.score(X_test_tfidf, y_test)

    return model, vectorizer, accuracy

# -----------------------------
# HIGH LEVEL: load model (preferred) or train (fallback)
# -----------------------------
def get_model_and_vectorizer():
    # 1) If pre-saved artifacts exist, load them (fast)
    if MODEL_PATH.exists() and VECT_PATH.exists():
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECT_PATH)
        # also try to load class mapping
        mapping_path = BASE_DIR / "num_to_emotion.pkl"
        if mapping_path.exists():
            num_to_emotion = joblib.load(mapping_path)
        else:
            # if mapping available not saved, we will create after training fallback
            num_to_emotion = None
        return model, vectorizer, num_to_emotion, None

    # 2) Else fallback: train from train.txt (only for testing)
    df, emotion_to_num, num_to_emotion = load_and_prepare_data(TRAIN_PATH)
    model, vectorizer, accuracy = train_model_from_df(df)

    # Save artifacts for next runs (optional)
    try:
        joblib.dump(model, MODEL_PATH)
        joblib.dump(vectorizer, VECT_PATH)
        joblib.dump(num_to_emotion, BASE_DIR / "num_to_emotion.pkl")
    except Exception:
        # ignore saving errors (e.g., read-only FS) but warn
        st.warning("Could not save model files to disk; app will retrain on restart.")

    return model, vectorizer, num_to_emotion, accuracy

# -----------------------------
# STREAMLIT UI
# -----------------------------
def main():
    st.title("Emotion Detection NLP App")
    st.write("Enter a sentence and the model will predict the emotion.")

    with st.spinner("Loading model (or training if no saved model found)..."):
        model, vectorizer, num_to_emotion, accuracy = get_model_and_vectorizer()

    if accuracy is not None:
        st.success(f"Model trained. Test accuracy: {accuracy:.2%}")
    else:
        st.success("Model loaded from saved artifacts.")

    user_text = st.text_area(
        "Type your sentence here:",
        placeholder="Example: I am feeling very happy today!"
    )

    if st.button("Predict Emotion"):
        if user_text.strip() == "":
            st.warning("Please enter some text.")
        else:
            clean = preprocess_text(user_text)
            vec = vectorizer.transform([clean])
            pred_num = model.predict(vec)[0]

            # build mapping if missing
            if num_to_emotion is None:
                # try to recover mapping from saved file
                mapping_path = BASE_DIR / "num_to_emotion.pkl"
                if mapping_path.exists():
                    num_to_emotion = joblib.load(mapping_path)
                else:
                    st.error("Label mapping is missing.")
                    return

            pred_label = num_to_emotion[pred_num]

            st.subheader("Prediction:")
            st.info(f"Emotion: **{pred_label}**")

if __name__ == "__main__":
    main()
