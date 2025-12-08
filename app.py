import streamlit as st
import numpy as np
import pandas as pd
import nltk
import string

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# -----------------------------
# DOWNLOAD NLTK RESOURCES
# -----------------------------
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)

stop_words = set(stopwords.words("english"))

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
def load_and_prepare_data():
    df = pd.read_csv("train.txt", sep=";", header=None, names=["text", "emotion"])
    df["emotion"] = df["emotion"].astype(str).str.strip()

    # Unique emotion labels
    unique_emotions = df["emotion"].unique()

    # FIXED VERSION — no syntax error:
    emotion_to_num = {emo: i for i, emo in enumerate(unique_emotions)}
    num_to_emotion = {v: k for k, v in emotion_to_num.items()}

    df["emotion_num"] = df["emotion"].map(emotion_to_num)

    df["clean_text"] = df["text"].apply(preprocess_text)

    return df, emotion_to_num, num_to_emotion

# -----------------------------
# TRAIN THE MODEL
# -----------------------------
@st.cache_resource(show_spinner=True)
def train_model():
    df, emotion_to_num, num_to_emotion = load_and_prepare_data()

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

    return model, vectorizer, num_to_emotion, accuracy

# -----------------------------
# STREAMLIT UI
# -----------------------------
def main():
    st.title("Emotion Detection NLP App 😃😞😡")
    st.write("Enter a sentence and the model will predict the emotion.")

    with st.spinner("Training / Loading model..."):
        model, vectorizer, num_to_emotion, accuracy = train_model()

    st.success(f"Model Ready! Accuracy: {accuracy:.2%}")

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
            pred_label = num_to_emotion[pred_num]

            st.subheader("Prediction:")
            st.info(f"Emotion: **{pred_label}**")

if __name__ == "__main__":
    main()
