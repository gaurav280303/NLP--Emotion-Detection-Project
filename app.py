import os
import re

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# 1️⃣ Simple text cleaning – NO NLTK, so no LookupError
def clean_text(text: str) -> str:
    # lowercase
    text = text.lower()
    # keep only letters and spaces
    text = re.sub(r"[^a-z\s]", " ", text)
    # remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


# 2️⃣ Load and prepare data (cached)
@st.cache_data
def load_and_prepare_data():
    # make path robust so it works on Streamlit Cloud too
    here = os.path.dirname(__file__)
    data_path = os.path.join(here, "train.txt")

    # your file: "text;emotion" format
    df = pd.read_csv(data_path, sep=";", header=None, names=["text", "emotion"])

    # clean text
    df["clean_text"] = df["text"].astype(str).apply(clean_text)

    # make label mapping
    emotions = sorted(df["emotion"].unique())
    emotion_to_num = {emo: i for i, emo in enumerate(emotions)}
    num_to_emotion = {i: emo for emo, i in emotion_to_num.items()}

    df["label"] = df["emotion"].map(emotion_to_num)

    return df, emotion_to_num, num_to_emotion


# 3️⃣ Train model (cached, runs once per restart)
@st.cache_resource
def train_model():
    df, emotion_to_num, num_to_emotion = load_and_prepare_data()

    X = df["clean_text"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # text to numbers
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=5000,
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # simple classifier
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)

    return model, vectorizer, num_to_emotion, acc


# 4️⃣ Predict function
def predict_emotion(model, vectorizer, num_to_emotion, text: str):
    clean = clean_text(text)
    X_vec = vectorizer.transform([clean])

    probs = model.predict_proba(X_vec)[0]
    pred_idx = int(np.argmax(probs))
    emotion = num_to_emotion[pred_idx]
    confidence = float(probs[pred_idx])

    return emotion, confidence


# 5️⃣ Streamlit UI
def main():
    st.title("Emotion Detection NLP App")

    st.write(
        "Type a message below and the model will predict the **emotion**."
    )

    # train model (cached)
    model, vectorizer, num_to_emotion, acc = train_model()

    st.sidebar.header("Model Info")
    st.sidebar.write(f"Validation accuracy: **{acc:.2%}**")
    st.sidebar.write("Uses TF-IDF + Logistic Regression.")

    user_text = st.text_area("Enter your text here:")

    if st.button("Predict emotion"):
        if not user_text.strip():
            st.warning("Please type something first.")
        else:
            emotion, conf = predict_emotion(
                model, vectorizer, num_to_emotion, user_text
            )
            st.success(f"Predicted emotion: **{emotion}**")
            st.write(f"Confidence: `{conf * 100:.1f}%`")


if __name__ == "__main__":
    main()
