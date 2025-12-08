# app.py

import streamlit as st
from nlp import predict   # we import the function from nlp.py

st.set_page_config(page_title="My NLP App", layout="centered")

st.title("NLP Model Web App")
st.write("Type some text below and let the model predict!")

# 1. Text input from user
user_text = st.text_area("Enter your text here:", height=150)

# 2. Button to run prediction
if st.button("Predict"):
    if user_text.strip() == "":
        st.warning("Please enter some text first.")
    else:
        # 3. Call your model
        result = predict(user_text)

        # 4. Show prediction
        st.success(f"Model prediction: **{result}**")
