import joblib
# suppose `logistic_model` is your trained model and `tfidf_vectorizer` is the vectorizer
joblib.dump(logistic_model, "model.pkl")
joblib.dump(tfidf_vectorizer, "vectorizer.pkl")

# Save mapping too (if you computed emotion_numbers)
joblib.dump(emotion_numbers, "emotion_to_num.pkl")
# For quick lookup by numeric index -> emotion string:
num_to_emotion = {v:k for k,v in emotion_numbers.items()}
joblib.dump(num_to_emotion, "num_to_emotion.pkl")
