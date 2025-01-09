import streamlit as st
import pickle

# Load the saved model and vectorizer
with open("model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("vectorizer.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Streamlit app title
st.title("Deceptive Review Detector")
st.write("Enter a review to predict whether it's truthful or deceptive.")

# Input from the user
user_review = st.text_area("Review Text:", "")

# Predict the result
if st.button("Predict"):
    if user_review.strip():
        # Transform the input text using the saved vectorizer
        review_vector = vectorizer.transform([user_review])

        # Predict using the loaded model
        prediction = model.predict(review_vector)[0]
        result = "Deceptive" if prediction == 1 else "Truthful"

        # Display the result
        st.success(f"The review is: {result}")
    else:
        st.error("Please enter a valid review.")

# About section
st.sidebar.header("About")
st.sidebar.write(
    """
    This app uses a machine learning model to classify reviews as **truthful** or **deceptive**.
    The model was trained using Logistic Regression and TF-IDF vectorization.
    """
)
