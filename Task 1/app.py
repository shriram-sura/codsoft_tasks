import streamlit as st
import joblib
import re


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

tfidf = joblib.load("tfidf_vectorizer.pkl")
model = joblib.load("movie_genre_model.pkl")


# --------------------------------------------------
# TEXT CLEANING
# --------------------------------------------------

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


# --------------------------------------------------
# PAGE SETTINGS
# --------------------------------------------------

st.set_page_config(
    page_title="Movie Genre Classifier",
    page_icon="🎬",
    layout="centered"
)


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("🎬 Movie Genre Classifier")

st.write(
    "Enter a movie plot or description and the machine learning "
    "model will predict its genre."
)


# --------------------------------------------------
# PROJECT STATISTICS
# --------------------------------------------------

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Training Movies", "54,214")

with col2:
    st.metric("Genres", "27")

with col3:
    st.metric("Test Accuracy", "58.63%")


st.markdown("---")


# --------------------------------------------------
# INPUT
# --------------------------------------------------

st.subheader("🎥 Enter Movie Description")

movie_description = st.text_area(
    "Movie Description",
    height=180,
    placeholder=(
        "Example: A detective investigates a mysterious murder "
        "in a small town..."
    ),
    label_visibility="collapsed"
)


# --------------------------------------------------
# EXAMPLE
# --------------------------------------------------

st.caption(
    "Example: A detective investigates a series of murders "
    "and uncovers a dangerous criminal conspiracy."
)


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if st.button(
    "🎯 Predict Genre",
    use_container_width=True
):

    if movie_description.strip() == "":
        st.warning("Please enter a movie description.")

    else:

        cleaned_description = clean_text(
            movie_description
        )

        text_vector = tfidf.transform(
            [cleaned_description]
        )

        prediction = model.predict(
            text_vector
        )[0]

        st.success(
            f"🎬 Predicted Genre: {prediction.upper()}"
        )


# --------------------------------------------------
# MODEL INFORMATION
# --------------------------------------------------

st.markdown("---")

st.subheader("📊 Model Information")

st.write(
    "**Feature Extraction:** TF-IDF"
)

st.write(
    "**Classifier:** Linear Support Vector Machine (SVM)"
)

st.write(
    "**Number of Genres:** 27"
)

st.write(
    "**Final Test Accuracy:** 58.63%"
)


# --------------------------------------------------
# ABOUT
# --------------------------------------------------

st.markdown("---")

st.subheader("ℹ️ About the Project")

st.write(
    "This project is an NLP-based multi-class classification "
    "system. The model learns patterns from movie plot "
    "descriptions and predicts one of 27 possible genres."
)

st.write(
    "Three major classifiers were evaluated during development: "
    "Naive Bayes, Logistic Regression, and Linear SVM. "
    "Linear SVM achieved the best validation performance among "
    "the primary models tested."
)