import streamlit as st
import joblib
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from preprocess import clean_text

# Page config
st.set_page_config(
    page_title="Amazon Sentiment Analyser",
    page_icon="🛍️",
    layout="centered"
)

# Load models
@st.cache_resource
def load_models():
    base = os.path.join(os.path.dirname(__file__), '..', 'models')
    model = joblib.load(os.path.join(base, 'logistic_model.pkl'))
    vectorizer = joblib.load(os.path.join(base, 'tfidf_vectorizer.pkl'))
    return model, vectorizer

model, vectorizer = load_models()

# Header
st.title("🛍️ Amazon Review Sentiment Analyser")
st.markdown("This app uses a machine learning model trained on **500,000+ Amazon reviews** to predict whether a review is positive or negative.")
st.divider()

# Input section
st.subheader("📝 Enter a Review")
user_input = st.text_area(
    label="Paste any product review below:",
    height=150,
    placeholder="e.g. This product was absolutely amazing, great quality and fast delivery..."
)

col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    analyse = st.button("🔍 Analyse Sentiment", use_container_width=True)

# Prediction
if analyse:
    if not user_input.strip():
        st.warning("⚠️ Please enter a review before analysing.")
    else:
        cleaned = clean_text(user_input)
        vectorised = vectorizer.transform([cleaned])
        prediction = model.predict(vectorised)[0]
        confidence = model.predict_proba(vectorised)[0]

        st.divider()
        st.subheader("📊 Results")

        if prediction == 1:
            st.success("✅ **Positive Sentiment** — This review appears to be positive.")
        else:
            st.error("❌ **Negative Sentiment** — This review appears to be negative.")

        col1, col2 = st.columns(2)
        col1.metric(
            label="😊 Positive Confidence",
            value=f"{confidence[1]:.1%}"
        )
        col2.metric(
            label="😞 Negative Confidence",
            value=f"{confidence[0]:.1%}"
        )

        st.markdown("**Confidence Score:**")
        st.progress(float(confidence[1]))
        st.caption(f"The model is {max(confidence):.1%} confident in this prediction.")

# Footer
st.divider()
st.markdown(
    """
    <div style='text-align: center; color: grey; font-size: 0.85em;'>
    Trained on Amazon Fine Food Reviews dataset | 
    Logistic Regression + TF-IDF | 94% Accuracy
    </div>
    """,
    unsafe_allow_html=True
)