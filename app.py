import streamlit as st
import joblib

# Load saved model and vectorizer
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Page settings
st.set_page_config(
    page_title="Fake News Detection System",
    page_icon="📰"
)

# Title
st.title("📰 Fake News Detection System")

st.write(
    "Enter a news article below and the model will predict whether it is Real or Fake."
)

# Sidebar
st.sidebar.header("Project Details")
st.sidebar.write("Model: Logistic Regression")
st.sidebar.write("Accuracy: 98.59%")
st.sidebar.write("Feature Extraction: TF-IDF")

# User Input
news_text = st.text_area(
    "Enter News Article Text",
    height=250
)

# Predict Button
if st.button("Predict"):

    if news_text.strip() == "":
        st.warning("Please enter some news text.")
    else:

        transformed_text = vectorizer.transform([news_text])

        prediction = model.predict(transformed_text)

        probability = model.predict_proba(transformed_text)

        confidence = max(probability[0]) * 100

        st.write(f"Confidence Score: {confidence:.2f}%")

        if prediction[0] == 1:
            st.success("✅ REAL NEWS")
        else:
            st.error("🚨 FAKE NEWS")