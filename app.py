import streamlit as st
import joblib
import requests
import os

from dotenv import load_dotenv
from bs4 import BeautifulSoup

# ==========================
# LOAD ENV VARIABLES
# ==========================
load_dotenv()

NEWS_API_KEY = os.getenv("NEWS_API_KEY")

# ==========================
# LOAD MODEL
# ==========================
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# ==========================
# PAGE CONFIG
# ==========================
st.set_page_config(
    page_title="Fake News Detection System",
    page_icon="📰"
)

st.title("📰 Fake News Detection System")
st.write("Detect whether a news article is Real or Fake using Machine Learning")

# ==========================
# PREDICTION FUNCTION
# ==========================
def predict_news(text):
    transformed = vectorizer.transform([text])

    prediction = model.predict(transformed)[0]

    probabilities = model.predict_proba(transformed)[0]

    st.write("Prediction Value:", prediction)
    st.write("Probabilities:", probabilities)

    confidence = max(probabilities) * 100

    return prediction, confidence


# ==========================
# URL EXTRACTION FUNCTION
# ==========================
def extract_article(url):
    try:
        headers = {
            "User-Agent": "Mozilla/5.0"
        }

        response = requests.get(
            url,
            headers=headers,
            timeout=10
        )

        soup = BeautifulSoup(
            response.content,
            "html.parser"
        )

        paragraphs = soup.find_all("p")

        text = " ".join(
            [p.get_text() for p in paragraphs]
        )

        return text

    except Exception:
        return ""


# ==========================
# LIVE NEWS FUNCTION
# ==========================
def get_live_news():

    url = (
        f"https://newsapi.org/v2/top-headlines?"
        f"country=us&pageSize=10&apiKey={NEWS_API_KEY}"
    )

    response = requests.get(url)

    data = response.json()

    return data.get("articles", [])


# ==========================
# SIDEBAR
# ==========================
mode = st.sidebar.selectbox(
    "Choose Mode",
    [
        "Manual Text",
        "Article URL",
        "Live News Feed"
    ]
)

# ==========================
# MANUAL TEXT MODE
# ==========================
if mode == "Manual Text":

    st.subheader("Enter News Text")

    news_text = st.text_area(
        "Paste article content"
    )

    if st.button("Predict"):

        if news_text.strip() == "":
            st.warning("Please enter news text.")

        else:

            prediction, confidence = predict_news(
                news_text
            )

            st.write(
                f"Confidence Score: {confidence:.2f}%"
            )

            if prediction == 1:
                st.success("✅ REAL NEWS")
            else:
                st.error("❌ FAKE NEWS")

# ==========================
# URL MODE
# ==========================
elif mode == "Article URL":

    st.subheader("Analyze News Article URL")

    article_url = st.text_input(
        "Enter article URL"
    )

    if st.button("Fetch & Predict"):

        article_text = extract_article(
            article_url
        )

        if article_text == "":
            st.error(
                "Could not extract article."
            )

        else:

            prediction, confidence = predict_news(
                article_text
            )

            st.write(
                f"Confidence Score: {confidence:.2f}%"
            )

            if prediction == 1:
                st.success("✅ REAL NEWS")
            else:
                st.error("❌ FAKE NEWS")

# ==========================
# LIVE NEWS MODE
# ==========================
elif mode == "Live News Feed":

    st.subheader("🌎 Latest News")

    if st.button("Fetch Latest News"):

        articles = get_live_news()

        for i, article in enumerate(articles):

            st.markdown(
                f"### {article['title']}"
            )

            st.write(
                article.get(
                    "description",
                    "No description available"
                )
            )

            if st.button(
                f"Analyze Article {i+1}",
                key=f"btn_{i}"
            ):

                text = (
                    article["title"]
                    + " "
                    + str(
                        article.get(
                            "description",
                            ""
                        )
                    )
                )

                prediction, confidence = predict_news(
                    text
                )

                st.write(
                    f"Confidence Score: {confidence:.2f}%"
                )

                if prediction == 1:
                    st.success("✅ REAL NEWS")
                else:
                    st.error("❌ FAKE NEWS")

                st.write(
                    f"Source: {article['url']}"
                )

# ==========================
# FOOTER
# ==========================
st.markdown("---")
st.caption(
    "Built using Streamlit, NLP, TF-IDF and Machine Learning"
)