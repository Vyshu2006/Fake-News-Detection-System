📰 Fake News Detection System
📌 Overview

The Fake News Detection System is a Machine Learning + NLP web application that classifies news articles as Real or Fake.
It uses TF-IDF vectorization and Logistic Regression / Naive Bayes models and provides a real-time interactive web interface using Streamlit.

🎯 Objective

To detect and classify fake news articles using machine learning techniques and help users identify misinformation in online content.

🚀 Features
📰 Fake News Detection
✅ Real News Detection
📊 Confidence Score Prediction
📈 Confusion Matrix Visualization
☁️ Word Cloud Analysis (Fake vs Real News)
🤖 Multiple ML Model Comparison
🌐 Interactive Streamlit Web App
📂 Dataset
Fake News Dataset (Fake.csv)
True News Dataset (True.csv)

Total Records: ~44,000+

🧠 Machine Learning Workflow
Data Collection
Data Preprocessing
TF-IDF Vectorization
Train-Test Split
Model Training (Logistic Regression & Naive Bayes)
Model Evaluation
Visualization (Confusion Matrix + Word Clouds)
Deployment using Streamlit
📊 Model Performance
Logistic Regression Accuracy: ~98%
Naive Bayes Accuracy: ~94%
📷 Visualizations
Confusion Matrix

Shows correct and incorrect predictions of the model.

Word Clouds
Fake News Word Cloud
Real News Word Cloud
🛠 Tech Stack
Python
Pandas
Scikit-learn
Streamlit
Matplotlib
Seaborn
WordCloud
Joblib
▶️ How to Run
1. Install dependencies
pip install -r requirements.txt
2. Train model
python train.py
3. Run web app
python -m streamlit run app.py
📁 Project Structure
FakeNewsDetection/
│
├── data/
├── app.py
├── train.py
├── fake_news_model.pkl
├── vectorizer.pkl
├── confusion_matrix.png
├── fake_wordcloud.png
├── real_wordcloud.png
├── requirements.txt
├── README.md
└── screenshots/
👨‍💻 Author

Devalapalli Vaishnavi

⭐ Outcome

This project helps users identify fake news using AI, improving awareness about misinformation on digital platforms.