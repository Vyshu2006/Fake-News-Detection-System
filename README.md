📰 Fake News Detection System
📌 Overview

The Fake News Detection System is a Machine Learning and Natural Language Processing (NLP) based web application that classifies news articles as Real or Fake. The system analyzes textual content using TF-IDF vectorization and machine learning classification models to help identify misinformation.

The application supports manual news analysis, URL-based article analysis, and live news fetching using News APIs.

🎯 Objective

The objective of this project is to detect fake news by applying machine learning techniques on news articles and provide users with an easy-to-use interface for real-time prediction and analysis.

🚀 Features
Core Features
Fake News Detection
Real News Detection
Dataset Preprocessing
TF-IDF Text Vectorization
Machine Learning Classification
Model Evaluation
Advanced Features
Manual News Text Prediction
News Article URL Analysis
Live News Fetching using News API
Confidence Score Display
Multiple Prediction Modes
Interactive Streamlit Interface
Visual Analytics
Confusion Matrix Visualization
Fake News Word Cloud
Real News Word Cloud
Model Performance Comparison
📂 Dataset

The project uses:

Fake.csv
True.csv

Dataset Size:

Fake News Articles: 23,481
Real News Articles: 21,417

Total Articles: ~44,000+

🧠 Machine Learning Workflow
Load Fake and True news datasets
Assign labels to each class
Merge datasets
Preprocess text data
Convert text into numerical vectors using TF-IDF
Split data into training and testing sets
Train machine learning models
Evaluate model performance
Save trained model and vectorizer
Deploy using Streamlit
🤖 Models Used
Logistic Regression

Used as the primary classification model.

Multinomial Naive Bayes

Used for comparison and evaluation.

📊 Model Performance
Model	Accuracy
Logistic Regression	~98%
Naive Bayes	~94%
📈 Visualizations
Confusion Matrix

Displays correct and incorrect classifications.

Fake News Word Cloud

Shows frequently occurring words in fake news articles.

Real News Word Cloud

Shows frequently occurring words in real news articles.

🌐 Real-Time News Analysis

The application supports:

Manual Text Analysis

Users can paste news content directly.

URL-Based Analysis

Users can provide a news article URL and the system extracts article content automatically for prediction.

Live News Feed

Latest news headlines can be fetched using NewsAPI and analyzed through the prediction system.

🛠️ Tech Stack
Programming Language
Python
Machine Learning
Scikit-learn
Data Processing
Pandas
NumPy
NLP
TF-IDF Vectorizer
Web Framework
Streamlit
Visualization
Matplotlib
Seaborn
WordCloud
APIs & Web Scraping
NewsAPI
Requests
BeautifulSoup
Model Persistence
Joblib
📁 Project Structure
FakeNewsDetection/
│
├── data/
│   ├── Fake.csv
│   └── True.csv
│
├── app.py
├── train.py
├── fake_news_model.pkl
├── vectorizer.pkl
│
├── confusion_matrix.png
├── fake_wordcloud.png
├── real_wordcloud.png
│
├── requirements.txt
├── README.md
├── .gitignore
└── screenshots/
▶️ Installation
Clone Repository
git clone <repository-url>
cd FakeNewsDetection
Create Virtual Environment
python -m venv venv
Activate Environment

Windows:

.\venv\Scripts\Activate.ps1
Install Dependencies
pip install -r requirements.txt
▶️ Train Model
python train.py

This generates:

fake_news_model.pkl
vectorizer.pkl
confusion_matrix.png
fake_wordcloud.png
real_wordcloud.png
▶️ Run Application
python -m streamlit run app.py
🔒 Security Note

The News API key is stored in a .env file and is excluded from GitHub using .gitignore.

👨‍💻 Author

Devalapalli Vaishnavi

🎯 Outcome

This project demonstrates the application of Machine Learning and NLP techniques for fake news detection. It provides real-time analysis capabilities through URL extraction and live news fetching while offering visual insights into model performance.