# 🛍️ Review Sentiment Analyser

> An end-to-end NLP web app that classifies product reviews as positive or negative — 
> upload any CSV, select your review column, and get instant sentiment insights.

## 🔗 Live Demo
**[👉 Try the app here](YOUR_STREAMLIT_URL)**

---

## 📌 Overview

This project builds a complete sentiment analysis pipeline trained on 500,000+ real 
Amazon Fine Food reviews. Two models are trained, compared, and evaluated — with the 
best model deployed as an interactive business intelligence web application.

The app goes beyond simple classification. It lets users upload their own review data, 
automatically detects the review column, analyses sentiment at scale, and breaks down 
results across product categories, regions, or any other grouping in the dataset.

---

## ✨ App Features

- 📁 **Upload any review CSV** — works with Amazon, Yelp, wine, app store, or any product reviews
- 🔍 **Smart column detection** — automatically identifies and ranks likely review columns
- 🗂️ **Multi-category analysis** — compare sentiment across product type, country, rating, brand
- 🔥 **Cross-category heatmap** — see positive sentiment % at the intersection of two categories
- 📅 **Trend analysis** — plots sentiment over time if your data contains a date column
- 🔑 **Top keywords** — most frequent words driving positive and negative sentiment
- 🔎 **Filter & explore** — filter by sentiment and category simultaneously
- ⬇️ **Download results** — export all results or filtered views as CSV

---

## 📊 Model Results

| Model | Accuracy | Macro F1 | Notes |
|-------|----------|----------|-------|
| Logistic Regression | 94% | 0.88 | Strong, fast baseline — used in deployment |
| Neural Network | 94% | 0.89 | Marginal improvement at significantly higher compute cost |

### Performance Breakdown (Logistic Regression)

| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| Negative | 0.87 | 0.72 | 0.79 |
| Positive | 0.95 | 0.98 | 0.96 |
| **Overall** | **0.94** | **0.94** | **0.94** |

### Key Finding
The logistic regression baseline matches the neural network at 94% overall accuracy. 
This demonstrates that strong feature engineering (TF-IDF + bigrams) can be just as 
powerful as deep learning for text classification — while being faster, lighter, and 
easier to deploy.

The model performs better on positive reviews (F1: 0.96) than negative ones (F1: 0.79), 
which is expected given the class imbalance — 85% of reviews in the dataset are positive.

---

## 💡 Business Insights

Based on analysis of 500,000+ reviews:

- **Negative reviews** most commonly mention delivery delays, misleading product 
  descriptions, and quality inconsistencies — clear operational improvement areas
- **Positive reviews** consistently highlight taste, value for money, and repeat 
  purchase intent — key drivers of customer loyalty
- Automatic classification enables customer support teams to **prioritise negative 
  reviews** without manual reading
- The multi-category breakdown allows businesses to identify **which product lines 
  or regions have the highest dissatisfaction rates**
- Sentiment trend analysis over time helps teams **measure the impact** of product 
  or service changes

---

## 🗂️ Project Structure

amazon-sentiment-analysis/
- app/
  - app.py                 # Streamlit web application
- src/
  - preprocess.py          # Text cleaning and preprocessing
- models/
  - logistic_model.pkl     # Trained ML model
  - tfidf_vectorizer.pkl   # TF-IDF vectorizer
- notebooks/
  - 01_eda_baseline.ipynb  # EDA + Logistic Regression
  - 02_neural_network.ipynb# Neural Network model
- data/
  - (dataset files)        # Raw data & outputs
- requirements.txt         # Dependencies
- README.md                # Project documentation              
---
---

## 🛠️ Tech Stack

| Area | Tools |
|------|-------|
| Language | Python 3.11 |
| Data Processing | Pandas, NumPy |
| NLP & Feature Engineering | TF-IDF, bigrams, regex, custom preprocessing |
| Machine Learning | Scikit-learn, Logistic Regression |
| Deep Learning | TensorFlow, Keras |
| Visualisation | Matplotlib, Seaborn, WordCloud |
| Web App | Streamlit |
| Deployment | Streamlit Community Cloud |
| Version Control | Git, GitHub |

---

## 🚀 Run Locally

**1. Clone the repository**
```bash
git clone https://github.com/danielakbank/amazon-sentiment-analysis.git
cd amazon-sentiment-analysis
```

**2. Create and activate a virtual environment**
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Run the app**
```bash
streamlit run app/app.py
```

Visit `http://localhost:8501`

---

## 📁 Try It With Your Own Data

The app works with **any CSV file** containing text reviews. To test it:

1. Download the [Wine Reviews dataset from Kaggle](https://www.kaggle.com/datasets/zynicide/wine-reviews) (~50MB)
2. Visit the [live app](YOUR_STREAMLIT_URL)
3. Upload the CSV
4. Select `description` as your review column
5. Select `country` and `variety` as category columns
6. Click Analyse

You can also use Amazon, Yelp, app store, or any other review dataset in CSV format.

---

## 📈 Training Data

| Property | Value |
|----------|-------|
| Dataset | Amazon Fine Food Reviews |
| Source | [Kaggle](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews) |
| Total reviews | 568,454 |
| After cleaning | 363,836 |
| Positive (4–5 stars) | 306,766 (84%) |
| Negative (1–2 stars) | 57,070 (16%) |
| Train / Test split | 80% / 20% stratified |

---

## 🧠 Modelling Approach

### Preprocessing
- Lowercasing and HTML tag removal
- Punctuation and number stripping
- Custom reusable `clean_text` function in `src/preprocess.py`

### Feature Engineering
- TF-IDF vectorisation with 10,000 features
- Unigrams and bigrams (`ngram_range=(1,2)`)
- Fitted on training data only to prevent data leakage

### Baseline Model
- Logistic Regression with `max_iter=1000`
- Achieves 94% accuracy — strong baseline for this task

### Neural Network
- Feedforward network: Dense(256) → BatchNorm → Dropout(0.4) → Dense(128) → Dropout(0.3) → Dense(1)
- Binary cross-entropy loss, Adam optimiser
- Early stopping (patience=3) to prevent overfitting
- Training stopped at epoch 2 — validation loss began increasing, indicating overfitting

### Why Logistic Regression Was Chosen for Deployment
- Matches the neural network at 94% accuracy
- Significantly faster inference — important for real-time app use
- Smaller memory footprint — better for cloud deployment
- Easier to maintain and debug

---

## 📸 Screenshots

> *(Add screenshots of your app here once deployed)*

| Dashboard | Category Analysis | Heatmap |
|-----------|------------------|---------|
| ![dashboard](data/dashboard.png) | ![categories](data/categories.png) | ![heatmap](data/heatmap.png) |

---

## 👤 Author

**Dean** *(update with your full name)*

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat&logo=linkedin)](YOUR_LINKEDIN_URL)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?style=flat&logo=github)](YOUR_GITHUB_URL)

---

## 📄 Licence

This project is open source and available under the [MIT Licence](LICENSE).