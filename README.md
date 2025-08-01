# 🏨 Hotel Review Sentiment Analysis

A machine learning project to classify hotel reviews as **positive** or **negative** based on their sentiment. The project leverages NLP techniques to clean, preprocess, and analyze textual data using traditional ML models and/or deep learning.

---
## 📖 Project Overview

This project performs sentiment analysis on hotel reviews to determine whether a review expresses a **positive** or **negative** sentiment. This can be used by hotel management systems to:

* Monitor customer satisfaction
* Highlight common issues
* Improve service quality

---

## 📂 Dataset

* **Source:** review_hotel.csv
* **Size:** \~10000 reviews
* **Fields:**

  * `Review`: Raw customer review text
  * `Sentiment`: Label (Positive/Negative)

---

## ⚙️ Tech Stack

* **Language:** Python
* **NLP Libraries:** NLTK,Scikit-learn
* **ML Models:** Random Forest Classifier, AdaBoost Classifier
* **Visualization:** Matplotlib, Seaborn, WordCloud

---

## 🛠 How it Works

1. **Data Preprocessing**

   * Lowercasing
   * Stopword removal(stopwords)
   * Lemmatization(WordNetLemmatizer)
   * Tokenization(WhitespaceTokenizer)

2. **Vectorization**

   * TF-IDF / TfidfVectorizer / Word2Vec

3. **Model Training**

   * Train-test split
   * Fit model(s) on training data

4. **Evaluation**

   * Accuracy

5. **Prediction**

   * Predict sentiment of new reviews

---

## 📊 Model Performance

| Model               | Accuracy |
| ------------------- | -------- |
| Random Forest       | 80.0%    |
| AdaBoost Classifier | 85.0%    |


