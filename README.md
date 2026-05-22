# COVID-19-Twitter-Sentiment-Semantic-Search-Engine
Here we apply different NLP algorithms and different machine/deeplearning models for learning purposes

When a global crisis hits, social media becomes a real-time gauge of human emotion. This project goes beyond basic classification—it acts as an end-to-end semantic search engine. By combining NLP classification models with vector similarity search, you can input a thought, predict its sentiment, and immediately retrieve the most semantically similar historical tweets from the early days of the pandemic.

---

## 🚀 Core Engineering Contributions (CV Highlights)

*   **Geospatial Data Engineering:** Cleansed and standardized highly unstructured location strings using `geonamescache`; implemented a distributed random imputation strategy matching the original top-50 country probability distribution to handle missing values without skewing metrics.
*   **Pipeline Development:** Built comprehensive text preprocessing pipelines handling lowercase normalization, URL/mention removal, and NLTK stopword filtering to isolate core vocabulary.
*   **Iterative Architecture:** Engineered and evaluated four separate NLP systems, scaling from baseline statistical classifiers (TF-IDF + Linear SVC) to Deep Learning sequence architectures (Word2Vec/GloVe embedded Bidirectional LSTMs).
*   **Multi-Modal Inference:** Designed an end-to-end inference framework matching sentiment classification predictions with custom Word2Vec vector similarity scoring to recommend contextually related historical tweets.

---

## 📊 Model Architecture & Evolution

This project takes an iterative approach to NLP, building from classical ML baselines up to Bidirectional Neural Networks to systematically break through accuracy limits.

| Model Version | Text Representation | Classifier Engine | Best For |
| :--- | :--- | :--- | :--- |
| **Model 1** | TF-IDF Vectorizer | Linear SVC | Fast, highly interpretable baseline with keyword-based similarity. |
| **Model 2** | Pre-trained GloVe (100d) | Bidirectional LSTM | Understanding complex linguistic context and sequences. |
| **Model 3** | Custom Word2Vec | Logistic Regression | Lightweight semantic averaging with custom-trained embeddings. |
| **Model 4** | Custom Word2Vec | Fine-Tuned Bi-LSTM | Capturing deep temporal dependencies using custom vocabulary constraints. |

---

## 🛠️ Tech Stack

*   **Data Processing & EDA:** Pandas, NumPy, Geonamescache, Matplotlib, Seaborn
*   **Machine Learning:** Scikit-Learn (TF-IDF, LinearSVC, Logistic Regression)
*   **Deep Learning & NLP:** TensorFlow / Keras (Bidirectional LSTM), Gensim (Word2Vec), NLTK

---

## ⚙️ How to Run the Inference Engine

**1. Install Dependencies**
```bash
pip install pandas numpy nltk matplotlib seaborn scikit-learn tensorflow gensim geonamescache

import nltk
nltk.download('stopwords')

# Example Usage
user_input = "The supermarket shelves are completely empty, this is scary."
predict_and_recommend(user_input, num_similar=3)
