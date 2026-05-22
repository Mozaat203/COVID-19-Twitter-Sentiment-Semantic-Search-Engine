# COVID-19 Twitter Sentiment & Semantic Search Engine

When a global crisis hits, social media becomes a real-time gauge of human emotion. This project goes beyond basic classification—it is an end-to-end semantic search engine and interactive AI dashboard. By combining NLP classification models with vector similarity search, users can input a thought, predict its sentiment in real-time, and immediately retrieve the most semantically similar historical tweets from the early days of the pandemic.

---

## 🚀 Core Engineering Contributions (CV Highlights)

*   **Interactive AI Dashboard:** Architected and deployed a multi-page Streamlit web application enabling real-time deep learning inference, interactive exploratory data analysis (EDA), and live semantic search.
*   **Geospatial Data Engineering:** Cleansed and standardized highly unstructured location strings using `geonamescache`; implemented a distributed random imputation strategy matching the original top-50 country probability distribution to handle missing values without skewing metrics.
*   **Pipeline Development:** Built comprehensive text preprocessing pipelines handling lowercase normalization, URL/mention removal, and NLTK stopword filtering to isolate core vocabulary.
*   **Iterative Architecture:** Engineered and evaluated four separate NLP systems, scaling from baseline statistical classifiers to Deep Learning sequence architectures (Word2Vec embedded Bidirectional LSTMs).

---

## 💻 Streamlit App Features

The project features a fully interactive UI built with Streamlit, divided into dedicated modules:

*   📊 **Interactive EDA:** Dynamic visualizations including sentiment distributions, tweet length KDE plots, and sentiment-specific Word Clouds generated on the fly.
*   🤖 **Real-Time Inference:** A live prediction engine powered by a Bidirectional LSTM. Users input custom text and receive instant sentiment classification alongside confidence interval bar charts.
*   🔍 **Semantic Search:** A search interface utilizing TF-IDF vectorization and Cosine Similarity to retrieve the top 5 most contextually relevant historical tweets based on user queries.
*   ⚙️ **Dynamic Retraining:** Built-in application logic to detect missing model weights and automatically trigger a background retraining pipeline with Early Stopping optimization.

---

## 📊 Model Architecture & Evolution

This project takes an iterative approach to NLP, building from classical ML baselines up to Bidirectional Neural Networks to systematically break through accuracy limits.

| Model Version | Text Representation | Classifier Engine | Best For |
| :--- | :--- | :--- | :--- |
| **Model 1** | TF-IDF Vectorizer | Linear SVC | Fast, highly interpretable baseline with keyword-based similarity. |
| **Model 2** | Pre-trained GloVe (100d) | Bidirectional LSTM | Understanding complex linguistic context and sequences. |
| **Model 3** | Custom Word2Vec | Logistic Regression | Lightweight semantic averaging with custom-trained embeddings. |
| **Model 4 (Deployed)** | Custom Word2Vec | Fine-Tuned Bi-LSTM | Capturing deep temporal dependencies using custom vocabulary constraints. |

---

## 🛠️ Tech Stack

*   **Frontend / UI:** Streamlit
*   **Data Processing & EDA:** Pandas, NumPy, Geonamescache, Matplotlib, Seaborn, WordCloud
*   **Machine Learning:** Scikit-Learn (TF-IDF, LinearSVC, Logistic Regression)
*   **Deep Learning & NLP:** TensorFlow / Keras (Bidirectional LSTM), Gensim (Word2Vec), NLTK

---

## ⚙️ How to Run the Application

**1. Install Dependencies**
```bash
pip install streamlit pandas numpy nltk matplotlib seaborn scikit-learn tensorflow gensim geonamescache wordcloud
```
**2. Download NLTK Stopwords
(The Streamlit app handles this automatically on startup, but you can run it manually if needed)
```bash
import nltk
nltk.download('stopwords')
```
**3. Launch the Streamlit Dashboard
Ensure Corona_NLP_train_Top50.csv is in your root directory, then start the server:
```bash
streamlit run app.py
```
**4. Using the App

    Navigate via the sidebar to explore the dataset in the Overview tab.

    Check out the EDA tab for data distribution and Word Clouds.

    Go to Prediction to test the Bi-LSTM model with your own sentences.

    Use Semantic Search to query the historical database for similar tweets.
