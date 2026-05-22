# ==============================
# Covid-19 Sentiment Analysis App
# Updated for coronavirus_tweets_top50.csv
# ==============================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os
import pickle

# NLP & DL
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
from gensim.models import Word2Vec

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(
    page_title="Covid-19 Sentiment AI",
    page_icon="🧬",
    layout="wide"
)

# ==============================
# CONSTANTS
# ==============================
MAX_LENGTH = 100
EMBEDDING_DIM = 100
# Updated filenames for the specific dataset
DATA_FILENAME = "Corona_NLP_train_Top50.csv"
MODEL_PATH = "model4_lstm_finetuned.keras"
TOK_PATH = "model4_tokenizer.pkl"
ENC_PATH = "model4_label_encoder.pkl"

# ==============================
# NLTK SETUP
# ==============================
@st.cache_resource
def setup_nltk():
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords")   
    return set(stopwords.words("english"))

STOP_WORDS = setup_nltk()

# ==============================
# TEXT CLEANING
# ==============================
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    return " ".join(w for w in text.split() if w not in STOP_WORDS)

# ==============================
# DATA LOADING
# ==============================
@st.cache_data
def load_dataset():
    # Updated to check for the top50 file
    if not os.path.exists(DATA_FILENAME):
        st.error(f"File '{DATA_FILENAME}' not found. Please ensure it is in the app directory.")
        return None
    
    # Using latin1 encoding as seen in the training notebook
    df = pd.read_csv(DATA_FILENAME, encoding="latin1")
    
    # Clean text using the defined function
    df["clean_tweets"] = df["OriginalTweet"].astype(str).apply(clean_text)
    
    # Standardize sentiments into 3 categories
    df["Sentiment_3"] = df["Sentiment"].replace({
        "Extremely Positive": "Positive",
        "Positive": "Positive",
        "Extremely Negative": "Negative",
        "Negative": "Negative"
    }).fillna("Neutral")
    return df

# ==============================
# MODEL TRAIN / LOAD
# ==============================
@st.cache_resource
def load_or_train_model(force=False):
    if not force and os.path.exists(MODEL_PATH):
        try:
            model = load_model(MODEL_PATH)
            tokenizer = pickle.load(open(TOK_PATH, "rb"))
            le = pickle.load(open(ENC_PATH, "rb"))
            return model, tokenizer, le
        except Exception:
            st.warning("Saved model incompatible or corrupted. Retraining...")

    df = load_dataset()
    if df is None:
        return None, None, None

    # Encoding labels
    le = LabelEncoder()
    y = le.fit_transform(df["Sentiment_3"])

    # Tokenizing text
    tokenizer = Tokenizer(num_words=15000)
    tokenizer.fit_on_texts(df["clean_tweets"])
    X = pad_sequences(tokenizer.texts_to_sequences(df["clean_tweets"]), maxlen=MAX_LENGTH)

    # Word2Vec Embeddings
    sentences = [t.split() for t in df["clean_tweets"]]
    w2v = Word2Vec(sentences, vector_size=EMBEDDING_DIM, min_count=1) # Reduced min_count for small datasets

    vocab_size = len(tokenizer.word_index) + 1
    embedding_matrix = np.zeros((vocab_size, EMBEDDING_DIM))
    for word, i in tokenizer.word_index.items():
        if word in w2v.wv:
            embedding_matrix[i] = w2v.wv[word]

    # Model Architecture
    model = Sequential([
        Embedding(vocab_size, EMBEDDING_DIM, weights=[embedding_matrix], trainable=False),
        Bidirectional(LSTM(64)),
        Dropout(0.4),
        Dense(32, activation="relu"),
        Dense(len(le.classes_), activation="softmax") # Dynamic output size
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # Training logic
    with st.spinner("Training model on top 50 tweets..."):
        model.fit(
            X, y,
            epochs=10, # Increased epochs for small dataset
            batch_size=8,
            callbacks=[EarlyStopping(patience=3, restore_best_weights=True)],
            verbose=0
        )

    # Save artifacts
    model.save(MODEL_PATH)
    pickle.dump(tokenizer, open(TOK_PATH, "wb"))
    pickle.dump(le, open(ENC_PATH, "wb"))

    return model, tokenizer, le

# ==============================
# SIDEBAR NAVIGATION
# ==============================
st.sidebar.title("🦠 COVID-19 NLP")
page = st.sidebar.radio("Navigation", [
    "Overview",
    "EDA",
    "Prediction",
    "Semantic Search"
])

# ==============================
# OVERVIEW PAGE
# ==============================
if page == "Overview":
    st.title("📘 Project Overview")
    st.write(f"Sentiment analysis using {DATA_FILENAME} (Deep Learning - LSTM).")
    df = load_dataset()
    if df is not None:
        st.write(f"Displaying top {len(df)} records:")
        st.dataframe(df[["OriginalTweet", "Sentiment_3"]].head())

# ==============================
# EDA PAGE
# ==============================
elif page == "EDA":
    st.title("📊 Exploratory Data Analysis")
    df = load_dataset()
    
    if df is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Sentiment Distribution")
            fig, ax = plt.subplots()
            sns.countplot(x="Sentiment_3", data=df, ax=ax, palette="viridis")
            st.pyplot(fig)

        with col2:
            st.subheader("Tweet Length Distribution")
            lengths = df["clean_tweets"].str.split().apply(len)
            fig2, ax2 = plt.subplots()
            sns.histplot(lengths, bins=15, ax=ax2, kde=True)
            st.pyplot(fig2)

        st.subheader("Word Cloud")
        sentiment = st.selectbox("Select Sentiment", df["Sentiment_3"].unique())
        subset = df[df["Sentiment_3"] == sentiment]["clean_tweets"]
        if not subset.empty:
            text = " ".join(subset)
            wc = WordCloud(width=800, height=400, background_color="white").generate(text)
            st.image(wc.to_array(), use_container_width=True)
        else:
            st.info("No data available for this sentiment.")

# ==============================
# PREDICTION PAGE
# ==============================
elif page == "Prediction":
    st.title("🤖 Sentiment Prediction")
    model, tokenizer, le = load_or_train_model()

    if model:
        text = st.text_area("Enter Tweet", "The vaccine distribution is helping.")
        if st.button("Predict"):
            cleaned = clean_text(text)
            seq = tokenizer.texts_to_sequences([cleaned])
            pad = pad_sequences(seq, maxlen=MAX_LENGTH)
            pred = model.predict(pad)
            
            idx = np.argmax(pred)
            label = le.inverse_transform([idx])[0]
            confidence = pred[0][idx]
            
            st.success(f"Prediction: **{label}** (Confidence: {confidence:.2%})")
            st.bar_chart(pd.Series(pred[0], index=le.classes_))

        if st.button("Force Retrain"):
            if os.path.exists(MODEL_PATH): os.remove(MODEL_PATH)
            load_or_train_model.clear()
            st.rerun()

# ==============================
# SEMANTIC SEARCH PAGE
# ==============================
elif page == "Semantic Search":
    st.title("🔍 Semantic Similarity Search")
    df = load_dataset()

    if df is not None:
        tfidf = TfidfVectorizer(max_features=1500)
        matrix = tfidf.fit_transform(df["clean_tweets"])

        query = st.text_input("Search Query", "lockdown")
        if st.button("Search"):
            qv = tfidf.transform([clean_text(query)])
            sims = cosine_similarity(qv, matrix).flatten()
            
            # Show top 5 results
            results = sims.argsort()[-5:][::-1]
            for i in results:
                if sims[i] > 0:
                    st.info(f"Score: {sims[i]:.2f} \n\n {df.iloc[i]['OriginalTweet']}")