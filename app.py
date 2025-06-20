import streamlit as st
from newspaper import Article
import nltk
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# NLTK setup
nltk.download('punkt')
nltk.download('stopwords')

# Load model and tokenizer
model = load_model("model.h5")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Label map
labels_map = {0: "World", 1: "Sports", 2: "Business", 3: "Science/Tech"}

# Clean and preprocess
def clean_text(text):
    stop_words = set(nltk.corpus.stopwords.words("english"))
    return " ".join([word for word in text.lower().split() if word not in stop_words])

def preprocess(text):
    cleaned = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=200)
    return padded

# Streamlit UI
st.title("📰 Article Insights Analyzer")
st.write("Paste a news article URL to classify it into a topic.")

url = st.text_input("Enter article URL:")

if st.button("🔍 Classify Article"):
    if url:
        try:
            article = Article(url)
            article.download()
            article.parse()

            st.subheader("Title:")
            st.write(article.title)

            st.subheader("Article Preview:")
            st.write(article.text[:800] + "...")

            padded_input = preprocess(article.text)
            prediction = model.predict(padded_input)
            predicted_class = np.argmax(prediction)
            label = labels_map[predicted_class]

            st.success(f"✅ **Predicted Category:** {label}")

        except Exception as e:
            st.error(f"⚠️ Error processing article: {e}")
    else:
        st.warning("Please enter a valid article URL.")
