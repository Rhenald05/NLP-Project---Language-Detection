import streamlit as st
import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import LabelEncoder
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# Download NLTK data
nltk.download('punkt')

# Load dataset
dataset = pd.read_csv("Language Detection.csv")

# Tokenization function
def tokenize_text(text):
    return word_tokenize(text)

# Preprocess text data
def preprocess_text(text):
    text = re.sub(r'[!@#$(),\n"%^&*:;~0-9]', ' ', text)
    text = re.sub('[\[\]]', ' ', text)
    text = text.lower()
    return text

# Load and preprocess data
X = dataset['Text'].apply(preprocess_text)
y = dataset['Language']

# Label Encoding
le = LabelEncoder()
y = le.fit_transform(y)

# TF-IDF vectorization
tfidf_v = TfidfVectorizer(tokenizer=tokenize_text)
X = tfidf_v.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Naive Bayes classifier
nb = MultinomialNB()
nb.fit(X_train, y_train)

# Prediction function
def predict_language(text):
    processed_text = preprocess_text(text)
    vectorized_text = tfidf_v.transform([processed_text])
    prediction = nb.predict(vectorized_text)[0]
    return le.inverse_transform([prediction])[0]

# Streamlit app
def main():
    st.title("Language Detection App for Beginners")
    st.write("Enter a text to detect its language.")

    text_input = st.text_input("Text Input")
    if st.button("Predict"):
        prediction = predict_language(text_input)
        st.write("Predicted Language:", prediction)

if __name__ == "__main__":
    main()
