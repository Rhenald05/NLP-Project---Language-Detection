import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load dataset
dataset = pd.read_csv("Language Detection.csv")

# Label Encoding
le = LabelEncoder()
dataset['Language'] = le.fit_transform(dataset['Language'])

# Regular Expression
dataset['Text'] = dataset['Text'].apply(lambda text: re.sub(r'[!@#$(),\n"%^&*:;~0-9]', ' ', text))
dataset['Text'] = dataset['Text'].apply(lambda text: re.sub('[\[\]]', ' ', text))
dataset['Text'] = dataset['Text'].str.lower()

# TF-IDF
tfidf_v = TfidfVectorizer()
X = tfidf_v.fit_transform(dataset['Text'])
y = dataset['Language']

# Naive Bayes
nb = MultinomialNB()
nb.fit(X, y)

def predict_language(text):
    processed_text = re.sub(r'[!@#$(),\n"%^&*:;~0-9]', ' ', text)
    processed_text = re.sub('[\[\]]', ' ', processed_text)
    processed_text = processed_text.lower()
    vectorized_text = tfidf_v.transform([processed_text])
    prediction = nb.predict(vectorized_text)[0]
    return le.inverse_transform([prediction])[0]

def main():
    st.title("Language Detection App for Language Learners")
    st.write("Enter a word or text to detect its language")

    text_input = st.text_input("Input the word or text here")
    if st.button("Detect Language"):
        prediction = predict_language(text_input)
        st.write("The language is ", prediction)

if __name__ == "__main__":
    main()
