import pandas as pd
import numpy as np
import pickle
import streamlit as st
import datetime

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from wordcloud import WordCloud


def greeting():
    current_time = datetime.datetime.now().time()
    morning_start = datetime.time(6)
    afternoon_start = datetime.time(12)
    evening_start = datetime.time(18)
    night_start = datetime.time(22)
    if current_time < morning_start:
        greeting = "Good Night!"
    elif current_time < afternoon_start:
        greeting = "Good Morning!"
    elif current_time < evening_start:
        greeting = "Good Afternoon!"
    elif current_time < night_start:
        greeting = "Good Evening!"
    else:
        greeting = "Good Night!"
    return greeting

# load model
pkl = open('pickle_model.pkl', 'rb')
model = pickle.load(pkl)

# predict
def prediction(s, model=model):
    pred = model.predict([s])
    return pred[0]
	

def main():
	st.title('Hello, ' + greeting())
	
	# define front end : font, background color, padding, text
	html_temp = """
	<div style ="background-color:pink;padding:13px">
	<h2 style ="color:black;text-align:center;">Product Review Sentiment Analysis</h2>
	</div>
	"""
	
	# front end
	st.markdown(html_temp, unsafe_allow_html = True)
	
	# text boxes for user input
	user_review = st.text_input("Check Review")
	result =""
	
	# panggil function prediction to make prediction
	if st.button("Predict"):
		result = prediction(user_review)
		st.success(f'The output is {result}')
	
if __name__=='__main__':
	main()