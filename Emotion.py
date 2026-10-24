from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import pipeline
import streamlit as st
model = AutoModelForSequenceClassification.from_pretrained("boltuix/bert-emotion")
tokenizer = AutoTokenizer.from_pretrained("boltuix/bert-emotion")

# Load the fine-tuned BERT-Emotion model
sentiment_analysis = pipeline("text-classification", model="boltuix/bert-emotion")

# Analyze emotion
with st.container():
    st.chat_input("Enter a test message to analyze emotion")
    text_given = st.text_input("What is your message")
    result = sentiment_analysis(text_given)
    st.write("Emotion Analysis Result:")
    st.write(result)
