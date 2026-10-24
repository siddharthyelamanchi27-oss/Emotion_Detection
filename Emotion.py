from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import pipeline
model = AutoModelForSequenceClassification.from_pretrained("boltuix/bert-emotion")
tokenizer = AutoTokenizer.from_pretrained("boltuix/bert-emotion")

# Load the fine-tuned BERT-Emotion model
sentiment_analysis = pipeline("text-classification", model="boltuix/bert-emotion")

# Analyze emotion
text_given = input("Enter a text to analyze its emotion: ")
result = sentiment_analysis(text_given)
print(result)
