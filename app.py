import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
import string
from nltk.stem import PorterStemmer

nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()  # Convert to lowercase
    text = nltk.word_tokenize(text)  # Tokenization

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)  # Remove all special characters

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))  # Stemming

    return " ".join(y)

tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title("Email/SMS Spam Classifier")

input_sms = st.text_input("Enter the message")

if st.button("Classify"):
    # Check if input is not empty
    if input_sms.strip() != "":
        # Preprocess
        transformed_sms = transform_text(input_sms)
        # Fit TF-IDF vectorizer and transform
        vector_input = tfidf.transform([transformed_sms])
        # Predict
        result = model.predict(vector_input)[0]
        # Display result
        if result == 1:
            st.header("Spam")
        else:
            st.header("Not Spam")
    else:
        st.error("Please enter a message to classify.")
