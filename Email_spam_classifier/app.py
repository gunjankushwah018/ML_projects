import streamlit as st
import pickle
import sklearn
import nltk
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


tfidf = pickle.load(open('vectorizer (1).pkl','rb'))
model = pickle.load(open('model (1).pkl','rb'))

st.title("Email Spam Classifier")
input_email = st.text_area("Enter Email")
if st.button('Predict'):

    # 1. Preprocess
    transformed_email = transform_text(input_email)

    # 2. Vectorize
    vector_input = tfidf.transform([transformed_email])

    # 3. Predict
    result = model.predict(vector_input)

    # 4. Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")