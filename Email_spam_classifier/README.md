# Email Spam Classifier

An end-to-end Machine Learning project that classifies email and SMS messages as **Spam** or **Ham (Not Spam)** using **TF-IDF Vectorization** and **Multinomial Naive Bayes**.

The project demonstrates a complete machine learning workflow, including data preprocessing, text cleaning, feature extraction, model training, model evaluation, model serialization, and deployment using Streamlit.

---

## Live Demo

Try the deployed application:

[Open Email Spam Classifier](https://mlprojects-kzqfqeceg93dztblnm7hwm.streamlit.app/)

---

## GitHub Repository

[View Source Code on GitHub](https://github.com/gunjankushwah018/ML_projects/tree/main/Email_spam_classifier)

---

## Problem Statement

Spam messages are unwanted messages that may contain promotional content, fraudulent offers, or other irrelevant information.

The goal of this project is to build a machine learning model that can automatically classify a given email or SMS message into one of two categories:

- **Ham** — Legitimate message
- **Spam** — Unwanted or spam message

This is treated as a **binary text classification problem**.

---

## Project Overview

The project takes a text message as input and processes it through an NLP and machine learning pipeline.

The text is first cleaned and preprocessed using NLTK. After preprocessing, TF-IDF is used to convert the text into numerical features. These features are then passed to a Multinomial Naive Bayes classifier to predict whether the message is Spam or Ham.

The trained model and TF-IDF vectorizer are saved using Pickle and loaded by the Streamlit application for real-time predictions.

---

## Machine Learning Workflow

```text
Raw Message
     ↓
Text Cleaning
     ↓
Tokenization
     ↓
Stopword Removal
     ↓
Punctuation Removal
     ↓
Stemming
     ↓
TF-IDF Vectorization
     ↓
Multinomial Naive Bayes
     ↓
Spam / Ham Prediction
