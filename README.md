# 🎬 IMDB Sentiment Analysis using RNN

This project implements a **Recurrent Neural Network (RNN)** for sentiment analysis on the **IMDB movie review dataset**.  
The model predicts whether a movie review is **positive or negative**.  
A **Streamlit web interface** is added so users can enter a review and instantly get sentiment predictions.

---

## 📌 Overview

Sentiment analysis is a Natural Language Processing (NLP) task that determines the emotional tone behind a text.  
In this project:

- Movie reviews are cleaned and preprocessed 🧹
- Text is converted into numerical features using **TF-IDF vectorization**
- A **PyTorch RNN model** is trained for binary classification 🤖
- A **Streamlit UI** allows interactive prediction 🌐

---

## 📊 Dataset

The dataset used is the **IMDB Movie Reviews Dataset**, which contains **50,000 reviews** labeled as positive or negative.

Each record includes:
- `review` – text of the movie review
- `sentiment` – label (`positive` or `negative`)

---

## 🧹 Text Preprocessing

Several preprocessing steps are applied before training the model:

- Convert text to lowercase
- Remove URLs
- Remove punctuation
- Remove HTML tags
- Remove stopwords using NLTK
- Apply stemming using PorterStemmer

These steps help clean the raw text and improve model performance.

---

## 🔢 Feature Engineering

The cleaned reviews are converted into numerical vectors using **TF-IDF (Term Frequency–Inverse Document Frequency)**.

TF-IDF helps:
- Reduce importance of common words
- Highlight informative words
- Convert text into machine learning features

Maximum vocabulary size used:

```
5000 features
```

---

## 🧠 Model Architecture

The model is implemented using **PyTorch** and consists of:

- **RNN Layer**
  - Processes sequential input
  - Captures contextual relationships in text

- **Fully Connected Layer**
  - Converts hidden representation into a binary output

Architecture type:

```
Many-to-One Sequence Model
```

Meaning a sequence of words produces **one sentiment prediction**.

---

## 🏋️ Training

The model is trained using:

- **Loss Function:** Binary Cross Entropy Loss
- **Optimizer:** Adam
- **Epochs:** 15
- **Batch Size:** 64

The dataset is split into:
- **80% Training**
- **20% Testing**

---

## 📈 Model Evaluation

Model performance is evaluated using **classification accuracy** on the test set.

Accuracy is computed as:

```
Accuracy = Correct Predictions / Total Predictions
```

---

## 🌐 Streamlit Web Application

A Streamlit interface allows users to interact with the trained model.

Users can:
1. Enter a movie review ✍️
2. Click the **Predict Sentiment** button
3. See whether the review is **Positive 😊 or Negative 😞**

The interface loads the saved:
- trained RNN model
- TF-IDF vectorizer

and performs prediction in real time.

---

## 🛠 Technologies Used

- Python 🐍
- PyTorch
- Streamlit
- Scikit-learn
- NLTK
- Pandas
- NumPy

---

## ⚙️ How It Works

1. User enters a movie review
2. Text preprocessing is applied
3. Review is converted into TF-IDF vector
4. Vector is passed to the trained RNN
5. Model predicts sentiment probability
6. Output is displayed in the Streamlit UI

---

## 🧪 Example Prediction

Input review:

```
This movie was absolutely amazing with great acting and storyline
```

Prediction:

```
Positive Review
Confidence: 0.91
```

---

## 🚀 Future Improvements

Possible improvements include:

- Using **LSTM or GRU instead of simple RNN**
- Implementing **word embeddings (Word2Vec / GloVe)**
- Using **BERT-based transformers**
- Deploying the application online

---

## Deployment 

Live Demo: https://rnnsentimentanalysis-m29imnyjyygtw5mjaydzhl.streamlit.app
