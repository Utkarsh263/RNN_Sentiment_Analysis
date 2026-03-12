import streamlit as st
import torch
import torch.nn as nn
import pickle
import re

# Load TF-IDF
with open("tfidf_vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)

# RNN model (same architecture as training)
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out

# Load model
input_size = 5000
model = RNN(input_size)
model.load_state_dict(torch.load("rnn_sentiment_model.pth", map_location="cpu"))
model.eval()

# text cleaning
def clean_text(text):
    text = text.lower()
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^A-Za-z0-9\s]", "", text)
    return text

# Streamlit UI
st.title("🎬 IMDB Sentiment Analysis")

review = st.text_area("Enter Movie Review")

if st.button("Predict Sentiment"):

    text = clean_text(review)

    vector = tfidf.transform([text]).toarray()

    X = torch.tensor(vector).float().unsqueeze(1)

    with torch.no_grad():
        output = model(X)
        prob = torch.sigmoid(output).item()

    if prob > 0.5:
        st.success("Positive Review 😊")
    else:
        st.error("Negative Review 😞")

    st.write("Confidence:", prob)