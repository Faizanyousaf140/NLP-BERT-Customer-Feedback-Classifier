import streamlit as st
import torch
import numpy as np
import joblib
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import re

# -----------------------------
# Load Model + Tokenizer + Encoder
# -----------------------------
@st.cache_resource
def load_model():
    model = DistilBertForSequenceClassification.from_pretrained("model/")
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    label_encoder = joblib.load("label_encoder.pkl")
    return model, tokenizer, label_encoder

model, tokenizer, label_encoder = load_model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# -----------------------------
# Preprocessing Function
# -----------------------------
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = " ".join(text.split())
    return text

# -----------------------------
# Prediction Function
# -----------------------------
def predict_sentiment(text):
    clean_text = preprocess_text(text)

    inputs = tokenizer(
        clean_text,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt"
    )

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]

    predicted_idx = np.argmax(probabilities)
    predicted_label = label_encoder.inverse_transform([predicted_idx])[0]
    confidence = probabilities[predicted_idx]

    return predicted_label, confidence, probabilities

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("💬 Urdu Sentiment Analysis Chatbot (DistilBERT)")
st.write("Enter text and get sentiment prediction instantly!")

user_input = st.text_area("Enter text:", height=150)

if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        label, conf, probs = predict_sentiment(user_input)

        st.subheader("🔍 Prediction")
        st.write(f"**Sentiment:** {label}")
        st.write(f"**Confidence:** {conf:.3f}")

        st.subheader("📊 Class Probabilities")
        for i, cls in enumerate(label_encoder.classes_):
            st.write(f"• **{cls}**: {probs[i]:.3f}")
