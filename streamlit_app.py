import streamlit as st
import torch
import numpy as np
import pickle
from transformers import DistilBertForSequenceClassification, AutoTokenizer
import re

# -----------------------------
# LOAD MODEL – FIXED FOR 2025
# -----------------------------
@st.cache_resource
def load_model():
    model_path = "model"  # ← no ./, no trailing /

    model = DistilBertForSequenceClassification.from_pretrained(
        model_path,
        local_files_only=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True
    )
    with open("model/label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)

    return model, tokenizer, label_encoder


# Load model
model, tokenizer, label_encoder = load_model()
device = torch.device("cpu")
model.to(device)
model.eval()

# -----------------------------
# Preprocessing & Prediction
# -----------------------------
def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"@\w+|#\w+|<.*?>", "", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = " ".join(text.split())
    return text

def predict_sentiment(text: str):
    clean = preprocess_text(text)
    inputs = tokenizer(clean, truncation=True, padding="max_length", max_length=128, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    pred_idx = np.argmax(probs)
    label = label_encoder.inverse_transform([pred_idx])[0]
    confidence = float(probs[pred_idx])

    return label, confidence, probs

# -----------------------------
# UI
# -----------------------------
st.title("Customer Feedback Sentiment Analyzer")
st.write("Fine-tuned DistilBERT model (Positive / Negative)")

user_input = st.text_area("Enter customer feedback:", height=150)

if st.button("Analyze Sentiment"):
    if not user_input.strip():
        st.warning("Please enter some text.")
    else:
        with st.spinner("Analyzing..."):
            label, conf, probs = predict_sentiment(user_input)

        st.success(f"**{label}** (confidence: {conf:.3f})")
        st.bar_chart(
            dict(zip(label_encoder.classes_, probs)),
            use_container_width=True
        )
