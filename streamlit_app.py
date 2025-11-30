
import os
os.environ["HF_HUB_OFFLINE"] = "1"       
os.environ["TRANSFORMERS_OFFLINE"] = "1"
# ==============================================================

import streamlit as st
import torch
import numpy as np
import pickle
import re
from transformers import DistilBertForSequenceClassification, AutoTokenizer

# -----------------------------
# Load everything
# -----------------------------
@st.cache_resource
def load_model():
    # No local_files_only, no ./, no tricks — just plain path + offline mode
    model = DistilBertForSequenceClassification.from_pretrained("model")
    tokenizer = AutoTokenizer.from_pretrained("model")
    
    with open("model/label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    
    return model, tokenizer, label_encoder


# Load model (this will now work 100%)
try:
    model, tokenizer, label_encoder = load_model()
    device = torch.device("cpu")
    model.to(device)
    model.eval()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# -----------------------------
# Preprocessing & Prediction
# -----------------------------
def preprocess_text(text: str) -> str:
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"[@#]\w+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = " ".join(text.split())
    return text

def predict_sentiment(text: str):
    clean = preprocess_text(text)
    if not clean.strip():
        return "Neutral", 0.0, np.array([0.5, 0.5])

    inputs = tokenizer(clean, truncation=True, padding="max_length", max_length=128, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        logits = model(input_ids, attention_mask=attention_mask).logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    pred_idx = np.argmax(probs)
    label = label_encoder.inverse_transform([pred_idx])[0]
    confidence = float(probs[pred_idx])

    return label, confidence, probs

# -----------------------------
# UI
# -----------------------------
st.title("Customer Feedback Sentiment Analyzer")
st.write("Fine-tuned DistilBERT • Positive vs Negative")

user_input = st.text_area("Enter feedback text:", height=150, placeholder="The service was excellent and staff very helpful!")

if st.button("Analyze Sentiment", type="primary"):
    if not user_input.strip():
        st.warning("Please enter some text.")
    else:
        with st.spinner("Thinking..."):
            label, conf, probs = predict_sentiment(user_input)

        st.success(f"**Predicted: {label}**")
        st.metric("Confidence", f"{conf:.3f}")

        chart_data = dict(zip(label_encoder.classes_, probs))
        st.bar_chart(chart_data)
