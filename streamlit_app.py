# ============ BULLETPROOF OFFLINE FIX FOR STREAMLIT CLOUD ============
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
# =====================================================================

import streamlit as st
import torch
import numpy as np
import pickle
import re
import os.path  # For path validation
from transformers import DistilBertForSequenceClassification, AutoTokenizer

# -----------------------------
# Load everything (with validation)
# -----------------------------
@st.cache_resource
def load_model():
    # Validate model folder exists and has config.json
    model_path = "model"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model folder '{model_path}' not found! Check your GitHub repo.")
    
    config_path = os.path.join(model_path, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Missing 'config.json' in '{model_path}'. Re-save the model.")
    
    # List files for debugging (shows in Streamlit logs)
    st.info(f"Model folder contents: {os.listdir(model_path)}")
    
    # Resolve to absolute path (fixes Streamlit path issues)
    abs_model_path = os.path.abspath(model_path)
    
    # Load with absolute path + offline mode
    model = DistilBertForSequenceClassification.from_pretrained(abs_model_path)
    tokenizer = AutoTokenizer.from_pretrained(abs_model_path)
    
    le_path = os.path.join(model_path, "label_encoder.pkl")
    with open(le_path, "rb") as f:
        label_encoder = pickle.load(f)
    
    return model, tokenizer, label_encoder


# Load model
try:
    model, tokenizer, label_encoder = load_model()
    device = torch.device("cpu")
    model.to(device)
    model.eval()
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.info("💡 Debug: Check if 'model/config.json' exists in your GitHub repo.")
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
st.title("🧠 Customer Feedback Sentiment Analyzer")
st.markdown("Powered by fine-tuned DistilBERT (Positive/Negative classification)")

user_input = st.text_area(
    "Enter your customer feedback:", 
    height=150, 
    placeholder="e.g., 'The delivery was super fast and the quality exceeded expectations!'"
)

if st.button("🔍 Analyze Sentiment", type="primary"):
    if not user_input.strip():
        st.warning("⚠️ Please enter some text to analyze.")
    else:
        with st.spinner("🤔 Analyzing sentiment..."):
            label, conf, probs = predict_sentiment(user_input)

        st.success(f"**Predicted Sentiment: {label}**")
        col1, col2 = st.columns([3, 1])
        with col1:
            st.metric("Confidence Score", f"{conf:.1%}")
        with col2:
            st.caption(f"({conf:.3f})")

        # Probability bar chart
        st.subheader("📈 Probability Breakdown")
        chart_data = pd.DataFrame({
            "Sentiment": label_encoder.classes_,
            "Probability": probs
        })
        st.bar_chart(chart_data.set_index("Sentiment"))

        # Raw probs expander
        with st.expander("🔍 View detailed probabilities"):
            for cls, prob in zip(label_encoder.classes_, probs):
                st.write(f"**{cls}:** {prob:.4f} ({prob:.1%})")
