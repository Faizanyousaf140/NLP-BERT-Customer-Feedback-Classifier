import streamlit as st
import torch
import numpy as np
import pickle  # ← changed from joblib to pickle (more reliable with LabelEncoder)
import os
from transformers import DistilBertForSequenceClassification, AutoTokenizer
import re

# -----------------------------
# Load Model + Tokenizer + Label Encoder (FIXED!)
# -----------------------------
@st.cache_resource
def load_model():
    # Critical fixes:
    model_path = "./model"  # ← leading ./ forces local folder
    # OR use "model" + local_files_only=True (both work)

    model = DistilBertForSequenceClassification.from_pretrained(
        model_path,
        local_files_only=True   # ← prevents Hugging Face Hub validation error
    )

    # Use AutoTokenizer (more reliable) and load from same folder
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True
    )

    # Load label encoder (must be saved with pickle, not joblib if you used sklearn LabelEncoder)
    with open(os.path.join(model_path, "label_encoder.pkl"), "rb") as f:
        label_encoder = pickle.load(f)

    return model, tokenizer, label_encoder


# Load everything
try:
    model, tokenizer, label_encoder = load_model()
    device = torch.device("cpu")  # Streamlit Cloud has NO GPU
    model.to(device)
    model.eval()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# -----------------------------
# Preprocessing Function
# -----------------------------
def preprocess_text(text: str) -> str:
    if not text:
        return ""
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
def predict_sentiment(text: str):
    clean_text = preprocess_text(text)
    if not clean_text.strip():
        return "Neutral", 0.0, np.array([0.5, 0.5])  # fallback

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

    return predicted_label, float(confidence), probabilities


# -----------------------------
# Streamlit UI
# -----------------------------
st.title("Urdu Sentiment Analysis Chatbot (DistilBERT)")
st.markdown("""
    Fine-tuned DistilBERT model for **Positive/Negative** sentiment classification on customer feedback.
""")

st.info("Note: This model was trained on English text. It works best with clear, natural English feedback.")

user_input = st.text_area("Enter customer feedback:", height=150, placeholder="e.g. The product is amazing and delivery was fast!")

if st.button("Analyze Sentiment", type="primary"):
    if not user_input.strip():
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner("Analyzing sentiment..."):
            label, conf, probs = predict_sentiment(user_input)

        st.success("Analysis Complete!")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Predicted Sentiment", label)
        with col2:
            st.metric("Confidence", f"{conf:.3f}")

        st.subheader("Class Probabilities")
        prob_df = {
            "Sentiment": label_encoder.classes_,
            "Probability": [f"{p:.3f}" for p in probs]
        }
        st.bar_chart(prob_df, x="Sentiment", y="Probability")

        # Optional: show raw probabilities
        with st.expander("View raw probabilities"):
            for cls, prob in zip(label_encoder.classes_, probs):
                st.write(f"**{cls}**: {prob:.4f}")
