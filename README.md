📘 Customer Feedback Sentiment Classification using BERT

This project fine-tunes an Encoder-Only BERT model to classify customer feedback into
Positive, Negative, or Neutral sentiments.

Dataset used:
🔗 https://www.kaggle.com/datasets/vishweshsalodkar/customer-feedback-dataset

✨ Features

✔ Preprocessing & tokenization pipeline

✔ Fine-tuning BERT for classification

✔ Comprehensive evaluation:

Accuracy

Precision, Recall, F1-Score

Confusion Matrix

✔ Example predictions

✔ Clean, modular Python scripts

🚀 How to Run the Project
1️⃣ Install Dependencies
pip install -r requirements.txt

2️⃣ Run Preprocessing (optional)
python preprocessing.py

3️⃣ Fine-Tune the BERT Model
python train.py

4️⃣ Evaluate the Model
python evaluate.py


This will generate:

confusion_matrix.png

metrics.txt

sample_predictions.txt

🧪 Inference Example

To test the model on new customer feedback:

python inference_example.py

📊 Evaluation Metrics Included

Accuracy

Precision / Recall / F1-score

Confusion Matrix 

Example predictions

📦 Requirements
transformers
torch
pandas
numpy
scikit-learn
matplotlib
seaborn

📝 License

MIT License
