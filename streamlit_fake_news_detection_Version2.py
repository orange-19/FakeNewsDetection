import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

# --- Model Definition (must match training) ---
class FakeNewsClassifier(nn.Module):
    def __init__(self, input_dim):
        super(FakeNewsClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.model(x)

# --- Load assets ---
@st.cache_resource
def load_assets():
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    scaler = joblib.load("scaler.pkl")
    input_dim = vectorizer.max_features if hasattr(vectorizer, "max_features") else len(vectorizer.get_feature_names_out())
    model = FakeNewsClassifier(input_dim)
    model.load_state_dict(torch.load("model_state_dict.pt", map_location=torch.device("cpu")))
    model.eval()
    return vectorizer, scaler, model

vectorizer, scaler, model = load_assets()

# --- Streamlit UI ---
st.title("📰 Fake News Detector")
st.write(
    """
    Enter a **news article** (title and/or text) below.
    This app will analyze the news and predict whether it is **FAKE** or **REAL**.
    """
)

user_title = st.text_input("News Title")
user_text = st.text_area("News Text")

if st.button("Detect"):
    combined_text = user_title.strip() + " " + user_text.strip()
    if not combined_text.strip():
        st.warning("Please enter the news title and/or text.")
    else:
        # Vectorize and scale
        X_input_vec = vectorizer.transform([combined_text]).toarray()
        X_input_scaled = scaler.transform(X_input_vec)
        X_input_tensor = torch.tensor(X_input_scaled, dtype=torch.float32)

        # Predict
        with torch.no_grad():
            output = model(X_input_tensor).item()
            pred_label = 1 if output >= 0.5 else 0
            confidence = output if pred_label == 1 else 1 - output

        if pred_label == 1:
            st.success(f"🟢 This news is predicted to be **REAL**. (Confidence: {confidence:.2%})")
        else:
            st.error(f"🔴 This news is predicted to be **FAKE**. (Confidence: {confidence:.2%})")

        st.markdown("---")
        st.write("**Model Confidence:**", f"{confidence:.2%}")
        st.write("**(REAL = 1, FAKE = 0)**")

st.markdown(
    """
    ---
    **Note:** This tool is for educational use only. Model predictions should not be solely relied upon for critical decisions.
    """
)