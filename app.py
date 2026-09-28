import re
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="BBC Topic Classifier", page_icon="📰", layout="centered")

# ---------- load saved model (cached so it loads once, not on every click) ----------
@st.cache_resource
def load_artifacts():
    vectorizer = joblib.load("vectorizer.pkl")
    lda_model = joblib.load("lda_model.pkl")
    topic_to_category = joblib.load("topic_to_category.pkl")
    return vectorizer, lda_model, topic_to_category

vectorizer, lda_model, topic_to_category = load_artifacts()


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def predict_topic(text):
    cleaned = clean_text(text)
    vectorized = vectorizer.transform([cleaned])
    topic_probs = lda_model.transform(vectorized)[0]
    predicted_topic = topic_probs.argmax()
    return predicted_topic, topic_probs

CATEGORY_EMOJI = {
    "business": "💼", "politics": "🏛️", "sport": "⚽",
    "tech": "💻", "entertainment": "🎬",
}

# ---------------------------- UI ----------------------------
st.title("📰 BBC News Topic Classifier")
st.caption("LDA topic model trained on the BBC news dataset — paste an article and see the predicted category.")

text = st.text_area("Paste a news article:", height=200, placeholder="Type or paste article text here...")

if st.button("Classify", type="primary", use_container_width=True):
    if not text.strip():
        st.warning("Please paste some text first.")
    else:
        topic_idx, probs = predict_topic(text)
        category = topic_to_category[topic_idx]
        emoji = CATEGORY_EMOJI.get(category, "📄")

        st.success(f"{emoji} **Predicted category: {category.title()}**")

        # probability breakdown as a simple bar chart
        prob_df = pd.DataFrame({
            "category": [topic_to_category[i] for i in range(len(probs))],
            "probability": probs,
        }).sort_values("probability", ascending=False)
        st.bar_chart(prob_df.set_index("category"))

st.divider()
st.caption("Model: CountVectorizer + LatentDirichletAllocation (scikit-learn) · 5 topics")
