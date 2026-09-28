"""
Run this ONCE (locally, not on every app start) to train the LDA model
from your notebook and save it to disk. Streamlit will just load the
saved files — it never retrains on each run.

Usage:
    python train_model.py
Needs bbc-text.csv in the same folder.
Produces: vectorizer.pkl, lda_model.pkl, topic_to_category.pkl
"""
import re
import joblib
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("punkt", quiet=True)

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def preprocess_tokens(tokens):
    return [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words and len(w) > 2]


def main():
    df = pd.read_csv("bbc-text.csv")
    df = df.drop_duplicates()

    df["clean_text"] = df["text"].apply(clean_text)
    df["tokens"] = df["clean_text"].apply(str.split)
    df["processed_tokens"] = df["tokens"].apply(preprocess_tokens)
    df["final_text"] = df["processed_tokens"].apply(lambda t: " ".join(t))

    vectorizer = CountVectorizer(max_df=0.9, min_df=5, stop_words="english")
    doc_term_matrix = vectorizer.fit_transform(df["final_text"])

    n_topics = 5
    lda_model = LatentDirichletAllocation(
        n_components=n_topics, random_state=42, max_iter=10, learning_method="online"
    )
    lda_model.fit(doc_term_matrix)

    # Map each LDA topic number to the real category it aligns with best
    doc_topics = lda_model.transform(doc_term_matrix)
    df["dominant_topic"] = doc_topics.argmax(axis=1)
    comparison = pd.crosstab(df["dominant_topic"], df["category"])
    topic_to_category = comparison.idxmax(axis=1).to_dict()

    joblib.dump(vectorizer, "vectorizer.pkl")
    joblib.dump(lda_model, "lda_model.pkl")
    joblib.dump(topic_to_category, "topic_to_category.pkl")
    print("Saved vectorizer.pkl, lda_model.pkl, topic_to_category.pkl")
    print("Topic mapping:", topic_to_category)


if __name__ == "__main__":
    main()
