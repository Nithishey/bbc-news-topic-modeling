# 📰 BBC News Topic Modeling with LDA + Streamlit App

An NLP project that discovers hidden topics in BBC news articles using **Latent Dirichlet Allocation (LDA)** and serves predictions through an interactive **Streamlit** web app. A companion **Power BI** dashboard explores the same dataset.

## 🎯 Problem Statement
News articles arrive unlabeled and in huge volumes. Can an unsupervised model group them into meaningful topics, and how closely do those topics match the real editorial categories (business, entertainment, politics, sport, tech)?

## 📊 Dataset
- BBC News dataset: 2,225 articles across 5 categories
- 2,126 articles remained after removing duplicates

## 🛠️ Tech Stack
Python · pandas · NLTK · scikit-learn · Matplotlib · WordCloud · Streamlit · Power BI

## 🔄 Workflow
1. **Data cleaning:** duplicate removal, lowercasing, regex cleaning
2. **Preprocessing:** tokenization, stopword removal, lemmatization
3. **Feature extraction:** `CountVectorizer` (bag-of-words)
4. **Modeling:** 5-topic `LatentDirichletAllocation`
5. **Evaluation:** topic-vs-category crosstab, perplexity diagnostics
6. **Visualization:** top-word bar charts and word clouds per topic
7. **Deployment:** Streamlit app that classifies new articles in real time
8. **BI layer:** Power BI dashboard (category distribution, text length by category, LDA results, slicer)

## ✅ Results
- Mapping each LDA topic to its best-matching real category gave **89.98% alignment** with the true labels.
- Tested on 5 new unseen sample articles (one per category): all predicted correctly.
- Note: LDA is unsupervised, and this alignment was measured on the same dataset the model was fitted on, so it shows how well topics match categories rather than generalization accuracy.

## 🚀 Run Locally
```bash
pip install -r requirements.txt
python train_model.py          # trains the model and creates the .pkl files
python -m streamlit run app.py # launches the web app
```

## 📁 Project Structure
```
├── app.py                          # Streamlit web app
├── train_model.py                  # trains and saves the LDA pipeline
├── News_Article_Topic_Modeling.ipynb  # full analysis notebook
├── bbc-text.csv                    # dataset
├── vectorizer.pkl / lda_model.pkl / topic_to_category.pkl  # saved model
└── requirements.txt
```

## 🔭 Future Improvements
- Compare against a supervised baseline (TF-IDF + Logistic Regression)
- Show top keywords driving each prediction in the app
- Tune the number of topics using coherence scores

## 👤 Author
**Nithish** · [LinkedIn](https://www.linkedin.com/in/nithishey) · [GitHub](https://github.com/Nithishey)
