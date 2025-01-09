import joblib
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Tuple, Set
from flask import Flask, request, render_template, jsonify
from collections import Counter

# Load files
def load_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.readlines()

positive_reviews = load_file('positive-reviews.txt')
negative_reviews = load_file('negative-reviews.txt')

positive_reviews[:5], negative_reviews[:5]

positive_words = set(load_file('positive-words.txt'))
negative_words = set(load_file('negative-words.txt'))

def extract_features(reviews):
    features = []
    for i, review in enumerate(reviews):
        review = review.lower()
        tokens = re.findall(r'\b\w+\b', review)

        pos_count = sum(1 for word in tokens if word in positive_words)
        neg_count  = sum(1 for word in tokens if word in negative_words)
        contains_no = int('no' in tokens)
        pronoun_count = sum(1 for word in tokens if word in ['i', 'me', 'my', 'you', 'your'])
        contains_exclamation = int('!' in review)
        log_length = np.log(len(tokens) + 1)

        features.append([pos_count, neg_count, contains_no, pronoun_count, contains_exclamation, log_length])
    
    return np.array(features)

positive_labels = [1] * len(positive_reviews)
negative_labels = [0] * len(negative_reviews)

reviews = positive_reviews + negative_reviews
labels = positive_labels + negative_labels

# Load the model
model = joblib.load("Logistic Regression TF-IDF Features.pkl")

tfidf_vectorizer = TfidfVectorizer(max_features=5000)
tfidf_features = tfidf_vectorizer.fit_transform(reviews).toarray()


word_counts = Counter()
positive_counts = Counter()
negative_counts = Counter()

for review, label in zip(reviews, labels):
    tokens = set(re.findall(r'\b\w+\b', review.lower()))
    word_counts.update(tokens)
    if label == 1:
        positive_counts.update(tokens)
    else:
        negative_counts.update(tokens)

total_words = sum(word_counts.values())
positive_total = sum(positive_counts.values())
negative_total = sum(negative_counts.values())

vocab = tfidf_vectorizer.get_feature_names_out()

pmi_scores = {}
for word in vocab:
    p_word = word_counts[word] / total_words
    p_word_positive = (positive_counts[word] / positive_total) if word in positive_counts else 0
    p_word_negative = (negative_counts[word] / negative_total) if word in negative_counts else 0

    if p_word_positive > 0:
        pmi_scores[word] = np.log2(p_word_positive / p_word)
    elif p_word_negative > 0:
        pmi_scores[word] = -np.log2(p_word_negative / p_word)
    else:
        pmi_scores[word] = 0

def predict_review(review:str)->str:
    features = extract_features([review])
    tfidf_features = tfidf_vectorizer.transform([review]).toarray()
    # pmi_features = np.array([[pmi_scores.get(word, 0) for word in vocab]])
    # X = np.hstack([features, tfidf_features, pmi_features])
    X = np.hstack([features, tfidf_features])
    y_pred = model.predict(X)
    return 'Positive' if y_pred[0] == 1 else 'Negative'


app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/predict", methods=["POST"])
def predict():
    text = request.json["review"]
    result = predict_review(text)
    return jsonify({"prediction": result, "review": text})


if __name__ == "__main__":
    app.run(debug=True)
