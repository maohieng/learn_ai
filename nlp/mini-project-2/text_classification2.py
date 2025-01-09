import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from collections import Counter

# Load files
def load_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.readlines()

positive_reviews = load_file('positive-reviews.txt')
negative_reviews = load_file('negative-reviews.txt')
positive_words = set(load_file('positive-words.txt'))
negative_words = set(load_file('negative-words.txt'))

# Preprocess data
def extract_features(reviews):
    features = []
    for review in reviews:
        review = review.lower()
        tokens = re.findall(r'\b\w+\b', review)
        
        pos_count = sum(1 for word in tokens if word in positive_words)
        neg_count = sum(1 for word in tokens if word in negative_words)
        contains_no = int('no' in tokens)
        pronoun_count = sum(1 for word in tokens if word in {'i', 'me', 'my', 'you', 'your'})
        contains_exclamation = int('!' in review)
        log_length = np.log(len(tokens) + 1)

        features.append([pos_count, neg_count, contains_no, pronoun_count, contains_exclamation, log_length])

    return np.array(features)

# Compute PMI scores
def compute_pmi(reviews, labels, vocab):
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

    return pmi_scores

# Create labels and split data
positive_labels = [1] * len(positive_reviews)
negative_labels = [0] * len(negative_reviews)

reviews = positive_reviews + negative_reviews
labels = positive_labels + negative_labels

# Extract manual features
manual_features = extract_features(reviews)

# Extract TF-IDF features
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
tfidf_features = tfidf_vectorizer.fit_transform(reviews).toarray()

# Compute PMI features
vocab = tfidf_vectorizer.get_feature_names_out()
pmi_scores = compute_pmi(reviews, labels, vocab)
pmi_features = np.array([[pmi_scores.get(word, 0) for word in vocab] for review in reviews])

# Combine all features
X = np.hstack([manual_features, tfidf_features, pmi_features])
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Train and evaluate models
def train_and_evaluate_model(model, model_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy of {model_name}: {accuracy:.4f}")

models = [
    (LogisticRegression(), "Logistic Regression"),
    # (RandomForestClassifier(), "Random Forest"),
    # (SVC(), "Support Vector Machine")
]

for model, model_name in models:
    train_and_evaluate_model(model, model_name)
