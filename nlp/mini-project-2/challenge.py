import logging
import joblib
import os
import math
import numpy as np
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Tuple, Set


# Load data (Assuming positive-reviews.txt and negative-reviews.txt exist)
def load_data(positive_file: str, negative_file: str) -> Tuple[List[str], List[str]]:
    with open(positive_file, "r") as pos_file:
        positive_reviews = pos_file.readlines()
    with open(negative_file, "r") as neg_file:
        negative_reviews = neg_file.readlines()
    return positive_reviews, negative_reviews


# Feature Extraction
def extract_features(
    reviews: List[str],
    positive_words: Set[str],
    negative_words: Set[str],
    tfidf_vectorizer: TfidfVectorizer,
) -> np.ndarray:
    features = []
    tfidf_matrix = tfidf_vectorizer.transform(reviews)
    for i, review in enumerate(reviews):
        # Count of positive words
        positive_count = sum(1 for word in review.split() if word in positive_words)
        # Count of negative words
        negative_count = sum(1 for word in review.split() if word in negative_words)
        # Presence of the word "no"
        contains_no = int("no" in review.split())
        # Count of first and second pronouns
        pronouns = {"I", "me", "my", "you", "your"}
        pronoun_count = sum(1 for word in review.split() if word in pronouns)
        # Presence of "I"
        contains_I = int("I" in review.split())
        # Logarithm of the length of the review
        review_length_log = math.log(len(review.split()) + 1)
        # TF-IDF features
        tfidf_features = tfidf_matrix[i].toarray()[0]
        # Append all features
        features.append(
            [
                positive_count,
                negative_count,
                contains_no,
                pronoun_count,
                contains_I,
                review_length_log,
                *tfidf_features,
            ]
        )
    return np.array(features)


# Feature Extraction without TF-IDF
def extract_features_without_tfidf(
    reviews: List[str],
    positive_words: Set[str],
    negative_words: Set[str],
) -> np.ndarray:
    features = []
    for review in reviews:
        # Count of positive words
        positive_count = sum(1 for word in review.split() if word in positive_words)
        # Count of negative words
        negative_count = sum(1 for word in review.split() if word in negative_words)
        # Presence of the word "no"
        contains_no = int("no" in review.split())
        # Count of first and second pronouns
        pronouns = {"I", "me", "my", "you", "your"}
        pronoun_count = sum(1 for word in review.split() if word in pronouns)
        # Presence of "I"
        contains_I = int("I" in review.split())
        # Logarithm of the length of the review
        review_length_log = math.log(len(review.split()) + 1)
        # Append all features
        features.append(
            [
                positive_count,
                negative_count,
                contains_no,
                pronoun_count,
                contains_I,
                review_length_log,
            ]
        )
    return np.array(features)


# Load positive and negative word lists
def load_word_lists(
    positive_file: str, negative_file: str
) -> Tuple[Set[str], Set[str]]:
    with open(positive_file, "r") as pos_file:
        positive_words = set(pos_file.read().splitlines())
    with open(negative_file, "r") as neg_file:
        negative_words = set(neg_file.read().splitlines())
    return positive_words, negative_words


positive_reviews: List[str]
negative_reviews: List[str]
positive_reviews, negative_reviews = load_data(
    "positive-reviews.txt", "negative-reviews.txt"
)
positive_words: Set[str]
negative_words: Set[str]
positive_words, negative_words = load_word_lists(
    "positive-words.txt", "negative-words.txt"
)

# Label data (1 for positive, 0 for negative)
positive_labels: List[int]
negative_labels: List[int]
positive_labels = [1] * len(positive_reviews)
negative_labels = [0] * len(negative_reviews)

# Combine data and labels
all_reviews: List[str]
all_labels: List[int]
all_reviews = positive_reviews + negative_reviews
all_labels = positive_labels + negative_labels

tfidf_vectorizer = TfidfVectorizer()
tfidf_vectorizer.fit(all_reviews)


def test_challenge(
    model_path: str,
    test_file: str,
    positive_words: Set[str],
    negative_words: Set[str],
    tfidf_vectorizer: TfidfVectorizer = None,
    use_tfidf: bool = True,
) -> None:
    # Load the model
    if model_path.endswith(".pkl"):
        model = joblib.load(model_path)
    elif model_path.endswith(".keras"):
        model = tf.keras.models.load_model(model_path)
    else:
        raise ValueError("Unsupported model format")

    # Load test data
    with open(test_file, "r") as file:
        test_reviews = file.readlines()

    # Extract features
    if use_tfidf:
        test_features = extract_features(
            test_reviews, positive_words, negative_words, tfidf_vectorizer
        )
    else:
        test_features = extract_features_without_tfidf(
            test_reviews, positive_words, negative_words
        )

    # Predict
    if model_path.endswith(".keras"):
        predictions = model.predict(test_features)
        predictions = np.argmax(predictions, axis=1)
    else:
        predictions = model.predict(test_features)

    if os.path.exists("challenge_output.txt"):
        os.remove("challenge_output.txt")

    # Print predictions
    for review, prediction in zip(test_reviews, predictions):
        with open("challenge_output.txt", "a") as file:
            file.write(f"{prediction}")


test_challenge(
    "models/log_reg_model.pkl",
    "challenge_data.txt",
    positive_words,
    negative_words,
    tfidf_vectorizer,
    use_tfidf=True,
)
