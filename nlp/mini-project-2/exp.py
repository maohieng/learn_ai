import logging
import joblib
import os
import math
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Tuple, Set

# Configure logging
logging.basicConfig(
    filename="model_evaluation.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Constants
LOGGING_ENABLED = True
SAVE_MODELS = True


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

# Split into training and testing sets (80% train, 20% test)
X_train: List[str]
X_test: List[str]
y_train: List[int]
y_test: List[int]
X_train, X_test, y_train, y_test = train_test_split(
    all_reviews, all_labels, test_size=0.2, random_state=42
)

# Initialize TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer()
tfidf_vectorizer.fit(all_reviews)

# Extract features
X_train_features: np.ndarray
X_test_features: np.ndarray
X_train_features = extract_features(
    X_train, positive_words, negative_words, tfidf_vectorizer
)
X_test_features = extract_features(
    X_test, positive_words, negative_words, tfidf_vectorizer
)

# Extract features without TF-IDF
X_train_features_no_tfidf = extract_features_without_tfidf(
    X_train, positive_words, negative_words
)
X_test_features_no_tfidf = extract_features_without_tfidf(
    X_test, positive_words, negative_words
)

# Convert labels to categorical for DNN
y_train_categorical: np.ndarray
y_test_categorical: np.ndarray
y_train_categorical = to_categorical(y_train)
y_test_categorical = to_categorical(y_test)

# Ensure the models directory exists
os.makedirs("models", exist_ok=True)

# Train and evaluate a Logistic Regression model
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train_features, y_train)
log_reg_predictions = log_reg.predict(X_test_features)
log_reg_accuracy = accuracy_score(y_test, log_reg_predictions)
if LOGGING_ENABLED:
    logging.info(f"Logistic Regression Accuracy: {log_reg_accuracy:.2f}")
if SAVE_MODELS:
    joblib.dump(log_reg, "models/log_reg_model.pkl")

# Train and evaluate a Logistic Regression model without TF-IDF
log_reg_no_tfidf = LogisticRegression(max_iter=1000)
log_reg_no_tfidf.fit(X_train_features_no_tfidf, y_train)
log_reg_no_tfidf_predictions = log_reg_no_tfidf.predict(X_test_features_no_tfidf)
log_reg_no_tfidf_accuracy = accuracy_score(y_test, log_reg_no_tfidf_predictions)
if LOGGING_ENABLED:
    logging.info(
        f"Logistic Regression without TF-IDF Accuracy: {log_reg_no_tfidf_accuracy:.2f}"
    )
if SAVE_MODELS:
    joblib.dump(log_reg_no_tfidf, "models/log_reg_model_no_tfidf.pkl")

# Train and evaluate a Naive Bayes model
nb_model = MultinomialNB()
nb_model.fit(X_train_features, y_train)
nb_predictions = nb_model.predict(X_test_features)
nb_accuracy = accuracy_score(y_test, nb_predictions)
if LOGGING_ENABLED:
    logging.info(f"Naive Bayes Accuracy: {nb_accuracy:.2f}")
if SAVE_MODELS:
    joblib.dump(nb_model, "models/nb_model.pkl")

# Train and evaluate a Naive Bayes model without TF-IDF
nb_model_no_tfidf = MultinomialNB()
nb_model_no_tfidf.fit(X_train_features_no_tfidf, y_train)
nb_no_tfidf_predictions = nb_model_no_tfidf.predict(X_test_features_no_tfidf)
nb_no_tfidf_accuracy = accuracy_score(y_test, nb_no_tfidf_predictions)
if LOGGING_ENABLED:
    logging.info(f"Naive Bayes without TF-IDF Accuracy: {nb_no_tfidf_accuracy:.2f}")
if SAVE_MODELS:
    joblib.dump(nb_model_no_tfidf, "models/nb_model_no_tfidf.pkl")

# Train and evaluate a Random Forest model
rf_model = RandomForestClassifier()
rf_model.fit(X_train_features, y_train)
rf_predictions = rf_model.predict(X_test_features)
rf_accuracy = accuracy_score(y_test, rf_predictions)
if LOGGING_ENABLED:
    logging.info(f"Random Forest Accuracy: {rf_accuracy:.2f}")
if SAVE_MODELS:
    joblib.dump(rf_model, "models/rf_model.pkl")

# Train and evaluate a Random Forest model without TF-IDF
rf_model_no_tfidf = RandomForestClassifier()
rf_model_no_tfidf.fit(X_train_features_no_tfidf, y_train)
rf_no_tfidf_predictions = rf_model_no_tfidf.predict(X_test_features_no_tfidf)
rf_no_tfidf_accuracy = accuracy_score(y_test, rf_no_tfidf_predictions)
if LOGGING_ENABLED:
    logging.info(f"Random Forest without TF-IDF Accuracy: {rf_no_tfidf_accuracy:.2f}")
if SAVE_MODELS:
    joblib.dump(rf_model_no_tfidf, "models/rf_model_no_tfidf.pkl")

# Define and compile the DNN model
dnn_model = Sequential()
dnn_model.add(Dense(64, input_dim=X_train_features.shape[1], activation="relu"))
dnn_model.add(Dense(32, activation="relu"))
dnn_model.add(Dense(2, activation="softmax"))

dnn_model.compile(
    loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
)

# Train the DNN model
dnn_model.fit(
    X_train_features, y_train_categorical, epochs=10, batch_size=32, verbose=0
)

# Evaluate the DNN model
dnn_loss, dnn_accuracy = dnn_model.evaluate(
    X_test_features, y_test_categorical, verbose=0
)
if LOGGING_ENABLED:
    logging.info(f"DNN Accuracy: {dnn_accuracy:.2f}")
if SAVE_MODELS:
    dnn_model.save("models/dnn_model.keras")

# Define and compile the DNN model without TF-IDF
dnn_model_no_tfidf = Sequential()
dnn_model_no_tfidf.add(
    Dense(64, input_dim=X_train_features_no_tfidf.shape[1], activation="relu")
)
dnn_model_no_tfidf.add(Dense(32, activation="relu"))
dnn_model_no_tfidf.add(Dense(2, activation="softmax"))

dnn_model_no_tfidf.compile(
    loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
)

# Train the DNN model without TF-IDF
dnn_model_no_tfidf.fit(
    X_train_features_no_tfidf, y_train_categorical, epochs=10, batch_size=32, verbose=0
)

# Evaluate the DNN model without TF-IDF
dnn_no_tfidf_loss, dnn_no_tfidf_accuracy = dnn_model_no_tfidf.evaluate(
    X_test_features_no_tfidf, y_test_categorical, verbose=0
)
if LOGGING_ENABLED:
    logging.info(f"DNN without TF-IDF Accuracy: {dnn_no_tfidf_accuracy:.2f}")
if SAVE_MODELS:
    dnn_model_no_tfidf.save("models/dnn_model_no_tfidf.keras")


def load_model_and_test(
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

    # Print predictions
    for review, prediction in zip(test_reviews, predictions):
        print(
            f"Review: {review.strip()}\nPrediction: {'Positive' if prediction == 1 else 'Negative'}\n"
        )


# Test the Logistic Regression model
load_model_and_test(
    "models/log_reg_model.pkl",
    "test-reviews.txt",
    positive_words,
    negative_words,
    tfidf_vectorizer,
    use_tfidf=True,
)

# Test the Logistic Regression model without TF-IDF
load_model_and_test(
    "models/log_reg_model_no_tfidf.pkl",
    "test-reviews.txt",
    positive_words,
    negative_words,
    use_tfidf=False,
)

# Test the Naive Bayes model
load_model_and_test(
    "models/nb_model.pkl",
    "test-reviews.txt",
    positive_words,
    negative_words,
    tfidf_vectorizer,
    use_tfidf=True,
)

# Test the Naive Bayes model without TF-IDF
load_model_and_test(
    "models/nb_model_no_tfidf.pkl",
    "test-reviews.txt",
    positive_words,
    negative_words,
    use_tfidf=False,
)

# Test the Random Forest model
load_model_and_test(
    "models/rf_model.pkl",
    "test-reviews.txt",
    positive_words,
    negative_words,
    tfidf_vectorizer,
    use_tfidf=True,
)

# Test the Random Forest model without TF-IDF
load_model_and_test(
    "models/rf_model_no_tfidf.pkl",
    "test-reviews.txt",
    positive_words,
    negative_words,
    use_tfidf=False,
)

# Test the DNN model
load_model_and_test(
    "models/dnn_model.keras",
    "test-reviews.txt",
    positive_words,
    negative_words,
    tfidf_vectorizer,
    use_tfidf=True,
)

# Test the DNN model without TF-IDF
load_model_and_test(
    "models/dnn_model_no_tfidf.keras",
    "test-reviews.txt",
    positive_words,
    negative_words,
    use_tfidf=False,
)
