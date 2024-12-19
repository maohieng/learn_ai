from collections import Counter
from itertools import islice
import json

def compute_ngrams(text: str, n=1):
    """
    Compute n-grams (unigrams, bigrams, trigrams) and return statistics.
    
    Args:
        text (str): The input text.
        n (int): The n-gram size (1 for unigrams, 2 for bigrams, 3 for trigrams).
    
    Returns:
        dict: A dictionary with statistics:
              - number of words (tokens)
              - number of different words (unique tokens)
              - words appearing n times
              - probability of each n-gram
    """
    # Tokenize the text into words
    words = text.lower().replace('.', '').split()
    
    # Generate n-grams
    ngrams = zip(*[islice(words, i, None) for i in range(n)])
    ngrams = [" ".join(ngram) for ngram in ngrams]
    
    # Compute frequency distribution
    ngram_counts = Counter(ngrams)
    total_ngrams = sum(ngram_counts.values())
    
    # Count the occurrences of n-grams appearing n times
    n_times = Counter(ngram_counts.values())
    
    # Calculate probabilities
    probabilities = {ngram: count / total_ngrams for ngram, count in ngram_counts.items()}
    
    return {
        "number_of_words": len(words),
        "number_of_different_words": len(set(words)),
        "ngrams_appearing_n_times": dict(n_times),
        "probabilities": probabilities,
    }

# Example usage
text = "CADT is the best engineering school in Cambodia in the field of computer science. The teachers are very friendly and competent."
unigrams = compute_ngrams(text, n=1)
bigrams = compute_ngrams(text, n=2)
trigrams = compute_ngrams(text, n=3)

with open("result.json", "w") as file:
    json.dump({
        "uni-gram": unigrams,
        "bi-grams": bigrams,
        "tri-grams": trigrams
    }, file)
