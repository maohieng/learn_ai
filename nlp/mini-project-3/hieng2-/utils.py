import numpy as np
from khmernltk import word_tokenize

EMBEDDING_DIM = 50
CONTEXT_WINDOW = 4
NEGATIVE_SAMPLES = 2
N = 5
H = 512

UNKNOWN_TOKEN = "<ចម>"

def create_embedding_array(words_embedding, word_to_index):
    # Create an array of array 50-dimensional with zeros
    embeddings_array = np.zeros((len(words_embedding), EMBEDDING_DIM))

    # Fill the array with the embeddings
    for word, embedding in words_embedding.items():
        index = word_to_index[word]
        embeddings_array[index] = embedding

    return embeddings_array

def map_embbedings_to_word(embbedings, word_to_index):
    """
    Map the embeddings to the word.
    """
    word_to_embbedings = {}
    for word, index in word_to_index.items():
        word_to_embbedings[word] = embbedings[index]
    return word_to_embbedings

def predict_next_word(model, sentence, word_to_index, index_to_word, vocabs):
    _tokens = word_tokenize(sentence)
    if len(_tokens) < N:
        raise ValueError(f"Expected {N} words, got {len(_tokens)}")
    
    last_2_words = _tokens[-2:]
    # Take the last N words
    _tokens = _tokens[-N:]

    x = np.array([[word_to_index[w] if w in vocabs else word_to_index[UNKNOWN_TOKEN] for w in _tokens]])
    y = model.predict(x)

    # Get 5 words with the highest probability
    top_indices = np.argsort(y[0])[::-1][:10]

    # Get the words
    top_words = [index_to_word[i] for i in top_indices]

    for w in top_words:
        if w not in last_2_words and w != UNKNOWN_TOKEN:
            return w
    
    return top_words[-1]

def generate_text(model, seed, word_to_index, index_to_word, vocabs, n_words=100):
    sentence = seed
    for _ in range(n_words):
        sentence += predict_next_word(model, sentence, word_to_index, index_to_word, vocabs)
    return sentence