import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pathlib
from utils import create_embedding_array

def reduce_and_plot_data(word_to_embeddings, word_to_index, index_to_word, graph_name='2D PCA of Word Embeddings'):
    # Create an array of array 50-dimensional with zeros
    embeddings_array = create_embedding_array(word_to_embeddings, word_to_index)

    # Fit PCA on the embeddings
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(embeddings_array)

    reduced_word_embeddings = {}
    plt.figure(figsize=(20, 10))
    for i, embedding in enumerate(reduced_embeddings):
        word = index_to_word[i]
        reduced_word_embeddings[word] = embedding
        x, y = embedding
        plt.scatter(x, y)
        plt.text(x + 0.01, y + 0.01, word, fontsize=9, font=pathlib.Path('KhmerOSContent-Regular.ttf'))

    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title(graph_name)
    plt.show()
    
    return reduced_word_embeddings

def load_and_plot_data(file_word_embbedings, file_word_to_index, graph_name='2D PCA of Word Embeddings'):
    word_to_embeddings = np.load(file_word_embbedings, allow_pickle=True).item()
    
    word_to_index = np.load(file_word_to_index, allow_pickle=True).item()
    index_to_word = {v: k for k, v in word_to_index.items()}

    return reduce_and_plot_data(word_to_embeddings, word_to_index, index_to_word, graph_name)

def similarity(word, reduced_embeddings):
    '''
    Calculate the similarity of the word with other words (top 10).
    '''
    word1_embedding = reduced_embeddings[word]
    similarities = {}
    for w, embedding in reduced_embeddings.items():
        score = np.dot(word1_embedding, embedding) / (np.linalg.norm(word1_embedding) * np.linalg.norm(embedding))
        similarities[w] = score

    sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    return sorted_similarities[:10]