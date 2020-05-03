from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation as LDA
from sklearn.manifold import TSNE
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns


def get_ngrams(corpus, ngram_range=(1, 1)):
    vec = CountVectorizer(ngram_range=ngram_range).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return words_freq


class TopicModeling(object):
    def __init__(self, n_topics, method='LSA'):
        assert method in ['LSA', 'LDA']
        if method == 'LDA': raise NotImplementedError
        self.method = method
        self.n_topics = n_topics
        self.tfidf = TfidfVectorizer()
        if method == 'LSA':
            self.model = TruncatedSVD(n_components=n_topics)
        else:
            self.model = LDA(n_components=n_topics)

    def __call__(self, corpus):
        self.term_matrix = self.tfidf.fit_transform(corpus)
        self.topic_matrix = self.model.fit_transform(self.term_matrix)
        self.topic_keys = self.topic_matrix.argmax(axis=1).tolist()

    def get_count_pairs(self):
        return np.unique(self.topic_keys, return_counts=True)

    def get_top_n_words(self, n):
        '''
        returns a list of n_topic strings, where each string contains the n most common
        words in a predicted category (topic), in order
        '''
        top_word_indices = []
        for topic in range(self.n_topics):
            temp_vector_sum = 0
            for j in range(len(self.topic_keys)):
                if self.topic_keys[j] == topic:
                    temp_vector_sum += self.term_matrix[j]
            temp_vector_sum = temp_vector_sum.toarray()
            top_n_word_indices = np.flip(np.argsort(temp_vector_sum)[0][-n:], 0)
            top_word_indices.append(top_n_word_indices)
        top_words = []
        for topic in top_word_indices:
            topic_words = []
            for index in topic:
                temp_word_vector = np.zeros((1, self.term_matrix.shape[1]))
                temp_word_vector[:, index] = 1
                the_word = self.tfidf.inverse_transform(temp_word_vector)[0][0]
                topic_words.append(the_word.encode('ascii').decode('utf-8'))
            top_words.append(" ".join(topic_words))
        return top_words

    def plot_tsne(self, n_components=2):
        topic_embedding = TSNE(n_components=n_components).fit_transform(self.topic_matrix)
        _, ax = plt.subplots(figsize=(16, 10))
        scatter = ax.scatter(topic_embedding[:, 0], topic_embedding[:, 1], c=self.topic_keys, cmap='tab20')
        legend = ax.legend(*scatter.legend_elements(), title='Topics')
        ax.add_artist(legend)
