import sys
assert sys.version_info[0]==3
assert sys.version_info[1] >= 5

from gensim.models import KeyedVectors
from gensim.test.utils import datapath
import pprint
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [10, 5]
import nltk
from nltk.corpus import reuters
import numpy as np
import random
import scipy as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA
from gensim.scripts.glove2word2vec import glove2word2vec

START_TOKEN = '<START>'
END_TOKEN = '<END>'
glove_file = 'glove.6B.200d.txt'
word2vec_file = 'glove.6B.200d.word2vec.txt'

def read_corpus(category="crude"):
    files = reuters.fileids(category)
    return [[START_TOKEN] + [w.lower() for w in list(reuters.words(f))] + [END_TOKEN] for f in files]

def distinct_words(corpus):
    corpus_words = []
    num_corpus_words = -1
    # ------------------
    # Write your implementation here.
    corpus_words = [y for x in corpus for y in x]
    corpus_words = list(set(corpus_words))
    corpus_words = sorted(corpus_words)
    num_corpus_words = len(corpus_words)
    # ------------------
    return corpus_words, num_corpus_words

def compute_co_occurrence_matrix(corpus, window_size=4):
    words, num_words = distinct_words(corpus)
    #print(words)
    M = None
    word2Ind = {}
    # ------------------
    # Write your implementation here.
    M = np.zeros((num_words,num_words))
    for y in corpus:
        #print(y)
        for i in range(len(y)):
            if i-window_size < 0:
                start = 0
            else:
                start = i-window_size
            if i+window_size > len(y):
                end = len(y)
            else:
                end = i+window_size
            phrase = y[start:end+1]
            index = words.index(y[i])
            for x in phrase:
                if x != y[i]:
                    now = words.index(x)
                    M[now][index] += 1

    word2Ind = dict(zip(words, range(len(words))))
    # ------------------
    return M, word2Ind

def reduce_to_k_dim(M, k=2):
    n_iters = 10     # Use this parameter in your call to `TruncatedSVD`
    M_reduced = None
    print("Running Truncated SVD over %i words..." % (M.shape[0]))
        # ------------------
        # Write your implementation here.
    svd = TruncatedSVD(n_components=k, n_iter=n_iters, random_state=123, tol=0.0)
    M_reduced =svd.fit_transform(M)
    # M_reduced = 
        # ------------------
    print("Done.")
    return M_reduced

def plot_embeddings(M_reduced, word2Ind, words):
    # ------------------
    # Write your implementation here.
    fig, ax = plt.subplots()
    plt.xlabel('x')
    plt.ylabel('y')
    #plt.scatter(M_reduced[:,0],M_reduced[:,1])
    #word2Ind = sorted(word2Ind.items(), key = lambda item:item[1])
    #word_name = word2Ind.keys()
    #print(word_name)
    plot_cord = []
    for x in words:
        direct = word2Ind[x]
        plot_cord.append(M_reduced[direct])
        #print(plot_cord)
    # print(word2Ind)
    for i in range(len(words)):
        plt.scatter(plot_cord[i][0], plot_cord[i][1])
        plt.annotate(words[i], xy=(plot_cord[i][0], plot_cord[i][1]), xytext=(plot_cord[i][0], plot_cord[i][1]))

    plt.savefig("test.jpg") 

    # ------------------

def get_matrix_of_vectors(wv_from_bin, required_words=['barrels', 'bpd', 'ecuador', 'energy', 'industry', 'kuwait', 'oil', 'output', 'petroleum', 'venezuela']):
    import random
    words = list(wv_from_bin.vocab.keys())
    print("Shuffling words ...")
    random.seed(224)
    random.shuffle(words)
    words = words[:10000]
    print("Putting %i words into word2Ind and matrix M..." % len(words))
    word2Ind = {}
    M = []
    curInd = 0
    for w in words:
        try:
            M.append(wv_from_bin.word_vec(w))
            word2Ind[w] = curInd
            curInd += 1
        except KeyError:
            continue
    for w in required_words:
        if w in words:
            continue
        try:
            M.append(wv_from_bin.word_vec(w))
            word2Ind[w] = curInd
            curInd += 1
        except KeyError:
            continue
    print(np.shape(M))
    M = np.stack(M)
    print(np.shape(M))
    print("Done.")
    return M, word2Ind

wv_from_bin = KeyedVectors.load_word2vec_format(word2vec_file)
# -----------------------------------------------------------------
# Run Cell to Reduce 200-Dimensional Word Embeddings to k Dimensions
# Note: This should be quick to run
# -----------------------------------------------------------------
# words = glove_model.most_similar('woman')
# print(words)
# print('-'*80)
# words = glove_model.most_similar('man')
# print(words)

# w1 = 'happy'
# w2 = 'cheerful'
# w3 = 'sad'

# d1 = glove_model.distance(w1, w2)
# d2 = glove_model.distance(w1, w3)

# print(d1, d2)

# pprint.pprint(glove_model.most_similar(positive=['woman', 'king'], negative=['man']))
pprint.pprint(wv_from_bin.most_similar(positive=['woman', 'worker'], negative=['man']))
print()
pprint.pprint(wv_from_bin.most_similar(positive=['man', 'worker'], negative=['woman']))