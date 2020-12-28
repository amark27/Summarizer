from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

import nltk
import string
import pandas as pd
import numpy as np
from unidecode import unidecode

#download assets from nltk
# nltk.download('stopwords')
# nltk.download('punkt')

def tfidf(corpus):
    '''
    Computes the TF-IDF (term frequency - inverse document frequency) matrix

    Args
    - corpus: a list of sentences (documents) that need to be summarized (m x n matrix)
    m = number of different terms used in the documents, n = number of documents (not 0)

    Returns
    - tfidf_vec: an m x n matrix of the corpus
    - vocab: all the unique words used in the corpus, excluding stop words
    '''
    # calculate term frequency matrix
    num_docs = len(corpus)
    stop_words = stopwords.words('english')
    punctuation = string.punctuation + "''" + "..." + "``"
    word_sentence = []
    vocab = set()

    # sanitize text and break up each sentence into individual words
    for doc in corpus:
        #sanitize_text = doc.translate(str.maketrans('', '', string.punctuation))
        sanitize_text = doc
        tokenized = [word.lower() for word in word_tokenize(sanitize_text)]
        tokenized = [word for word in tokenized if word not in stop_words and word not in punctuation]
        word_sentence.append(tokenized)    
        vocab = vocab.union(set(tokenized))
    
    word_ind = {word : i for i, word in enumerate(vocab)}
    tf = np.zeros((len(vocab), num_docs))

    for i, words in enumerate(word_sentence):
        for word in words:
            tf[word_ind[word], i] += 1
    
    dft = np.sum(np.greater(tf, [0]).astype(float), axis=1)
    idf = np.log(np.divide([num_docs], dft))
    tfidf_vec= tf * np.expand_dims(idf, axis=1)

    return tfidf_vec, vocab

def svd(doc_term_matrix):
    '''
    Gives the singular value decomposition of an m x n matrix.
    A = U * sigma * V^t
    
    Args
    - doc_term_matrix: an m x n matrix. m = number of different terms used in the documents, n = number of documents

    Returns
    - u: an m x r matrix of left singular values (term-topic table). r = number of topics
    - sigma: an r x r diagonal matrix of singular values in decreasing order across the diagonal
    - v_t: an n x r matrix of right singular values (document-topic table)
    '''

    u, sigma, v_t = np.linalg.svd(doc_term_matrix, full_matrices=False)
    return u, sigma, v_t

def weigh_sentence_importance(v_t, sigma):
    '''
    Uses the LSA enhancement described by Josef Steinberg, et al. to weigh
    sentence importance from topics
    Takes all topics that have singular values > half of the largest singular value

    Compute s_k = sqrt(sum(v_ki^2 * sigma_i^2) from i = 1 to n) for all sentences
    s_k is the length of the vector of the kth sentence
    n is the number of topics 

    Args
    - v_t, sigma matrices from SVD

    Returns
    - Vector of each sentence weight as calculated above (1 x m)
    '''

    #look for the sigma value range that we need to consider using binary search
    #sigma array is sorted in descending order and will never be empty
    l, r, target = 0, len(sigma), sigma[0]/2
    while l < r:
        mid = l + (r-l)//2

        if sigma[mid] < target:
            r = mid
        else:
            l = mid + 1
    sigma_bound = l

    v_t_slice = v_t[:, :sigma_bound]
    sigma_slice = sigma[:sigma_bound]
    v_t_sq = np.square(v_t_slice)
    sig_sq = np.square(np.diag(sigma_slice))
    prod = np.matmul(v_t_sq, sig_sq)
    s = np.sqrt(np.sum(prod, axis = 1)).T

    return s

def get_important_sentences(v_t, sigma):
    '''
    Based on the sentence importance results, sort the indices to return indices that correspond to the
    most importance sentence to least important

    Args
    - v_t, sigma matrices from SVD

    Returns
    - Vector of sentence indices in descending order of weight (1 x m)
    '''

    return (-weigh_sentence_importance(v_t, sigma)).argsort()

def create_word_to_sentence_map(corpus):
    '''
    Creates a dictionary that maps a word from the vocab to all sentences with that word in the corpus.

    Args
    - corpus of sentences used in this summary

    Returns
    - the dictionary described
    '''
    
    word_to_sentence = {}
    stop_words = set(stopwords.words('english'))

    for i, doc in enumerate(corpus):
        #remove punctuation while preserving contractions in text
        sanitize_text = doc.translate(str.maketrans('', '', string.punctuation))
        tokenized = word_tokenize(sanitize_text)
        #remove duplicate words
        tokenized = list(set([word.lower() for word in tokenized]))

        for word in tokenized:
            if word not in stop_words:
                if word not in word_to_sentence:
                    word_to_sentence[word] = [i]
                else:
                    word_to_sentence[word].append(i)
    
    return word_to_sentence

def extract_summary(v_t, sigma, k, corpus):
    '''
    Helper method to get the text summary.

    Summary will be taken from the top k sentences from getImportantSentences()
    for each topic.

    Args
    - v_t, sigma from SVD
    - k: number of sentences to include in summary
    - corpus: the list of sentences

    Returns
    - the list of strings for the summary
    '''

    return [corpus[i] for i in get_important_sentences(v_t, sigma)[:k]]

def preprocess(block_text):
    '''
    Preprocesses the original text to be summarized by tokenizing the sentences and removing
    unnecessary characters.

    Args
    - block_text: text to be summarized

    Returns
    - list of sentences that can be used to create a summary
    '''

    tokenized = sent_tokenize(unidecode(block_text)) 
    return [token.replace('\n',' ') for token in tokenized]

def test_similarity(summary, u_orig, sigma_orig):
    '''
    Tests similarity by looking at the term significance of the original text and summary.
    Uses cosine similarity to do this.

    Args
    - summary: a list of strings that make up the summary
    - u_orig: the u matrix from SVD of the original text (n x r)
    - sigma_orig: the sigma matrix from SVD of the original text (1 x n)

    Returns
    - cosine similarity
    '''

    summary_corpus, _ = tfidf(summary)
    u_summary, sigma_summary, vt_summary = svd(summary_corpus)
    s_summary = weigh_sentence_importance(u_summary, sigma_summary)
    s_orig = weigh_sentence_importance(u_orig, sigma_orig)

    # summary will always be shorter vector than the original so scale down original
    s_orig = s_orig[:s_summary.shape[0]]

    # normalize both vectors (both should have non-zero magnitude)
    s_summary_norm = s_summary / np.linalg.norm(s_summary)
    s_orig_norm = s_orig / np.linalg.norm(s_orig)

    # dot product 2 normalized vectors = cosine similarity
    return np.dot(s_summary_norm, s_orig_norm)