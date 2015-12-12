import json
import sys
from collections import defaultdict


unknown_token = 'UNK'
dont_include = set([",", "\n"])

def buildVocab(corpus):
    counts = defaultdict(int)
    vocab = set()
    for review in corpus:
        for word in review['text']:
             counts[word] += 1

    for word, count in counts.iteritems():
        if count > 5:
            vocab.add(word)

    vocab.add(unknown_token)
    return vocab


def preProcess(corpus, vocab):
    for i, review in enumerate(corpus):
        for j, word in enumerate(review['text']):
            if word not in vocab:
                corpus[i]['text'][j] = unknown_token
    return corpus

def getReviews():
    with open("reviews.json", 'rb') as f:
        return json.load(f)

def getStopWords():
    with open("stopwords.txt", 'rb') as f:
        return set([x.strip('\n') for x in f.readlines()])
