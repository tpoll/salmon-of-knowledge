import json
import sys

unknown_token = 'UNK'
dont_include = set([",", "\n"])

def buildVocab(corpus):
    seen = set()
    vocabulary = set()
    for review in corpus:
        for word in review['text']:
            if word in seen and word not in dont_include:
                vocabulary.add(word)
            else:
                seen.add(word)
    vocabulary.add(unknown_token)

    return vocabulary

def preProcess(corpus, vocab):
    for i, review in enumerate(corpus):
        for j, word in enumerate(review['text']):
            if word not in vocab:
                corpus[i]['text'][j] = unknown_token
    return corpus

def getReviews():
    with open("reviews.json", 'rb') as f:
        return json.load(f)
