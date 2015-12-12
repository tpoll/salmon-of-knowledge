import json
import sys
from collections import defaultdict
from sets import ImmutableSet


unknown_token = 'UNK'
start_token   = '<S>'
end_token     = '</S>'
dont_include = set([",", "\n"])
positive = ImmutableSet([4, 5])
negative = ImmutableSet([1, 2, 3])

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
    vocab.add(start_token)
    vocab.add(end_token)

    return vocab

def posVocabLen(vocab, corpus):
    used = set()
    length = 0
    for review in corpus:
        if review['stars'] in positive:
            for word in review['text']:
                if word not in used:
                    used.add(word)
                    length += 1
    return length


def negVocabLen(vocab, corpus):
    used = set()
    length = 0
    for review in corpus:
        if review['stars'] in negative:
            for word in review['text']:
                if word not in used:
                    used.add(word)
                    length += 1
    return length

def preProcess(corpus, vocab):
    for i, review in enumerate(corpus):
        for j, word in enumerate(review['text']):
            if word not in vocab:
                corpus[i]['text'][j] = unknown_token
    return corpus

def preProcessN(corpus, vocab, N):
    processed = preProcess(corpus, vocab)
    for review in processed:

        for i in range(N - 1):
            review['text'].insert(0, start_token)
            review['text'].append(end_token)

    return processed

def getReviews():
    with open("reviews.json", 'rb') as f:
        return json.load(f)

def getStopWords():
    with open("stopwords.txt", 'rb') as f:
        return set([x.strip('\n') for x in f.readlines()])
