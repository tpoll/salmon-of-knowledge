import json
import sys
from collections import defaultdict
import spacy.en
from sets import ImmutableSet
STARS = 0
TEXT = 1

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
        for word in review[TEXT]:
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
        if review[STARS] in positive:
            for word in review[TEXT]:
                if word not in used:
                    used.add(word)
                    length += 1
    return length


def negVocabLen(vocab, corpus):
    used = set()
    length = 0
    for review in corpus:
        if review[STARS] in negative:
            for word in review[TEXT]:
                if word not in used:
                    used.add(word)
                    length += 1
    return length

def preProcess(corpus, vocab):
    for i, review in enumerate(corpus):
        for j, word in enumerate(review[TEXT]):
            if word not in vocab:
                corpus[i][TEXT][j] = unknown_token
    return corpus

def preProcessN(corpus, vocab, N):
    processed = preProcess(corpus, vocab)
    for review in processed:

        for i in range(N - 1):
            review[TEXT].insert(0, start_token)
            review[TEXT].append(end_token)

    return processed

def getReviewsTokenized():
    nlp = spacy.en.English(parser=False, tagger=False, entity=False)
    data = []
    with open("reviews.json", 'rb') as f:
        reviews = json.load(f)
        for review in reviews:
            token = nlp(unicode(review['text']))
            data.append([review['stars'], [tok.string for tok in token if not (tok.string.isspace() or '"' in tok.string)]])
    return data

def getStopWords():
    with open("stopwords.txt", 'rb') as f:
        return set([x.strip('\n') for x in f.readlines()])
