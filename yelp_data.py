import json
import sys

unknown_token = 'UNK'
start_token   = '<S>'
end_token     = '</S>'
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
