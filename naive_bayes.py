from yelp_data import getReviews
from collections import defaultdict
import json

unknown_token = 'UNK'

def buildVocab(corpus):
    seen = set()
    vocabulary = set()
    for review in corpus:
        for word in review['text']:
            if word in seen:
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


class NaiveBayes(object):
    """NaiveBayes for sentiment analysis"""
    def __init__(self, arg):
        self.postive_probs = defaultdict(lambda: float("-inf"))
        self.negative_probs = defaultdict(lambda: float("-inf"))

        
    def Train(training_set):
        pass


def main():
    reviews = getReviews()
    vocab = buildVocab(reviews)
    training_set_prep = preProcess(reviews, vocab)
    print training_set_prep[0]
    


if __name__ == '__main__':
    main()