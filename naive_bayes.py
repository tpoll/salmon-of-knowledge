from yelp_data import getReviews
from collections import defaultdict
from math import log
from sets import ImmutableSet
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
    def __init__(self, vocab):
        self.postiveProbs = {}
        self.negativeProbs = {}
        self.negativeReviews = 0
        self.postiveReviews = 0
        self.positive = ImmutableSet([4, 5])
        self.negative = ImmutableSet([1, 2, 3])
        self.vocab = vocab

        
    def Train(self, training_set):
        positive_counts = defaultdict(lambda: 1)
        negative_counts = defaultdict(lambda: 1)

        for review in training_set:
            if review['stars'] in self.positive:
                for word in review['text']:
                    self.postiveReviews += 1
                    positive_counts[word] += 1
            else:
                for word in review['text']:
                    self.negativeReviews += 1
                    negative_counts[word] += 1

        self.__buildLogProbs(positive_counts, self.postiveReviews, self.postiveProbs)
        self.__buildLogProbs(negative_counts, self.negativeReviews, self.negativeProbs)


    def __buildLogProbs(self, counts, reviewTotal, probDict):
        for word in self.vocab:
            probDict[word] = log(float(counts[word]) / float(reviewTotal))

    def PredictPositive(self, sent):
        p_positive = 0.0
        p_negative = 0.0

        for word in sent['text']:
            p_positive += self.postiveProbs[word]
            p_negative += self.negativeProbs[word]

        if p_positive > p_negative:
            return True
        else:
            return False

def main():
    total = 0.0
    right = 0.0
    reviews = getReviews()
    vocab = buildVocab(reviews[0:8000])
    training_set_prep = preProcess(reviews, vocab)
    naiveBayes = NaiveBayes(vocab)
    naiveBayes.Train(training_set_prep)
    
    #Test accuracy
    for review in reviews[8001:9999]:
        total += 1.0
        if review['stars'] in naiveBayes.positive and naiveBayes.PredictPositive(review):
            right += 1.0
        elif review['stars'] in naiveBayes.negative and not naiveBayes.PredictPositive(review):
            right += 1.0

    print ((right/total) * 100)


if __name__ == '__main__':
    main()