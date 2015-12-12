import yelp_data
from collections import defaultdict
from math import log
from sets import ImmutableSet
import json


class NaiveBayes(object):
    """NaiveBayes for sentiment analysis"""
    def __init__(self, vocab, stopwords):
        self.stopwords = stopwords
        self.postiveProbs = {}
        self.negativeProbs = {}
        self.totalNegativeWords = 0
        self.totalPostiveWords = 0
        self.positive = ImmutableSet([4, 5])
        self.negative = ImmutableSet([1, 2, 3])
        self.vocab = vocab

    
    def Train(self, training_set):
        positiveCounts = defaultdict(lambda: 1)
        negativeCounts = defaultdict(lambda: 1)

        # get positive and negative counts
        # for each word using review ratings.
        for review in training_set:
            if review['stars'] in self.positive:
                for word in review['text']:
                    self.totalPostiveWords += 1
                    positiveCounts[word] += 1
            else:
                for word in review['text']:
                    self.totalNegativeWords += 1
                    negativeCounts[word] += 1

        # use counts to get positive and negative probabilities for each word
        self.__buildLogProbs(positiveCounts, self.totalPostiveWords, self.postiveProbs)
        self.__buildLogProbs(negativeCounts, self.totalNegativeWords, self.negativeProbs)


    def __buildLogProbs(self, counts, reviewTotal, probDict):
        for word in self.vocab:
            probDict[word] = log(float(counts[word]) / (float(reviewTotal) + len(self.vocab)))
            
    def PredictPositive(self, sent):
        p_positive = 0.0
        p_negative = 0.0

        for word in sent['text']:
            if word not in self.stopwords:
                p_positive += self.postiveProbs[word]
                p_negative += self.negativeProbs[word]

        if p_positive > p_negative:
            return True
        else:
            return False

def main():
    total = 0.0
    right = 0.0
    reviews = yelp_data.getReviews()
    training_set = reviews[0:1000]
    test_set     = reviews[1001:2000]
    vocab = yelp_data.buildVocab(training_set)
    stopwords = yelp_data.getStopWords()
    training_set_prep = yelp_data.preProcess(training_set, vocab)
    test_set_prep = yelp_data.preProcess(test_set, vocab)
    naiveBayes = NaiveBayes(vocab, stopwords)
    naiveBayes.Train(training_set_prep)
    
    #Test accuracy
    for review in test_set_prep:
        total += 1.0
        if review['stars'] in naiveBayes.positive and naiveBayes.PredictPositive(review):
            right += 1.0
        elif review['stars'] in naiveBayes.negative and not naiveBayes.PredictPositive(review):
            right += 1.0

    print ((right/total) * 100)


if __name__ == '__main__':
    main()