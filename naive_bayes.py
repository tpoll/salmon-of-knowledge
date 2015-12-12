import yelp_data
from collections import defaultdict
from math import log
from sets import ImmutableSet
import json


class NaiveBayes(object):
    """NaiveBayes for sentiment analysis"""
    def __init__(self, vocab):
        self.positiveCounts = defaultdict(lambda: defaultdict(lambda: 1))
        self.negativeCounts = defaultdict(lambda: defaultdict(lambda: 1))
        self.negativeNgrams = defaultdict(lambda: len(vocab))
        self.positiveNgrams = defaultdict(lambda: len(vocab))
        self.positive = ImmutableSet([4, 5])
        self.negative = ImmutableSet([1, 2, 3])
        self.vocab = vocab

    
    def Train(self, training_set, nGram=1):

        for N in range(1, nGram + 1):

            # get positive and negative counts
            # for each word using review ratings.
            for review in training_set:
                if review['stars'] in self.positive:
                    for i, word in enumerate(review['text'][nGram - N:]):
                        if word is not "</S>":

                            gram = tuple(review['text'][i - N:i])
                            self.positiveNgrams[N] += 1
                            self.positiveCounts[N][gram] += 1
                else:
                    for i, word in enumerate(review['text'][nGram - N:]):
                        if word is not "</S>":
                            gram = tuple(review['text'][i - N:i])
                            self.negativeNgrams[N] += 1
                            self.negativeCounts[N][gram] += 1
    
    # predict probability of positive using linear interpolation            
    def PredictPositive(self, review, maxN, weights):
        p_positive = 0.0
        p_negative = 0.0

        for N in range(1, maxN + 1):
            for i, word in enumerate(review['text'][maxN - N:]):
                if word is not "</S>":
                    gram = tuple(review['text'][i - N:i])
                    p_negative += weights[N - 1] * log(float(self.negativeCounts[N][gram]) / float(self.negativeNgrams[N]))
                    p_positive += weights[N - 1] * log(float(self.positiveCounts[N][gram]) / float(self.positiveNgrams[N]))

        if p_positive > p_negative:
            return True
        else:
            return False

def main():

    maxN = 3
    reviews = yelp_data.getReviews()
    training_set = reviews[0:50000]
    test_set     = reviews[50001:100001]
    vocab = yelp_data.buildVocab(training_set)
    training_set_prep = yelp_data.preProcessN(training_set, vocab, maxN)
    test_set_prep = yelp_data.preProcessN(test_set, vocab, maxN)
    naiveBayes = NaiveBayes(vocab)
    naiveBayes.Train(training_set_prep, maxN)
    
    #Test accuracy
    total = 0.0
    right = 0.0
    interpWeights = [.25, .70, .05]
    for review in test_set_prep:
        total += 1.0
        if review['stars'] in naiveBayes.positive and naiveBayes.PredictPositive(review, maxN, interpWeights):
            right += 1.0
        elif review['stars'] in naiveBayes.negative and not naiveBayes.PredictPositive(review, maxN, interpWeights):
            right += 1.0

    print ((right/total) * 100)


if __name__ == '__main__':
    main()