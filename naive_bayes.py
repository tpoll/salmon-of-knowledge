import yelp_data
from collections import defaultdict
from math import log
from sets import ImmutableSet
import json


class NaiveBayes(object):
    """NaiveBayes for sentiment analysis"""
    def __init__(self, vocab, stopwords):
        self.positiveCounts = defaultdict(lambda: defaultdict(lambda: 1))
        self.negativeCounts = defaultdict(lambda: defaultdict(lambda: 1))
        self.stopwords = stopwords
        self.positive = ImmutableSet([4, 5])
        self.negative = ImmutableSet([1, 2, 3])
        self.vocab = vocab

    
    def Train(self, training_set, nGram=1):

        self.negativeNgrams = defaultdict(lambda: 1 + yelp_data.posVocabLen(self.vocab, training_set))
        self.positiveNgrams = defaultdict(lambda: 1 + yelp_data.negVocabLen(self.vocab, training_set))

        for N in range(1, nGram + 1):

            # get positive and negative counts
            # for each word using review ratings.
            for review in training_set:
                if review['stars'] in self.positive:
                    for i, word in enumerate(review['text'][nGram - N:]):
                        if word is not "</S>" and word is not "<S>":
                            gram = tuple(review['text'][i - N:i])
                            self.positiveNgrams[N] += 1
                            self.positiveCounts[N][gram] += 1
                else:
                    for i, word in enumerate(review['text'][nGram - N:]):
                        if word is not "</S>" and word is not "<S>":
                            gram = tuple(review['text'][i - N:i])
                            self.negativeNgrams[N] += 1
                            self.negativeCounts[N][gram] += 1

    def __fuckingStupidBackoff(self, ngram, n, positive):
        alpha = 0.4
        if positive:
            gramCounts = self.positiveCounts
            total = self.positiveNgrams
        else:
            gramCounts = self.negativeCounts
            total = self.negativeNgrams
        if n > 1:
            if gramCounts[n][ngram] > 8:
                return log(float(gramCounts[n][ngram]) / float(gramCounts[n-1][ngram[0:n-1]]))
            else:
                return log(alpha) + self.__fuckingStupidBackoff(ngram[0:n-1], n-1, positive)
        else:
            return log(float(gramCounts[n][ngram]) / float(total[n]))
    
    # predict probability of positive using weighted linear interpolation            
    def PredictPositive(self, review, maxN, weights):
        p_positive = 0.0
        p_negative = 0.0

        for i, word in enumerate(review['text'][maxN - 1:]):
            gram = tuple(review['text'][i-maxN:i])
            positive = 0.0
            negative = 0.0
            for n in range(maxN, 1, -1):
                top = gram[0:n]
                p_positive += self.__fuckingStupidBackoff(top, n, True)
                p_negative += self.__fuckingStupidBackoff(top, n, False)

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
    stopwords = yelp_data.getStopWords()
    naiveBayes = NaiveBayes(vocab, stopwords)
    naiveBayes.Train(training_set_prep, maxN)
    
    #Test accuracy
    total = 0.0
    right = 0.0
    for review in test_set_prep:
        total += 1.0
        if review['stars'] in naiveBayes.positive and naiveBayes.PredictPositive(review, maxN, interpWeights):
            right += 1.0
        elif review['stars'] in naiveBayes.negative and not naiveBayes.PredictPositive(review, maxN, interpWeights):
            right += 1.0

    print ((right/total) * 100)


if __name__ == '__main__':
    main()