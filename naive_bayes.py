import yelp_data
from collections import defaultdict
from math import log
from sets import ImmutableSet
import json
STARS = 0
TEXT = 1

class NaiveBayes(object):
    """NaiveBayes for sentiment analysis"""
    def __init__(self, vocab, stopwords):
        self.stopwords = stopwords
        self.positive = ImmutableSet([4, 5])
        self.negative = ImmutableSet([1, 2])
        self.vocab = vocab
    
    def Train(self, training_set, N=1):

        self.totalPositiveWords = len(self.vocab)
        self.totalNegativeWords = len(self.vocab)
        self.positiveCounts = defaultdict(lambda: defaultdict(lambda: 1))
        self.negativeCounts = defaultdict(lambda: defaultdict(lambda: 1))

        for review in training_set:
            if review[STARS] in self.positive:
                self.totalPositiveWords += len(review[1]) - N * 2
            elif review[STARS] in self.negative:
                self.totalNegativeWords += len(review[1]) - N * 2

            for i, word in enumerate(review[1]):
                for n in range(N, 0, -1):
                    if i + n < len(review[1]):
                        gram = tuple(review[1][i+N-n:i+N])
                        if review[STARS] in self.positive:
                            self.positiveCounts[n][gram] += 1
                        elif review[STARS] in self.negative:
                            self.negativeCounts[n][gram] += 1


    def __stupidBackoff(self, ngram, n, positive):
        alpha = 0.4
        if positive:
            gramCounts = self.positiveCounts
            total = self.totalPositiveWords
        else:
            gramCounts = self.negativeCounts
            total = self.totalNegativeWords
        if n > 1:
            if gramCounts[n][ngram] > 8:
                return log(float(gramCounts[n][ngram]) / float(gramCounts[n-1][ngram[0:n-1]]))
            else:
                return log(alpha) + self.__stupidBackoff(ngram[0:n-1], n-1, positive)
        else:
            return log(float(gramCounts[n][ngram]) / float(total))
    

    def PredictPositiveStupidBackoff(self, review, maxN):
        p_positive = 0.0
        p_negative = 0.0

        for i, word in enumerate(review[TEXT]):
            if i + maxN < len(review[TEXT]):
                gram = tuple(review[TEXT][i:i+maxN])
                p_positive += self.__stupidBackoff(gram, maxN, True)
                p_negative += self.__stupidBackoff(gram, maxN, False)

        return p_positive > p_negative


def main():


    maxN = 4
    reviews = yelp_data.getReviewsTokenized()
    training_set = reviews[0:9000]
    test_set     = reviews[9001:18000]
    vocab = yelp_data.buildVocab(training_set)
    training_set_prep = yelp_data.preProcessN(training_set, vocab, maxN)
    test_set_prep = yelp_data.preProcessN(test_set, vocab, maxN)
    stopwords = yelp_data.getStopWords()
    naiveBayes = NaiveBayes(vocab, stopwords)

    print "----------------------------------"
    print "Sentiment analysis of Yelp reviews"
    print "----------------------------------"
    print "Todd Pollak, Teddy Cleveland"
    print "----------------------------------"
    print maxN, "- gram model"
    print "----------------------------------"
    print "Training model...."
    print "----------------------------------"

    naiveBayes.Train(training_set_prep, maxN)

    print len(training_set), "reviews used in training", len(test_set), "reviews used in test set"
    print "----------------------------------"
    print "Running test data on constructed model...."
    print "----------------------------------"
    

    # Test accuracy
    total = 0.0
    right = 0.0
    for review in test_set_prep:
        total += 1.0
        if review[STARS] in naiveBayes.positive and naiveBayes.PredictPositiveStupidBackoff(review, maxN):
            right += 1.0
        elif review[STARS] in naiveBayes.negative and not naiveBayes.PredictPositiveStupidBackoff(review, maxN):
            right += 1.0

    print "Percent Accuracy using Stupid Backoff and Laplace Smoothing:", ((right/total) * 100), "\n"


if __name__ == '__main__':
    main()