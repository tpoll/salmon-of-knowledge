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
        self.neutralProbs = {}
        self.totalNegativeWords = 0
        self.totalPostiveWords = 0
        self.totalNeutralWords = 0
        self.positive = ImmutableSet([4, 5])
        self.negative = ImmutableSet([1, 2,])
        self.neutral = ImmutableSet([3])
        self.vocab = vocab

        
    def Train(self, training_set):
        positiveCounts = defaultdict(lambda: 1)
        negativeCounts = defaultdict(lambda: 1)
        neutralCounts = defaultdict(lambda: 1)


        for review in training_set:
            if review['stars'] in self.positive:
                for word in review['text']:
                    self.totalPostiveWords += 1
                    positiveCounts[word] += 1
            elif review['stars'] in self.negative:
                for word in review['text']:
                    self.totalNegativeWords += 1
                    negativeCounts[word] += 1
            else:
                for word in review['text']:
                    self.totalNeutralWords += 1
                    neutralCounts[word] += 1                


        self.__buildLogProbs(positiveCounts, self.totalPostiveWords, self.postiveProbs)
        self.__buildLogProbs(negativeCounts, self.totalNegativeWords, self.negativeProbs)
        self.__buildLogProbs(neutralCounts, self.totalNeutralWords, self.neutralProbs)



    def __buildLogProbs(self, counts, reviewTotal, probDict):
        for word in self.vocab:
            probDict[word] = log(float(counts[word]) / float(reviewTotal))

    def predictClass(self, sent):
        p_positive = 0.0
        p_negative = 0.0
        p_neutral = 0.0

        for word in sent['text']:
            p_positive += self.postiveProbs[word]
            p_negative += self.negativeProbs[word]
            p_neutral += self.neutralProbs[word]


        if p_positive > p_negative and p_positive > p_neutral:
            return "pos"
        elif p_negative > p_positive and p_negative > p_neutral:
            return "neg"
        elif p_neutral > p_positive and p_neutral > p_negative:
            return "neut"

def main():
    total = 0.0
    right = 0.0
    reviews = getReviews()
    vocab = buildVocab(reviews[0:480000])
    training_set_prep = preProcess(reviews[0:480000], vocab)
    test_set_prep = preProcess(reviews[480001:499999], vocab)
    naiveBayes = NaiveBayes(vocab)
    naiveBayes.Train(training_set_prep)
    
    #Test accuracy
    for review in test_set_prep:
        total += 1.0
        if review['stars'] in naiveBayes.positive and naiveBayes.predictClass(review) == "pos":
            right += 1.0
        elif review['stars'] in naiveBayes.negative and naiveBayes.predictClass(review) == "neg":
            right += 1.0
        elif review['stars'] in naiveBayes.neutral and naiveBayes.predictClass(review) == "neut":
            right += 1.0

    print ((right/total) * 100)


if __name__ == '__main__':
    main()