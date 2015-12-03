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
    def __init__(self):
        self.postive_probs = defaultdict(lambda: float('-inf'))
        self.negative_probs = defaultdict(lambda: float('-inf'))
        self.negative_reviews = 0
        self.postive_reviews = 0
        self.positive = ImmutableSet([4, 5])
        self.negative = ImmutableSet([1, 2, 3])

        
    def Train(self, training_set):
        positive_counts = defaultdict(int)
        negative_counts = defaultdict(int)

        for review in training_set:
            if review['stars'] in self.positive:
                self.postive_reviews += 1
                for word in review['text']:
                    positive_counts[word] += 1
            else:
                self.negative_reviews += 1
                for word in review['text']:
                    negative_counts[word] += 1

        self.__buildLogProbs(positive_counts, self.postive_reviews, self.postive_probs)
        self.__buildLogProbs(negative_counts, self.negative_reviews, self.negative_probs)


    def __buildLogProbs(self, counts, reviewTotal, probDict):
        for word, count in counts.iteritems():
            probDict[word] = log(float(count) / float(reviewTotal))



def main():
    reviews = getReviews()
    vocab = buildVocab(reviews)
    training_set_prep = preProcess(reviews, vocab)
    naiveBayes = NaiveBayes()
    naiveBayes.Train(training_set_prep)
    


if __name__ == '__main__':
    main()