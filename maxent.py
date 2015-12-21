import yelp_data
import operator
import codecs
import os
from collections import defaultdict
from collections import Counter
from math import log
from sets import ImmutableSet
import json

unknown_token = 'UNK'
positive_class = "positive"
negative_class = "negative"
STARS = 0
TEXT = 1

class Maxent(object):
    def __init__(self, vocab, stopwords):
        self.vocab    = vocab
        self.stopwords = stopwords
        self.features = {}

    def buildFeatures(self, ngrams, N):
        counter = 0
        for i in range(1, N + 1):
            for feature, count in ngrams.counts[i].iteritems():
                if (N==2 and count > 1) or (N==3 and count > 1) or N==1:
                    self.features[feature] = counter
                    counter += 1

    def buildData(self, dataset, ngrams, nGram):
        matrix = [defaultdict(int) for x in xrange(len(dataset))]
        for i, sent in enumerate(dataset):
            for N in range(nGram + 1):
                for j, word in enumerate(sent[TEXT][nGram - N:]):
                    if word is not "</S>" and word is not "<S>":
                        gram = tuple(sent[TEXT][j - N:j])
                        if gram in self.features:
                            matrix[i][self.features[gram]] += 1
        return matrix

    def getSentiment(self, sentence):
        if sentence[STARS] >= 4:
            return str(len(self.features)) + " positive"
        else:
            return str(len(self.features)) + " negative"

    def buildARFFfile(self, dataset, filename, ngrams, nGram):
        num_features = len(self.features)
        with codecs.open(filename, 'wb', encoding='utf-8') as f:
            f.write("@relation maxent\n\n")
            features = sorted(self.features.items(), key=operator.itemgetter(1))

            for feature in features:
                f.write("@attribute \"" + ' '.join(feature[0]) + "\" NUMERIC\n")
            f.write("@attribute __sentiment__ {positive, negative}\n\n")
            f.write("@data\n")
            dataMatrix = self.buildData(dataset, ngrams, nGram)

            for i, sent in enumerate(dataMatrix):
                f.write("{")
                for feature in sorted(sent.iteritems()):
                    f.write(str(feature[0]) + " " + str(feature[1]) + ",")
                f.write(self.getSentiment(dataset[i]) + "}\n")

class Ngrams(object):
    """NaiveBayes for sentiment analysis"""
    def __init__(self):
        self.counts = defaultdict(lambda: defaultdict(int))

    
    def Train(self, training_set, nGram=1):
        for N in range(1, nGram + 1):
            for review in training_set:
                    for i, word in enumerate(review[TEXT][nGram - N:]):
                        if word is not "</S>" and word is not "<S>":
                            gram = tuple(review[TEXT][i - N:i])
                            if gram:
                                self.counts[N][gram] += 1


def main():
    reviews = yelp_data.getReviewsTokenized()
    training_set = reviews[0:8000]
    test_set     = reviews[8001:16000]
    vocab = yelp_data.buildVocab(training_set)
    training_set_prep = yelp_data.preProcess(training_set, vocab)
    test_set_prep = yelp_data.preProcess(test_set, vocab)
    ngrams = Ngrams()
    ngrams.Train(training_set_prep, 2)
    stopwords = yelp_data.getStopWords()
    me = Maxent(vocab, stopwords)
    me.buildFeatures(ngrams, 2)
    me.buildARFFfile(training_set_prep, "yelp_maxent_training.arff", ngrams, 1)
    me.buildARFFfile(test_set_prep, "yelp_maxent_test.arff", ngrams, 1)


if __name__ == '__main__':
    main()