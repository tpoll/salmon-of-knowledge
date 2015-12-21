import yelp_data
import operator
import codecs
import os
import operator
import nltk
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
    def __init__(self, vocab):
        self.vocab    = vocab
        self.features = {}

    def buildFeatures(self, ngrams, N):
        counter = 0
        for i in range(1, N + 1):
            for feature, count in ngrams.counts[i].iteritems():
                if (N==2 and count > 8) or (N==3 and count > 10) or N==1:
                    self.features[feature] = counter
                    counter += 1

    def buildData(self, dataset, nGram):
        matrix = [defaultdict(int) for x in xrange(len(dataset))]
        for i, sent in enumerate(dataset):
            for N in range(1, nGram + 1):
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

    def buildARFFfile(self, dataset, filename, nGram):
        num_features = len(self.features)
        with codecs.open(filename, 'wb', encoding='utf-8') as f:
            f.write("@relation maxent\n\n")
            features = sorted(self.features.items(), key=operator.itemgetter(1))

            for feature in features:
                f.write("@attribute \"" + ' '.join(feature[0]) + "\" NUMERIC\n")
            f.write("@attribute __sentiment__ {positive, negative}\n\n")
            f.write("@data\n")
            dataMatrix = self.buildData(dataset, nGram)

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
    def CalculateNgramPMI(self, k, N):
        nSum = sum([self.counts[N][x] for x in self.counts[N]])
        unSum = sum([self.counts[1][x] for x in self.counts[1]])

        wordProbs = {x[0]: float(self.counts[1][x]) / unSum for x in self.counts[1]} # word probabilities(w1 and w2)
        jointProbs = {x: float(self.counts[N][x]) / nSum for x in self.counts[N] if self.counts[N][x] > 10 } # joint probabilites (w1&w2)

        probs = {}

        for nGram, jProb in jointProbs.iteritems():
            indvSum = 1.0
            for i in range(0, N):
                indvSum *= float(wordProbs[nGram[i]])
            probs[nGram] = log((jProb / indvSum), 2)
        topK = sorted(probs.iteritems(), key=operator.itemgetter(1), reverse=True)[:k]

        self.counts[N] = {key[0]: self.counts[N][key[0]] for key in topK} # Replace Bigrams with high information features


def main():
    N = 3
    reviews = yelp_data.getReviewsTokenized()
    training_set = reviews[0:3000]
    test_set     = reviews[3001:6000]
    vocab = yelp_data.buildVocab(training_set)
    training_set_prep = yelp_data.preProcess(training_set, vocab)
    test_set_prep = yelp_data.preProcess(test_set, vocab)
    
    ngrams = Ngrams()
    ngrams.Train(training_set_prep, N)
    ngrams.CalculateNgramPMI(600, 2)
    ngrams.CalculateNgramPMI(100, 3)

    
    me = Maxent(vocab)
    me.buildFeatures(ngrams, N)
    me.buildARFFfile(training_set_prep, "yelp_maxent_training.arff", N)
    me.buildARFFfile(test_set_prep, "yelp_maxent_test.arff", N)


if __name__ == '__main__':
    main()