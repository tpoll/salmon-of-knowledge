import yelp_data
import operator
import codecs
from collections import defaultdict
from math import log
from sets import ImmutableSet
import json

unknown_token = 'UNK'
positive_class = "positive"
negative_class = "negative"

class Maxent(object):
    def __init__(self, vocab):
        self.vocab    = vocab
        self.features = {}

    def buildFeatures(self):
        for i, word in enumerate(self.vocab):
            self.features[word] = i

    def buildData(self, dataset):
        matrix = [[] for x in xrange(len(dataset))]
        for i, sent in enumerate(dataset):
            for word in sent["text"]:
                matrix[i].append(self.features[word])
        return matrix

    def getSentiment(self, sentence):
        if sentence["stars"] >= 4:
            return str(len(self.features)) + " positive"
        else:
            return str(len(self.features)) + " negative"

    def buildARFFfile(self, dataset, filename):
        num_features = len(self.features)
        with codecs.open(filename, 'wb', encoding='utf-8') as f:
            f.write("@relation maxent\n\n")
            features = sorted(self.features.items(), key=operator.itemgetter(1))
            for feature in features:
                f.write("@attribute \"" + feature[0] + "\" {0, 1}\n")
            f.write("@attribute __sentiment__ {positive, negative}\n\n")
            f.write("@data\n")
            dataMatrix = self.buildData(dataset)
            for i, sent in enumerate(dataMatrix):
                f.write("{")
                sent.sort()
                for feature in sent:
                    f.write(str(feature) + " " + str(1) + ",")
                f.write(self.getSentiment(dataset[i]) + "}\n")


def main():
    reviews = yelp_data.getReviews()
    training_set = reviews[0:5000]
    test_set     = reviews[5001:10000]
    vocab = yelp_data.buildVocab(training_set)
    training_set_prep = yelp_data.preProcess(training_set, vocab)
    test_set_prep = yelp_data.preProcess(test_set, vocab)
    me = Maxent(vocab)
    me.buildFeatures()
    me.buildARFFfile(training_set_prep, "yelp_maxent_training.arff")
    me.buildARFFfile(test_set_prep, "yelp_maxent_test.arff")


if __name__ == '__main__':
    main()