import yelp_data
import operator
import codecs
from collections import defaultdict
from collections import Counter
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

    def buildNormalizedFeatures(self, dataset):
        counts = defaultdict(int)

        for review in dataset:
            for word in review['text']:
                 counts[word] += 1

        counter = 0

        # print len(counts)
        # sorted_keys =  sorted(counts.items(), key=lambda x: x[1])
        # for x in sorted_keys:
            # print x
        for word, count in counts.iteritems():
            if count > 135:
                self.features[word] = counter
                counter += 1

    def buildData(self, dataset):
        matrix = [defaultdict(int) for x in xrange(len(dataset))]
        for i, sent in enumerate(dataset):
            for word in sent["text"]:
                if word in self.features:
                    matrix[i][self.features[word]] += 1
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
                f.write("@attribute \"" + feature[0] + "\" NUMERIC\n")
            f.write("@attribute __sentiment__ {positive, negative}\n\n")
            f.write("@data\n")
            dataMatrix = self.buildData(dataset)
            for i, sent in enumerate(dataMatrix):
                f.write("{")
                for feature in sorted(sent.iteritems()):
                    f.write(str(feature[0]) + " " + str(feature[1]) + ",")
                f.write(self.getSentiment(dataset[i]) + "}\n")

def main():
    reviews = yelp_data.getReviews()
    training_set = reviews[0:1000]
    test_set     = reviews[1001:2000]
    vocab = yelp_data.buildVocab(training_set)
    training_set_prep = yelp_data.preProcess(training_set, vocab)
    test_set_prep = yelp_data.preProcess(test_set, vocab)
    me = Maxent(vocab)
    # me.buildNormalizedFeatures(training_set_prep)
    me.buildFeatures()
    me.buildARFFfile(training_set_prep, "yelp_maxent_training.arff")
    me.buildARFFfile(test_set_prep, "yelp_maxent_test.arff")


if __name__ == '__main__':
    main()