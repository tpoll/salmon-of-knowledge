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
    def __init__(self, vocab, stopwords):
        self.vocab    = vocab
        self.stopwords = stopwords
        self.features = {}

    def buildFeatures(self, ngrams):
        counter = 0
        for feature, count in ngrams.counts[1].iteritems():
            self.features[feature] = counter
            counter += 1

    def buildData(self, dataset, ngrams):
        matrix = [defaultdict(int) for x in xrange(len(dataset))]
        for i, sent in enumerate(dataset):
            for j, word in enumerate(sent['text'][1 - 1:]):
                if word is not "</S>" and word is not "<S>":
                    gram = tuple(sent['text'][j - 1:j])
                    matrix[i][self.features[gram]] += 1
        return matrix

    def getSentiment(self, sentence):
        if sentence["stars"] >= 4:
            return str(len(self.features)) + " positive"
        else:
            return str(len(self.features)) + " negative"

    def buildARFFfile(self, dataset, filename, ngrams):
        num_features = len(self.features)
        with codecs.open(filename, 'wb', encoding='utf-8') as f:
            f.write("@relation maxent\n\n")
            features = sorted(self.features.items(), key=operator.itemgetter(1))

            for feature in features:
                f.write("@attribute \"" + ' '.join(feature[0]) + "\" NUMERIC\n")
            f.write("@attribute __sentiment__ {positive, negative}\n\n")
            f.write("@data\n")
            dataMatrix = self.buildData(dataset, ngrams)

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

            # get positive and negative counts
            # for each word using review ratings.
            for review in training_set:
                    for i, word in enumerate(review['text'][nGram - N:]):
                        if word is not "</S>" and word is not "<S>":
                            gram = tuple(review['text'][i - N:i])
                            self.counts[N][gram] += 1


def main():
    reviews = yelp_data.getReviews()
    training_set = reviews[0:1000]
    test_set     = reviews[1000:10001]
    vocab = yelp_data.buildVocab(training_set)
    training_set_prep = yelp_data.preProcess(training_set, vocab)
    test_set_prep = yelp_data.preProcess(test_set, vocab)
    ngrams = Ngrams()
    ngrams.Train(training_set_prep)
    stopwords = yelp_data.getStopWords()
    me = Maxent(vocab, stopwords)
    me.buildFeatures(ngrams)
    me.buildARFFfile(training_set_prep, "yelp_maxent_training.arff", ngrams)
    me.buildARFFfile(test_set_prep, "yelp_maxent_test.arff", ngrams)


if __name__ == '__main__':
    main()