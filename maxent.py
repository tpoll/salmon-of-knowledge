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
import spacy.en
from sets import ImmutableSet

unknown_token = 'UNK'
positive_class = "positive"
negative_class = "negative"
STARS = 0
TEXT = 1
TAG = 2
CHUNK = 3

class Maxent(object):
    def __init__(self, vocab, nlp):
        self.vocab    = vocab
        self.features = {}
        self.chunks = defaultdict(int)
        self.PosGrams = ImmutableSet([nlp.vocab.strings['JJ'], nlp.vocab.strings['VB'], nlp.vocab.strings['RB'], 
                            nlp.vocab.strings['RBR'], nlp.vocab.strings['JJR'], nlp.vocab.strings['JJS'], nlp.vocab.strings['RBS'],
                            nlp.vocab.strings['VBN'], nlp.vocab.strings['VBD'], nlp.vocab.strings['VBP']])

    def buildChunks(self, dataset):
        for review in dataset:
            for chunk in review[CHUNK]:
                self.chunks[chunk] += 1

    def buildFeatures(self, ngrams, N):
        counter = 0
        for i in range(1, N + 1):
            for feature, count in ngrams.counts[i].iteritems():
                if (i==2) or (i==3) or (i==1 and ngrams.tags[feature][0] in self.PosGrams):
                    self.features[feature] = counter
                    counter += 1

        for feature, count in self.chunks.iteritems():
            if count > 7 and len(feature) > 1 and feature not in self.features:
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
            
            for chunk in sent[CHUNK]:
                if chunk in self.features:
                    matrix[i][self.features[chunk]] += 1
        
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
    def __init__(self, nlp):
        self.counts = defaultdict(lambda: defaultdict(int))
        self.tags = {}
        self.Verbs = ImmutableSet([nlp.vocab.strings['VB'], nlp.vocab.strings['VBN'], nlp.vocab.strings['VBD'], nlp.vocab.strings['VBP']])
        self.Adj = ImmutableSet([nlp.vocab.strings['JJ'], nlp.vocab.strings['JJR'], nlp.vocab.strings['JJS']])
        self.Nouns = ImmutableSet([nlp.vocab.strings['NN']])
        self.Adverbs = ImmutableSet([nlp.vocab.strings['RB'], nlp.vocab.strings['RBR'], nlp.vocab.strings['RBS']])
        self.PosGrams = ImmutableSet([nlp.vocab.strings['JJ'], nlp.vocab.strings['NN'], nlp.vocab.strings['VB'], nlp.vocab.strings['RB'], 
                            nlp.vocab.strings['RBR'], nlp.vocab.strings['JJR'], nlp.vocab.strings['JJS'], nlp.vocab.strings['RBS'],
                            nlp.vocab.strings['VBN'], nlp.vocab.strings['VBD'], nlp.vocab.strings['VBP'] ])
    
    def Train(self, training_set, nGram=1):
        for N in range(1, nGram + 1):
            for review in training_set:
                    for i, word in enumerate(review[TEXT][nGram - N:]):
                        if word is not "</S>" and word is not "<S>":
                            gram = tuple(review[TEXT][i - N:i])
                            if gram:
                                self.tags[gram] = review[TAG][i - N:i]
                                self.counts[N][gram] += 1

    #Calculate Pointwise Mutual information of N-grams
    def CalculateNgramPMI(self, k, N):
        nSum = sum([self.counts[N][x] for x in self.counts[N]])
        unSum = sum([self.counts[1][x] for x in self.counts[1]])

        wordProbs = {x[0]: float(self.counts[1][x]) / unSum for x in self.counts[1]} # word probabilities(w1 and w2)
        jointProbs = {x: float(self.counts[N][x]) / nSum for x in self.counts[N] if self.counts[N][x] > 15 } # joint probabilites (w1&w2)

        probs = {}

        for nGram, jProb in jointProbs.iteritems():
            indvSum = 1.0
            for i in range(0, N):
                indvSum *= float(wordProbs[nGram[i]])
            probs[nGram] = log((jProb / indvSum), 2)

        topK = sorted(probs.iteritems(), key=operator.itemgetter(1), reverse=True)
        newK = []

        for gram in topK:
            if all([self.tags[gram[0]][i] in self.PosGrams for i in range(0,N)]):
                if all([self.tags[gram[0]][i] not in self.Nouns for i in range(0,N)]):
                    newK.append(gram)

        newK = newK[0:k]
        self.counts[N] = {key[0]: self.counts[N][key[0]] for key in newK} # Replace nGrams with high information features


def main():
    N = 3
    (reviews, nlp) = yelp_data.getReviewsTokenizedandTagged(30000)
    training_set = reviews[0:15000]
    test_set     = reviews[15001:30000]
    vocab = yelp_data.buildVocab(training_set)
    training_set_prep = yelp_data.preProcess(training_set, vocab)
    test_set_prep = yelp_data.preProcess(test_set, vocab)
    
    ngrams = Ngrams(nlp)
    ngrams.Train(training_set_prep, N)
    ngrams.CalculateNgramPMI(700, 2)
    ngrams.CalculateNgramPMI(700, 3)


    
    me = Maxent(vocab, nlp)
    me.buildChunks(training_set_prep)
    me.buildFeatures(ngrams, N)
    me.buildARFFfile(training_set_prep, "yelp_maxent_training.arff", N)
    me.buildARFFfile(test_set_prep, "yelp_maxent_test.arff", N)


if __name__ == '__main__':
    main()