import yelp_data
from collections import defaultdict
from math import log
from sets import ImmutableSet
import json
import sys

STARS = 0
TEXT = 1

class NaiveBayes(object):
    """NaiveBayes for sentiment analysis"""
    def __init__(self, vocab):
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
            if gramCounts[n][ngram] > 3:
                return log(float(gramCounts[n][ngram]) / float(gramCounts[n-1][ngram[0:n-1]]))
            else:
                return log(alpha) + self.__stupidBackoff(ngram[0:n-1], n-1, positive)
        else:
            return log(float(gramCounts[n][ngram]) / float(total))
    
    def __noBackoff(self, ngram, n, positive):
        if positive:
            gramCounts = self.positiveCounts
            total = self.totalPositiveWords
        else:
            gramCounts = self.negativeCounts
            total = self.totalNegativeWords
        if n > 1:
            return log(float(gramCounts[n][ngram]) / float(gramCounts[n-1][ngram[0:n-1]]))
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

    def PredictPositive(self, review, maxN):
        p_positive = 0.0
        p_negative = 0.0
        for i, word in enumerate(review[TEXT]):
            if i + maxN < len(review[TEXT]):
                gram = tuple(review[TEXT][i:i+maxN])
                p_positive += self.__noBackoff(gram, maxN, True)
                p_negative += self.__noBackoff(gram, maxN, False)

        return p_positive > p_negative

def WriteEntryHeader(file, test_config):
    file.write(test_config["name"] +"\n")
    file.write("------------------------\n\n")


def WriteResultHeader(file, entry_config):
    if entry_config["backoff"]:
        backoff = "Stupid Backoff"
    else:
        backoff = "No Backoff"
    file.write(str(entry_config["N"]) + "-gram model " + "with " + backoff + "\n")
    file.write("training indices " + str(entry_config["training"]["start"]) + " to " + str(entry_config["training"]["end"]) + "\n")
    file.write("test indices " + str(entry_config["test"]["start"]) + " to " + str(entry_config["test"]["end"]) + "\n")

def WriteResult(file, total, right, false_pos, false_neg):
    file.write("Results:\n")
    file.write("Classificaiton Accuracy: " + str((right / total) * 100) + " %\n")
    file.write("Negative Reviews Tagged as Positive: " + str(false_pos) + " (" + str((false_pos / total) * 100) + " %)\n")
    file.write("Positive Reviews Tagged as Negative: " + str(false_neg) + " (" + str((false_neg / total) * 100) + " %)\n")
    file.write("------------------------\n\n")

def main():

    reviews = yelp_data.getReviewsTokenized()

    with open(sys.argv[1], 'rb') as f:
        config = json.loads(f.read())
        for test_group in config["tests"]:
            outfile = open("naive_bayes/results/test_results_"+str(test_group["id"])+".txt", "wb")
            WriteEntryHeader(outfile, test_group)
            for i, entry in enumerate(test_group["entries"]):
                WriteResultHeader(outfile, entry)
                maxN = entry["N"]
                training_set = reviews[int(entry["training"]["start"]):int(entry["training"]["end"])]
                test_set     = reviews[int(entry["test"]["start"]):int(entry["test"]["end"])]
                vocab = yelp_data.buildVocab(training_set)
                training_set_prep = yelp_data.preProcessN(training_set, vocab, maxN)
                test_set_prep = yelp_data.preProcessN(test_set, vocab, maxN)
                naiveBayes = NaiveBayes(vocab)

                print "----------------------------------"
                print "Training model for test", test_group["id"], "entry", i
                print "----------------------------------"
                naiveBayes.Train(training_set_prep, maxN)
                print "----------------------------------"
                print "Running test data on constructed model...."
                print "----------------------------------"
                
                # Test accuracy
                total = 0.0
                right = 0.0
                false_pos = 0.0
                false_neg = 0.0
                for review in test_set_prep:
                    total += 1.0

                    if entry["backoff"]:
                        is_positive = naiveBayes.PredictPositiveStupidBackoff(review, maxN)
                    else:
                        is_positive = naiveBayes.PredictPositive(review, maxN)

                    if review[STARS] in naiveBayes.positive and is_positive:
                        right += 1.0
                    elif review[STARS] in naiveBayes.positive and not is_positive:
                        false_neg += 1.0
                    elif review[STARS] in naiveBayes.negative and not is_positive:
                        right += 1.0
                    elif review[STARS] in naiveBayes.negative and is_positive:
                        false_pos += 1.0

                WriteResult(outfile, total, right, false_pos, false_neg)

                print "Test complete...."
                print "----------------------------------\n\n\n"

if __name__ == '__main__':
    main()