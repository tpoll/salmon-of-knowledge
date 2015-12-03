import json
import sys
from nltk import word_tokenize

#Usage python parse_data.py <start range> <end range> <path to dataset>

def main():
    data = []
    new = open("reviews.json", "wb")
    with open(sys.argv[3], 'rb') as f:
        for x in xrange(int(sys.argv[1]),int(sys.argv[2])):
            review  = json.loads(f.readline())
            data.append({'text': word_tokenize(review["text"]), 'stars': review["stars"]})


    json.dump(data, new)
    new.close()


if __name__ == '__main__':
    main()