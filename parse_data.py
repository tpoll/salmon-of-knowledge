import json
import sys

#Usage python parse_data.py <start range> <end range> <path to dataset>

def main():
    data = []
    new = open("reviews.json", "wb")
    with open(sys.argv[3], 'rb') as f:
        for x in xrange(int(sys.argv[1]),int(sys.argv[2])):
            review  = json.loads(f.readline())
            data.append({'text': review["text"], 'stars': review["stars"]})


    json.dump(data, new)
    new.close()


if __name__ == '__main__':
    main()