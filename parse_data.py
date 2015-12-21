import json
import sys

#Usage python parse_data.py <start range> <end range> <path to dataset>

def main():
    data = []
    new = open("reviews.json", "wb")
    with open(sys.argv[3], 'rb') as f:
        count = int(sys.argv[1])
        upper = int(sys.argv[2])
        while count < upper:
            review  = json.loads(f.readline())
            if review["stars"] == 3:
                continue
            data.append({'text': review["text"], 'stars': review["stars"]})
            count += 1


    json.dump(data, new)
    new.close()


if __name__ == '__main__':
    main()