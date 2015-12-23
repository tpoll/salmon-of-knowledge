import json
import sys

#Usage python parse_data.py <start range> <end range> <path to dataset>

def main():
    data = []
    new = open("reviews.json", "wb")
    onPos = True
    onNegative = False
    
    with open(sys.argv[3], 'rb') as f:
        count = int(sys.argv[1])
        upper = int(sys.argv[2])

        while count < upper:
            review  = json.loads(f.readline())
            if review["stars"] == 3:
                continue
            elif review["stars"] >= 4 and onPos:
                data.append({'text': review["text"], 'stars': review["stars"]})
                onPos = False
                onNegative = True
                count += 1
            elif review['stars'] <= 2 and onNegative:
                data.append({'text': review["text"], 'stars': review["stars"]})
                onPos = True
                onNegative = False
                count += 1

    json.dump(data, new)
    new.close()

if __name__ == '__main__':
    main()