import json
import sys


reviews = {}

def main():
    global reviews

    with open("reviews.json", 'rb') as f:
        reviews = json.load(f)

if __name__ == '__main__':
    main()