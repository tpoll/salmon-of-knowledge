import json
import sys



def getReviews():
    with open("reviews.json", 'rb') as f:
        return json.load(f)
