import json
from nltk_utils import tokenize, stem

with open("intents.json", 'r') as f:
    intents = json.load(f)

all_words = []
xy = []
tags = []

for intent in intents['intents']:
    tag = intent['tags']
    tags.append(tag)
    for pattern in intents['pattern']:
        pass