import json
from nltk_utils import tokenize, stem
import numpy as np

with open("intents.json", 'r') as f:
    intents = json.load(f)

all_words = []
xy = []
tags = []

for intent in intents['intents']:
    tag = intent['tags']
    tags.append(tag)
<<<<<<< HEAD
    for pattern in intents['pattern']:
        pass
=======
    for pattern in intent['pattern']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w,tag))
        
ignore_words = ['?', '!', '.' , ',']
all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

X_train = []
Y_train = []
for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    
    label = tags.index(tag)
    Y_train.append(label)

X_train = np.array(X_train)
Y_train = np.array(Y_train)    
    
>>>>>>> b5af64cfc3e05b705b05050b3ea484dde40fa5bd
