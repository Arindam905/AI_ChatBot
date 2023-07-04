import json
from nltk_utils import tokenizer, stem
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import random
from nltk_utils import bag_of_words, tokenizer, stem


with open("intents.json", 'r') as f:
    intents = json.load(f)

all_words = []
xy = []
tags = []

for intent in intents['intents']:
    tag = intent['tags']
    tags.append(tag)
    for pattern in intent['pattern']:
        w = tokenizer(pattern)
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

class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = Y_train
       
    #dataset[idx]
    def __getitem__(self, index):
        return  self.x_data[index], self.y_data[index]   
    
    def __len__(self):
        return self.n_samples
    
batch_size =8
    
dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, branch_size=batch_size, shuffles=True, num_workers=0)