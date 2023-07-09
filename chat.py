import random
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenizer
import pyttsx3
import speech_recognition as sr
import pyaudio


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

engine = pyttsx3.init('sapi5')
voices = engine.getProperty('voices') 
engine.setProperty('voice', voices[0].id)
def speak(audio):   
    engine.say(audio)    
    engine.runAndWait() 


def takeCommand():
     
    r = sr.Recognizer()
     
    with sr.Microphone() as source:
         
        print("Listening...")
        r.pause_threshold = 1
        audio = r.listen(source)
  
    try:
        print("Recognizing...")   
        query = r.recognize_google(audio, language ='en-in')
        print(f"User said: {query}\n")
  
    except Exception as e:
        print(e)   
        print("Unable to Recognize your voice.") 
        return "None"
    return query

with open("AI_ChatBot\starwarsintents.json", 'r') as f:
    intents = json.load(f)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)

model.load_state_dict(model_state)
model.eval()

bot_name = "Levi"
print("Let's chat! (type 'quit' to exit)")
while True:
    
    sentence = takeCommand()
    if sentence == "quit":
        break

    sentence = tokenizer(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                print(f"{bot_name}: {random.choice(intent['responses'])}")
                speak(random.choice(intent['responses']))
    else:
        print(f"{bot_name}: I do not understand...")