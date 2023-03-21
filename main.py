# from time import sleep
from autocorrect import Speller
import tensorflow
import numpy
import tflearn
import pickle
import json
import random
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
spell = Speller(lang='en')
# import pyttsx3
# engine = pyttsx3.init()
#import pyaudio
import speech_recognition


with open('intents.json') as file:
    data = json.load(file)

with open("data.pickle", "rb") as f:
    words, labels, training, output = pickle.load(f)

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)
model.load("model.tflearn")

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]
    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return numpy.array(bag)

def process(message):
    inp = message
    corrected_inp = spell(inp)
    results = model.predict([bag_of_words(corrected_inp, words)])
    results_index = numpy.argmax(results)
    tag = labels[results_index]
    for tg in data["intents"]:
        if tg['tag'] == tag:
            responses = tg['responses']

    # if responses == []: # get from firestore

    # sleep(3)
    bot = random.choice(responses)
    return(bot)

from flask import Flask, render_template, request
app = Flask(__name__, template_folder='template')

@app.route('/')
def home():
    return render_template("good.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    return str(process(userText))
if __name__ == "__main__":
    app.run(debug=True)

