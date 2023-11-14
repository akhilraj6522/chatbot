import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
import os
import numpy
import tflearn
import tensorflow
import random
import pickle
import json

#open the json file
with open('intents.json') as file:
    data = json.load(file)


all_words = []
labels = []
docs_x = []
docs_y = []

# loop through the intents
for intent in data['intents']:
    # loop through patterns of each intent
    for pattern in intent['patterns']:
    
        # tokenize pattern: "whats on the menu" --> ["whats", "on", "the", "menu"]
        words_in_pattern = nltk.word_tokenize(pattern)
        
        # append tokenized words
        all_words.extend(words_in_pattern)
        
        docs_x.append(words_in_pattern)
        docs_y.append(intent["tag"])
        
    if intent['tag'] not in labels:
        labels.append(intent['tag'])
print(all_words, '\n\n\n', docs_x, '\n\n\n', docs_y, '\n\n\n', labels)

# stemming taking each word in entire word set and bring it down to the root word     told / talking -> talk
all_words = [stemmer.stem(w.lower()) for w in all_words if w != "?"]

#remove duplicate words
all_words = sorted(list(set(all_words)))

labels = sorted(labels)

training = []
output = []

out_empty = [0 for _ in range(len(labels))]

for x, doc in enumerate(docs_x):
    bag = []

    wrds = [stemmer.stem(w.lower()) for w in doc]

    for w in all_words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)

    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1

    training.append(bag)
    output.append(output_row)


training = numpy.array(training) #one hot encoded list of patterns
output = numpy.array(output)     #one hot encoded list of tags

with open("data.pickle", "wb") as f:
    pickle.dump((all_words, labels, training, output), f)

tensorflow.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

if os.path.exists("./model.tflearn.index"):
    model.load("./model.tflearn")
else:
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("./model.tflearn")
    
    
    
def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
            
    return numpy.array(bag)


def chat():
    print("Start talking with the bot (type quit to stop)!")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break

        results = model.predict([bag_of_words(inp, all_words)])
        results_index = numpy.argmax(results)
        tag = labels[results_index]

        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']

        print(random.choice(responses))

chat()