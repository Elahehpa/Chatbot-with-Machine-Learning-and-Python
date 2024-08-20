# Chatbot-with-Machine-Learning-and-Python
This project demonstrates how to create a simple chatbot using Python and machine learning. The chatbot uses a neural network model to classify user intents based on predefined patterns and respond accordingly.
## Project Overview

1. **Define Intents and Responses**: Define various intents with their associated patterns and responses in a JSON file.
2. **Data Preparation**: Import necessary libraries, process data, and prepare it for training.
3. **Model Training**: Create and train a neural network model to classify user intents.
4. **Save the Model**: Save the trained model, tokenizer, and label encoder for future use.
5. **Chat Function**: Implement a chat function to interact with the chatbot using the trained model.

## Files

- `intents.json`: Contains intents, patterns, and responses for the chatbot.
- `chat_model`: Saved neural network model.
- `tokenizer.pickle`: Saved tokenizer object.
- `label_encoder.pickle`: Saved label encoder object.
- `chatbot.py`: Main script for training and interacting with the chatbot.

## Installation

To run this chatbot, you need to have Python and the required libraries installed. You can install the necessary libraries using `pip`:

```bash
pip install numpy tensorflow scikit-learn keras colorama

## Usage
Training the Model
Prepare the Data: Ensure intents.json is in the same directory as your script.

Run the Training Script:
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import pickle

# Load and process data
with open('intents.json') as file:
    data = json.load(file)

training_sentences = []
training_labels = []
labels = []
responses = []

for intent in data['intents']:
    for pattern in intent['patterns']:
        training_sentences.append(pattern)
        training_labels.append(intent['tag'])
    responses.append(intent['responses'])
    if intent['tag'] not in labels:
        labels.append(intent['tag'])

num_classes = len(labels)

lbl_encoder = LabelEncoder()
lbl_encoder.fit(training_labels)
training_labels = lbl_encoder.transform(training_labels)

vocab_size = 1000
embedding_dim = 16
max_len = 20
oov_token = "<OOV>"

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
tokenizer.fit_on_texts(training_sentences)
sequences = tokenizer.texts_to_sequences(training_sentences)
padded_sequences = pad_sequences(sequences, truncating='post', maxlen=max_len)

# Define and train the model
model = Sequential()
model.add(Embedding(vocab_size, embedding_dim, input_length=max_len))
model.add(GlobalAveragePooling1D())
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

epochs = 500
history = model.fit(padded_sequences, np.array(training_labels), epochs=epochs)

# Save the model and tokenizer
model.save("chat_model")

with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('label_encoder.pickle', 'wb') as ecn_file:
    pickle.dump(lbl_encoder, ecn_file, protocol=pickle.HIGHEST_PROTOCOL)

Running the Chatbot
Start the Chatbot: Execute the chatbot.py script to start the chatbot.
import json
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
import colorama
colorama.init()
from colorama import Fore, Style
import random
import pickle

with open("intents.json") as file:
    data = json.load(file)

def chat():
    model = keras.models.load_model('chat_model')

    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    with open('label_encoder.pickle', 'rb') as enc:
        lbl_encoder = pickle.load(enc)

    max_len = 20

    while True:
        print(Fore.LIGHTBLUE_EX + "User: " + Style.RESET_ALL, end="")
        inp = input()
        if inp.lower() == "quit":
            break

        result = model.predict(keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences([inp]), truncating='post', maxlen=max_len))
        tag = lbl_encoder.inverse_transform([np.argmax(result)])

        for i in data['intents']:
            if i['tag'] == tag:
                print(Fore.GREEN + "ChatBot:" + Style.RESET_ALL, np.random.choice(i['responses']))

print(Fore.YELLOW + "Start messaging with the bot (type quit to stop)!" + Style.RESET_ALL)
chat()
