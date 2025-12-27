import json
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

# Load Data
with open('intents.json', encoding='utf-8') as f:
    data = json.load(f)

training_sentences = []
training_labels = []
labels = []

for intent in data['intents']:
    for pattern in intent['patterns']:
        training_sentences.append(pattern.lower())
        training_labels.append(intent['tag'])
    if intent['tag'] not in labels:
        labels.append(intent['tag'])

# Encoding
le = LabelEncoder()
training_labels = le.fit_transform(training_labels)

# Tokenizing
vocab_size = 1000
max_len = 20
tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(training_sentences)
sequences = tokenizer.texts_to_sequences(training_sentences)
padded = pad_sequences(sequences, truncating='post', maxlen=max_len)

# Model Building
model = Sequential([
    Embedding(vocab_size, 16, input_length=max_len),
    GlobalAveragePooling1D(),
    Dense(16, activation='relu'),
    Dense(16, activation='relu'),
    Dense(len(labels), activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(padded, np.array(training_labels), epochs=500, verbose=0)

# Save
model.save("chatbot_model.h5")
with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)
with open('labelencoder.pkl', 'wb') as f:
    pickle.dump(le, f)

print("✅ Training Complete! Files generated.")