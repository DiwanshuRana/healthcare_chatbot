from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
import pickle
import json
import random
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

model = tf.keras.models.load_model("chatbot_model.h5")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)
with open("labelencoder.pkl", "rb") as f:
    le = pickle.load(f)
with open("intents.json", encoding='utf-8') as f:
    intents = json.load(f)

def get_response(text):
    max_len = 20
    seq = tokenizer.texts_to_sequences([text.lower()])
    padded = pad_sequences(seq, truncating='post', maxlen=max_len)
    pred = model.predict(padded)
    idx = np.argmax(pred)
    tag = le.inverse_transform([idx])[0]
    
    if pred[0][idx] < 0.3: # Agar model sure nahi hai
        return "I am not sure, please consult a doctor."

    for intent in intents['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get", methods=["POST"])
def chatbot():
    msg = request.form["msg"]
    return jsonify({"response": get_response(msg)})

if __name__ == "__main__":
    app.run(debug=True)