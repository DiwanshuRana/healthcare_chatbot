import streamlit as st
import numpy as np
import tensorflow as tf
import pickle
import json
import random
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- PAGE SETUP ---
st.set_page_config(page_title="Healthcare Chatbot", page_icon="🏥")

# --- LOAD MODELS (Optimized for Streamlit) ---
@st.cache_resource
def load_my_model():
    model = tf.keras.models.load_model("chatbot_model.h5")
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    with open("labelencoder.pkl", "rb") as f:
        le = pickle.load(f)
    with open("intents.json", encoding='utf-8') as f:
        intents = json.load(f)
    return model, tokenizer, le, intents

# Loading assets
try:
    model, tokenizer, le, intents = load_my_model()
except Exception as e:
    st.error(f"Files load nahi ho pa rahi hain. Please check karein: {e}")
    st.stop()

# --- PREDICTION LOGIC ---
def get_response(text):
    max_len = 20
    # Preprocessing
    seq = tokenizer.texts_to_sequences([text.lower()])
    padded = pad_sequences(seq, truncating='post', maxlen=max_len)
    
    # Model Prediction
    pred = model.predict(padded)
    idx = np.argmax(pred)
    tag = le.inverse_transform([idx])[0]
    
    # Confidence Score check (Aapka 0.3 wala logic)
    if pred[0][idx] < 0.3:
        return "I am not sure, please consult a doctor."

    # Response nikalna
    for intent in intents['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])
    
    return "Maaf kijiye, main samajh nahi paaya."

# --- STREAMLIT UI ---
st.title("🏥 AI Healthcare Chatbot")
st.markdown("---")

# Chat History maintain karne ke liye (Session State)
if "messages" not in st.session_state:
    st.session_state.messages = []

# Purani chats screen par dikhane ke liye
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User se input lene ke liye
if prompt := st.chat_input("Apne symptoms batayein..."):
    # 1. User ka message dikhao
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Bot ka response generate karo
    with st.chat_message("assistant"):
        response = get_response(prompt)
        st.markdown(response)
    
    # 3. History mein save karo
    st.session_state.messages.append({"role": "assistant", "content": response})
