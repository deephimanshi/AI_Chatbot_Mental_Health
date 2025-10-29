import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

st.title("💬 AI Mental Health Support Chatbot")
st.write("Hi there 👋 I’m here to listen and support you. This chatbot uses NLP to respond empathetically to your feelings.")

tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")

def preprocess_input(text):
    negative_words = ["sad", "depressed", "angry", "anxious", "hopeless", "lonely"]
    if any(word in text.lower() for word in negative_words):
        return text + " Please respond with empathy."
    return text

if "history" not in st.session_state:
    st.session_state["history"] = None

user_input = st.text_input("You:")

if st.button("Send") and user_input:
    new_input_ids = tokenizer.encode(preprocess_input(user_input) + tokenizer.eos_token, return_tensors='pt')
    bot_input_ids = torch.cat([st.session_state["history"], new_input_ids], dim=-1) if st.session_state["history"] is not None else new_input_ids
    st.session_state["history"] = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(st.session_state["history"][:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    st.text_area("Chatbot:", value=response, height=150)
