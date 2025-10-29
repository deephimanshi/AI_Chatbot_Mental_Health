from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

print("Loading model... (this may take a few seconds)")
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")

def preprocess_input(text):
    negative_words = ["sad", "depressed", "angry", "anxious", "hopeless", "lonely"]
    if any(word in text.lower() for word in negative_words):
        return text + " Please respond with empathy and kindness."
    return text

def chat():
    print("Chatbot: Hello, I’m here to listen. How are you feeling today?")
    chat_history_ids = None
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["quit", "exit", "bye"]:
            print("Chatbot: Take care. You’re doing your best 💛")
            break
        new_input_ids = tokenizer.encode(preprocess_input(user_input) + tokenizer.eos_token, return_tensors='pt')
        bot_input_ids = torch.cat([chat_history_ids, new_input_ids], dim=-1) if chat_history_ids is not None else new_input_ids
        chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
        response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
        print(f"Chatbot: {response}")

if __name__ == "__main__":
    chat()
