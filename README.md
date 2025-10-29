# 💬 AI Chatbot for Mental Health Support

## 🧠 Overview
The **AI Chatbot for Mental Health Support** is a conversational AI project built using Natural Language Processing (NLP).  
It aims to provide empathetic, supportive, and non-judgmental responses to users expressing their emotions.

While it is **not a replacement for professional therapy**, it demonstrates how AI can be leveraged to create safe, comforting spaces for people to talk and feel heard.

---

## 🎯 Objectives
- To simulate human-like conversations with emotional awareness.  
- To create a simple, user-friendly chatbot interface.  
- To explore real-world applications of NLP using pre-trained language models.

---

## ⚙️ Tools & Technologies Used
- **Python**
- **Hugging Face Transformers**
- **PyTorch**
- **Streamlit** (for web interface)
- **NLTK** (for text preprocessing)

---

## 🧩 Project Workflow
1. **Model Selection:**  
   Used the pre-trained **DialoGPT-small** model from Hugging Face for conversational dialogue generation.

2. **Preprocessing:**  
   Implemented basic NLP filters to detect negative emotions and prompt empathetic responses.

3. **Response Generation:**  
   Used transformer-based architecture to create context-aware, human-like replies.

4. **Interface Design:**  
   Built a simple **Streamlit** app for real-time chatting with the AI.

5. **Testing & Evaluation:**  
   Conducted sample user interactions to validate tone, empathy, and contextual accuracy.

---

## 🖥️ Installation & Setup
```bash
# Clone this repository
git clone https://github.com/yourusername/AI-Mental-Health-Chatbot.git
cd AI-Mental-Health-Chatbot

# Install dependencies
pip install transformers torch streamlit nltk

# (Optional) Download NLTK data
python -m nltk.downloader punkt
