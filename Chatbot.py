import os
import nltk
import ssl
import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download('punkt')

intents = [
    {
        "tag": "greeting",
        "patterns":["Hi", "Hello", "Hey", "How are you", "What's up"],
        "responses":["Hi there", "Hello", "Hey", "I am fine, thank you", "Nothing much"]
    },
    {
        "tag": "goodbye",
        "patterns":["Bye", "See you later", "Goodbye", "Take care"],
        "responses":["Goodbye", "See you later", "Take care"]
    },
    {
        "tag": "thanks",
        "patterns":["Thank you", "Thanks", "Thanks a lot", "I appreciate it"],
        "responses":["You're welcome", "No problem", "Glad i could help"]
    },
    {
        "tag": "about",
        "patterns":["What can you do", "Who are you", "What are you", "What is your purpose"],
        "responses":["I am a chatbot", "My purpose is to assist you", "I can answer questions and provide assistance"]
    },
    {
        "tag": "help",
        "patterns":["Help", "I need help", "Can you help me", "What should i do"],
        "responses":["Sure, what do you need to help with", "I am here to help. What's the problem?", "How can i assist you?"]
    },
    {
        "tag": "age",
        "patterns":["How old are you", "What's your age"],
        "responses":["I dont have an age. I'm a chatbot.", "I was just born in the digital world", "Age is just a number for me"]
    },
    {
        "tag": "weather",
        "patterns":["What's the weather like", "How's the weather today"],
        "responses":["I'm sorry, i cannot provide real-time weather information.", "You can check the weather on a weather app or website"]
    },
    {
        "tag": "budget",
        "patterns":["How can I make a budget", "What's a good budgeting strategy", "How do I create a budget"],
        "responses":["To make budget, start by tracking your income and expenses. Then, allocate your income towards essential expenses."]
    },
]

# Create the vectorizer and classifier
vectorizer = TfidfVectorizer()
clf = LogisticRegression(random_state=0, max_iter=10000)

# Preprocess the data
tags = []
patterns = []
for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)

# Train the model
x = vectorizer.fit_transform(patterns)
clf.fit(x,tags)

def chatbot(user_input):
    user_input_vector = vectorizer.transform([user_input])
    predicted_tag = clf.predict(user_input_vector)[0]

    for intent in intents:
        if intent['tag'] == predicted_tag:
            response = random.choice(intent['responses'])
            return response
    return "I'm sorry, i didn't understand that."

counter = 0

def main():
    global counter
    st.title("Chatbot")
    st.write("Welcome to the chatbot. Please type a message and press Enter to start the conversation.")

    counter += 1
    user_input = st.text_input("You:", key=f"user_input_{counter}")

    if user_input:
        response = chatbot(user_input)
        st.text_area("Chatbot:", value=response, height=100, max_chars = None, key=f"chatbot_response_{counter}")
    
        if response.lower() in ["goodbye", "bye"]:
            st.write("Thank you for chatting with me. Have a great day!")
            st.stop()

if __name__ == '__main__':
    main()
    

