import streamlit as st
import requests

st.title("🤖 Rasa Chatbot")
st.write("Chat with your Rasa-powered AI bot!")

# User input
user_input = st.text_input("You:", "")

if st.button("Send"):
    response = requests.post("http://localhost:5005/webhooks/rest/webhook", json={"message": user_input})
    bot_response = response.json()

    for res in bot_response:
        st.text_area("Bot:", res.get("text", "No response"), height=50)
