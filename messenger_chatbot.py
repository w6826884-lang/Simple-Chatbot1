from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
load_dotenv()
model = HuggingFaceEndpoint(
    repo_id = "dolphin-2.9.2-qwen2-72b",
    task='text-generation'
)
model = ChatHuggingFace(model=model)
messages=[
    SystemMessage(Content="You are a  political satirist globally and you are aware about the latest political memes in social media." \
    "You have to generate god level political sattire to the questions asked to you")
]
chat_history = []
while True:
    user_input = input('You :')
    chat_history.append(user_input)
    if user_input=='exit':
        break
    result = model.invoke(chat_history)
    print("AI: ", result.content)
    chat_history.append(result)
    
