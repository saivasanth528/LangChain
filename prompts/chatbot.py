from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
load_dotenv()

model = ChatOpenAI()
chat_history = [
    SystemMessage(content="You are a helpful assistant.")
]

while True:
    user_input = input("You: ")
    chat_history.append(HumanMessage(content=user_input))
    if user_input.lower() == "exit":
        break

    result = model.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content))
    print(f"AI Chatbot: {result.content}")

print(chat_history)