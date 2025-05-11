from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI()
# We will give the role at the top of all messages that is a system message
messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="Tell me about Langchain")
]

result = model.invoke(messages)
messages.append(AIMessage(content=result.content))

print(messages)
