from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()

model = ChatOpenAI()

template1 = PromptTemplate(
    template='Write a detailed report on {topic} ',
    input_variables=['topic']
)

template2 = PromptTemplate(
    template='Write a 5 line summary on the following text. \n {text}',
    input_variables=['text']
)

parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser

"""
Now what is the difference here, if you observe for with out parsers, in the above chain we can only chain
the first two steps, extract the result then again we need to form the second chain. But through this, it became
much easier and simple to form a single chain and to play with the models
"""

result = chain.invoke({'topic': 'The impact of AI on modern society'})
print(result)
