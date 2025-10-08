from langchain.schema.runnable import RunnableLambda
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableSequence, RunnableLambda, RunnableParallel, RunnablePassthrough

from Runnables.runnable_passthrough import joke_gen_chain


def word_counter(text):
    return len(text.split())

runnable_word_counter = RunnableLambda(word_counter)

#print(runnable_word_counter.invoke('Hi, there man'))

load_dotenv()

prompt = PromptTemplate(
    template = 'Write a joke about {topic}',
    input_variables=['topic']
)

model = ChatOpenAI()

parser = StrOutputParser()

joke_gen_chain = RunnableSequence(prompt, model, parser)

parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'word_count': RunnableLambda(word_counter),
})

final_chain = RunnableSequence(joke_gen_chain, parallel_chain)

result = final_chain.invoke({'topic': 'AI'})

print(result['joke'], result['word_count'])