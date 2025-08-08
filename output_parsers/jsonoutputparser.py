from langchain_core.prompts import PromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
    )


model = ChatHuggingFace(llm=llm)
parser = JsonOutputParser()

template=PromptTemplate(
    template = 'Give the name, age and city of a fictional person \n {format_instruction}',
    input_variables=[],
    partial_variables={'format_instruction': parser.get_format_instructions()}, # this will get filled before the run time
)

prompt = template.format()
"""
Give the name, age and city of aictional person 
 Return a JSON object.
 
 This is the output of the above prompt, the last line was get filled through parser
"""
print(prompt)
result = model.invoke(prompt)

print(result)



final_result = parser.parse(result.content)
# print(final_result)
# print(type(final_result))


# The above lines can be simply replaced with chains, thats the beauty of parsers

chain = template | model | parser

result = chain.invoke({})

print(result)