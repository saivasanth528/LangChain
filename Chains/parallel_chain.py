from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint


load_dotenv()

model1 = ChatOpenAI()

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
    )

model2 = ChatHuggingFace(llm=llm)

prompt1 = PromptTemplate(
    template = 'Generate the short and crisp notes from the following text \n {text}',
    input_variables=['text']
)

prompt2 = PromptTemplate(
    template = 'Generate 5 short question and answers from the following text \n {text}',
    input_variables=['text']
)

prompt3 = PromptTemplate(
    template = 'Merge the provided notes and quiz into a single document \n notes -> {notes} and quiz -> {quiz}',
    input_variables=['notes', 'quiz']
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'notes': prompt1 | model1 | parser,
    'quiz': prompt2 | model2 | parser
})

merge_chain = prompt3 | model1 | parser

chain = parallel_chain | merge_chain


text = """
Linear regression is one of the most fundamental and commonly used statistical techniques in data science and machine learning. It is used to model the relationship between a dependent variable (also called the response or output) and one or more independent variables (also called predictors or features). The idea is to find a linear relationship — one that can be represented by a straight line — that best fits the data.

In simple linear regression, there is only one independent variable, and the model tries to fit a straight line to the data using the equation:
y=mx+b
y=mx+b

Here, yy is the predicted value, xx is the input, mm is the slope of the line (showing the rate of change), and bb is the y-intercept (the value of yy when x=0x=0).

The main goal of linear regression is to find the best-fitting line that minimizes the difference between the actual data points and the predicted values. This difference is called the residual, and the model tries to minimize the sum of squared residuals, also known as the mean squared error (MSE). This process is called the least squares method.

Linear regression can also be extended to handle more than one independent variable. This is known as multiple linear regression. In this case, the model fits a hyperplane in a multi-dimensional space and the equation looks like:


Linear regression is widely used in real-world applications such as predicting house prices, forecasting sales, estimating demand, analyzing trends, and identifying relationships between variables. It's particularly popular because of its simplicity, interpretability, and efficiency.

However, it comes with some important assumptions: the relationship between variables must be linear; the residuals (errors) should be normally distributed and have constant variance (homoscedasticity); the input variables should not be too highly correlated (no multicollinearity); and the data should be free from significant outliers that can distort the results.

Despite its limitations, linear regression is a valuable tool for understanding data and building predictive models. It also serves as the building block for more advanced regression techniques and machine learning models.

In summary, linear regression is a simple yet powerful method that helps in making predictions and understanding the influence of different variables on an outcome. It is an essential concept in both statistics and machine learning.
"""

result = chain.invoke({'text': text})

print(result)

chain.get_graph().print_ascii()



