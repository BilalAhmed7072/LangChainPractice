from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel

load_dotenv()

model1 = ChatOpenAI(model_name=' ', temperature= 0.3)
model2 = ChatAnthropic(model_name=' ', temperature= 0.3)

prompt1 = PromptTemplate(
    template=" give short and concise notes on the topic. \n {text}",
    input_variables= [ 'text']
)
prompt2 = PromptTemplate(
    template=" give 5 short question answer form the notes. \n {text}",
    input_variables=['text']
)

prompt3 = PromptTemplate(
    template= " merge the both notes {notes} and  quiz {quiz} in a single document.",
    input_variables= ['notes','quiz']
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'notes': prompt1 | model1 | parser ,
    'quiz' : prompt2 | model2 | parser
})

merge_chain =  prompt3 | model2 | parser

chain = parallel_chain | merge_chain

text = """"Still effective in cases where number of dimensions is greater than the number of samples.

Uses a subset of training points in the decision function (called support vectors), so it is also memory efficient.

Versatile: different Kernel functions can be specified for the decision function. Common kernels are provided, but it is also possible to specify custom kernels.

The disadvantages of support vector machines include:

If the number of features is much greater than the number of samples, avoid over-fitting in choosing Kernel functions and regularization term is crucial.

SVMs do not directly provide probability estimates, these are calculated using an expensive five-fold cross-validation (see Scores and probabilities, below).


"""

result = chain.invoke({'text': text})
print(result.content)

chain.get_graph().print_ascii()