from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()
template = " give me 4 interesting facts about {topic}"

prompt = PromptTemplate(
    template=template,
    input_variables=['topic']
)

model = ChatOpenAI()

parser = StrOutputParser()
chain = prompt | model | parser
result = chain.invoke({'topic':'pakistan'})

print(result.content)

#for chain visualization
chain.get_graph().print_ascii()