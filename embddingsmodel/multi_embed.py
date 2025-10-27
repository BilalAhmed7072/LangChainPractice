from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()

embedding = OpenAIEmbeddings(model='',
                             dimentions=32)
documents = ["pakistan's capital is islamabad",
             "lahore is the capital of punjab"]

result = embedding.embed_documents(documents)
print(str(result))