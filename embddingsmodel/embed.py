from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()
embedding = OpenAIEmbeddings(model='',
                             dimensions=32)
prompt = 'who was quiad e azam?'
result = embedding.aembed_query(prompt)
print(str(result))