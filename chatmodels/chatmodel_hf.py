from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from dotenv import load_dotenv
load_dotenv()

llm = HuggingFaceEndpoint(repo_id='skyxiaobaibai/TinyLlma-1.1B-alpaca_2k-Q4_0-GGUF',
                    task="text-generation")
model = ChatHuggingFace(llm=llm)

prompt = 'who is bilal ahemd and what you know about this name?'

response = model.invoke(prompt)
print(response.content)

