from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

load_dotenv()

model = ChatAnthropic(model='claude-sonnet-4-5-20250929', temperature=0.6, max_completion_tokens =200)
prompt = 'who is imran khan and write his political struggle.'
result = model.invoke(prompt)
print(result.content)