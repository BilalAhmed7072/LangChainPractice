from langchain_core.messages import HumanMessage , AIMessage,SystemMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()

model = ChatOpenAI()
messages = [SystemMessage(content="you are the helpful assistant"),
            HumanMessage(content="who is imran khan")]


result = model.invoke(messages)
messages.append(AIMessage(content=result.content))
print(messages)