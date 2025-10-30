from langchain_openai import ChatOpenAI
from langchain.agents import Tool , AgentType, initialize_agent
from langchain_community.tools import WikipediaQueryRun 
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.memory import ConvsersationBufferMemory

def Calculate(expression: str)->str:
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"error : {str(e)}"

wiki = WikipediaQueryRun(api_wrapper = WikipediaAPIWrapper())

tools = [
    Tool(
        name = "calculater",
        funct = Calculate ,
        description = " this will calculate ......"
    ),
    Tool(
        name = " wikipedia",
        funct = wiki ,
        description = " will call the and do on wikipedia"
    )
]

model =  ChatOpenAI(model = "gpt-3.5-turbo", temperature = 0.5)
memory = ConvsersationBufferMemory(memory_key = "chat_history")

agent = initialize_agent(
    llm = model,
    tools= tools,
    agent_type = AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    memory = memory,
    verbose = True
)

agent.run("put the query here.........")
