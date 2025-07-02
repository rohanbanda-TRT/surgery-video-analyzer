from langchain.agents import (
    AgentExecutor,
    create_openai_tools_agent,
)
from langchain_core.messages import HumanMessage


def create_agent(llm, tools: list, system_prompt):

    agent = create_openai_tools_agent(llm, tools, system_prompt)
    executor = AgentExecutor(agent=agent, tools=tools)

    return executor

def agent_node(state, agent, name):
    result = agent.invoke(state)
    return {"messages": [HumanMessage(content=result["output"], name=name)]}

def create_supervisor():
    pass
