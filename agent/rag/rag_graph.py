
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
import os
from langgraph.checkpoint.memory import InMemorySaver
from typing import List, TypedDict
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage
import logging

load_dotenv()
logger = logging.getLogger(__name__)

class AgentState(TypedDict):
    messages: List[BaseMessage]

def build_rag_agent_graph():
    """
    Builds a simple conversational agent with memory, but no tools.
    Its purpose is to generate a response based on the provided message history.
    """
    deployment = os.getenv("AZURE_DEPLOYMENT_NAME")
    api_version = os.getenv("OPENAI_API_VERSION")
    api_key = os.getenv("OPENAI_API_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

    if not all([deployment, api_version, api_key, endpoint]):
        raise ValueError("Missing Azure environment variables")

    llm = AzureChatOpenAI(
        azure_deployment=deployment,
        openai_api_version=api_version,
        azure_endpoint=endpoint,
        api_key=api_key,
        temperature=0,
    )

    def agent_node(state):
        """
        Invokes the LLM with the current state and appends the response.
        """
        response = llm.invoke(state["messages"])
        return {"messages": state["messages"] + [response]}

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", agent_node)
    workflow.set_entry_point("agent")
    workflow.add_edge("agent", END)

    checkpointer = InMemorySaver()
    graph = workflow.compile(checkpointer=checkpointer)
    return graph
