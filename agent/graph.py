


# agent/graph.py
# --- FINAL VERSION ---

from langgraph.graph import StateGraph, END
from langchain_openai import AzureChatOpenAI
from langgraph_codeact import create_codeact
# Import all the new, granular tools from tools.py
from .tools import (
    analyze_document_type,
    extract_facture,
    extract_contrat_de_bail,
    extract_contrat_de_marche,
    extract_bordereau_de_versement,
    extract_contrat_acquisition,
    extract_contrat_de_maintenance
)
from dotenv import load_dotenv
import os
from langgraph.checkpoint.memory import InMemorySaver
import builtins
import contextlib
import io
from typing import Any

load_dotenv()

class State(dict):
    pass

def build_agent_graph(thread_id_key: str = "thread_id"):
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

    # Provide the agent with the full list of specialized tools
    tools = [
        analyze_document_type,
        extract_facture,
        extract_contrat_de_bail,
        extract_contrat_de_marche,
        extract_bordereau_de_versement,
        extract_contrat_acquisition,
        extract_contrat_de_maintenance,
    ]

    def eval(code: str, _locals: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        original_keys = set(_locals.keys())
        try:
            with contextlib.redirect_stdout(io.StringIO()) as f:
                exec(code, builtins.__dict__, _locals)
            result = f.getvalue()
            if not result:
                result = "<code ran, no output printed to stdout>"
        except Exception as e:
            result = f"Error during execution: {repr(e)}"
        new_keys = set(_locals.keys()) - original_keys
        new_vars = {key: _locals[key] for key in new_keys}
        return result, new_vars

    codeact_agent = create_codeact(
        model=llm,
        tools=tools,
        eval_fn=eval,
    )

    checkpointer = InMemorySaver()
    agent = codeact_agent.compile(checkpointer=checkpointer)
    return agent
