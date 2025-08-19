# app.py
# --- FINAL VERSION ---

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from agent.extract.extract_graph import build_agent_graph
from agent.rag.rag_graph import build_rag_agent_graph 
from agent.rag.rag_tools import search_document_chunks, transform_query
from langchain_openai import AzureChatOpenAI
import tempfile
import shutil
from langchain_core.messages import HumanMessage
import logging
import os
import json
from dotenv import load_dotenv
from agent.rag.graphRag_tools import query_graph
# --- Setup ---
load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI()
agent_app = build_agent_graph()
rag_agent_app = build_rag_agent_graph() 

@app.get("/")
async def root():
    return {"status": "ok"}

@app.post("/extract")
async def extract(file: UploadFile = File(...), id: str = Form(...)):
    """
    This endpoint gives the agent a high-level goal. The agent will decide which tools
    to use and in what order. The final response from the agent's tool is returned directly.
    """
    suffix = os.path.splitext(file.filename)[-1].lower()
    temp_file_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            temp_file_path = tmp.name

        logger.info(f"Saved uploaded file to temporary path: {temp_file_path}")
        logger.info(f"Starting agent with Thread ID: {id}")

        config = {"configurable": {"thread_id": id}}

        prompt = (
            "You are a robotic document processing pipeline. Your only function is to execute a sequence of tasks and return the final result."
            "\nYour tasks are:"
            "\n1. Execute the `analyze_document_type` tool on the document at this path: "
            f"'{temp_file_path}'."
            "\n2. Based on the document type, execute the single, correct extraction tool (e.g., `extract_contrat_de_bail`)."
            "\n3. Take the final output from that extraction tool. This output will be a Python dictionary."
            "\n4. Your final answer to the user MUST be ONLY the verbatim dictionary returned by the tool. Do not explain it, do not summarize it, do not reformat it. If the tool returns `{'status': 'Success', ...}`, your response is that dictionary. If the tool returns `{'error': '...', ...}`, your response is that error dictionary."
            "\n\nYour final response must be a dictionary and nothing else."
        )

        messages = [HumanMessage(content=prompt)]
        
        final_state = agent_app.invoke({"messages": messages}, config)
        
        final_tool_output = final_state["messages"][-1].content
        
        logger.info("Agent finished. Returning final dictionary from tool.")
        
        if final_tool_output:
            return final_tool_output
        else:
            logger.warning("Agent response was not valid")
            return {
                "error": "Agent response was not valid",
                "raw_output": final_tool_output
            }

    except Exception as e:
        logger.exception("Extraction process failed", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            logger.info(f"Cleaned up file: {temp_file_path}")

@app.post("/chat")
async def chat(prompt: str = Form(...), id: str = Form(...)):
    """
    Endpoint for the conversational RAG pipeline with memory, query transformation, and Graph RAG.
    """
    logger.info(f"Starting RAG pipeline with Thread ID: {id} and prompt: '{prompt}'")
    
    config = {"configurable": {"thread_id": id}}

    try:
        # Get chat history and transform the query
        # thread_state = rag_agent_app.get_state(config)
        # chat_history = thread_state.values.get('messages', []) if thread_state else []
        # history_for_prompt = [{"type": msg.type, "data": {"content": msg.content}} for msg in chat_history]
        # standalone_query = transform_query(prompt, history_for_prompt)
        
        # ### NEW LOGIC ###: First, try to answer using the knowledge graph.
        graph_results = query_graph(prompt)
        
        context = ""
        sources = []
        
        if graph_results and "error" not in graph_results[0]:
            logger.info("Found relevant information in the knowledge graph.")
            context = json.dumps(graph_results)
            sources = graph_results
        else:
            # Fallback to vector search if the graph doesn't have the answer
            logger.info("No direct answer in graph, falling back to vector search.")
            retrieved_chunks = search_document_chunks(prompt)
            context = "\n\n---\n\n".join([chunk['contents'] for chunk in retrieved_chunks])
            sources = retrieved_chunks

        # Construct the final prompt for the agent
        final_prompt = (
            "You are a helpful assistant who answers questions based on the provided context. "
            "If the context is a JSON object from a graph database, interpret the relationships to form a natural language answer. "
            "If the context is plain text, summarize the key information to answer the question.\n\n"
            "--- CONTEXT ---\n"
            f"{context}\n\n"
            "--- QUESTION ---\n"
            f"{prompt}\n\n"
            "--- ANSWER ---\n"
        )
        
        # Invoke the agent to get the final answer
        messages = [HumanMessage(content=final_prompt)]
        final_state = rag_agent_app.invoke({"messages": messages}, config)
        
        final_response = final_state["messages"][-1].content
        
        logger.info("RAG pipeline finished. Returning conversational response.")
        
        return {
            "response": final_response,
            "sources": sources # Return whichever source was used
        }

    except Exception as e:
        logger.exception("Chat process failed", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
