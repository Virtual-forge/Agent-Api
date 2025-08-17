



# app.py
# --- FINAL VERSION ---

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from agent.graph import build_agent_graph
import tempfile
import shutil
from langchain_core.messages import HumanMessage
import logging
import os
import json

# --- Setup ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI()
agent_app = build_agent_graph()
SEPARATOR = "---JSON_AND_TEXT_SEPARATOR---"

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

        # This prompt correctly tells the agent to return the raw tool output.
        # In app.py, inside the /extract endpoint

# This prompt is designed to force the agent to be a simple data forwarder.
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
        
        # The final message content is now the clean dictionary from your tool.
        final_tool_output = final_state["messages"][-1].content
        
        logger.info("Agent finished. Returning final dictionary from tool.")
        
        if final_tool_output:
            return final_tool_output
        else:
            # If the output isn't a dictionary, it's likely an error message string.
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
