# agent/rag/rag_tools.py

from langchain_core.tools import tool
import logging
from typing import List, Dict, Any
import psycopg2
from psycopg2.extras import Json
from agent.extract.extract_tools import create_embedding, get_db_connection, llm # Re-using existing helpers
from langchain_core.messages import HumanMessage


logger = logging.getLogger(__name__)

def transform_query(question: str, chat_history: List[Dict[str, Any]]) -> str:
    """
    Rewrites a follow-up question into a standalone question using chat history.
    """
    if not chat_history:
        return question

    history_str = ""
    for turn in chat_history:
        if turn['type'] == 'human':
            history_str += f"Human: {turn['data']['content']}\n"
        elif turn['type'] == 'ai':
            history_str += f"AI: {turn['data']['content']}\n"

    prompt = (
        "Given the following conversation history and a follow-up question, "
        "rephrase the follow-up question to be a standalone question that can be used for a vector search. "
        "Your response should ONLY be the rephrased question and only use keywords from the chat history DO NOT improvise.\n\n"
        "--- CHAT HISTORY ---\n"
        f"{history_str}\n\n"
        "--- FOLLOW-UP QUESTION ---\n"
        f"{question}\n\n"
        "--- STANDALONE QUESTION ---\n"
    )
    
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        standalone_question = response.content
        logger.info(f"Transformed query from '{question}' to '{standalone_question}'")
        return standalone_question
    except Exception as e:
        logger.error(f"Error during query transformation: {e}")
        return question # Fallback to the original question on error



def search_document_chunks(query: str) -> List[Dict[str, Any]]:
    """
    Searches for relevant document chunks in the database based on a user's query.
    """
    logger.info(f"RAG Tool: Searching for chunks related to query: '{query}'")
    
    embedding_response = create_embedding(query)
    if not embedding_response:
        return [{"error": "Failed to create embedding for the query."}]
    
    query_embedding = embedding_response.data[0].embedding

    sql = """
        SELECT parent_id, contents, metadata, embedding <-> %s::halfvec AS score
        FROM child_doc
        ORDER BY score
        LIMIT 5;
    """
    
    conn = None
    results = []
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute(sql, (str(query_embedding),))
            rows = cur.fetchall()
            for row in rows:
                raw_distance = row[3]
                similarity = 1 / (1 + raw_distance**2)
                score_percentage = round(similarity * 100, 2)

                chunk_metadata = row[2]
                chunk_metadata['score'] = score_percentage

                results.append({
                    "parent_id": row[0],
                    "contents": row[1],
                    "metadata": chunk_metadata
                })
        logger.info(f"RAG Tool: Found {len(results)} relevant chunks.")
    except (Exception, psycopg2.DatabaseError) as error:
        logger.error(f"RAG Tool: Error searching for chunks: {error}", exc_info=True)
        return [{"error": "A database error occurred during search."}]
    finally:
        if conn is not None:
            conn.close()
            
    return results
