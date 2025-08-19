# agent/rag/graph_rag_tools.py

import logging
from typing import List, Dict, Any
import psycopg2
from psycopg2.extras import Json
from agent.extract.extract_tools import llm, get_db_connection # Re-using existing helpers
from langchain_core.messages import HumanMessage
import os
import json
from agent.extract.extract_tools import Document
logger = logging.getLogger(__name__)

# --- PostgreSQL Connection is used for everything ---

def extract_and_ingest_graph(chunks: List[Document]):
    """
    Extracts entities and relationships from text chunks and ingests them into PostgreSQL.
    """
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            for chunk in chunks:
                content = chunk.page_content # Use page_content from Document object
                
                prompt = (
                    "You are a data extraction expert. From the following text, extract key entities (like companies, people, contract references) and their relationships. "
                    "Format your response as a JSON object with two keys: 'entities' and 'relationships'. "
                    "'entities' should be a list of dictionaries, each with 'name' and 'type'. "
                    "'relationships' should be a list of dictionaries, each with 'source', 'target', and 'type'.\n\n"
                    f"--- TEXT ---\n{content}"
                )
                
                response = llm.invoke([HumanMessage(content=prompt)])
                graph_data = json.loads(response.content)

                # Ingest entities into graph_nodes table
                for entity in graph_data.get('entities', []):
                    cur.execute(
                        "INSERT INTO graph_nodes (name, type) VALUES (%s, %s) ON CONFLICT (name, type) DO NOTHING;",
                        (entity['name'], entity['type'])
                    )
                
                # Ingest relationships into graph_edges table
                for rel in graph_data.get('relationships', []):
                    # Get the IDs of the source and target nodes
                    cur.execute("SELECT id FROM graph_nodes WHERE name = %s AND type = %s;", (rel['source'], 'Client')) # Assuming type for now
                    source_id_result = cur.fetchone()
                    cur.execute("SELECT id FROM graph_nodes WHERE name = %s AND type = %s;", (rel['target'], 'Prestataire')) # Assuming type for now
                    target_id_result = cur.fetchone()

                    if source_id_result and target_id_result:
                        source_id = source_id_result[0]
                        target_id = target_id_result[0]
                        cur.execute(
                            """
                            INSERT INTO graph_edges (source_id, target_id, type)
                            VALUES (%s, %s, %s);
                            """,
                            (source_id, target_id, rel['type'])
                        )
            conn.commit()
        logger.info("Successfully ingested graph data into PostgreSQL.")
    except (Exception, psycopg2.DatabaseError) as error:
        logger.error(f"Error during graph ingestion: {error}")
    finally:
        if conn is not None:
            conn.close()


def query_graph(query: str) -> List[Dict[str, Any]]:
    """
    Translates a natural language query into a SQL query for the graph tables and executes it.
    """
    conn = None
    try:
        # Use an LLM to generate a SQL query
        prompt = (
            "You are a PostgreSQL expert. Given the schema (tables: 'graph_nodes' with columns 'id', 'name', 'type'; and 'graph_edges' with columns 'id', 'source_id', 'target_id', 'type'), "
            "translate the following user question into a SQL query that joins these tables to find the answer. "
            "Respond with only the SQL query.\n\n"
            f"--- QUESTION ---\n{query}"
        )
        
        response = llm.invoke([HumanMessage(content=prompt)])
        sql_query = response.content.strip().replace('`', '').replace('sql\n', '') # Clean up LLM output

        logger.info(f"Generated SQL for graph query: {sql_query}")

        # Execute the query
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute(sql_query)
            # Fetch column names from the cursor description
            colnames = [desc[0] for desc in cur.description]
            results = [dict(zip(colnames, row)) for row in cur.fetchall()]
            return results
    except (Exception, psycopg2.DatabaseError) as error:
        logger.error(f"Error querying graph: {error}")
        return [{"error": "Failed to query the knowledge graph."}]
    finally:
        if conn is not None:
            conn.close()
