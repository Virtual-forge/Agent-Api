




# agent/tools.py
# --- REFACTORED & COMPLETED ---

from langchain_core.tools import tool
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage
import base64
import mimetypes
import os
from dotenv import load_dotenv
from pdf2image import convert_from_path
from PIL import Image
import tempfile
import io
import json
import logging
from typing import Any, Dict, List
from openai import AzureOpenAI
import re
import tiktoken
import psycopg2
from psycopg2.extras import Json
from agent.rag.graphRag_tools import extract_and_ingest_graph
# --- Setup (No changes needed here) ---
load_dotenv()

api_version = os.getenv("OPENAI_API_VERSION")
deployment = os.getenv("AZURE_DEPLOYMENT_NAME")
apikey = os.getenv("OPENAI_API_KEY")
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")


azure_endpoint = os.getenv("AZURE_OPENAI_EMBEDDING_ENDPOINT")
api_key = os.getenv("AZURE_OPENAI_API_KEY")
deployment_name = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")

db_name = os.getenv("DB_NAME")
user = os.getenv("DB_USER")
password = os.getenv("DB_PASSWORD")
host = os.getenv("DB_HOST")
port = os.getenv("DB_PORT")
logger = logging.getLogger("tools")

encoding = tiktoken.encoding_for_model("text-embedding-3-large")

client = AzureOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            api_version="2023-05-15"  # Use a suitable API version
        )

llm = AzureChatOpenAI(
    azure_deployment=deployment,
    openai_api_version=api_version,
    azure_endpoint=endpoint,
    api_key=apikey,
    temperature=0.0,
)

class Document:
    def __init__(self, page_content: str, metadata: Dict[str, Any]):
        self.page_content = page_content
        self.metadata = metadata

    def __repr__(self):
        return f"Document(page_content='{self.page_content[:50]}...', metadata={self.metadata})"

def summarize_document(full_text: str) -> str:
    """
    Summarizes the full text of a document using an LLM to prepare it for embedding.
    This is used when the original text is too long and would exceed token limits.
    """
    logging.info(f"Document text is long ({len(full_text)} chars). Generating summary for embedding...")
    
    prompt = (
        "You are a legal document summarization expert. Your task is to create a dense, factually-correct summary of the following document text. "
        "The summary MUST retain all key information: names of parties, dates, monetary amounts, specific obligations, clause titles, and core legal concepts. "
        "The goal is to create a shorter version of the text that has the same semantic meaning, suitable for a vector embedding. "
        "Do not add any commentary. Respond only with the summarized text."
        f"\n\n--- DOCUMENT TEXT TO SUMMARIZE ---\n\n{full_text}"
    )
    
    try:
        message = HumanMessage(content=prompt)
        response = llm.invoke([message])
        summary = response.content if isinstance(response.content, str) else str(response.content)
        logging.info(f"Successfully generated summary ({len(summary)} chars).")
        return summary
    except Exception as e:
        logging.error(f"Error during summarization: {e}")
        # Fallback: return a truncated version of the text if summarization fails
        return full_text[:12000] # Truncate to a safe length for embedding

def get_db_connection():
    """Establishes a connection to the PostgreSQL database."""
    # IMPORTANT: Use environment variables for connection details in production.
    conn = psycopg2.connect(
        dbname=db_name,
        user=user,
        password=password,
        host=host,
        port=port
    )
    return conn

def save_parent_to_db(embedding_response: Any, all_structured_data: Any) -> int:
    """Saves the parent document embedding to the database and returns its ID."""
    sql = """
        INSERT INTO parent_doc (embedding, metadata)
        VALUES (%s, %s) RETURNING id;
    """
    conn = None
    parent_id = None
    
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            # Extract the embedding vector and other metadata from the response
            embedding_vector = embedding_response.data[0].embedding
            
            cur.execute(sql, (embedding_vector,Json(all_structured_data)))
            parent_id = cur.fetchone()[0]
            conn.commit()
            logging.info(f"Successfully saved parent document to DB with ID: {parent_id}")
    except (Exception, psycopg2.DatabaseError) as error:
        logging.error(f"Error saving parent document: {error}")
    finally:
        if conn is not None:
            conn.close()
    return parent_id

def save_child_to_db(chunk: Any, parent_id: int):
    """Embeds a child chunk and saves it to the database."""
    sql = """
        INSERT INTO child_doc (parent_id, contents, embedding, metadata)
        VALUES (%s, %s, %s, %s);
    """
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            # Embed the child chunk's content
            child_embedding_response = create_embedding(chunk.page_content)
            if not child_embedding_response:
                logging.error(f"Failed to create embedding for chunk: {chunk.metadata}")
                return

            child_embedding_vector = child_embedding_response.data[0].embedding
            
            # Add the parent_id to the chunk's metadata before saving
            chunk.metadata["parent_id"] = parent_id
            
            cur.execute(sql, (parent_id, chunk.page_content, child_embedding_vector, Json(chunk.metadata)))
            conn.commit()
    except (Exception, psycopg2.DatabaseError) as error:
        logging.error(f"Error saving child chunk for parent_id {parent_id}: {error}")
    finally:
        if conn is not None:
            conn.close()

def simple_text_splitter(text: str, chunk_size: int) -> List[str]:
    """A basic splitter for oversized chunks."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 < chunk_size:
            current_chunk += sentence + " "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def split_markdown_semantically(full_text_with_markers: str, max_chars_per_chunk: int = 2000) -> List[Document]:
    """
    Splits markdown text with embedded page markers into semantic chunks,
    correctly assigning page numbers to metadata by processing the document sequentially.

    Args:
        full_text_with_markers (str): The markdown content including page break markers.
        max_chars_per_chunk (int): The character limit for a chunk.

    Returns:
        A list of Document objects, each with correct page number metadata.
    """
    final_chunks = []
    page_break_marker_regex = r"\|\|--PAGE_BREAK:--\|\|(\d+)"
    
    # Split the entire text by the page break markers, keeping the page numbers.
    # This results in a list like: ['text from page 1', '1', 'text from page 2', '2', ...]
    parts = re.split(page_break_marker_regex, full_text_with_markers)
    
    current_page = 1
    
    # Process the text parts. The list contains text and page numbers interleaved.
    for i in range(0, len(parts), 2):
        text_block = parts[i]
        
        # The next item in the list should be the page number that this text block belongs to.
        if (i + 1) < len(parts):
            current_page = int(parts[i+1])
        
        if not text_block.strip():
            continue

        # Now, split this page's text block by semantic headings
        sections = re.split(r'(?=###\s)', text_block)
        for section_text in sections:
            clean_content = section_text.strip()
            if not clean_content:
                continue

            # Determine the header for the metadata
            is_article = clean_content.startswith("###")
            header = clean_content.split('\n', 1)[0].strip() if is_article else "Preamble"
            
            metadata = {"header": header, "source_page": current_page}

            # Handle oversized chunks
            if len(clean_content) <= max_chars_per_chunk:
                final_chunks.append(Document(page_content=clean_content, metadata=metadata))
            else:
                sub_chunks = simple_text_splitter(clean_content, max_chars_per_chunk)
                for j, sub_chunk_text in enumerate(sub_chunks):
                    sub_chunk_metadata = metadata.copy()
                    sub_chunk_metadata["part"] = f"{j + 1}/{len(sub_chunks)}"
                    final_chunks.append(Document(page_content=sub_chunk_text, metadata=sub_chunk_metadata))

    return final_chunks

def create_embedding(text_content: str):
    """
    Generates an embedding for a given text using Azure OpenAI.

    Args:
        text_content (str): The text to be embedded.

    Returns:
        openai.types.CreateEmbeddingResponse: The full response object from the
        Azure OpenAI API, containing the embedding and other metadata, or None
        if an error occurs.
    """
    try:
        

        logger.info(f"Generating embedding with deployment '{deployment_name}'...")
        # Call the embeddings API
        response = client.embeddings.create(
            model=deployment_name,
            input=text_content
        )
        
        logger.info("Embedding generation complete.")
        
        # Return the entire response object from the API call
        return response

    except Exception as e:
        logger.info(f"An error occurred: {e}")
        return None

def ask_vlm(file_path: str, prompt: str) -> str:
    """ Send a prompt and file path to LLM/VLM
    """
    mime_type, _ = mimetypes.guess_type(file_path)
    with open(file_path, "rb") as f:
        b64_data = base64.b64encode(f.read()).decode("utf-8")

    content = [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64_data}"}},
    ]

    message = HumanMessage(content=content)
    response = llm.invoke([message])
    return response.content

def split_llm_response(response_content: str):
    separator = "---JSON_AND_TEXT_SEPARATOR---"

    try:
        # Split the response using the separator
        parts = response_content.split(separator)

        if len(parts) != 2:
            raise ValueError("Unexpected response format: separator not found or multiple separators")

        json_part = parts[0].strip()
        text_part = parts[1].strip()

        # If the JSON is wrapped in ```json ... ```, strip it
        if json_part.startswith("```json"):
            json_part = json_part.strip("`").replace("json\n", "", 1).strip()

        # Now parse the JSON part
        structured_data = json.loads(json_part)

        return {
            "structured_data": structured_data,
            "raw_text": text_part
        }

    except (ValueError, json.JSONDecodeError) as e:
        logger.error(f"Failed to split/parse LLM response: {e}")
        return {
            "error": "Failed to split or parse structured data from the document.",
            "raw_output": response_content
        }

def process_document(file_path: str) -> List[str]:
    """
    Processes a document (PDF or image) and returns a list of paths to temporary image files, one for each page.
    This function is the single source of truth for handling file inputs.
    The caller is responsible for cleaning up the temporary files.
    """
    page_image_paths = []
    
    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type == 'application/pdf':
        logger.info(f"Processing PDF: {file_path}")
        images = convert_from_path(file_path, dpi=300)
        if not images:
            raise ValueError("PDF is empty or could not be converted.")

        for i, image in enumerate(images):
            # Note: The temp files are NOT deleted on close, so they can be used by other functions.
            tmp_img = tempfile.NamedTemporaryFile(delete=False, suffix=f"_page_{i+1}.png")
            image.save(tmp_img.name, "PNG")
            page_image_paths.append(tmp_img.name)
        logger.info(f"Converted PDF to {len(page_image_paths)} page images.")
    elif mime_type and mime_type.startswith('image/'):
        logger.info(f"Processing single image: {file_path}")
        # For images, we don't create a temp file, we just use the original path.
        page_image_paths.append(file_path)
    else:
        raise ValueError(f"Unsupported file type: {mime_type}")

    return page_image_paths

def _cleanup_temp_files(file_paths: List[str]):
    """Helper function to delete a list of temporary files."""
    for path in file_paths:
        # Ensure we don't try to delete the original non-temp file
        if os.path.basename(path).startswith(os.path.basename(tempfile.gettempdir())):
            try:
                os.remove(path)
                logger.info(f"Cleaned up temp file: {path}")
            except OSError as e:
                logger.error(f"Error cleaning up file {path}: {e}")

def merge_small_chunks(chunks: List[Any], min_chunk_size_tokens: int = 200) -> List[Any]:
    """
    Merges chunks that are smaller than a specified token count with the subsequent chunk.
    """
    if not chunks:
        return []

    merged_chunks = []
    i = 0
    while i < len(chunks):
        current_chunk = chunks[i]
        num_tokens = len(encoding.encode(current_chunk.page_content))
        if num_tokens < min_chunk_size_tokens and (i + 1) < len(chunks):
            next_chunk = chunks[i + 1]
            merged_content = current_chunk.page_content + "\n\n" + next_chunk.page_content
            new_chunk = Document(page_content=merged_content, metadata=current_chunk.metadata)
            merged_chunks.append(new_chunk)
            i += 2
        else:
            merged_chunks.append(current_chunk)
            i += 1
    logging.info(f"Chunk merging complete. Original: {len(chunks)} chunks, Merged: {len(merged_chunks)} chunks.")
    return merged_chunks

def _extract_from_document(file_path: str, extraction_prompt: str, batch_size: int = 3) -> Dict[str, Any]:
    """
    Extracts, chunks, embeds, and saves document content, including a step to merge small chunks.
    """
    temp_files = []
    all_raw_text_parts = []
    all_structured_data = []

    try:
        page_image_paths = process_document(file_path)
        if file_path.lower().endswith('.pdf'):
            temp_files.extend(page_image_paths)

        num_pages = len(page_image_paths)
        num_batches = (num_pages + batch_size - 1) // batch_size
        logging.info(f"Document has {num_pages} pages. Processing in {num_batches} batches of size {batch_size}.")

        for i in range(0, num_pages, batch_size):
            batch_num = (i // batch_size) + 1
            batch_paths = page_image_paths[i:i + batch_size]
            logging.info(f"Processing Batch {batch_num}/{num_batches} ({len(batch_paths)} pages)...")
            start_page_num = i + 1
            batch_prompt = (
                f"{extraction_prompt}\n\n"
                "---INSTRUCTIONS---\n"
                "1. First, provide the structured JSON data.\n"
                "2. Then, include the separator '---JSON_AND_TEXT_SEPARATOR---'.\n"
                "3. Finally, provide the full, verbatim markdown text from the images.\n"
                "4. You MUST insert a page break marker of the form '||--PAGE_BREAK:--||X' (where X is the page number) at the end of the text for each page.\n"
                "5. IMPORTANT: When you encounter tables, you must extract every single row completely. Do not truncate tables or use '...'. Your output will be programmatically checked for completeness, and truncated responses will be rejected.\n"
                f"The first page in this batch is page {start_page_num}."
            )
            content = [{"type": "text", "text": batch_prompt}]
            for path in batch_paths:
                mime_type, _ = mimetypes.guess_type(path)
                with open(path, "rb") as f:
                    b64_data = base64.b64encode(f.read()).decode("utf-8")
                content.append({"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64_data}"}})
            
            message = HumanMessage(content=content)
            response = llm.invoke([message])
            
            try:
                response_content = response.content if isinstance(response.content, str) else str(response.content)
                parsed_batch = split_llm_response(response_content)
                all_structured_data.append(parsed_batch["structured_data"])
                all_raw_text_parts.append(parsed_batch["raw_text"])
                logging.info(f"Successfully processed and parsed Batch {batch_num}.")
            except (json.JSONDecodeError, KeyError) as e:
                logging.error(f"Failed to parse response for Batch {batch_num}. Error: {e}", exc_info=True)

        logging.info("All batches processed. Starting post-processing and chunking...")
        full_text_with_markers = "\n\n".join(all_raw_text_parts)
        
        initial_chunks = split_markdown_semantically(full_text_with_markers, max_chars_per_chunk=2000)
        logging.info(f"Document initially split into {len(initial_chunks)} semantic chunks.")

        document_chunks = merge_small_chunks(initial_chunks, min_chunk_size_tokens=25)

        full_markdown_text = "".join(all_raw_text_parts)

        num_tokens = len(encoding.encode(full_markdown_text))
        TOKEN_LIMIT = 8000
        
        text_to_embed = full_markdown_text
        if num_tokens > TOKEN_LIMIT:
            text_to_embed = summarize_document(full_markdown_text)

        parent_embedding_response = create_embedding(text_to_embed)
        
        parent_id = None
        if parent_embedding_response:
            parent_id = save_parent_to_db(parent_embedding_response, all_structured_data)
        else:
            raise Exception("Parent embedding creation failed.")

        if parent_id:
            for chunk in document_chunks:
                save_child_to_db(chunk, parent_id)
            logging.info("Successfully processed and saved all child chunks.")
            extract_and_ingest_graph(document_chunks)

        return {
            "status": "Success",
            "parent_id": parent_id,
            "total_chunks_saved": len(document_chunks)
        }
    except Exception as e:
        logging.error(f"An unexpected error occurred during document extraction: {e}", exc_info=True)
        return {
            "error": "An unexpected error occurred during the extraction process.",
            "details": str(e)
        }
    finally:
        _cleanup_temp_files(temp_files)


@tool
def analyze_document_type(file_path: str) -> str:
    """
    Analyzes a document (PDF or image) to determine its content type.
    This should ALWAYS be the first tool used for any document processing task.
    It processes the first page to classify the entire document.
    Returns a single string representing the document type (e.g., 'facture', 'contrat_de_bail').
    """
    logger.info(f"Starting document analysis for: {file_path}")
    temp_files = []
    try:
        page_image_paths = process_document(file_path)
        if not page_image_paths:
            return "unknown_document_type"
        
        # If the original was a PDF, the generated images are temporary.
        if file_path.lower().endswith('.pdf'):
            temp_files.extend(page_image_paths)

        prompt = (
            "Classify the type of this document. The possible types are:\n"
            "facture\n"
            "contrat_de_bail\n"
            "contrat_de_marche\n"
            "bordereau_de_versement\n"
            "contrat_acquésition\n"
            "contrat_de_maintenance\n"
            "Respond with only one of these types, and nothing else."
        )
        
        # We only need the first page to classify.
        doc_type = ask_vlm(page_image_paths[0], prompt)
        logger.info(f"Classified document '{file_path}' as type: {doc_type}")
        return doc_type.strip()

    except Exception as e:
        logger.error(f"Error in analyze_document_type: {e}", exc_info=True)
        return "error_during_classification"
    finally:
        _cleanup_temp_files(temp_files)
@tool
def extract_facture(file_path: str) -> str:
    """
    Extracts structured data AND full raw text from a document PREVIOUSLY IDENTIFIED as a 'facture'.
    Processes all pages and returns response status , parent_id and number of chunks.
    Do not call this tool before `analyze_document_type` has been used.
    """
    prompt = """Tu es un assistant d'extraction d'informations. Ta tâche est double et tu dois la réaliser en analysant l'ensemble des images fournies qui représentent les pages d'un document unique.

1.  **Synthétiser les informations de TOUTES les pages** pour créer un **unique objet JSON consolidé**. Ne crée pas un JSON par page.
2.  Extraire l'intégralité du texte brut de toutes les pages et le concatener en Markdown.

Réponds en respectant EXACTEMENT le format suivant, sans aucune explication :
[L'OBJET JSON UNIQUE ET CONSOLIDÉ ICI]
---JSON_AND_TEXT_SEPARATOR---
[LE TEXTE BRUTE COMPLET EN MARKDOWN DE TOUTES LES PAGES ICI]

Voici la structure JSON à utiliser :
{
  "facture": {
    "identification": { "numero_facture": "string", "date": "string", "reference_commande": "string" },
    "fournisseur": { "nom": "string", "adresse": "string", "identifiants_fiscaux": { "ICE": "string", "IF": "string", "TP": "string" } },
    "client": { "nom": "string", "adresse": "string", "identifiants_fiscaux": { "ICE": "string", "IF": "string", "TP": "string" } },
    "ligne_facture": [ { "description": "string", "quantite": float, "prix_unitaire": float, "montant_HT": float, "TVA": float, "montant_TTC": float } ],
    "totaux": { "total_HT": float, "total_TVA": float, "total_TTC": float },
    "paiement": { "mode": "string", "delai": "string" }
  }
}
Commence ta réponse directement avec l'accolade `{` du JSON. Si une information est absente, mets `null`."""
    return _extract_from_document(file_path, prompt)
@tool
def extract_contrat_de_bail(file_path: str) -> str:
    """
    Extracts structured data AND full raw text from a document PREVIOUSLY IDENTIFIED as a 'contrat_de_bail'.
    returns response status , parent_id and number of chunks.
    Do not call this tool before `analyze_document_type` has been used.
    """
    prompt = """Tu es un assistant d'extraction d'informations. Ta tâche est double et tu dois la réaliser en analysant l'ensemble des images fournies qui représentent les pages d'un document unique.

1.  **Synthétiser les informations de TOUTES les pages** pour créer un **unique objet JSON consolidé**. Ne crée pas un JSON par page.
2.  Extraire l'intégralité du texte brut de toutes les pages et le concatener en Markdown.

Réponds en respectant EXACTEMENT le format suivant, sans aucune explication :
[L'OBJET JSON UNIQUE ET CONSOLIDÉ ICI]
---JSON_AND_TEXT_SEPARATOR---
[LE TEXTE BRUTE COMPLET EN MARKDOWN DE TOUTES LES PAGES ICI]

Voici la structure JSON à utiliser :
{
  "contrat_bail": {
    "bailleur": { "nom": "string", "identite": { "type": "string", "numero": "string" } },
    "locataire": { "nom": "string", "identite": { "type": "string", "numero": "string" } },
    "bien_loue": { "adresse": "string", "type_bien": "string", "usage": "string" },
    "conditions": { "date_debut": "string", "date_fin": "string", "loyer_mensuel": float, "charges": "string", "depot_garantie": float, "mode_paiement": "string" },
    "clauses": { "resiliation": "string", "renouvellement": "string", "entretien": "string", "assurance_obligatoire": "string" },
    "dates": { "date_signature": "string" }
  }
}
Commence ta réponse directement avec l'accolade `{` du JSON. Si une information est absente, mets `null`."""
    return _extract_from_document(file_path, prompt)
@tool
def extract_contrat_de_marche(file_path: str) -> str:
    """
    Extracts structured data AND full raw text from a document PREVIOUSLY IDENTIFIED as a 'contrat_de_marche'.
    returns response status , parent_id and number of chunks.
    Do not call this tool before `analyze_document_type` has been used.
    """
    prompt = """Tu es un assistant d'extraction d'informations. Ta tâche est double et tu dois la réaliser en analysant l'ensemble des images fournies qui représentent les pages d'un document unique.

1.  **Synthétiser les informations de TOUTES les pages** pour créer un **unique objet JSON consolidé**. Ne crée pas un JSON par page.
2.  Extraire l'intégralité du texte brut de toutes les pages et le concatener en Markdown.

Réponds en respectant EXACTEMENT le format suivant, sans aucune explication :
[L'OBJET JSON UNIQUE ET CONSOLIDÉ ICI]
---JSON_AND_TEXT_SEPARATOR---
[LE TEXTE BRUTE COMPLET EN MARKDOWN DE TOUTES LES PAGES ICI]

Voici la structure JSON à utiliser :
{
  "contrat_marche": {
    "identification": { "nom_projet": "string", "reference_contrat": "string", "objet": "string" },
    "parties": { "maitre_ouvrage": { "nom": "string", "representant": "string" }, "titulaire": { "nom": "string", "representant": "string" } },
    "conditions": { "montant": float, "devise": "string", "duree_execution": "string", "mode_paiement": "string" },
    "clauses": { "penalites_retard": "string", "resiliation": "string", "garantie_bancaire": "string", "assurance": "string" },
    "dates": { "date_signature": "string", "date_debut_execution": "string", "date_fin_previsionnelle": "string" }
  }
}
Commence ta réponse directement avec l'accolade `{` du JSON. Si une information est absente, mets `null`."""
    return _extract_from_document(file_path, prompt)
@tool
def extract_bordereau_de_versement(file_path: str) -> str:
    """
    Extracts structured data AND full raw text from a document PREVIOUSLY IDENTIFIED as a 'bordereau_de_versement'.
    returns response status , parent_id and number of chunks.
    Do not call this tool before `analyze_document_type` has been used.
    """
    prompt = """Tu es un assistant d'extraction d'informations. Ta tâche est double et tu dois la réaliser en analysant l'ensemble des images fournies qui représentent les pages d'un document unique.

1.  **Synthétiser les informations de TOUTES les pages** pour créer un **unique objet JSON consolidé**. Ne crée pas un JSON par page.
2.  Extraire l'intégralité du texte brut de toutes les pages et le concatener en Markdown.

Réponds en respectant EXACTEMENT le format suivant, sans aucune explication :
[L'OBJET JSON UNIQUE ET CONSOLIDÉ ICI]
---JSON_AND_TEXT_SEPARATOR---
[LE TEXTE BRUTE COMPLET EN MARKDOWN DE TOUTES LES PAGES ICI]

Voici la structure JSON à utiliser :
{
  "bordereau": {
    "numero": "string",
    "date": "string",
    "agence": "string",
    "deposant": { "nom": "string", "piece_identite": { "type": "string", "numero": "string", "delivree_par": "string" } },
    "beneficiaire": { "intitule_compte": "string", "numero_compte": "string" },
    "operation": { "libelle": "string", "montant_operation": float, "montant_total": float }
  }
}
Commence ta réponse directement avec l'accolade `{` du JSON. Si une information est absente, mets `null`."""
    return _extract_from_document(file_path, prompt)
@tool
def extract_contrat_acquisition(file_path: str) -> str:
    """
    Extracts structured data AND full raw text from a document PREVIOUSLY IDENTIFIED as a 'contrat_acquésition'.
    returns response status , parent_id and number of chunks.
    Do not call this tool before `analyze_document_type` has been used.
    """
    prompt = """Tu es un assistant d'extraction d'informations. Ta tâche est double et tu dois la réaliser en analysant l'ensemble des images fournies qui représentent les pages d'un document unique.

1.  **Synthétiser les informations de TOUTES les pages** pour créer un **unique objet JSON consolidé**. Ne crée pas un JSON par page.
2.  Extraire l'intégralité du texte brut de toutes les pages et le concatener en Markdown.

Réponds en respectant EXACTEMENT le format suivant, sans aucune explication :
[L'OBJET JSON UNIQUE ET CONSOLIDÉ ICI]
---JSON_AND_TEXT_SEPARATOR---
[LE TEXTE BRUTE COMPLET EN MARKDOWN DE TOUTES LES PAGES ICI]

Voici la structure JSON à utiliser :
{
  "contrat_acquisition": {
    "identification": { "reference_contrat": "string", "objet": "string", "date_signature": "string" },
    "acheteur": { "nom": "string", "representant": "string" },
    "fournisseur": { "nom": "string", "representant": "string" },
    "details_commande": [ { "description": "string", "quantite": float, "prix_unitaire": float, "montant_total": float } ],
    "conditions": { "montant_total": float, "devise": "string", "delai_livraison": "string", "mode_paiement": "string", "garanties": "string", "conditions_resiliation": "string" }
  }
}
Commence ta réponse directement avec l'accolade `{` du JSON. Si une information est absente, mets `null`."""
    return _extract_from_document(file_path, prompt)
@tool
def extract_contrat_de_maintenance(file_path: str) -> str:
    """
    Extracts structured data AND full raw text from a document PREVIOUSLY IDENTIFIED as a 'contrat_de_maintenance'.
    returns response status , parent_id and number of chunks.
    Do not call this tool before `analyze_document_type` has been used.
    """
    prompt = """Tu es un assistant d'extraction d'informations. Ta tâche est double et tu dois la réaliser en analysant l'ensemble des images fournies qui représentent les pages d'un document unique.

1.  **Synthétiser les informations de TOUTES les pages** pour créer un **unique objet JSON consolidé**. Ne crée pas un JSON par page.
2.  Extraire l'intégralité du texte brut de toutes les pages et le concatener en Markdown.

Réponds en respectant EXACTEMENT le format suivant, sans aucune explication :
[L'OBJET JSON UNIQUE ET CONSOLIDÉ ICI]
---JSON_AND_TEXT_SEPARATOR---
[LE TEXTE BRUTE COMPLET EN MARKDOWN DE TOUTES LES PAGES ICI]

Voici la structure JSON à utiliser :
{
  "contrat_maintenance": {
    "identification": { "reference_contrat": "string" (example : N° 26/2019), "objet": "string" ( example : Mise en place d'unreseau local et d'une salle informatique pour la B.U Abattoirs de Casablanca ), "date_signature": "string" },
    "parties": { "client": { "nom": "string", "representant": "string" }, "prestataire": { "nom": "string", "representant": "string" } },
    "maintenance": { "type": "string", "frequence": "string", "equipements_couverts": "string", "lieu_intervention": "string" },
    "conditions": { "duree_contrat": "string", "montant_total": float, "mode_paiement": "string", "SLA": "string", "conditions_resiliation": "string", "assurance_responsabilite": "string" }
  }
}
Commence ta réponse directement avec l'accolade `{` du JSON. Si une information est absente, mets `null`."""
    return _extract_from_document(file_path, prompt)
