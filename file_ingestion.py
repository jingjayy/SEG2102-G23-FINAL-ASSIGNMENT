import os
import re
import json
import logging
import asyncio
import google.generativeai as genai
import time
from typing import List, Dict, Any, Tuple
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env")

genai.configure(api_key=GOOGLE_API_KEY)

# Models (User updated these)
LLM_MODEL = "gemini-2.5-flash"   
ENRICH_MODEL = "gemini-2.5-flash"    
OPTIMIZE_MODEL = "gemini-2.5-flash-lite" 
EMBEDDING_MODEL = "models/gemini-embedding-001"

MAX_CONCURRENCY = 10
LLM_TIMEOUT = 1200

# ---------------------------------------------------------------------------
# UTILITIES
# ---------------------------------------------------------------------------

def get_embedding(text: str) -> List[float]:
    try:
        time.sleep(1) # Rate limit safety
        result = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=text,
            task_type="retrieval_document",
            title="Ingestion"
        )
        embedding = result['embedding']
        return embedding[:1536]
    except Exception as e:
        logging.error(f"Embedding generation failed: {e}")
        return []

def normalize_fingerprint(text: str) -> str:
    if not text: return ""
    return re.sub(r'[^a-z0-9]', '', text.lower())

async def retry_with_backoff(func, *args, retries=5, **kwargs):
    for attempt in range(retries):
        try:
            await asyncio.sleep(2) # Reduced delay since concurrency is higher but model is faster
            return await func(*args, **kwargs)
        except Exception as e:
            if "429" in str(e):
                wait_time = (2 ** attempt) * 5
                logging.warning(f"Hit 429 Rate Limit. Waiting {wait_time}s...")
                await asyncio.sleep(wait_time)
            elif attempt == retries - 1:
                logging.error(f"API Call Failed after {retries} attempts: {e}")
                raise
            else:
                wait_time = 2 ** attempt 
                await asyncio.sleep(wait_time)

# ---------------------------------------------------------------------------
# ROBUST LOGIC (Optimized)
# ---------------------------------------------------------------------------

class DocumentFingerprint:
    """Pre-computed DNA for fast slicing."""
    def __init__(self, full_text: str):
        self.full_text = full_text
        self.dna = ""
        self.map = []
        
        # Build DNA once
        for i, char in enumerate(full_text):
            if char.isalnum():
                self.dna += char.lower()
                self.map.append(i)
                
    def find(self, signature: str, start_raw_idx: int) -> int:
        if not signature: return -1
        
        # Convert start_raw_idx to dna_idx
        # We need to find the first dna index where map[dna_idx] >= start_raw_idx
        start_dna_idx = 0
        # Binary search could be faster but linear scan from last position is okay if sequential
        # For simplicity, we just scan or use bisect if needed. 
        # Since we usually move forward, we can optimize.
        
        # Simple approximation: scan map to find start point
        # This is still slow if we do it every time from 0. 
        # But we can pass a hint or just search in the substring of DNA.
        
        # Optimization: We only care about DNA after the current position.
        # Let's find the DNA offset corresponding to start_raw_idx
        # This is fast enough if we assume sequential access, but let's be safe.
        
        # Find the first index in self.map that is >= start_raw_idx
        import bisect
        start_dna_idx = bisect.bisect_left(self.map, start_raw_idx)
        
        target_dna = normalize_fingerprint(signature)
        if not target_dna: return -1
        
        # Search in the DNA slice
        found_relative = self.dna.find(target_dna, start_dna_idx)
        
        # Fallback for short signatures
        if found_relative == -1 and len(target_dna) > 20:
             found_relative = self.dna.find(target_dna[:20], start_dna_idx)
             
        if found_relative != -1:
            return self.map[found_relative]
            
        return -1

async def get_document_structure(full_text: str) -> List[Dict[str, str]]:
    logging.info("Sending full document to LLM for STRUCTURAL ANALYSIS...")
    
    system_prompt = """
    You are a legal document analyzer. Break this document into **ATOMIC LEGAL SECTIONS**.
    Rules:
    1. Granularity: Identify EVERY numbered section (e.g., "1.", "60A.").
    2. The Anchor: Provide `start_signature` (first 30-50 chars).
    3. Tables: Identify "LIST OF AMENDMENTS" or "SCHEDULE".
    4. Naming: Include Section Number AND Title.
    
    Return JSON: { "segments": [ {"title": "...", "start_signature": "...", "type": "text"} ] }
    """

    model = genai.GenerativeModel(LLM_MODEL)
    
    async def _call_llm():
        response = await asyncio.to_thread(
            model.generate_content,
            f"{system_prompt}\n\nExtract structure (return JSON):\n\n{full_text}",
            generation_config={"response_mime_type": "application/json"}
        )
        return response

    try:
        response = await retry_with_backoff(_call_llm)
        plan = json.loads(response.text)
        segments = plan.get("segments", [])
        logging.info(f"LLM identified {len(segments)} atomic segments.")
        return segments
    except Exception as e:
        logging.error(f"Structural Analysis Failed: {e}")
        return []

def slice_text_by_plan(full_text: str, segments: List[Dict[str, str]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, str]]]:
    logging.info("Building Document Fingerprint...")
    doc_fingerprint = DocumentFingerprint(full_text)
    logging.info("Fingerprint built. Slicing...")
    
    chunks = []
    missing_segments = []
    current_idx = 0
    
    for i, segment in enumerate(segments):
        signature = segment["start_signature"]
        start_idx = doc_fingerprint.find(signature, current_idx)
        
        if start_idx == -1:
            logging.warning(f"Segment NOT FOUND in text: {segment.get('title', 'Unknown')} | Signature: '{signature}'")
            missing_segments.append(segment)
            continue
            
        if start_idx < current_idx: start_idx = current_idx
        
        end_idx = len(full_text)
        if i < len(segments) - 1:
            next_sig = segments[i+1]["start_signature"]
            next_idx = doc_fingerprint.find(next_sig, start_idx + 1)
            if next_idx != -1:
                end_idx = next_idx
        
        chunk_content = full_text[start_idx:end_idx]
        if len(chunk_content.strip()) > 20:
            chunks.append({
                "content": chunk_content,
                "metadata": {
                    "section_title": segment.get("title", ""),
                    "type": segment["type"]
                }
            })
        current_idx = end_idx
        
    logging.info(f"Slicing complete. Created {len(chunks)} chunks. Missed {len(missing_segments)} segments.")
    return chunks, missing_segments

async def enrich_chunk(chunk_data: Dict[str, Any], semaphore: asyncio.Semaphore, idx: int, total: int, document_name: str) -> Dict[str, Any]:
    content = chunk_data["content"]
    meta = chunk_data["metadata"]
    
    async with semaphore:
        if idx % 10 == 0:
            logging.info(f"Enriching chunk {idx}/{total}...")
            
        # Optimize & Extract Metadata
        prompt = f"""
        Analyze this legal section for Vector Search Optimization and Metadata Extraction.
        
        **Context:**
        Document Name: "{document_name}" (This is likely the Act Name)
        
        **Input Text:**
        {content}
        
        **Task:**
        1. **Optimize**: Rewrite the text to be search-friendly (synonyms, context, key entities).
        2. **Extract Metadata**:
           - `act_name`: Name of the Act. Use the Document Name as a strong hint. DO NOT add "(Assumed)" or "(Implied)". If uncertain, use the Document Name.
           - `section_number`: The section number.
           - `legal_domain`: The broad domain.
           - `part`: The Part/Chapter.
           - `keywords`: List of 5-10 semantic keywords. 
             **CRITICAL KEYWORD RULES:**
             - Include the **Section Title** verbatim.
             - Include the **Section Title** WITHOUT the section number (e.g., if "13. Termination", add "Termination").
             - Include a "loose" version of the title (e.g., if "Termination of contract without notice", add "Termination of contract").
             - Include the **Section Number**.
        
        **Output JSON:**
        {{
            "optimized_content": "...",
            "metadata": {{
                "act_name": "...",
                "section_number": "...",
                "legal_domain": "...",
                "part": "...",
                "keywords": ["..."]
            }}
        }}
        """
        
        model = genai.GenerativeModel(OPTIMIZE_MODEL)
        try:
            response = await retry_with_backoff(
                asyncio.to_thread,
                model.generate_content, 
                prompt,
                generation_config={"response_mime_type": "application/json"}
            )
            
            try:
                result = json.loads(response.text)
                if isinstance(result, list): result = result[0] if result else {}
            except:
                result = {}
            
            optimized_text = result.get("optimized_content", content)
            extracted_meta = result.get("metadata", {})
            
            # Normalize keywords to lowercase for loose matching
            if "keywords" in extracted_meta and isinstance(extracted_meta["keywords"], list):
                extracted_meta["keywords"] = [k.lower().strip() for k in extracted_meta["keywords"]]
                
            meta.update(extracted_meta)
            
        except Exception as e:
            logging.error(f"Enrichment failed for chunk {idx}: {e}")
            optimized_text = content
            meta.update({"act_name": "Unknown", "legal_domain": "General"})

        final_optimized = f"Section: {meta.get('section_title', '')}\n\n{optimized_text}"
        
        embedding = await asyncio.to_thread(get_embedding, final_optimized)
        
        return {
            "content": content,
            "optimized_content": final_optimized,
            "metadata": meta,
            "embedding": embedding
        }

async def process_robust(file_path: str) -> List[Dict[str, Any]]:
    try:
        logging.info(f"Robust Ingestion: {file_path}")
        if file_path.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
            pages = loader.load()
            full_text = "".join(page.page_content for page in pages)
        elif file_path.endswith('.docx'):
            loader = Docx2txtLoader(file_path)
            docs = loader.load()
            full_text = "\n".join(doc.page_content for doc in docs)
        else:
            raise ValueError("Unsupported file type")

        segments = await get_document_structure(full_text)
        if not segments: return []

        raw_slices, _ = slice_text_by_plan(full_text, segments)
        
        # Add document name to metadata
        doc_name = os.path.basename(file_path)
        for chunk in raw_slices:
            chunk['metadata']['document_name'] = doc_name

        semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
        tasks = [enrich_chunk(chunk, semaphore, i, len(raw_slices), doc_name) for i, chunk in enumerate(raw_slices)]
        results = await asyncio.gather(*tasks)
        
        return [r for r in results if r['embedding']]

    except Exception as e:
        logging.error(f"Robust Processing Error: {e}")
        return []

# ---------------------------------------------------------------------------
# CONVENTIONAL LOGIC
# ---------------------------------------------------------------------------

async def process_conventional(file_path: str) -> List[Dict[str, Any]]:
    try:
        logging.info(f"Conventional Ingestion: {file_path}")
        if file_path.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        elif file_path.endswith('.docx'):
            loader = Docx2txtLoader(file_path)
        else:
            raise ValueError("Unsupported file type")
            
        docs = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(docs)
        
        semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

        async def _process_split(split):
            async with semaphore:
                embedding = await asyncio.to_thread(get_embedding, split.page_content)
                if embedding:
                    return {
                        "content": split.page_content,
                        "metadata": {**split.metadata, "document_name": os.path.basename(file_path)},
                        "embedding": embedding
                    }
                return None

        tasks = [_process_split(split) for split in splits]
        results = await asyncio.gather(*tasks)
        
        return [r for r in results if r is not None]

    except Exception as e:
        logging.error(f"Conventional Processing Error: {e}")
        return []
