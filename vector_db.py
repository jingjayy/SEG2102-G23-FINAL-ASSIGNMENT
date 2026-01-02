import os
import psycopg2
from psycopg2.extras import Json
from pgvector.psycopg2 import register_vector
import logging
from dotenv import load_dotenv

load_dotenv()

DB_URL = os.getenv("DATABASE_URL")

def get_db_connection(autocommit=False):
    conn = psycopg2.connect(DB_URL)
    if autocommit:
        conn.autocommit = True
    try:
        register_vector(conn)
    except Exception as e:
        logging.warning(f"register_vector failed: {e}")
    return conn

def init_db():
    """Initialize the database with pgvector extension and tables."""
    conn = get_db_connection(autocommit=True)
    cur = conn.cursor()
    
    try:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        register_vector(conn)
        
        # 1. Robust Chunks (Metadata Rich, No Vector Index)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS robust_chunks (
                id SERIAL PRIMARY KEY,
                content TEXT,
                optimized_content TEXT,
                metadata JSONB,
                embedding vector(1536),
                
                -- Expanded Metadata Columns
                act_name TEXT,
                section_number TEXT,
                legal_domain TEXT,
                part TEXT,
                keywords TEXT[]
            );
        """)
        
        # B-Tree Indexes for Metadata Filtering
        cur.execute("CREATE INDEX IF NOT EXISTS robust_chunks_act_name_idx ON robust_chunks (act_name);")
        cur.execute("CREATE INDEX IF NOT EXISTS robust_chunks_legal_domain_idx ON robust_chunks (legal_domain);")
        cur.execute("CREATE INDEX IF NOT EXISTS robust_chunks_keywords_idx ON robust_chunks USING GIN (keywords);") # GIN for array search
        
        # 2. Conventional Chunks (Standard HNSW/IVFFlat)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS conventional_chunks (
                id SERIAL PRIMARY KEY,
                content TEXT,
                metadata JSONB,
                embedding vector(1536)
            );
        """)

        # 3. Conventional Chunks (IVFFlat Only)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS conventional_ivfflat_chunks (
                id SERIAL PRIMARY KEY,
                content TEXT,
                metadata JSONB,
                embedding vector(1536)
            );
        """)
        
        logging.info("Database initialized successfully.")
        
    except Exception as e:
        logging.error(f"DB Initialization failed: {e}")
        raise
    finally:
        cur.close()
        conn.close()

def create_indexes():
    """Create indexes. Only for Conventional table now."""
    conn = get_db_connection(autocommit=True)
    cur = conn.cursor()
    
    try:
        # Conventional Table Indexes (HNSW + IVFFlat)
        logging.info("Creating indexes for conventional_chunks...")
        cur.execute("""
            CREATE INDEX IF NOT EXISTS conventional_hnsw_idx 
            ON conventional_chunks USING hnsw (embedding vector_cosine_ops);
        """)
        
        # IVFFlat Table Index
        logging.info("Creating indexes for conventional_ivfflat_chunks...")
        cur.execute("""
            CREATE INDEX IF NOT EXISTS conventional_ivfflat_idx 
            ON conventional_ivfflat_chunks USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100);
        """)
        
        logging.info("Indexes created successfully.")
        
    except Exception as e:
        logging.error(f"Index creation failed: {e}")
    finally:
        cur.close()
        conn.close()

def insert_robust_chunks(chunks):
    """Insert chunks into robust_chunks table with metadata."""
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        for chunk in chunks:
            meta = chunk.get('metadata', {})
            cur.execute("""
                INSERT INTO robust_chunks (
                    content, optimized_content, metadata, embedding,
                    act_name, section_number, legal_domain, part, keywords
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                chunk['content'],
                chunk['optimized_content'],
                Json(meta),
                chunk['embedding'],
                meta.get('act_name'),
                meta.get('section_number'),
                meta.get('legal_domain'),
                meta.get('part'),
                meta.get('keywords', [])
            ))
        conn.commit()
    except Exception as e:
        conn.rollback()
        logging.error(f"Failed to insert robust chunks: {e}")
        raise
    finally:
        cur.close()
        conn.close()

def insert_conventional_chunks(chunks):
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        for chunk in chunks:
            cur.execute("""
                INSERT INTO conventional_chunks (content, metadata, embedding)
                VALUES (%s, %s, %s)
            """, (
                chunk['content'],
                Json(chunk['metadata']),
                chunk['embedding']
            ))
            
            # Insert into IVFFlat table as well
            cur.execute("""
                INSERT INTO conventional_ivfflat_chunks (content, metadata, embedding)
                VALUES (%s, %s, %s)
            """, (
                chunk['content'],
                Json(chunk['metadata']),
                chunk['embedding']
            ))
        conn.commit()
    except Exception as e:
        conn.rollback()
        logging.error(f"Failed to insert conventional chunks: {e}")
        raise
    finally:
        cur.close()
        conn.close()

def search_vectors_conventional(query_embedding, limit=5):
    """Standard HNSW Search."""
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT content, metadata, 1 - (embedding <=> %s::vector) as similarity
            FROM conventional_chunks
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """, (query_embedding, query_embedding, limit))
        
        results = cur.fetchall()
        return [
            {"content": r[0], "metadata": r[1], "similarity": float(r[2])} 
            for r in results
        ]
    except Exception as e:
        logging.error(f"Conventional Search failed: {e}")
        return []
    finally:
        cur.close()
        conn.close()

def search_vectors_ivfflat(query_embedding, limit=5):
    """IVFFlat Search on dedicated table."""
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT content, metadata, 1 - (embedding <=> %s::vector) as similarity
            FROM conventional_ivfflat_chunks
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """, (query_embedding, query_embedding, limit))
        
        results = cur.fetchall()
        return [
            {"content": r[0], "metadata": r[1], "similarity": float(r[2])} 
            for r in results
        ]
    except Exception as e:
        logging.error(f"IVFFlat Search failed: {e}")
        return []
    finally:
        cur.close()
        conn.close()

def delete_document(document_name):
    """Delete all chunks associated with a document name."""
    conn = get_db_connection(autocommit=True)
    cur = conn.cursor()
    try:
        logging.info(f"Deleting existing chunks for document: {document_name}")
        
        # Delete from Robust
        cur.execute("DELETE FROM robust_chunks WHERE metadata->>'document_name' = %s", (document_name,))
        
        # Delete from Conventional
        cur.execute("DELETE FROM conventional_chunks WHERE metadata->>'document_name' = %s", (document_name,))
        
        # Delete from Conventional IVFFlat
        cur.execute("DELETE FROM conventional_ivfflat_chunks WHERE metadata->>'document_name' = %s", (document_name,))
        
        logging.info("Deletion complete.")
        
    except Exception as e:
        logging.error(f"Failed to delete document {document_name}: {e}")
    finally:
        cur.close()
        conn.close()

def search_partitioned(query_embedding, filters, limit=5):
    """
    Intelligent Partitioned Search:
    1. Filter by Metadata (Exact Match).
    2. Exact Vector Search (Brute Force) on filtered subset.
    """
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        query_parts = ["1=1"]
        params = []
        
        # Build Dynamic WHERE Clause
        if filters.get('legal_domain'):
            query_parts.append("legal_domain = %s")
            params.append(filters['legal_domain'])
            
        if filters.get('act_name'):
            query_parts.append("act_name = %s")
            params.append(filters['act_name'])
            
        if filters.get('part'):
            query_parts.append("part = %s")
            params.append(filters['part'])
            
        # Keyword Array Overlap
        if filters.get('keywords'):
            # Check if any of the query keywords exist in the chunk's keywords
            query_parts.append("keywords && %s::text[]")
            params.append(filters['keywords'])

        where_clause = " AND ".join(query_parts)
        
        # Correct Parameter Order for SQL Placeholders:
        # 1. %s::vector (SELECT clause) -> query_embedding
        # 2. %s (WHERE clause) -> filter params (already in params list)
        # 3. %s::vector (ORDER BY clause) -> query_embedding
        # 4. %s (LIMIT clause) -> limit
        
        final_params = [query_embedding] + params + [query_embedding, limit]
        
        sql = f"""
            SELECT content, metadata, 1 - (embedding <=> %s::vector) as similarity,
                   act_name, legal_domain
            FROM robust_chunks
            WHERE {where_clause}
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """
        
        cur.execute(sql, tuple(final_params))
        results = cur.fetchall()
        
        if not results:
             logging.info("Partitioned Search: No results found with filters. Falling back to brute force.")
             # Fallback: Search without metadata filters
             sql_fallback = """
                SELECT content, metadata, 1 - (embedding <=> %s::vector) as similarity,
                       act_name, legal_domain
                FROM robust_chunks
                ORDER BY embedding <=> %s::vector
                LIMIT %s
             """
             cur.execute(sql_fallback, (query_embedding, query_embedding, limit))
             results = cur.fetchall()
             
             return [
                {
                    "content": r[0], 
                    "metadata": r[1], 
                    "similarity": float(r[2]),
                    "detected_act": r[3],
                    "detected_domain": r[4],
                    "fallback": True
                } 
                for r in results
            ]

        return [
            {
                "content": r[0], 
                "metadata": r[1], 
                "similarity": float(r[2]),
                "detected_act": r[3],
                "detected_domain": r[4],
                "fallback": False
            } 
            for r in results
        ]
    except Exception as e:
        logging.error(f"Partitioned Search failed: {e}")
        return []
    finally:
        cur.close()
        conn.close()

def search_by_keywords(keywords, query_embedding, limit=5):
    """
    Search for chunks that contain ALL the provided keywords.
    """
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        # The @> operator checks if the array column contains the query array
        # We cast the input list to a text array
        sql = """
            SELECT content, metadata, 1 - (embedding <=> %s::vector) as similarity,
                   act_name, legal_domain
            FROM robust_chunks
            WHERE keywords @> %s::text[]
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """
        
        cur.execute(sql, (query_embedding, keywords, query_embedding, limit))
        results = cur.fetchall()
        
        return [
            {
                "content": r[0], 
                "metadata": r[1], 
                "similarity": float(r[2]),
                "detected_act": r[3],
                "detected_domain": r[4],
                "matched_keywords": keywords
            } 
            for r in results
        ]
    except Exception as e:
        logging.error(f"Keyword Search failed: {e}")
        return []
    finally:
        cur.close()
        conn.close()
