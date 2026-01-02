import os
import psycopg2
from psycopg2.extras import Json
from vector_db import get_db_connection, init_db

def reproduce_keyword_issue():
    print("Reproducing Keyword Mismatch Issue...")
    
    conn = get_db_connection()
    cur = conn.cursor()
    
    try:
        # 1. Setup: Ensure DB is clean and has one record
        # Note: We won't run init_db() to avoid wiping existing data if we can avoid it, 
        # but for a clean repro we might want to insert a specific test record.
        # Let's just insert a test record.
        
        # Create a 1536-dim zero vector
        dummy_vector = "[" + ",".join(["0.0"] * 1536) + "]"
        
        test_keywords = ["Peninsular Malaysia", "Employment Act"]
        
        print(f"Inserting chunk with keywords: {test_keywords}")
        cur.execute("""
            INSERT INTO robust_chunks (
                content, optimized_content, metadata, embedding, keywords
            ) VALUES (
                'keyword_repro_test', 
                'keyword_repro_test', 
                '{"test": true}', 
                %s,
                %s
            ) RETURNING id;
        """, (dummy_vector, test_keywords))
        chunk_id = cur.fetchone()[0]
        conn.commit()
        
        # 2. Test Cases
        test_cases = [
            (["Peninsular Malaysia"], True, "Exact match"),
            (["Malaysia"], True, "Partial word match (Case Sensitive)"), 
            (["malaysia"], True, "Partial word match (Case Insensitive)"),
            (["PENINSULAR"], True, "Partial word match (Upper Case)"),
            (["Employment"], True, "Other keyword match")
        ]
        
        print("\nRunning Search Tests...")
        for query_keywords, expected, desc in test_cases:
            # Current logic in vector_db.py uses: keywords && %s::text[]
            # This checks for EXACT overlap of array elements.
            
            cur.execute("""
                SELECT count(*) 
                FROM robust_chunks 
                WHERE id = %s AND keywords && %s::text[]
            """, (chunk_id, query_keywords))
            
            count = cur.fetchone()[0]
            found = count > 0
            
            status = "PASS" if found == expected else "FAIL"
            # For this repro, we EXPECT failure on partial/case-insensitive matches with current logic
            # So if found is False but expected is True, that confirms the issue.
            
            print(f"[{status}] {desc}: Query={query_keywords} -> Found={found} (Expected={expected})")
            
        # Cleanup
        cur.execute("DELETE FROM robust_chunks WHERE id = %s", (chunk_id,))
        conn.commit()
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        cur.close()
        conn.close()

if __name__ == "__main__":
    reproduce_keyword_issue()
