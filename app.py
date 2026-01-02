import os
import asyncio
import json
import logging
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import google.generativeai as genai
from file_ingestion import process_robust, process_conventional, get_embedding
from vector_db import init_db, create_indexes, insert_robust_chunks, insert_conventional_chunks, search_vectors_conventional, search_partitioned, search_vectors_ivfflat, delete_document

# Configure Logging
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize DB on startup
try:
    init_db()
    create_indexes()
except Exception as e:
    logging.error(f"Startup DB Error: {e}")

# Combinatorial Search Logic
from itertools import combinations
from vector_db import search_by_keywords
import concurrent.futures
from collections import defaultdict

STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "if", "because", "as", "what",
    "when", "where", "how", "which", "who", "whom", "this", "that", "these",
    "those", "am", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having", "do", "does", "did", "doing",
    "at", "by", "for", "from", "in", "into", "of", "off", "on", "onto",
    "out", "over", "up", "with", "to", "about", "against", "between",
    "during", "before", "after", "above", "below", "under", "again",
    "further", "then", "once", "here", "there", "all", "any", "both",
    "each", "few", "more", "most", "other", "some", "such", "no", "nor",
    "not", "only", "own", "same", "so", "than", "too", "very", "can",
    "will", "just", "should", "now",
    # Added common legal terms to stop words to prevent generic matches
    "employee", "employer", "contract", "service"
}

def generate_combinations(query):
    """
    Generates all keyword combinations from the query, sorted by length (descending).
    Filters out common stop words to reduce combinatorial explosion.
    """
    # Simple tokenization: remove non-alphanumeric (except spaces) and split
    import re
    clean_query = re.sub(r'[^a-zA-Z0-9\s]', '', query.lower())
    tokens = clean_query.split()
    
    # Filter stop words
    tokens = [t for t in tokens if t not in STOP_WORDS]
    
    # Limit to 8 words to prevent explosion (reduced from 10)
    tokens = tokens[:8] 
    
    all_combos = []
    # Generate combinations of length 5 down to 1
    for r in range(min(len(tokens), 5), 0, -1):
        for combo in combinations(tokens, r):
            all_combos.append(list(combo))
            
    return all_combos

import concurrent.futures

def combinatorial_search(query, embedding):
    """
    Greedy Combinatorial Search.
    Prioritizes longer keyword matches.
    Parallelized by batching combinations of the same length.
    """
    combos = generate_combinations(query)
    final_results = []
    seen_content_ids = set()
    successful_subsets = set() # Track subsets that are covered by successful supersets
    
    # Group combos by length
    from collections import defaultdict
    combos_by_len = defaultdict(list)
    for c in combos:
        combos_by_len[len(c)].append(c)
        
    # Process lengths in descending order (5, 4, 3...)
    sorted_lengths = sorted(combos_by_len.keys(), reverse=True)
    
    # Helper to check if a combo is a subset of an already successful combo
    def is_subset_of_successful(combo):
        combo_set = set(combo)
        for success in successful_subsets:
            if combo_set.issubset(success):
                return True
        return False

    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        for length in sorted_lengths:
            current_batch = combos_by_len[length]
            
            # Filter out subsets of already successful searches BEFORE running
            # This is an optimization, but since we run in batches, we only know about success from PREVIOUS batches (longer lengths)
            filtered_batch = [c for c in current_batch if not is_subset_of_successful(c)]
            
            if not filtered_batch:
                continue
                
            # Submit batch to executor
            future_to_combo = {
                executor.submit(search_by_keywords, combo, embedding, limit=5): combo 
                for combo in filtered_batch
            }
            
            # Process results as they complete for this batch
            for future in concurrent.futures.as_completed(future_to_combo):
                combo = future_to_combo[future]
                try:
                    results = future.result()
                    if results:
                        # Mark this combo as successful
                        successful_subsets.add(frozenset(combo))
                        
                        for res in results:
                            content_sig = res['content'][:50]
                            if content_sig not in seen_content_ids:
                                final_results.append(res)
                                seen_content_ids.add(content_sig)
                except Exception as exc:
                    logging.error(f"Search generated an exception for combo {combo}: {exc}")
            
            # Check if we have enough results after this batch
            if len(final_results) >= 10:
                break
                
    return final_results[:5]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
        
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Run Ingestion Async
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # 0. Clean up old data for this file
            delete_document(filename)
            
            # 1. Robust Ingestion
            robust_chunks = loop.run_until_complete(process_robust(filepath))
            if robust_chunks:
                insert_robust_chunks(robust_chunks)
                
            # 2. Conventional Ingestion
            conventional_chunks = loop.run_until_complete(process_conventional(filepath))
            if conventional_chunks:
                insert_conventional_chunks(conventional_chunks)
                
            return jsonify({
                "message": "Ingestion Complete",
                "robust_count": len(robust_chunks),
                "conventional_count": len(conventional_chunks)
            })
            
        except Exception as e:
            logging.error(f"Ingestion Failed: {e}")
            return jsonify({"error": str(e)}), 500

def calculate_metrics(results, start_time):
    """Calculates execution time and average similarity."""
    time_taken = time.time() - start_time
    if not results:
        return {"time": f"{time_taken:.4f}s", "accuracy": "0.00%"}
    
    # Assuming results have a 'similarity' field (0-1 float)
    similarities = [r.get('similarity', 0) for r in results]
    avg_similarity = sum(similarities) / len(similarities) if similarities else 0
    
    return {
        "time": f"{time_taken:.4f}s",
        "accuracy": f"{avg_similarity * 100:.2f}%"
    }

@app.route('/search/robust', methods=['POST'])
def search_robust():
    data = request.json
    query = data.get('query')
    if not query: return jsonify({"error": "No query"}), 400
    
    start_time = time.time()
    try:
        embedding = get_embedding(query)
        if not embedding: return jsonify({"error": "Embedding failed"}), 500
        
        # Extract keywords for "Exact Search" (Partitioned)
        # Simple tokenization and stop word removal
        import re
        clean_query = re.sub(r'[^a-zA-Z0-9\s]', '', query.lower())
        tokens = clean_query.split()
        keywords = [t for t in tokens if t not in STOP_WORDS]
        
        # Use Partitioned Search (Exact Search on Robust Chunks)
        # This filters by keywords (overlap) and then ranks by vector similarity
        filters = {
            "keywords": keywords
        }
        
        results = search_partitioned(embedding, filters, limit=5)
        
        metrics = calculate_metrics(results, start_time)
        return jsonify({
            "results": results,
            "filters": filters,
            "metrics": metrics
        })
    except Exception as e:
        logging.error(f"Robust Search Failed: {e}")
        return jsonify({"error": str(e)}), 500



import time

@app.route('/search/conventional', methods=['POST'])
def search_conventional():
    data = request.json
    query = data.get('query')
    if not query: return jsonify({"error": "No query"}), 400
    
    start_time = time.time()
    try:
        embedding = get_embedding(query)
        if not embedding: return jsonify({"error": "Embedding failed"}), 500
        
        results = search_vectors_conventional(embedding, limit=5)
        metrics = calculate_metrics(results, start_time)
        
        return jsonify({
            "results": results,
            "metrics": metrics
        })
    except Exception as e:
        logging.error(f"Conventional Search Failed: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/search/ivfflat', methods=['POST'])
def search_ivfflat():
    data = request.json
    query = data.get('query')
    if not query: return jsonify({"error": "No query"}), 400
    
    start_time = time.time()
    try:
        embedding = get_embedding(query)
        if not embedding: return jsonify({"error": "Embedding failed"}), 500
        
        results = search_vectors_ivfflat(embedding, limit=5)
        metrics = calculate_metrics(results, start_time)
        
        return jsonify({
            "results": results,
            "metrics": metrics
        })
    except Exception as e:
        logging.error(f"IVFFlat Search Failed: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
