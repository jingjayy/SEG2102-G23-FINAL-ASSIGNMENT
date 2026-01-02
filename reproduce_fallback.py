import requests
import json
import time

def test_fallback():
    url = "http://127.0.0.1:5000/search/robust"
    
    # Query unlikely to match any keywords in a legal database
    query = "supercalifragilisticexpialidocious random nonsense that definitely has no keywords"
    
    payload = {"query": query}
    headers = {"Content-Type": "application/json"}
    
    print(f"Sending query: {query}")
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        
        data = response.json()
        results = data.get("results", [])
        
        if not results:
            print("FAILURE: No results returned.")
            return
            
        print(f"Received {len(results)} results.")
        
        # Check for fallback flag
        is_fallback = any(r.get('fallback') is True for r in results)
        
        if is_fallback:
            print("SUCCESS: Fallback triggered (fallback flag found).")
        else:
            print("WARNING: Fallback flag NOT found. (Maybe it found keywords? or fallback logic failed to set flag?)")
            print("Results sample:", results[0] if results else "None")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Wait a bit for app to reload if needed
    time.sleep(2) 
    test_fallback()
