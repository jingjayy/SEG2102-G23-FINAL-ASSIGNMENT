import requests
import time

def benchmark_search():
    url = "http://127.0.0.1:5000/search/robust"
    
    # Query with multiple potential keywords to trigger combinatorial search
    query = "minimum notice period for termination of employment contract due to bankruptcy"
    
    payload = {"query": query}
    headers = {"Content-Type": "application/json"}
    
    print(f"Sending complex query: {query}")
    
    start_time = time.time()
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        end_time = time.time()
        
        data = response.json()
        results = data.get("results", [])
        metrics = data.get("metrics", {})
        
        print(f"Total Request Time: {end_time - start_time:.4f}s")
        print(f"Server Reported Time: {metrics.get('time')}")
        print(f"Results Count: {len(results)}")
        
        if results:
            print("Top Result:", results[0].get('content')[:100])
            print("Matched Keywords:", results[0].get('matched_keywords'))
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    time.sleep(2)
    benchmark_search()
