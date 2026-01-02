import json
import time
import urllib.request
import urllib.error

QUERIES = {
    "Contracts and Termination (Part II)": [
        "An employee who has worked for us for three years is resigning. Our employment contract is silent on the notice period. What is the minimum notice period we are legally required to accept from them, and what is the notice period we would have to give if we terminated their contract?",
        "Can we terminate an employee's contract immediately without requiring them to serve out their notice period, and how is the indemnity amount calculated for this 'pay in lieu of notice'?",
        "We believe an employee has committed serious misconduct. What steps must we take, including the 'due inquiry' process, before we can proceed with dismissal, downgrading, or suspension without notice under the Act?",
        "One of our employees has been absent from work for three consecutive working days without notifying us. Can we consider their contract as broken, and what is the legal justification?"
    ],
    "Wages and Deductions (Parts III & IV)": [
        "We pay our employees monthly. What is the latest day of the following month that we are legally required to pay their wages? Also, is the payment deadline different for overtime work or rest day work?",
        "How should we calculate the wages for a new employee who started halfway through the month, or for an employee whose service was terminated before the month end?",
        "We want to deduct a sum from an employee's wages for a company loan we provided. What types of deductions are automatically lawful, and for which ones do we need the employee's written consent or the Director General's permission?",
        "Are we legally required to pay wages electronically into a bank account, or can an employee request payment by cheque or cash?"
    ],
    "Working Hours and Leave (Part XII)": [
        "What are the legal limits on daily and weekly working hours for an employee, and how long must the rest period be after continuous hours of work?",
        "What is the minimum rate of pay for work performed that exceeds the normal hours of work (overtime)?",
        "How many days of paid annual leave is an employee entitled to after two years of continuous service? What if they are dismissed before completing 12 months in the final year?",
        "What are the conditions for a married male employee to be eligible for paid paternity leave, and what is the duration of this leave?",
        "An employee is not hospitalized. What is their entitlement for paid sick leave after four years of service, and what is the requirement for them to notify the employer of their sick leave?"
    ],
    "Maternity and Pregnancy (Part IX)": [
        "What is the length of the eligible period for maternity leave for a female employee?",
        "Is a female employee who has five surviving children still entitled to the maternity allowance, even if she is still entitled to maternity leave?",
        "Can we terminate the contract of a pregnant female employee for reasons other than misconduct, wilful breach of contract, or business closure, and where does the burden of proof lie if the termination is challenged?"
    ]
}

URLS = {
    "Robust (Partitioned/Exact)": "http://127.0.0.1:5000/search/robust",
    "Conventional (HNSW)": "http://127.0.0.1:5000/search/conventional",
    "IVFFlat": "http://127.0.0.1:5000/search/ivfflat"
}
OUTPUT_FILE = "test_results_comparison.txt"

def run_query(url, query):
    data = json.dumps({"query": query}).encode('utf-8')
    req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
    
    start_time = time.time()
    try:
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode('utf-8'))
            end_time = time.time()
            return result, end_time - start_time
    except urllib.error.URLError as e:
        return {"error": str(e)}, time.time() - start_time

def main():
    # Collect timing data for summary
    timing_data = {method: [] for method in URLS.keys()}
    query_list = []
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("Test Results Comparison: Robust vs Conventional vs IVFFlat\n")
        f.write("========================================================\n\n")
        
        for category, queries in QUERIES.items():
            f.write(f"Category: {category}\n")
            f.write("-" * (len(category) + 10) + "\n")
            
            for q in queries:
                f.write(f"Query: {q}\n")
                print(f"Running query: {q[:50]}...")
                query_list.append(q[:60] + "..." if len(q) > 60 else q)
                
                for method_name, url in URLS.items():
                    f.write(f"\n  Method: {method_name}\n")
                    result, duration = run_query(url, q)
                    timing_data[method_name].append(duration)
                    
                    f.write(f"  Time Taken: {duration:.4f} seconds\n")
                    
                    if "error" in result:
                        f.write(f"  Error: {result['error']}\n")
                    else:
                        metrics = result.get("metrics", {})
                        f.write(f"  Reported Accuracy (Similarity): {metrics.get('accuracy', 'N/A')}\n")
                        
                        results_list = result.get("results", [])
                        if not results_list:
                            f.write("    No results found.\n")
                        
                        for i, res in enumerate(results_list[:3], 1): # Limit to top 3 for brevity in comparison
                            content = res.get('content', 'No content').replace('\n', ' ')
                            similarity = res.get('similarity', 'N/A')
                            source = res.get('source', 'N/A') # For robust
                            f.write(f"    {i}. [Sim: {similarity}] [Source: {source}] {content[:150]}...\n")
                
                f.write("\n" + "="*50 + "\n\n")
        
        # Time Taken Per Query Summary Section
        f.write("\n" + "="*80 + "\n")
        f.write("TIME TAKEN PER QUERY SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        # Header row
        methods = list(URLS.keys())
        header = f"{'Query #':<10}"
        for method in methods:
            header += f"{method:<25}"
        f.write(header + "\n")
        f.write("-" * (10 + 25 * len(methods)) + "\n")
        
        # Data rows
        for i, q_short in enumerate(query_list):
            row = f"{'Q' + str(i+1):<10}"
            for method in methods:
                row += f"{timing_data[method][i]:.4f}s{'':<19}"
            f.write(row + "\n")
        
        f.write("-" * (10 + 25 * len(methods)) + "\n")
        
        # Totals row
        totals_row = f"{'TOTAL':<10}"
        for method in methods:
            total = sum(timing_data[method])
            totals_row += f"{total:.4f}s{'':<19}"
        f.write(totals_row + "\n")
        
        # Averages row
        avg_row = f"{'AVERAGE':<10}"
        for method in methods:
            avg = sum(timing_data[method]) / len(timing_data[method]) if timing_data[method] else 0
            avg_row += f"{avg:.4f}s{'':<19}"
        f.write(avg_row + "\n\n")
        
        # Detailed query legend
        f.write("Query Legend:\n")
        for i, q_short in enumerate(query_list):
            f.write(f"  Q{i+1}: {q_short}\n")
                
    print(f"Testing complete. Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
