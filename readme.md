# Scaling Strategies for High-Dimensional Vector Databases
### Intelligent Partitioned Vector Search for Malaysian Legal Documents

**Subject:** SEG2102 Database Management Systems  
**Institution:** Sunway University  
**Year:** 2024/2025  

---

## Group Members

| Student Name             | Student ID | Contribution Focus |
| :----------------------- | :--------- | :----------------- |
| **Jing Jay Hong**        | 22008338   | Methodology, Pipeline Architecture, Analysis |
| **Vicky Leow Ming Fong** | 22009591   | Literature Review, Report Synthesis |
| **Wei Ting Tan**         | 21058664   | Data Collection, Formatting, Abstract/Conclusion |

---

# Project Overview

This repository contains the source code, benchmarking scripts, and experimental data for the research report: **"Scaling Strategies for High-Dimensional Vector Databases."**

The project challenges the hardware-centric "Memory Wall" paradigm in vector search. Instead of relying on expensive, memory-heavy global indexes (like HNSW), we implemented a **software-architectural solution** titled **"Intelligent Partitioned Vector Search."**

The system was developed using **PostgreSQL (pgvector)** and **Large Language Models (Google Gemini)** to index and retrieve Malaysian legal statutes (e.g., Employment Act 1955). The goal was to solve the problem of **"Semantic Drift"**—where standard vector engines confuse legally distinct concepts (e.g., "Misconduct" vs. "Sexual Harassment").

---

## Repository Structure

### Core Application Logic
*   **`app.py`**: The Flask backend acting as the "Smart Gatekeeper." It handles query intent classification and orchestrates the search logic.
*   **`vector_db.py`**: The database interface layer. Contains the SQL logic for:
    *   **Conventional Search:** Standard HNSW indexing.
    *   **Robust Search:** The proposed Partitioned (Filter-Then-Compute) logic.
*   **`file_ingestion.py`**: The ingestion pipeline. It splits legal PDFs, uses LLMs to generate "Atomic Legal Section" metadata, and populates the database.

### Experimentation & Benchmarking
*   **`benchmark_search.py`**: The primary script used to generate the Latency and Precision data found in the final report.
*   **`test_queries.py`**: A suite of 16 complex legal queries used to stress-test the system against semantic traps.
*   **`reproduce_keyword_mismatch.py`**: A specific test script demonstrating cases where HNSW fails but the Robust method succeeds.

### Data & Results
*   **`test_results_comparison.txt`**: Raw logs of the experiment runs, detailing execution time and similarity scores for HNSW vs. IVFFlat vs. Robust.
*   **`analysis_report.md`**: A summary of the findings generated during the development phase.

---

## Technology Stack

*   **Language:** Python 3.9+
*   **Database:** PostgreSQL 16 with `pgvector` extension
*   **AI Models:**
    *   *Embeddings:* Google Gemini Embedding-001 (1536 dimensions)
    *   *Structural Analysis:* Google Gemini 2.5 Flash
*   **Libraries:** Flask, LangChain, Psycopg2, Scikit-learn

---

## Setup & Installation

### 1. Prerequisites
Ensure you have **PostgreSQL** installed with the `pgvector` extension enabled:
```sql
CREATE EXTENSION vector;
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Environment Configuration
```ini
Locate the file named env.example in this repository.
Rename it to .env.
Open the file and fill in your actual credentials (API Key and Database Password):
# Replace with your actual Google AI API Key
GOOGLE_API_KEY=your_actual_api_key_here

# Update with your DB credentials
DATABASE_URL="postgresql://postgres:your_password@localhost:5432/Assignment"
```

### 4. Running the Benchmark
```bash
To replicate the results presented in the report:
python benchmark_search.py
```

## Key Research Findings

1.  **Precision Dominance:** The proposed **Intelligent Partitioned Framework** achieved **100% retrieval precision** on statutory queries.
2.  **Latency Efficiency:** The proposed framework achieved the **lowest average latency (1.47s)** compared to HNSW (1.50s) and IVFFlat (1.52s).
3.  **Failure Analysis:** The disk-based IVFFlat index proved unreliable, returning **0.00% accuracy** on complex queries.
