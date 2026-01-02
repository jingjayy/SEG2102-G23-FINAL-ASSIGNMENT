# Analysis of Test Results: Robust (Partitioned) vs Conventional (HNSW) vs IVFFlat

## Overview
We tested three search methods against 16 complex legal queries:
1.  **Robust (Partitioned/Exact)**: Uses `search_partitioned` with keyword filtering (Exact Keyword Overlap) on the `robust_chunks` table.
2.  **Conventional (HNSW)**: Pure Semantic Search using HNSW index on `conventional_chunks`.
3.  **IVFFlat**: Pure Semantic Search using IVFFlat index on `conventional_ivfflat_chunks`.

## Key Findings

### 1. Robust (Partitioned) Chunks are "Better" Despite Lower Accuracy Scores
The user's hypothesis was correct. Although the reported similarity scores for Robust (Partitioned) are often lower (e.g., ~70-80% vs ~80-85% for HNSW), the **content quality** and **precision** of the retrieved chunks are often superior for specific legal questions.

*   **Example: Pregnant Termination (Query 16)**
    *   **Robust (Partitioned)**: Found **Section 41A** (Sim: 0.89) - *"Where a female employee is pregnant... it shall be an offence for her employer to terminate..."*. This is the **exact** answer.
    *   **Conventional (HNSW)**: Found Section 47 and 25 (Sim: 0.84) - Related to termination and business changes, but missed the specific pregnancy protection section in the top results.
    *   **Verdict**: Robust wins on content quality.

*   **Example: Misconduct (Query 3)**
    *   **Robust (Partitioned)**: Found **Section 14** (Sim: 0.81) - *"An employer may, on the grounds of misconduct... dismiss without notice..."*. This is the **exact** section defining the due inquiry process.
    *   **Conventional (HNSW)**: Found Section 81D (Sim: 0.80) - Related to sexual harassment inquiry. This is a "semantic trap" - it matched "inquiry" but for the wrong topic.
    *   **Verdict**: Robust wins on content quality.

*   **Example: Notice Period (Query 1)**
    *   **Robust (Partitioned)**: Found Section 16 (Sim: 0.76) - Agricultural workers. This is a false positive due to keyword overlap.
    *   **Conventional (HNSW)**: Found Section 12/13 implicitly (Sim: 0.82) - "Contracts to be in writing...".
    *   **Verdict**: Mixed. Robust can be too specific/literal if the keywords are generic.

### 2. Why are Robust Chunks Better?
The `robust_chunks` table likely contains **cleaner, more structured chunks** (possibly preserving Section headers and full text blocks) compared to the `conventional_chunks` which might be sliding windows or less context-aware.
- The "Exact Search" (Partitioned) leverages the presence of specific keywords (like "misconduct", "pregnant") to filter down to the right set of chunks *before* ranking, whereas HNSW relies purely on vector proximity which can be misled by dominant but irrelevant themes (like "sexual harassment" dominating "inquiry").

### 3. IVFFlat Performance
- IVFFlat continues to be the weakest performer, often returning 0 results or less relevant chunks than HNSW.

## Conclusion

**"Are the chunks better?" -> YES.**

For specific legal queries where precise terminology is used (e.g., "misconduct", "maternity allowance", "pregnant"), the **Robust (Partitioned)** method retrieves the **exact legal sections** required to answer the question, whereas the Semantic (HNSW) search sometimes drifts into conceptually related but legally distinct topics.

**Recommendation**:
The ideal system should use the **Robust Chunks** but might benefit from a **Hybrid Ranking**:
1.  Use **Partitioned Search** to find candidates based on keyword overlap (High Recall for specific terms).
2.  Use **Vector Similarity** to rank them (as currently done).
3.  Potentially **relax** the keyword constraint if too few results are found (fallback to pure vector search on Robust table).
