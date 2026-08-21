# Phase 13 Implementation Plan — Citation Integrity & Evidence Verification Engine

## 1. Current Architecture Assessment

The RAG application is currently at the stable `phase-12-stable` baseline (Commit `cb6b33f`). The production system incorporates:

* **Retrieval & Diversity (Phase 5):** FAISS vector store (`sentence-transformers/all-MiniLM-L6-v2`) with `RETRIEVAL_K=15`, filtered by `RELEVANCE_THRESHOLD=1.50`, and MMR soft diversity selection (`DIVERSITY_PENALTY=0.15`) selecting up to `MAX_CONTEXT_CHUNKS=5`.
* **Frontend Rendering & Persistence (Phases 6 & 7):** Glassmorphism interface with `Marked.js` Markdown rendering, `DOMPurify` XSS sanitization, and browser `localStorage` chat persistence.
* **Evaluation Harness (Phase 8):** Deterministic evaluation dataset (`eval_dataset.json`) and test execution harness (`test_rag_evaluation.py`).
* **Token Economy & Resilience (Phase 9):** Session history pruning (last 2 QA pairs) and explicit HTTP 429 rate-limit trapping.
* **Semantic Response Cache (Phase 10):** In-memory cosine similarity cache (`GLOBAL_SEMANTIC_CACHE`, threshold `0.95`, LRU 100 entries, TTL 3600s, KB version validation via `index.faiss` mtime).
* **Production Observability (Phase 11):** Zero-dependency thread-safe `/metrics` telemetry endpoint tracking aggregate system, retrieval, cache, generation, and latency metrics.
* **Intra-Provider LLM Cascading (Phase 12):** Primary `llama-3.3-70b-versatile` model cascading to fallback `llama-3.1-8b-instant` on recoverable Groq API failures (429/500/503/timeout), with single-attempt retry bounds.

---

## 2. Evidence-Based Problem Identification

### Problem Statement
While Phase 12 solved LLM rate-limit resilience, an unverified gap remains between **retrieved context metadata** and **LLM-generated inline citations**.

Currently in `llama.py`:
1. The RAG prompt instructs the LLM to format inline citations as `[Source X, Page Y]`.
2. The LLM generates an answer containing string citations.
3. The server checks ONLY `fallback_sentence not in answer` for grounding, and directly returns the raw answer text to the client.

### Evidence of Failure Modes
* **Citation Index Off-by-One / Mismatch:** The prompt labels chunks as `[Source 1]`, `[Source 2]`, etc. (1-indexed chunk numbers), while `citation_sources` maps unique source documents `{"id": 1, "source": "file.pdf"}` (1-indexed document numbers). Small LLMs (such as fallback `llama-3.1-8b-instant`) occasionally confuse chunk indices with document IDs or cite source indices beyond the range of retrieved chunks (e.g., citing `[Source 6, Page 3]` when only 4 chunks were retrieved).
* **Page Number Hallucination:** In edge cases, the LLM generates a plausible page number `[Source 1, Page 15]` that does not exist in the retrieved chunk metadata for that source document.
* **Orphaned / Malformed Citations:** Unformatted citations such as `[Source 1]` (missing page) or `[Page 4]` (missing source) degrade the UI Evidence Viewer experience, as the frontend JavaScript relies on valid citation structures to highlight evidence cards.

---

## 3. Recommended Phase 13 Capability

### Selected Capability: **Citation Integrity & Evidence Verification Engine**

A deterministic, zero-latency post-generation verification engine in `llama.py` that parses, validates, and sanitizes inline citations before returning responses to the client or saving payloads to the semantic response cache.

### Comparative Evaluation of Candidates

| Candidate Capability | Value | Latency | Risk | Decision |
| :--- | :--- | :--- | :--- | :--- |
| **A. Citation Integrity & Evidence Verification Engine** | **Highest** | **0ms (Regex)** | **Low** | **RECOMMENDED** — Eliminates hallucinated/broken citations, guarantees evidence alignment for UI. |
| **B. Contextual Sentence-Level Compression** | Medium | +50-100ms (CPU) | Medium | DEFERRED — CPU sentence embedding computation adds latency on Render free-tier. |
| **C. Persistent Cache Disk Rehydration** | Medium | 0ms | Low | DEFERRED — In-memory Phase 10 cache is already fast; disk I/O rehydration adds deployment complexity. |
| **D. BM25 Sparse-Dense Hybrid Search** | Medium | +30ms | Medium | DEFERRED — Requires adding `rank_bm25` dependency and rebuilding local indexes. |

---

## 4. Exact Proposed Changes

### 1. `llama.py` (Backend Engine)
* **Add `verify_and_sanitize_citations(answer, sources_list, citation_sources_map)` function:**
  * Uses regex to extract all `\[Source\s+(\d+),\s*Page\s*([^\]]+)\]` and `\[Source\s+(\d+)\]` patterns from `answer`.
  * Validates source index against retrieved chunk range `[1..len(sources_list)]` and document IDs.
  * Validates page number against actual metadata `page` of the referenced chunk/source.
  * Replaces invalid/hallucinated citations with verified citations or strips orphaned references.
  * Updates `final_payload` with verified citation flags.
* **Update `/metrics` Telemetry:**
  * Add `citation_verified_count`, `citation_sanitized_count`, and `citation_hallucination_count` under `generation` / `citations` metrics.

### 2. `test_server.py` (Regression Suite)
* **Add Test 9 (Citation Verification & Sanitization Tests):**
  * Test valid citations pass untouched.
  * Test out-of-bounds source index `[Source 99, Page 1]` is detected and sanitized.
  * Test hallucinated page number is corrected to valid metadata page.
  * Test fallback model output verification.
  * Test `/metrics` citation counters increment correctly.

### 3. `test_rag_evaluation.py` (Evaluation Harness)
* **Enhance Citation Evaluation Logic:**
  * Verify that citation evaluation checks not just string presence (`"[Source"`), but structural validity against retrieved sources.

---

## 5. Architectural Pipeline Integration

The capability integrates seamlessly without breaking any prior phase:

```
User Query 
  → Semantic Cache Lookup (Phase 10)
  → FAISS Similarity Search (k=15) 
  → MMR Diversity Selection (Phase 5) 
  → Grounding Check (Phase 1/3) 
  → Primary LLM / Fallback Cascade (Phase 12) 
  → [NEW] Citation Integrity & Evidence Verification Engine (Phase 13) 
  → Add to GLOBAL_SESSIONS (Phase 4/9) 
  → Save to GLOBAL_SEMANTIC_CACHE (Phase 10) 
  → Increment /metrics (Phase 11) 
  → JSON Response to Frontend (Phase 6/7)
```

---

## 6. Metrics & Telemetry

Expose additional aggregate telemetry in `GET /metrics`:

```json
{
  "citations": {
    "total_citations_parsed": 0,
    "verified_valid_citations": 0,
    "sanitized_citations": 0,
    "hallucinated_citations_rejected": 0
  }
}
```

* **Target Goal:** 100% citation validity on all served grounded responses; 0 invalid/out-of-bounds citations delivered to the browser UI.

---

## 7. Regression Protection & Verification Plan

### Automated Regression Verification
Run `python test_server.py` and confirm all existing assertions PASS:
* FAISS retrieval & MMR diversity
* Grounding & fallback sentences
* Primary `llama-3.3-70b-versatile` generation
* Fallback `llama-3.1-8b-instant` 429 cascade
* Semantic cache hit / miss / eviction / KB versioning
* Conversational pronoun resolution (`resolve_conversational_query`)
* `GLOBAL_SESSIONS` thread-safety & LRU cleanup
* Markdown & DOMPurify frontend compatibility
* `/metrics` endpoint backward compatibility

---

## 8. Rollback Strategy

If any regression or unexpected behavior is detected during Phase 13 execution:

```powershell
git reset --hard phase-12-stable
```

`phase-12-stable` (Commit `cb6b33f`) remains the untouched rollback baseline.

---

## 9. Open Questions & User Approval Gate

> [!IMPORTANT]
> **User Feedback Requested:** Please review this proposed Phase 13 scope. Upon your explicit approval, implementation will proceed according to this plan.
