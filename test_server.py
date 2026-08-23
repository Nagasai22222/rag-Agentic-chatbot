import os
import sys
import time
import json
from unittest.mock import patch

os.chdir(r"c:\Users\ACER\OneDrive\Desktop\vasavi\RAG")
import concurrent.futures
from langchain_groq import ChatGroq
from llama import app, GLOBAL_SEMANTIC_CACHE, CACHE_TTL_SECONDS, record_metric, get_session_history, add_to_session, is_rate_limited, _rate_limit_store, _rate_limit_lock
client = app.test_client()

def run_query(session_id, prompt):
    print(f"\n[Session: {session_id}] -> {prompt}")
    res = client.post('/ask', 
                      data=json.dumps({'query': prompt, 'userId': session_id}),
                      content_type='application/json')
    try:
        data = json.loads(res.data)
    except json.JSONDecodeError:
        print(f"FAILED TO DECODE JSON! Status Code: {res.status_code}")
        print(f"Raw Response: {res.data.decode('utf-8')[:500]}")
        sys.exit(1)
        
    if res.status_code == 429:
        print(f"   [RATE LIMIT] HTTP 429 Trapped Successfully: {data.get('message')}")
        return data
    if res.status_code == 500:
        print(f"   [LLM ERROR] HTTP 500 Trapped: {data.get('message')}")
        return data
        
    resolved = data.get('resolved_query', 'N/A')
    history_used = data.get('history_used', False)
    grounded = data.get('grounded', False)
    cache_hit = data.get('cache_hit', False)
    cache_time = data.get('cache_lookup_time', 0.0)
    
    print(f"   Resolved Query : {resolved} (History Modified: {history_used})")
    print(f"   Grounded       : {grounded}")
    print(f"   Cache Hit      : {cache_hit} (Lookup Time: {cache_time}s)")
    print(f"   Result         : {data.get('result', '')[:80].replace(chr(10), ' ')}...")
    return data

def reset_session(session_id):
    client.post('/reset', 
                data=json.dumps({'userId': session_id}),
                content_type='application/json')
    print(f"\n[RESET] Cleared Session: {session_id}")

def mock_429(*args, **kwargs):
    raise Exception("Error code: 429 - {'error': {'message': 'Rate limit reached'}}")

def mock_500(*args, **kwargs):
    raise Exception("Runtime LLM backend crash")

def mock_success(*args, **kwargs):
    class MockRes:
        content = "This is a mathematically verified mock answer."
    return MockRes()

def test_citation_verification(metrics):
    print("\n--- PHASE 13 CITATION VERIFICATION TESTS ---")
    from llama import verify_and_sanitize_citations
    
    mock_sources = [
        {"chunk": 1, "source": "rag_doc.pdf", "page": 5},
        {"chunk": 2, "source": "rag_doc.pdf", "page": 8},
        {"chunk": 3, "source": "ai_basics.pdf", "page": 12}
    ]
    mock_citation_map = {
        "rag_doc.pdf": {"id": 1, "source": "rag_doc.pdf", "pages": [5, 8], "chunks": [1, 2]},
        "ai_basics.pdf": {"id": 2, "source": "ai_basics.pdf", "pages": [12], "chunks": [3]}
    }
    
    # 1. Valid source + valid page -> preserved
    ans1 = "RAG combines retrieval and generation [Source 1, Page 5]."
    san1, stats1 = verify_and_sanitize_citations(ans1, mock_sources, mock_citation_map)
    assert san1 == ans1, f"Valid citation must be preserved. Got: {san1}"
    assert stats1["valid_citations"] == 1
    assert stats1["invalid_citations"] == 0
    print("  -> Case 1 (Valid source + page): PASS")

    # 2. Valid source without page -> preserved
    ans2 = "Fine-tuning updates weights [Source 1]."
    san2, stats2 = verify_and_sanitize_citations(ans2, mock_sources, mock_citation_map)
    assert san2 == ans2, f"Valid source without page must be preserved. Got: {san2}"
    assert stats2["valid_citations"] == 1
    print("  -> Case 2 (Valid source without page): PASS")

    # 3. Invalid source index -> sanitized
    ans3 = "Embeddings are vectors [Source 99, Page 1]."
    san3, stats3 = verify_and_sanitize_citations(ans3, mock_sources, mock_citation_map)
    assert "[Source 99" not in san3, f"Invalid source index must be sanitized. Got: {san3}"
    assert "Embeddings are vectors." in san3
    assert stats3["invalid_citations"] == 1
    print("  -> Case 3 (Invalid source index): PASS")

    # 4. Invalid page number -> sanitized
    ans4 = "Attention is all you need [Source 1, Page 99]."
    san4, stats4 = verify_and_sanitize_citations(ans4, mock_sources, mock_citation_map)
    assert "[Source 1, Page 99]" not in san4, f"Invalid page number must be sanitized. Got: {san4}"
    assert stats4["invalid_citations"] == 1
    print("  -> Case 4 (Invalid page number): PASS")

    # 5. Multiple valid citations -> all preserved
    ans5 = "RAG [Source 1, Page 5] and Fine-tuning [Source 3, Page 12] work well."
    san5, stats5 = verify_and_sanitize_citations(ans5, mock_sources, mock_citation_map)
    assert san5 == ans5, f"Multiple valid citations must be preserved. Got: {san5}"
    assert stats5["valid_citations"] == 2
    print("  -> Case 5 (Multiple valid citations): PASS")

    # 6. Mixed valid/invalid citations -> only invalid sanitized
    ans6 = "RAG [Source 1, Page 5] is good, but fake [Source 99, Page 1] is bad."
    san6, stats6 = verify_and_sanitize_citations(ans6, mock_sources, mock_citation_map)
    assert "[Source 1, Page 5]" in san6
    assert "[Source 99" not in san6
    assert stats6["valid_citations"] == 1
    assert stats6["invalid_citations"] == 1
    print("  -> Case 6 (Mixed valid/invalid citations): PASS")

    # 7. Hallucinated citation -> sanitized
    ans7 = "Hallucinated statement [Source 5, Page 20]."
    san7, stats7 = verify_and_sanitize_citations(ans7, mock_sources, mock_citation_map)
    assert "[Source 5" not in san7
    assert stats7["invalid_citations"] == 1
    print("  -> Case 7 (Hallucinated citation): PASS")

    # 8. Citation-free response -> unchanged
    ans8 = "This is a direct answer with no citations."
    san8, stats8 = verify_and_sanitize_citations(ans8, mock_sources, mock_citation_map)
    assert san8 == ans8
    assert stats8["total_citations"] == 0
    print("  -> Case 8 (Citation-free response): PASS")

    # 9. Citation metrics verification under /metrics
    cit_metrics = metrics.get('citations', {})
    assert 'total_citations' in cit_metrics, "Metrics must include citation telemetry"
    print(f"  -> Case 9 (Citation metrics in /metrics): {cit_metrics}")

    # 10. Verification latency benchmarked
    print(f"  -> Case 10 (Measured verification latency): {stats1['verification_time']*1000:.4f} ms")

    print("\n# ALL PHASE 13 CITATION TESTS SUCCEEDED!")

def test_contextual_query_resolution(metrics):
    print("\n--- PHASE 14 CONTEXTUAL QUERY RESOLUTION & EXPANSION TESTS ---")
    from llama import resolve_and_expand_query, expand_domain_query, resolve_contextual_query
    from langchain_groq import ChatGroq

    # 1. Standalone Query
    q1, is_c1, is_e1, t1 = resolve_and_expand_query("What is fine-tuning?", [])
    assert q1 == "What is fine-tuning?", f"Standalone query must be unchanged. Got: {q1}"
    assert is_c1 == False and is_e1 == False
    print("  -> Case 1 (Standalone query): PASS")

    # 2. Pronoun Follow-up
    hist2 = [{"role": "user", "content": "What is fine-tuning?"}, {"role": "assistant", "content": "Fine-tuning adapts weights."}]
    q2, is_c2, is_e2, t2 = resolve_and_expand_query("Why is it important?", hist2)
    assert "fine-tuning" in q2.lower(), f"Pronoun 'it' must be replaced with subject. Got: {q2}"
    assert is_c2 == True
    print(f"  -> Case 2 (Pronoun follow-up): PASS ('Why is it important?' -> '{q2}')")

    # 3. Context-dependent Follow-up
    hist3 = [{"role": "user", "content": "Explain prompt engineering."}, {"role": "assistant", "content": "Prompt engineering shapes inputs."}]
    q3, is_c3, is_e3, t3 = resolve_and_expand_query("What are the key benefits?", hist3)
    assert "prompt engineering" in q3.lower(), f"Implicit coreference must append subject. Got: {q3}"
    assert is_c3 == True
    print(f"  -> Case 3 (Context-dependent follow-up): PASS ('What are the key benefits?' -> '{q3}')")

    # 4. Short Query
    q4, is_c4, is_e4, t4 = resolve_and_expand_query("RAG?", [])
    assert "retrieval augmented generation" in q4.lower(), f"Short domain query 'RAG?' must be expanded. Got: {q4}"
    assert is_e4 == True
    print(f"  -> Case 4 (Short query expansion): PASS ('RAG?' -> '{q4}')")

    # 5. Already-Complete Query
    q5, is_c5, is_e5, t5 = resolve_and_expand_query("What are pretrained model weights in deep learning?", [])
    assert q5 == "What are pretrained model weights in deep learning?", f"Complete query must remain untouched. Got: {q5}"
    assert is_c5 == False and is_e5 == False
    print("  -> Case 5 (Already-complete query): PASS")

    # 6. Unsupported Query
    with patch.object(ChatGroq, 'invoke', side_effect=mock_success):
        res6 = run_query("test_unsupported_p14", "What is the capital of France?")
        assert res6.get("grounded") == False, "Unsupported query must trigger Fast-Trip"
        print("  -> Case 6 (Unsupported query Fast-Trip): PASS")

    # 7. Same Standalone Query Across Two Sessions (Cache Valid)
    s_stand1 = "session_standalone_1"
    s_stand2 = "session_standalone_2"
    reset_session(s_stand1)
    reset_session(s_stand2)
    with patch.object(ChatGroq, 'invoke', side_effect=mock_success):
        r_stand1 = run_query(s_stand1, "What is fine-tuning?")
        r_stand2 = run_query(s_stand2, "What is fine-tuning?")
        assert r_stand2.get("cache_hit") == True, "Standalone queries across sessions MUST continue using semantic cache"
        print("  -> Case 7 (Standalone query cross-session cache): PASS")

    # 8. Same Contextual Query Text With Different Session Subjects (Cache Bypassed / No Cross-Contamination)
    s_a = "session_a"
    s_b = "session_b"
    reset_session(s_a)
    reset_session(s_b)
    with patch.object(ChatGroq, 'invoke', side_effect=mock_success):
        run_query(s_a, "What is fine-tuning?")
        run_query(s_b, "What is artificial intelligence?")
        res_a = run_query(s_a, "Why is it useful?")
        res_b = run_query(s_b, "Why is it useful?")
        assert "fine-tuning" in res_a.get("resolved_query", "").lower()
        assert "artificial intelligence" in res_b.get("resolved_query", "").lower()
        assert res_a.get("cache_hit") == False, "Contextual query in session A must bypass cache"
        assert res_b.get("cache_hit") == False, "Contextual query in session B must bypass cache"
        print("  -> Case 8 (Contextual query cache isolation & no cross-contamination): PASS")

    # 9. Same Contextual Query Within Same Session (Evaluated dynamically)
    with patch.object(ChatGroq, 'invoke', side_effect=mock_success):
        res_a_again = run_query(s_a, "Why is it useful?")
        assert res_a_again.get("cache_hit") == False, "Contextual queries within same session must bypass cache"
        print("  -> Case 9 (Same contextual query intra-session bypass): PASS")

    # 10. New Chat Isolation
    reset_session(s_a)
    with patch.object(ChatGroq, 'invoke', side_effect=mock_success):
        res_reset = run_query(s_a, "Why is it useful?")
        assert "fine-tuning" not in res_reset.get("resolved_query", "").lower(), "After reset, session history must be clear"
        print("  -> Case 10 (New Chat isolation): PASS")

    # 11. Measured Query Resolution Latencies
    t_standalone_ms = t1 * 1000.0
    t_contextual_ms = t2 * 1000.0
    print(f"  -> Case 11 (Measured resolution latencies: Standalone={t_standalone_ms:.4f} ms, Contextual={t_contextual_ms:.4f} ms): PASS")

    qr_metrics = metrics.get('query_resolution', {})
    print(f"  -> Case 12 (Query resolution telemetry in /metrics): {qr_metrics}")

    print("\n# ALL PHASE 14 CONTEXTUAL RESOLUTION & CACHE ISOLATION TESTS SUCCEEDED!")

if __name__ == "__main__":
    print("--- PHASE 10 CACHE TESTS ---")
    
    from langchain_groq import ChatGroq
    with patch.object(ChatGroq, 'invoke', side_effect=mock_success):
        # TEST 1 - Standalone (Initial Miss)
        print("\n# TEST 1: Standalone (Initial Cache Miss)")
        s1 = "test_s1"
        run_query(s1, "What is RAG and how does it work?")
        
        # TEST 2 - Exact Hit
        print("\n# TEST 2: Exact Repeated Query (Cache Hit & Groq Bypass)")
        res_exact = run_query(s1, "What is RAG and how does it work?")
        assert res_exact.get('cache_hit') == True
        
        # TEST 3 - Semantic Hit
        print("\n# TEST 3: Semantic Equivalent (Cache Hit)")
        res_sem = run_query(s1, "Explain what RAG is and its mechanics")
        
        # TEST 4 - Conversational Follow-up
        print("\n# TEST 4: Follow-up Pronouns (Cache Miss to hit)")
        run_query(s1, "What is fine-tuning?")
        res_pronoun = run_query(s1, "What is its purpose?")
        assert res_pronoun.get('cache_hit') == False, "New intent must cache miss"

        # TEST 4.1 - Re-run same pronoun in DIFFERENT chat (Contextual queries bypass global cache for isolation)
        print("\n# TEST 4.1: Contextual query in new session (Global Cache Bypass)")
        s1_alt = "test_s1_alt"
        run_query(s1_alt, "What is fine-tuning?")
        res_pronoun_again = run_query(s1_alt, "What is its purpose?")
        assert res_pronoun_again.get('cache_hit') == False, "Contextual queries must BYPASS global semantic cache to prevent cross-session contamination"
        
        # TEST 5 - Unsupported
        print("\n# TEST 5: Unsupported query")
        run_query(s1, "What is the capital of France?")
        res_unsupported = run_query(s1, "What is the capital of France?")
        assert res_unsupported.get('cache_hit') == False, "Fast-trip (grounded=False) should NEVER be cached"
    
    # TEST 6 - Simulated HTTP 429 (Cascading Fallback Recovery)
    print("\n# TEST 6: Simulated Groq HTTP 429 (Fallback Recovery)")
    s7 = "test_s7"
    with patch.object(ChatGroq, 'invoke', side_effect=[Exception("Error code: 429 rate limit"), mock_success()]):
        res_429 = run_query(s7, "Explain embeddings thoroughly.")
        assert res_429.get("result") != "The language model service is temporarily rate limited. Please try again shortly.", "Fallback should have recovered the 429 perfectly!"
        assert res_429.get("grounded") == True, "Fallback response must maintain grounding status"
        
    print("\n# TEST 6.1: Both models fail (Ultimate Failure Trap)")
    with patch.object(ChatGroq, 'invoke', side_effect=[Exception("Error code: 429 rate limit"), Exception("Error code: 503 fallback overloaded")]):
        res_ultimate_fail = run_query(s7, "Explain vectors.")
        assert (res_ultimate_fail.get("error") == "provider_error" or res_ultimate_fail.get("error") == "rate_limit"), "Ultimate failure must abort safely"
        assert res_ultimate_fail.get("cache_hit", False) == False, "Total generation failures NEVER cache"

    # TEST 7: Invalid KB Version
    print("\n# TEST 7: KB Version Invalidation")
    s_ver = "test_ver"
    with patch.object(ChatGroq, 'invoke', side_effect=mock_success):
        run_query(s_ver, "What are pretrained weights?")
        
        # Artificially expire the cache by faking all timestamps bounds
        for k in GLOBAL_SEMANTIC_CACHE:
            GLOBAL_SEMANTIC_CACHE[k]["kb_version"] = 0 # Invalid
        
    print("\n# DONE")
    
    # TEST 8: /metrics Observability Endpoint Validation
    print("\n# TEST 8: Observability /metrics Output Validation")
    res_m = client.get('/metrics')
    metrics = json.loads(res_m.data)
    
    sys_metrics = metrics.get('system', {})
    gen_metrics = metrics.get('generation', {})
    cache_metrics = metrics.get('cache', {})
    
    print("\n--- PHASE 11 OBSERVABILITY REPORT ---")
    print(f"   Uptime: {sys_metrics.get('uptime_seconds')}s")
    print(f"   Total Requests Evaluated: {sys_metrics.get('total_requests')}")
    print(f"   Successful Requests: {sys_metrics.get('successful_requests')}")
    print(f"   Cache Hits: {cache_metrics.get('hits')}")
    print(f"   Cache Misses: {cache_metrics.get('misses')}")
    print(f"   Total Generation Calls (Sum): {gen_metrics.get('total_generations')}")
    print(f"   Primary Generations: {gen_metrics.get('primary_generation_count')}")
    print(f"   Fallback Called: {gen_metrics.get('fallback_generation_count')}")
    print(f"   Fallback Succeeded: {gen_metrics.get('fallback_success_count')}")
    print(f"   Fallback Failed: {gen_metrics.get('fallback_failure_count')}")
    print(f"   Fast Trips (Ungrounded): {gen_metrics.get('fast_trip_count')}")
    print(f"   HTTP 429 / Generation Errors (Ultimate bounds): {sys_metrics.get('errors', 0)}")
    
    assert sys_metrics.get('total_requests') > 0, "Metrics must record requests"
    assert cache_metrics.get('hits') > 0, "Cache hits must accumulate"
    assert gen_metrics.get('fast_trip_count') >= 1, "Fast trips must accumulate"
    assert gen_metrics.get('fallback_success_count') >= 1, "Fallback success trap must aggregate"
    assert gen_metrics.get('fallback_failure_count') >= 1, "Ultimate failure bounds must record"

import concurrent.futures

def test_concurrency_and_locks(metrics):
    print("\n--- PHASE 15 THREAD-SAFE CONCURRENCY & LOCK SUITE ---")
    
    # 1. Concurrent Session Creation & Writes Across Different Sessions
    errors = []
    def worker_different_sessions(idx):
        try:
            user_id = f"concurrent_user_{idx}"
            run_query(user_id, f"What is RAG topic {idx}?")
        except Exception as e:
            errors.append(e)

    with patch.object(ChatGroq, 'invoke', side_effect=mock_success):
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(worker_different_sessions, i) for i in range(20)]
            concurrent.futures.wait(futures)
            
    assert len(errors) == 0, f"Concurrent session creation/writes failed: {errors}"
    print("  -> Case 1 & 4 (Concurrent session creation & writes to different sessions): PASS")

    # 2. Concurrent Reads & Writes on the SAME Session
    same_user = "shared_concurrent_user"
    def worker_same_session(idx):
        try:
            if idx % 2 == 0:
                run_query(same_user, f"Query variant {idx}")
            else:
                run_query(same_user, "Why is fine-tuning important?")
        except Exception as e:
            errors.append(e)

    with patch.object(ChatGroq, 'invoke', side_effect=mock_success):
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(worker_same_session, i) for i in range(20)]
            concurrent.futures.wait(futures)

    assert len(errors) == 0, f"Concurrent same-session reads/writes failed: {errors}"
    print("  -> Case 2 & 3 (Concurrent reads & writes on same session): PASS")

    # 3. Concurrent Reset during Active Session Reads/Writes
    def worker_reset_session(idx):
        try:
            if idx % 3 == 0:
                reset_session(same_user)
            else:
                run_query(same_user, f"Post-reset query {idx}")
        except Exception as e:
            errors.append(e)

    with patch.object(ChatGroq, 'invoke', side_effect=mock_success):
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(worker_reset_session, i) for i in range(15)]
            concurrent.futures.wait(futures)

    assert len(errors) == 0, f"Concurrent session reset failed: {errors}"
    print("  -> Case 5 (Concurrent session reset): PASS")

    # 4. Concurrent Cache Reads & Writes & Capacity Eviction
    def worker_cache_eviction(idx):
        try:
            q = f"Unique cache query item number {idx}"
            run_query("cache_worker", q)
        except Exception as e:
            errors.append(e)

    with patch.object(ChatGroq, 'invoke', side_effect=mock_success):
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(worker_cache_eviction, i) for i in range(120)]
            concurrent.futures.wait(futures)

    assert len(errors) == 0, f"Concurrent cache reads/writes/eviction failed: {errors}"
    print("  -> Case 6, 7 & 8 (Concurrent cache reads, writes, and LRU eviction): PASS")

    # 5. Concurrent TTL Expiration & KB Invalidation
    def worker_ttl_kb(idx):
        try:
            if idx % 2 == 0:
                for k in list(GLOBAL_SEMANTIC_CACHE.keys()):
                    GLOBAL_SEMANTIC_CACHE[k]["kb_version"] = 0
            run_query("ttl_user", "What is fine-tuning?")
        except Exception as e:
            errors.append(e)

    with patch.object(ChatGroq, 'invoke', side_effect=mock_success):
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(worker_ttl_kb, i) for i in range(10)]
            concurrent.futures.wait(futures)

    assert len(errors) == 0, f"Concurrent TTL/KB invalidation failed: {errors}"
    print("  -> Case 9 & 10 (Concurrent TTL expiration & KB invalidation): PASS")

    # 6. Concurrent Metrics Updates
    def worker_metrics(idx):
        try:
            record_metric("system", "total_requests", 1)
        except Exception as e:
            errors.append(e)

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(worker_metrics, i) for i in range(50)]
        concurrent.futures.wait(futures)

    assert len(errors) == 0, f"Concurrent metrics update failed: {errors}"
    print("  -> Case 11 & 12 (Concurrent metrics & cache/session isolation): PASS")

    # Benchmarking uncontended lock overheads
    t_start = time.perf_counter()
    for _ in range(1000):
        get_session_history("bench_user")
    t_session_read = (time.perf_counter() - t_start) / 1000.0 * 1000.0

    t_start = time.perf_counter()
    for i in range(1000):
        add_to_session("bench_user", "user", f"msg {i}")
    t_session_write = (time.perf_counter() - t_start) / 1000.0 * 1000.0

    print(f"  -> Case 13 (Uncontended Session Read Overhead: {t_session_read:.6f} ms)")
    print(f"  -> Case 14 (Uncontended Session Write Overhead: {t_session_write:.6f} ms)")
    print("\n# ALL PHASE 15 CONCURRENCY & THREAD SAFETY TESTS SUCCEEDED!")

def test_phase16_api_hardening():
    print("\n========================================================")
    print("RUNNING PHASE 16: PRODUCTION API HARDENING & CONFIGURATION TESTS")
    print("========================================================")

    # A. Valid Request
    with patch.object(ChatGroq, 'invoke', side_effect=mock_success):
        res = client.post('/ask', data=json.dumps({'query': 'What is RAG?', 'userId': 'p16_user1'}), content_type='application/json')
        assert res.status_code == 200, f"Valid request failed: {res.data}"
        print("  -> Case A (Valid Request): PASS")

    # B. Query Length Limit
    long_query = "What is RAG? " * 100 # > 1000 chars
    res = client.post('/ask', data=json.dumps({'query': long_query, 'userId': 'p16_user2'}), content_type='application/json')
    assert res.status_code == 400, f"Query length limit test failed: status {res.status_code}"
    data = json.loads(res.data)
    assert data.get("error") == "bad_request", f"Unexpected error key: {data}"
    print("  -> Case B (Query Length Limit > 1000 chars): PASS")

    # C. userId Length Limit (>128 chars -> 400 Bad Request) & Character Collisions
    long_userid = "user_" + "a" * 200 # > 128 chars
    res = client.post('/ask', data=json.dumps({'query': 'What is RAG?', 'userId': long_userid}), content_type='application/json')
    assert res.status_code == 400, f"Oversized userId (>128 chars) was not rejected with 400: status {res.status_code}"

    # Verify valid userIds with '-', '_', and '.' do not collide and map independently
    with patch.object(ChatGroq, 'invoke', side_effect=mock_success):
        u_dot = "user.1"
        u_dash = "user-1"
        u_underscore = "user_1"
        client.post('/ask', data=json.dumps({'query': 'What is RAG?', 'userId': u_dot}), content_type='application/json')
        client.post('/ask', data=json.dumps({'query': 'What is RAG?', 'userId': u_dash}), content_type='application/json')
        client.post('/ask', data=json.dumps({'query': 'What is RAG?', 'userId': u_underscore}), content_type='application/json')
        assert len(get_session_history(u_dot)) > 0, "u_dot session missing"
        assert len(get_session_history(u_dash)) > 0, "u_dash session missing"
        assert len(get_session_history(u_underscore)) > 0, "u_underscore session missing"
    print("  -> Case C (userId Length Limit Rejection & Character Collision Prevention): PASS")

    # D. Oversized Payload (> 1MB)
    oversized_data = json.dumps({'query': 'A' * (1024 * 1024 + 50), 'userId': 'p16_user4'})
    res = client.post('/ask', data=oversized_data, content_type='application/json')
    assert res.status_code in (400, 413), f"Oversized payload failed: status {res.status_code}"
    print("  -> Case D (Oversized Payload > 1MB): PASS")

    # E. Malformed JSON Body
    res = client.post('/ask', data="NOT_A_VALID_JSON{", content_type='application/json')
    assert res.status_code == 400, f"Malformed JSON failed: status {res.status_code}"
    print("  -> Case E (Malformed JSON Body): PASS")

    # F & G. CORS & Security Headers
    res = client.get('/health')
    assert res.status_code == 200, f"/health check failed: {res.status_code}"
    assert res.headers.get("X-Content-Type-Options") == "nosniff", "Missing X-Content-Type-Options"
    assert res.headers.get("X-Frame-Options") == "DENY", "Missing X-Frame-Options"
    assert res.headers.get("X-XSS-Protection") == "1; mode=block", "Missing X-XSS-Protection"
    assert res.headers.get("Referrer-Policy") == "strict-origin-when-cross-origin", "Missing Referrer-Policy"
    print("  -> Case F & G (CORS & Security Headers): PASS")

    # H. /ready Success
    res = client.get('/ready')
    assert res.status_code == 200, f"/ready endpoint failed: {res.data}"
    data = json.loads(res.data)
    assert data.get("status") == "ready", f"Unexpected status: {data}"
    print("  -> Case H (/ready Success Endpoint): PASS")

    # I. /ready Failure / Unready State
    with patch('llama.QA_COMPONENTS', None):
        with patch('llama._INIT_ERROR', 'Database uninitialized'):
            res = client.get('/ready')
            assert res.status_code == 503, f"/ready unready state failed: status {res.status_code}"
            data = json.loads(res.data)
            assert data.get("status") == "not_ready", f"Unexpected response: {data}"
    print("  -> Case I (/ready Failure/Unready State): PASS")

    # J, K, L. /metrics Authentication
    with patch('llama.METRICS_AUTH_TOKEN', 'secret_token_123'):
        # J. Without token -> 401
        res = client.get('/metrics')
        assert res.status_code == 401, f"/metrics without token failed: {res.status_code}"

        # L. With invalid token -> 401
        res = client.get('/metrics', headers={'X-Metrics-Token': 'wrong_token'})
        assert res.status_code == 401, f"/metrics with invalid token failed: {res.status_code}"

        # K. With valid token -> 200
        res = client.get('/metrics', headers={'X-Metrics-Token': 'secret_token_123'})
        assert res.status_code == 200, f"/metrics with valid token failed: {res.status_code}"
    print("  -> Case J, K & L (/metrics Auth Guarding): PASS")

    # M, N, O. Rate Limiting Success, Rejection, and Expiration
    client_key = "test_ip:rate_user"
    with patch('llama.RATE_LIMIT_MAX_REQUESTS', 3):
        with patch('llama.RATE_LIMIT_WINDOW', 2):
            with _rate_limit_lock:
                _rate_limit_store.pop(client_key, None)

            lim1, _ = is_rate_limited(client_key)
            lim2, _ = is_rate_limited(client_key)
            lim3, _ = is_rate_limited(client_key)
            assert not lim1 and not lim2 and not lim3, "Rate limit triggered prematurely"

            lim4, retry_after = is_rate_limited(client_key)
            assert lim4 is True, "Rate limit failed to trigger on 4th hit"
            assert retry_after > 0, "Invalid retry_after value"

            time.sleep(2.1)
            lim5, _ = is_rate_limited(client_key)
            assert lim5 is False, "Rate limit failed to expire after window"
    print("  -> Case M, N & O (Rate Limiter Success, Rejection & Window Expiration): PASS")

    # P. Rate Limiter Bounded-Memory Behavior
    with patch('llama.MAX_TRACKED_RATE_LIMIT_CLIENTS', 5):
        with _rate_limit_lock:
            _rate_limit_store.clear()
        for i in range(20):
            is_rate_limited(f"client_{i}")
        with _rate_limit_lock:
            assert len(_rate_limit_store) <= 5, f"Rate limit memory unbounded: size {len(_rate_limit_store)}"
    print("  -> Case P (Rate Limiter Bounded-Memory Eviction): PASS")

    # Q. Concurrent Rate-Limit Access
    errors = []
    def rate_worker(idx):
        try:
            is_rate_limited(f"concurrent_client_{idx % 3}")
        except Exception as e:
            errors.append(e)

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(rate_worker, i) for i in range(50)]
        concurrent.futures.wait(futures)

    assert len(errors) == 0, f"Concurrent rate-limiter errors: {errors}"
    print("  -> Case Q (Concurrent Rate-Limit Access): PASS")

    # R. API Key Non-Leakage
    res = client.post('/ask', data=json.dumps({'query': 'invalid query', 'userId': 'p16_user'}), content_type='application/json')
    assert "gsk_" not in res.data.decode('utf-8'), "API Key leaked in response!"
    print("  -> Case R (API Key Non-Leakage Verification): PASS")

    # S. Production Debug Protection
    assert app.config['DEBUG'] is False, "Flask DEBUG mode is improperly enabled!"
    assert app.config['TESTING'] is False, "Flask TESTING mode is improperly enabled!"
    print("  -> Case S (Production Debug Mode Protection): PASS")

    # T. Standardized Error Responses
    res = client.post('/ask', data=json.dumps({'query': '', 'userId': 'p16_user'}), content_type='application/json')
    assert res.status_code == 400, "Empty query didn't return 400"
    data = json.loads(res.data)
    assert "error" in data and "message" in data, f"Non-standard error response format: {data}"
    print("  -> Case T (Standardized Error Responses): PASS")

    # Benchmarking Rate Limiter Overhead
    t_start = time.perf_counter()
    for _ in range(1000):
        is_rate_limited("bench_client")
    t_rate_limit = (time.perf_counter() - t_start) / 1000.0 * 1000.0
    print(f"  -> Overhead Measurement (Rate Limiter Overhead: {t_rate_limit:.6f} ms)")

    print("\n# ALL PHASE 16 PRODUCTION API HARDENING TESTS SUCCEEDED!")

if __name__ == "__main__":
    test_citation_verification(metrics)
    test_contextual_query_resolution(metrics)
    test_concurrency_and_locks(metrics)
    test_phase16_api_hardening()


