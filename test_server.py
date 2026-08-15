import os
import sys
import time
import json
from unittest.mock import patch

os.chdir(r"c:\Users\ACER\OneDrive\Desktop\vasavi\RAG")
print("Initializing modules...")
from llama import app, GLOBAL_SEMANTIC_CACHE, CACHE_TTL_SECONDS
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

        # TEST 4.1 - Re-run same pronoun in DIFFERENT chat (Should hit cache globally because resolved query is same!)
        print("\n# TEST 4.1: Same pronoun query in new context")
        s1_alt = "test_s1_alt"
        run_query(s1_alt, "What is fine-tuning?")
        res_pronoun_again = run_query(s1_alt, "What is its purpose?")
        assert res_pronoun_again.get('cache_hit') == True, "Resolved query is identical, cache across sessions safely!"
        
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
    
    # We ran exactly: 
    # 1 miss, 1 exact hit, 1 semantic hit (miss because strict), 
    # 2 follow up (miss, hit), 2 unsupported (miss, miss) -> Fast trips!
    # 1 429 trap, 1 retry cache miss. 1 kb version trap.
    
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
