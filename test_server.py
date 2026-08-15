import os
import sys
import time
import json
from unittest.mock import patch

os.chdir(r"c:\Users\ACER\OneDrive\Desktop\vasavi\RAG")
print("Initializing modules...")
from llama import app
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
    
    chars = data.get('history_char_count')
    turns = data.get('generation_history_turns')
    
    print(f"   Resolved Query : {resolved} (History Modified: {history_used})")
    print(f"   Grounded       : {grounded}")
    print(f"   Result         : {data.get('result', '')[:80].replace(chr(10), ' ')}...")
    if chars is not None:
        print(f"   [Diagnostic] Generation Char Count: {chars}")
        print(f"   [Diagnostic] Iterative Turns Kept: {turns}")
    
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

if __name__ == "__main__":
    print("--- PHASE 4 & PHASE 9 TESTS ---")
    
    # TEST 1 - Standalone
    print("\n# TEST 1: Standalone")
    s1 = "test_s1"
    run_query(s1, "What is RAG and how does it work?")
    
    # TEST 2 - Follow-up
    print("\n# TEST 2: Follow-up")
    run_query(s1, "What is fine-tuning?")
    run_query(s1, "What is its purpose?")
    
    # TEST 3 - Pronoun Follow-up
    print("\n# TEST 3: Pronouns")
    s3 = "test_s3"
    run_query(s3, "What are pretrained weights?")
    run_query(s3, "Why are they important?")
    
    # TEST 4 - Context continuation
    print("\n# TEST 4: Context Continuation")
    s4 = "test_s4"
    run_query(s4, "What is RAG?")
    run_query(s4, "What are its main steps?")
    
    # TEST 5 - New Chat isolation
    print("\n# TEST 5: New Chat Isolation")
    s5_a = "test_s5_A"
    run_query(s5_a, "What is fine-tuning?")
    reset_session(s5_a)
    s5_b = "test_s5_B"
    run_query(s5_b, "What is its purpose?")
    
    # TEST 6 - Unsupported query
    print("\n# TEST 6: Unsupported query")
    run_query(s5_b, "What is the capital of France?")
    
    # TEST 7 - Simulated Groq HTTP 429
    print("\n# TEST 7: Simulated Groq HTTP 429")
    s7 = "test_s7"
    run_query(s7, "Explain embeddings.")
    
    from langchain_groq import ChatGroq
    with patch.object(ChatGroq, 'invoke', side_effect=mock_429):
        run_query(s7, "What is prompt engineering?")

    # TEST 8 - After simulated 429, follow-up
    print("\n# TEST 8: Recovery Follow-Up (State Preservation)")
    run_query(s7, "How are they created?")
    
    # TEST 9 - Simulated Generic Failure
    print("\n# TEST 9: Simulated Generic LLM failure")
    s9 = "test_s9"
    run_query(s9, "Explain fine tuning.")
    with patch.object(ChatGroq, 'invoke', side_effect=mock_500):
        # We trigger a full 500 error natively mapped
        run_query(s9, "What is the typical timeframe?")
        
    print("\n# DONE")
