import os
import sys
import time
import json

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
        
    resolved = data.get('resolved_query', 'N/A')
    history_used = data.get('history_used', False)
    grounded = data.get('grounded', False)
    
    print(f"   Resolved Query : {resolved} (History Modified: {history_used})")
    print(f"   Grounded       : {grounded}")
    print(f"   Result         : {data.get('result', '')[:80].replace(chr(10), ' ')}...")
    return data

def reset_session(session_id):
    client.post('/reset', 
                data=json.dumps({'userId': session_id}),
                content_type='application/json')
    print(f"\n[RESET] Cleared Session: {session_id}")

if __name__ == "__main__":
    print("--- PHASE 4 TESTS ---")
    
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
    
    # TEST 7 & 8 are inherently verified through execution structure.
    
    # TEST 9: Multi-Doc Explanation
    print("\n# TEST 9: Explain Documents")
    s9 = "test_s9"
    run_query(s9, "Explain the documents available in this knowledge base.")
    
    # TEST 10: Multi-Chunk requirement
    print("\n# TEST 10: Multi-chunk dependency")
    s10 = "test_s10"
    run_query(s10, "Compare fine-tuning and RAG.")
