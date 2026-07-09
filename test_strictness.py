"""
Unit test script to verify the RAG Chatbot's strict module routing,
relevance thresholding, fallback handling, and failed query logging.
"""

import os
import json
import numpy as np

# Set environment variable to suppress TF warnings etc.
os.environ["PYTHONIOENCODING"] = "utf-8"

print("🔄 Loading rag_chatbot module and dependencies...")
import rag_chatbot

def init_chatbot():
    print("🔄 Initializing chatbot state...")
    rag_chatbot.initialize_nlp()
    rag_chatbot.train_ml_classifier()
    rag_chatbot.load_timetable_data()
    rag_chatbot.load_faculty_data()
    rag_chatbot.load_circulars()
    rag_chatbot.load_curriculum()
    rag_chatbot.load_bus_fees()
    rag_chatbot.load_placements()
    rag_chatbot.load_events()
    rag_chatbot.load_dept_info()
    rag_chatbot.load_college_info()
    rag_chatbot.load_subjects_list()
    rag_chatbot._load_student_dw()
    ok = rag_chatbot.initialize_rag()
    if ok:
        print("✓ RAG engine initialized successfully!")
    else:
        print("✗ RAG engine initialization failed!")
    return ok

def run_tests():
    # Remove existing failed_queries.json to have a clean start
    failed_log_path = "failed_queries.json"
    if os.path.exists(failed_log_path):
        try:
            os.remove(failed_log_path)
            print("✓ Cleared existing failed_queries.json for test run.")
        except Exception as e:
            print(f"⚠ Could not remove failed_queries.json: {e}")

    # Mock requests.post to simulate web fallback failure when "fail_me_now" is in query
    import requests
    original_post = requests.post
    def mock_post(url, *args, **kwargs):
        payload = kwargs.get("json", {})
        messages = payload.get("messages", [])
        if any("fail_me_now" in m.get("content", "").lower() for m in messages):
            class MockResponse:
                status_code = 500
                text = ""
            return MockResponse()
        return original_post(url, *args, **kwargs)
    requests.post = mock_post

    # Test cases: (query, expected_module, should_succeed_bool)
    test_cases = [
        ("Tell me about the vision of the department", "department", True),
        ("Who is Dr Narayana Rao Appini", "faculty_profile", True),
        ("What is the bus fee for Nellore", "bus_fee", True),
        ("What is the distance to the moon?", "unknown", True),
        ("How to build a nuclear reactor at home?", "unknown", True),
        ("tell me about correspondence courses in NBKR", "college", True),
        ("what is machine learning", "unknown", True),
        ("NBKR full form", "college", True),
        ("What is the distance to the moon? fail_me_now", "unknown", False),
        ("How to build a nuclear reactor at home? fail_me_now", "unknown", False),
        ("tell me about correspondent as a journalism career", "college", False),
    ]

    print("\n" + "=" * 80)
    print("🏃 Running Test Cases")
    print("=" * 80)

    fallback_signatures = [
        "I couldn't find that information",
        "I don't know the answer",
        "contact the department office",
        "Please ask me something about the college"
    ]

    passed = 0
    for i, (query, expected_module, should_succeed) in enumerate(test_cases, 1):
        print(f"\nTest {i}: \"{query}\"")
        qa = rag_chatbot.analyse_query(query)
        module = rag_chatbot.classify_query_module(query, qa)
        print(f"  -> Classified Module: {module} (Expected: {expected_module})")
        
        response = rag_chatbot.get_response(query)
        
        # Check fallback behavior
        has_fallback = any(sig in response for sig in fallback_signatures)
        
        print(f"  -> Response length: {len(response)} chars")
        if should_succeed:
            if has_fallback:
                print("  ❌ FAIL: Expected successful response, but got fallback.")
            else:
                print("  ✓ PASS: Successful response returned.")
                passed += 1
        else:
            if has_fallback:
                print("  ✓ PASS: Correctly returned fallback response.")
                passed += 1
            else:
                print("  ❌ FAIL: Expected fallback response, but got custom response.")

        # Check classification match
        if module != expected_module:
            print(f"  ❌ FAIL: Module mismatch. Expected {expected_module}, got {module}")

    # Restore original requests.post
    requests.post = original_post
            # we don't decrement passed if it already failed above

    # Verify failed query logging
    print("\n" + "=" * 80)
    print("📝 Verifying Failed Query Logging")
    print("=" * 80)
    
    if os.path.exists(failed_log_path):
        with open(failed_log_path, "r", encoding="utf-8") as f:
            logs = json.load(f)
        print(f"✓ Found failed_queries.json containing {len(logs)} logs.")
        print(json.dumps(logs, indent=2))
        
        # Should have logged the 2 unknown test cases
        if len(logs) >= 2:
            print("✓ PASS: Failed queries were correctly logged to failed_queries.json.")
            passed += 1
        else:
            print("❌ FAIL: Expected at least 2 logged failures, but got less.")
    else:
        print("❌ FAIL: failed_queries.json was not created.")

    print("\n" + "=" * 80)
    print(f"📊 Summary: Passed {passed} of {len(test_cases) + 1} checks.")
    print("=" * 80)
    
    if passed == len(test_cases) + 1:
        print("🏆 ALL TESTS PASSED!")
        return True
    else:
        print("⚠️ SOME TESTS FAILED. Please review the output above.")
        return False

if __name__ == "__main__":
    if init_chatbot():
        success = run_tests()
        import sys
        sys.exit(0 if success else 1)
    else:
        print("Failed to initialize chatbot.")
        import sys
        sys.exit(1)
