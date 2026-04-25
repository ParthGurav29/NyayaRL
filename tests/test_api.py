"""
Test script for NyayaRL FastAPI server.

Tests all endpoints: /health, /reset, /step

Usage:
    # Start server first:
    uvicorn app:app --reload --port 8000

    # Then run tests:
    python tests/test_api.py
"""

import requests
import json

BASE_URL = "http://localhost:8000"
SESSION_ID = "test_session_001"


def test_health():
    """Test /health endpoint."""
    print("=" * 60)
    print("TEST 1: Health Check")
    print("=" * 60)

    response = requests.get(f"{BASE_URL}/health")

    if response.status_code == 200:
        data = response.json()
        print(f"Status: {data.get('status')}")
        print("[PASS] Health check PASSED\n")
        return True
    else:
        print(f"✗ Health check FAILED: {response.status_code}")
        print(f"Response: {response.text}")
        return False


def test_reset():
    """Test /reset endpoint."""
    print("=" * 60)
    print("TEST 2: Reset Episode")
    print("=" * 60)

    payload = {
        "session_id": SESSION_ID,
        "curriculum_level": 1
    }

    response = requests.post(
        f"{BASE_URL}/reset",
        json=payload,
        headers={"Content-Type": "application/json"}
    )

    if response.status_code == 200:
        data = response.json()
        print(f"Case ID: {data.get('case_file', {}).get('case_id')}")
        print(f"Accused Count: {data.get('case_file', {}).get('accused_count')}")
        print(f"Evidence Items: {len(data.get('case_file', {}).get('evidence_items', []))}")
        print(f"Witnesses: {len(data.get('case_file', {}).get('witness_statements', []))}")
        print(f"IPC Sections: {data.get('case_file', {}).get('applicable_ipc_sections')}")
        print(f"Curriculum Level: {data.get('curriculum_level')}")
        print(f"Is Done: {data.get('is_done')}")
        print("[PASS] Reset PASSED\n")
        return True
    else:
        print(f"✗ Reset FAILED: {response.status_code}")
        print(f"Response: {response.text}")
        return False


def test_step_actus_reus():
    """Test /step endpoint - Step 1: Actus Reus."""
    print("=" * 60)
    print("TEST 3: Step 1 - Actus Reus")
    print("=" * 60)

    payload = {
        "session_id": SESSION_ID,
        "step_type": "actus_reus",
        "anchored_evidence_ids": ["E1"]
    }

    response = requests.post(
        f"{BASE_URL}/step",
        json=payload,
        headers={"Content-Type": "application/json"}
    )

    if response.status_code == 200:
        data = response.json()
        step_result = data.get('step_result', {})
        print(f"Is Valid: {step_result.get('is_valid')}")
        print(f"Reward: {step_result.get('reward')}")
        print(f"Failure Reason: {step_result.get('failure_reason')}")
        print(f"Done: {data.get('done')}")

        if step_result.get('is_valid'):
            print("[PASS] Step 1 (Actus Reus) PASSED\n")
            return True
        else:
            print(f"✗ Step 1 FAILED: {step_result.get('failure_reason')}\n")
            return False
    else:
        print(f"✗ Step 1 FAILED: {response.status_code}")
        print(f"Response: {response.text}")
        return False


def test_step_mens_rea():
    """Test /step endpoint - Step 2: Mens Rea."""
    print("=" * 60)
    print("TEST 4: Step 2 - Mens Rea")
    print("=" * 60)

    payload = {
        "session_id": SESSION_ID,
        "step_type": "mens_rea",
        "anchored_witness_ids": ["W1"]
    }

    response = requests.post(
        f"{BASE_URL}/step",
        json=payload,
        headers={"Content-Type": "application/json"}
    )

    if response.status_code == 200:
        data = response.json()
        step_result = data.get('step_result', {})
        print(f"Is Valid: {step_result.get('is_valid')}")
        print(f"Reward: {step_result.get('reward')}")
        print(f"Prosecution Challenge: {step_result.get('prosecution_challenge')}")
        print(f"Done: {data.get('done')}")

        if step_result.get('is_valid'):
            print("[PASS] Step 2 (Mens Rea) PASSED\n")
            return True
        else:
            print(f"✗ Step 2 FAILED: {step_result.get('failure_reason')}\n")
            return False
    else:
        print(f"✗ Step 2 FAILED: {response.status_code}")
        print(f"Response: {response.text}")
        return False


def test_invalid_step():
    """Test /step endpoint - Invalid step type."""
    print("=" * 60)
    print("TEST 5: Invalid Step Type")
    print("=" * 60)

    # Start a new session for this test
    invalid_session = "test_invalid_001"

    # First reset
    requests.post(f"{BASE_URL}/reset", json={"session_id": invalid_session, "curriculum_level": 1})

    # Try to submit wrong step type (should be actus_reus first, not mens_rea)
    payload = {
        "session_id": invalid_session,
        "step_type": "mens_rea",  # Wrong! Should be actus_reus first
        "anchored_witness_ids": ["W1"]
    }

    response = requests.post(
        f"{BASE_URL}/step",
        json=payload,
        headers={"Content-Type": "application/json"}
    )

    if response.status_code == 400:
        print(f"Expected 400 error received")
        print(f"Error Detail: {response.json().get('detail', '')}")
        print("[PASS] Invalid step correctly rejected\n")
        return True
    else:
        print(f"✗ Invalid step test FAILED: Expected 400, got {response.status_code}")
        return False


def test_session_not_found():
    """Test /step with non-existent session."""
    print("=" * 60)
    print("TEST 6: Session Not Found")
    print("=" * 60)

    payload = {
        "session_id": "non_existent_session_xyz",
        "step_type": "actus_reus",
        "anchored_evidence_ids": ["E1"]
    }

    response = requests.post(
        f"{BASE_URL}/step",
        json=payload,
        headers={"Content-Type": "application/json"}
    )

    if response.status_code == 400:
        print(f"Expected 400 error received")
        print(f"Error Detail: {response.json().get('detail', '')}")
        print("[PASS] Non-existent session correctly rejected\n")
        return True
    else:
        print(f"✗ Session not found test FAILED: Expected 400, got {response.status_code}")
        return False


def test_full_episode():
    """Test complete 6-step episode."""
    print("=" * 60)
    print("TEST 7: Full Episode (6 steps)")
    print("=" * 60)

    session_id = "test_full_episode"

    # Reset
    requests.post(f"{BASE_URL}/reset", json={"session_id": session_id, "curriculum_level": 1})

    steps = [
        {"step_type": "actus_reus", "anchored_evidence_ids": ["E1"]},
        {"step_type": "mens_rea", "anchored_witness_ids": ["W1"]},
        {"step_type": "linkage", "anchored_evidence_ids": ["E2"], "cited_ipc_sections": ["302"]},
        {"step_type": "counter_argument", "anchored_evidence_ids": ["E1"]},
        {"step_type": "ipc_application", "cited_ipc_sections": ["302"]},
        {"step_type": "precedent_citation", "anchored_evidence_ids": ["ILDC_STUB_001"], "judgment": "convict"},
    ]

    all_passed = True
    for i, step_data in enumerate(steps, 1):
        payload = {"session_id": session_id, **step_data}
        response = requests.post(f"{BASE_URL}/step", json=payload)

        if response.status_code == 200:
            data = response.json()
            step_result = data.get('step_result', {})
            done = data.get('done')
            print(f"Step {i}: valid={step_result.get('is_valid')}, reward={step_result.get('reward')}, done={done}")

            if done and i == 6:
                print(f"Episode completed successfully!")
                print(f"Total reward would include terminal bonus")
        else:
            print(f"Step {i} FAILED: {response.status_code} - {response.text}")
            all_passed = False
            break

    if all_passed:
        print("[PASS] Full Episode PASSED\n")
    else:
        print("✗ Full Episode FAILED\n")

    return all_passed


def main():
    """Run all API tests."""
    print("\n" + "=" * 60)
    print("NYAYARL API TEST SUITE")
    print("Testing FastAPI Server Endpoints")
    print("=" * 60 + "\n")

    # Check if server is running first
    print("Checking if server is running...")
    try:
        requests.get(f"{BASE_URL}/health", timeout=2)
    except requests.exceptions.ConnectionError:
        print("\n" + "!" * 60)
        print("ERROR: Server is not running!")
        print(f"Start server with: uvicorn app:app --reload --port 8000")
        print("!" * 60 + "\n")
        return 1

    tests = [
        test_health,
        test_reset,
        test_step_actus_reus,
        test_step_mens_rea,
        test_invalid_step,
        test_session_not_found,
        test_full_episode,
    ]

    results = []
    for test_fn in tests:
        try:
            passed = test_fn()
            results.append((test_fn.__doc__, passed))
        except Exception as e:
            print(f"\n[FAIL] {test_fn.__doc__}: {e}\n")
            results.append((test_fn.__doc__, False))

    # Summary
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, p in results if p)
    total = len(results)

    for name, result in results:
        status = "PASSED" if result else "FAILED"
        print(f"  {name.strip()}: {status}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n=== ALL API TESTS PASSED ===")
        print("Server is ready for Docker deployment!")
        return 0
    else:
        print("\n=== SOME TESTS FAILED ===")
        return 1


if __name__ == "__main__":
    exit(main())
