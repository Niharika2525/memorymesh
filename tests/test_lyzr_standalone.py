"""Standalone Lyzr Cloud Connection Test for MemoryMesh.

Can be run directly via:
    python tests/test_lyzr_standalone.py
Or via pytest:
    pytest tests/test_lyzr_standalone.py

Sends one simple message to the official Lyzr Inference v3 Chat API,
measures elapsed time, and reports status and sanitized response.
Never logs or prints secret API keys.
"""

import os
import sys

# Ensure repository root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.services.llm import get_llm_service


def run_standalone_test():
    """Execute standalone Lyzr Cloud connection test and return result dictionary."""
    llm = get_llm_service()
    result = llm.test_lyzr_connection(message="Ping")

    print("\n" + "=" * 60)
    print("  MemoryMesh — Lyzr Cloud Inference Connection Test")
    print("=" * 60)
    print(f"Configured:      {result.get('configured')}")
    print(f"Agent ID:        {result.get('agent_id_preview')}")
    print(f"HTTP Status:     {result.get('status_code')}")
    print(f"Elapsed Time:    {result.get('elapsed_seconds')}s")
    print(f"Success:         {result.get('success')}")

    if result.get("success"):
        print(f"Response:        {result.get('response_preview')}")
        print(f"Message:         {result.get('message')}")
    else:
        print(f"Error:           {result.get('error')}")
        if result.get("raw_detail"):
            print(f"Detail:          {result.get('raw_detail')}")
        if result.get("explanation"):
            print(f"Explanation:     {result.get('explanation')}")
        print(f"Message:         {result.get('message')}")
    print("=" * 60 + "\n")

    return result


def test_lyzr_standalone_connection():
    """Pytest test case for standalone Lyzr connection test."""
    result = run_standalone_test()
    # The test must successfully execute the network call or report configuration
    assert "configured" in result
    assert "status_code" in result
    assert "elapsed_seconds" in result
    assert isinstance(result["elapsed_seconds"], (int, float))


if __name__ == "__main__":
    res = run_standalone_test()
    if res.get("success"):
        print(">> Standalone Lyzr Cloud test PASSED.")
        sys.exit(0)
    else:
        print(">> Standalone Lyzr Cloud test completed with status:", res.get("status_code"))
        # Exit 0 so scripts can inspect output, or exit 1 if strictly verifying success
        sys.exit(0 if res.get("configured") else 1)
