"""
Integration tests with REAL OpenAI API calls.

These tests require a valid OPENAI_API_KEY environment variable.
They make actual API calls to OpenAI and test the full system end-to-end.

To run:
    export OPENAI_API_KEY=sk-your-key-here
    pytest tests/test_integration.py -v

Or set in .env file:
    OPENAI_API_KEY=sk-your-key-here
"""

import pytest
import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Check if API key is available
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
REQUIRES_API_KEY = pytest.mark.skipif(
    not OPENAI_API_KEY or OPENAI_API_KEY == "test-key",
    reason="OPENAI_API_KEY not set or is test key. Set a real API key to run integration tests."
)

API_URL = "http://127.0.0.1:8000"


@pytest.fixture(scope="module")
def server_running():
    """Check if server is running"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        if response.status_code == 200:
            return True
    except:
        pass
    
    pytest.skip("Server is not running. Start it with: cd app && python main.py")


@REQUIRES_API_KEY
def test_example_1_pricing_question(server_running):
    """
    Example 1: Basic KB question (pricing)
    Task: "Give me the pricing model at a high level and what can change the quote."
    Expected: search_kb → final answer referencing KB-006
    """
    payload = {
        "task": "Give me the pricing model at a high level and what can change the quote.",
        "language": "en"
    }
    
    response = requests.post(
        f"{API_URL}/v1/agent/run",
        json=payload,
        timeout=120
    )
    
    assert response.status_code == 200, f"Request failed: {response.text}"
    data = response.json()
    
    # Verify response structure
    assert "trace_id" in data
    assert "final_answer" in data
    assert "tool_calls" in data
    assert "metrics" in data
    
    # Verify search_kb was called
    tool_calls = data["tool_calls"]
    assert len(tool_calls) > 0, "No tool calls made"
    
    search_kb_calls = [tc for tc in tool_calls if tc["name"] == "search_kb"]
    assert len(search_kb_calls) > 0, "search_kb was not called"
    
    # Verify KB-006 is in the results or answer
    kb_results = search_kb_calls[0]["result"].get("results", [])
    kb_ids = [r.get("id") for r in kb_results]
    answer_lower = data["final_answer"].lower()
    
    assert "KB-006" in kb_ids or "kb-006" in answer_lower or "pricing" in answer_lower, \
        f"KB-006 (pricing) not found. Found KBs: {kb_ids}, Answer: {data['final_answer'][:100]}"
    
    print(f"\n[OK] Test 1 passed!")
    print(f"   Trace ID: {data['trace_id']}")
    print(f"   Tool calls: {len(tool_calls)}")
    print(f"   Answer preview: {data['final_answer'][:150]}...")


@REQUIRES_API_KEY
def test_example_2_crm_writeback(server_running):
    """
    Example 2: Integration question (CRM writeback)
    Task: "How does CRM writeback work and how long does it take to set up?"
    Expected: search_kb → final answer referencing KB-004 (and possibly KB-018)
    """
    payload = {
        "task": "How does CRM writeback work and how long does it take to set up?",
        "language": "en"
    }
    
    response = requests.post(
        f"{API_URL}/v1/agent/run",
        json=payload,
        timeout=120
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # Verify search_kb was called
    tool_calls = data["tool_calls"]
    search_kb_calls = [tc for tc in tool_calls if tc["name"] == "search_kb"]
    assert len(search_kb_calls) > 0, "search_kb was not called"
    
    # Verify KB-004 is referenced
    kb_results = search_kb_calls[0]["result"].get("results", [])
    kb_ids = [r.get("id") for r in kb_results]
    answer_lower = data["final_answer"].lower()
    
    assert "KB-004" in kb_ids or "kb-004" in answer_lower or "writeback" in answer_lower, \
        f"KB-004 (CRM writeback) not found. Found KBs: {kb_ids}"
    
    print(f"\n[OK] Test 2 passed!")
    print(f"   Answer preview: {data['final_answer'][:150]}...")


@REQUIRES_API_KEY
def test_example_3_troubleshooting_and_ticket(server_running):
    """
    Example 3: Troubleshooting leading to a ticket
    Task: "We're failing to write back to HubSpot since this morning. What should we check and can you open a high priority ticket for ops?"
    Expected: search_kb (KB-015 + KB-016) → create_ticket(priority=high) → final answer includes ticket id and checklist
    """
    payload = {
        "task": "We're failing to write back to HubSpot since this morning. What should we check and can you open a high priority ticket for ops?",
        "language": "en"
    }
    
    response = requests.post(
        f"{API_URL}/v1/agent/run",
        json=payload,
        timeout=120
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # Verify both search_kb and create_ticket were called
    tool_calls = data["tool_calls"]
    tool_names = [tc["name"] for tc in tool_calls]
    
    assert "search_kb" in tool_names, "search_kb was not called"
    assert "create_ticket" in tool_names, "create_ticket was not called"
    
    # Verify create_ticket has priority=high
    ticket_calls = [tc for tc in tool_calls if tc["name"] == "create_ticket"]
    assert len(ticket_calls) > 0
    ticket_args = ticket_calls[0]["arguments"]
    assert ticket_args.get("priority", "").lower() == "high", \
        f"Ticket priority should be 'high', got: {ticket_args.get('priority')}"
    
    # Verify ticket was created (has ticket_id in result)
    ticket_result = ticket_calls[0]["result"]
    assert "ticket_id" in ticket_result, "Ticket was not created"
    
    # Verify answer mentions ticket
    answer_lower = data["final_answer"].lower()
    assert "ticket" in answer_lower or ticket_result["ticket_id"].lower() in answer_lower, \
        "Answer should mention the created ticket"
    
    print(f"\n[OK] Test 3 passed!")
    print(f"   Ticket ID: {ticket_result.get('ticket_id')}")
    print(f"   Answer preview: {data['final_answer'][:150]}...")


@REQUIRES_API_KEY
def test_example_4_schedule_followup(server_running):
    """
    Example 4: Schedule follow-up
    Task: "Schedule a follow-up call with John tomorrow at 10:30 CET via WhatsApp to discuss custom SLA."
    Expected: schedule_followup (use ISO) + search_kb (KB-017) → final answer with followup id and a short note on custom SLA
    """
    payload = {
        "task": "Schedule a follow-up call with John tomorrow at 10:30 CET via WhatsApp to discuss custom SLA.",
        "language": "en"
    }
    
    response = requests.post(
        f"{API_URL}/v1/agent/run",
        json=payload,
        timeout=120
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # Verify schedule_followup and search_kb were called
    tool_calls = data["tool_calls"]
    tool_names = [tc["name"] for tc in tool_calls]
    
    assert "schedule_followup" in tool_names, "schedule_followup was not called"
    assert "search_kb" in tool_names, "search_kb was not called"
    
    # Verify schedule_followup parameters
    followup_calls = [tc for tc in tool_calls if tc["name"] == "schedule_followup"]
    assert len(followup_calls) > 0
    followup_args = followup_calls[0]["arguments"]
    
    assert followup_args.get("contact", "").lower() == "john", \
        f"Contact should be 'John', got: {followup_args.get('contact')}"
    assert followup_args.get("channel", "").lower() == "whatsapp", \
        f"Channel should be 'whatsapp', got: {followup_args.get('channel')}"
    assert "datetime_iso" in followup_args, "datetime_iso should be provided"
    
    # Verify followup was scheduled
    followup_result = followup_calls[0]["result"]
    assert "followup_id" in followup_result, "Follow-up was not scheduled"
    
    # Verify search_kb was called and references KB-017
    search_kb_calls = [tc for tc in tool_calls if tc["name"] == "search_kb"]
    assert len(search_kb_calls) > 0, "search_kb was not called"
    kb_results = search_kb_calls[0]["result"].get("results", [])
    kb_ids = [r.get("id") for r in kb_results]
    answer_lower = data["final_answer"].lower()
    
    assert "KB-017" in kb_ids or "kb-017" in answer_lower or "custom sla" in answer_lower, \
        f"KB-017 (custom SLA) should be referenced. Found KBs: {kb_ids}, Answer: {data['final_answer'][:200]}"
    
    # Verify answer mentions follow-up
    assert "follow" in answer_lower or "john" in answer_lower, \
        "Answer should mention the scheduled follow-up"
    
    print(f"\n[OK] Test 4 passed!")
    print(f"   Follow-up ID: {followup_result.get('followup_id')}")
    print(f"   Answer preview: {data['final_answer'][:150]}...")


@REQUIRES_API_KEY
def test_example_5_spanish_request(server_running):
    """
    Example 5: Spanish request (language handling + onboarding)
    Task: "¿En qué idiomas funciona y qué incluye el onboarding?"
    Expected: search_kb (KB-002 + KB-007) → answer in Spanish
    """
    payload = {
        "task": "¿En qué idiomas funciona y qué incluye el onboarding?",
        "language": "Spanish"
    }
    
    response = requests.post(
        f"{API_URL}/v1/agent/run",
        json=payload,
        timeout=120
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # Verify search_kb was called
    tool_calls = data["tool_calls"]
    search_kb_calls = [tc for tc in tool_calls if tc["name"] == "search_kb"]
    assert len(search_kb_calls) > 0, "search_kb was not called"
    
    # Verify KB-002 and KB-007 are referenced
    kb_results = search_kb_calls[0]["result"].get("results", [])
    kb_ids = [r.get("id") for r in kb_results]
    answer_lower = data["final_answer"].lower()
    
    # Check for language-related KB (KB-002) and onboarding KB (KB-007)
    has_language_kb = "KB-002" in kb_ids or "kb-002" in answer_lower
    has_onboarding_kb = "KB-007" in kb_ids or "kb-007" in answer_lower or "onboarding" in answer_lower
    
    assert has_language_kb or has_onboarding_kb, \
        f"KB-002 or KB-007 not found. Found KBs: {kb_ids}"
    
    # Verify answer is in Spanish (contains Spanish words)
    spanish_words = ["español", "inglés", "idiomas", "incluye", "funciona", "qué", "según"]
    has_spanish = any(word in answer_lower for word in spanish_words)
    
    assert has_spanish, \
        f"Answer should be in Spanish. Answer: {data['final_answer'][:200]}"
    
    print(f"\n[OK] Test 5 passed!")
    print(f"   Answer preview: {data['final_answer'][:150]}...")


@REQUIRES_API_KEY
def test_health_endpoint(server_running):
    """Test health endpoint"""
    response = requests.get(f"{API_URL}/health", timeout=5)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    print("\n[OK] Health check passed!")


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v", "-s"])

