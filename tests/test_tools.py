import pytest
from app.tools import search_kb, create_ticket, schedule_followup


def test_search_kb_basic():
    """Test basic KB search functionality"""
    result = search_kb("pricing", top_k=3)
    
    assert "results" in result
    assert isinstance(result["results"], list)
    assert len(result["results"]) > 0
    
    # Should find KB-006 (pricing)
    found_pricing = any(r["id"] == "KB-006" for r in result["results"])
    assert found_pricing, "Should find KB-006 (pricing) entry"
    
    # Results should have required fields
    for entry in result["results"]:
        assert "id" in entry
        assert "title" in entry
        assert "score" in entry
        assert "snippet" in entry
        assert "tags" in entry


def test_search_kb_with_filters():
    """Test KB search with filters"""
    # Search with tag filter
    result = search_kb("pricing", top_k=5, filters={"tags": ["pricing"]})
    
    assert "results" in result
    # All results should have "pricing" in tags
    for entry in result["results"]:
        assert "pricing" in [tag.lower() for tag in entry["tags"]]
    
    # Search with audience filter
    result = search_kb("onboarding", top_k=5, filters={"audience": "customer"})
    
    assert "results" in result
    # All results should be for customer audience
    for entry in result["results"]:
        # Note: We can't directly check audience in results, but we can verify
        # that the search still works with filters
        assert "id" in entry


def test_search_kb_top_k_limit():
    """Test that top_k is respected"""
    result = search_kb("integration", top_k=2)
    
    assert len(result["results"]) <= 2


def test_create_ticket():
    """Test ticket creation"""
    result = create_ticket(
        title="Test ticket",
        body="This is a test ticket",
        priority="high"
    )
    
    assert "ticket_id" in result
    assert "status" in result
    assert result["status"] == "created"
    assert result["ticket_id"].startswith("TICK-")


def test_create_ticket_invalid_priority():
    """Test ticket creation with invalid priority"""
    with pytest.raises(ValueError, match="Invalid priority"):
        create_ticket(
            title="Test",
            body="Test",
            priority="invalid"
        )


def test_schedule_followup():
    """Test follow-up scheduling"""
    result = schedule_followup(
        datetime_iso="2025-12-15T10:30:00+01:00",
        contact="test@example.com",
        channel="email"
    )
    
    assert "scheduled" in result
    assert "followup_id" in result
    assert result["scheduled"] is True
    assert result["followup_id"].startswith("FUP-")


def test_schedule_followup_invalid_channel():
    """Test follow-up scheduling with invalid channel"""
    with pytest.raises(ValueError, match="Invalid channel"):
        schedule_followup(
            datetime_iso="2025-12-15T10:30:00+01:00",
            contact="test@example.com",
            channel="invalid"
        )


def test_schedule_followup_invalid_datetime():
    """Test follow-up scheduling with invalid datetime"""
    with pytest.raises(ValueError, match="Invalid datetime format"):
        schedule_followup(
            datetime_iso="invalid-datetime",
            contact="test@example.com",
            channel="email"
        )


def test_no_web_browsing_tools():
    """Verify that no web browsing tools are defined (OpenAI requirement)"""
    from app.tools import get_tool_definitions
    
    tools = get_tool_definitions()
    tool_names = [t["function"]["name"] for t in tools]
    
    # Verify only the 3 required local tools exist
    expected_tools = {"search_kb", "create_ticket", "schedule_followup"}
    assert set(tool_names) == expected_tools, f"Expected exactly {expected_tools}, got {set(tool_names)}"
    
    # Verify no web browsing tools
    web_browsing_keywords = ["web", "browse", "search_web", "internet", "url", "http"]
    for tool_name in tool_names:
        tool_name_lower = tool_name.lower()
        for keyword in web_browsing_keywords:
            assert keyword not in tool_name_lower, f"Tool '{tool_name}' contains web browsing keyword '{keyword}'"
    
    # Verify tool descriptions don't mention web browsing
    for tool in tools:
        description = tool["function"].get("description", "").lower()
        assert "web" not in description or "web browsing" not in description, \
            f"Tool '{tool['function']['name']}' description mentions web browsing"


def test_tools_are_local_only():
    """Verify that all tools operate on local data only (no external API calls)"""
    from app.tools import get_tool_definitions
    import inspect
    
    tools = get_tool_definitions()
    
    # Map tool names to their implementations
    tool_implementations = {
        "search_kb": search_kb,
        "create_ticket": create_ticket,
        "schedule_followup": schedule_followup
    }
    
    # Check source code of each tool for external API calls
    external_api_indicators = [
        "requests.get", "requests.post", "httpx.get", "httpx.post",
        "urllib.request", "http.client", "aiohttp", "websocket"
    ]
    
    for tool in tools:
        tool_name = tool["function"]["name"]
        if tool_name in tool_implementations:
            source = inspect.getsource(tool_implementations[tool_name])
            
            # Check for external API calls
            for indicator in external_api_indicators:
                assert indicator not in source, \
                    f"Tool '{tool_name}' contains external API call: {indicator}"
