"""
Tools module for the AI Agent Router API.

This module provides three main tools:
1. search_kb: Search the knowledge base for relevant information
2. create_ticket: Create support tickets with persistent storage
3. schedule_followup: Schedule follow-up contacts

Storage backends supported:
- memory: Fast in-memory storage (default, data lost on restart)
- file: JSON file-based persistence (simple, human-readable)
- sqlite: SQLite database (production-ready, ACID guarantees)
"""

import json
import os
import sqlite3
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
from contextlib import contextmanager
import uuid


# ============================================================================
# Storage Configuration
# ============================================================================
# 
# Choose your storage backend by setting the STORAGE_TYPE environment variable
# or by uncommenting one of the options below and commenting the default.
#
# Usage:
#   - Set environment variable: export STORAGE_TYPE="sqlite"
#   - Or edit this file: uncomment desired option, comment default
#
# Storage Types:
#   - memory: Fastest, no persistence (good for testing/development)
#   - file: Simple JSON file storage (good for small deployments)
#   - sqlite: Database with ACID guarantees (good for production)
#
# ============================================================================

# Default: In-Memory Storage (fast, but data lost on restart)
STORAGE_TYPE = os.getenv("STORAGE_TYPE", "memory").lower()

# Option 2: SQLite Database Storage (persistent, ACID guarantees)
# STORAGE_TYPE = os.getenv("STORAGE_TYPE", "sqlite").lower()

# Option 3: File-based JSON Storage (simple persistence)
# STORAGE_TYPE = os.getenv("STORAGE_TYPE", "file").lower()

# Storage-specific configuration (loaded for all storage types)
# These are used conditionally based on STORAGE_TYPE
STORAGE_FILE = os.getenv("STORAGE_FILE", "tickets_followups.json")  # Used by file storage
DB_PATH = os.getenv("DATABASE_PATH", "agent_router.db")                  # Used by SQLite storage

# In-memory storage for tickets and follow-ups (default backend)
# These are always maintained in memory for fast access, and synced to persistent
# storage when using file or sqlite backends
_tickets: Dict[str, Dict[str, Any]] = {}
_followups: Dict[str, Dict[str, Any]] = {}
_ticket_counter = 0
_followup_counter = 0


@contextmanager
def _get_db_connection():
    """
    Context manager for SQLite database connections.
    
    Provides safe database connection handling with automatic commit/rollback.
    Sets row_factory to sqlite3.Row for dict-like access to query results.
    
    Usage:
        with _get_db_connection() as conn:
            rows = conn.execute("SELECT * FROM tickets").fetchall()
            for row in rows:
                print(row["ticket_id"])  # Access by column name
    
    Raises:
        sqlite3.Error: If database operations fail (automatically rolled back)
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # Enable dict-like access: row["column_name"]
    try:
        yield conn
        conn.commit() 
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _init_sqlite_db():
    """
    Initialize SQLite database schema for tickets and follow-ups.
    
    Creates all necessary tables if they don't exist:
    - tickets: Stores support tickets with metadata
    - followups: Stores scheduled follow-up contacts
    - storage_counters: Tracks ticket and follow-up counters for ID generation
    
    This function is idempotent - safe to call multiple times.
    Uses CREATE TABLE IF NOT EXISTS to avoid errors on re-initialization.
    
    Usage:
        Called automatically when using SQLite storage backend.
        No manual invocation needed.
    """
    with _get_db_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS tickets (
                ticket_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                body TEXT NOT NULL,
                priority TEXT NOT NULL CHECK(priority IN ('low', 'medium', 'high')),
                status TEXT NOT NULL DEFAULT 'created',
                author TEXT,
                created_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS followups (
                followup_id TEXT PRIMARY KEY,
                datetime_iso TEXT NOT NULL,
                contact TEXT NOT NULL,
                channel TEXT NOT NULL CHECK(channel IN ('email', 'phone', 'whatsapp')),
                scheduled INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS storage_counters (
                counter_name TEXT PRIMARY KEY,
                counter_value INTEGER NOT NULL DEFAULT 0
            )
        """)
        # Initialize counters if they don't exist
        conn.execute("""
            INSERT OR IGNORE INTO storage_counters (counter_name, counter_value) 
            VALUES ('ticket_counter', 0), ('followup_counter', 0)
        """)


def _load_storage() -> tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]], int, int]:
    """
    Load tickets and follow-ups from persistent storage.
    
    Supports multiple storage backends:
    - file: Loads from JSON file (STORAGE_FILE)
    - sqlite: Loads from SQLite database (DB_PATH)
    - memory: Returns current in-memory state (no loading needed)
    
    Returns:
        Tuple of (tickets_dict, followups_dict, ticket_counter, followup_counter)
        - tickets_dict: Dictionary mapping ticket_id -> ticket data
        - followups_dict: Dictionary mapping followup_id -> followup data
        - ticket_counter: Current ticket counter value
        - followup_counter: Current follow-up counter value
    
    Usage:
        Called automatically on module initialization when using persistent storage.
        Returns empty dicts and zero counters if storage file/database doesn't exist.
    """
    if STORAGE_TYPE == "file":
        if not os.path.exists(STORAGE_FILE):
            return {}, {}, 0, 0
        
        try:
            with open(STORAGE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                tickets = data.get("tickets", {})
                followups = data.get("followups", {})
                ticket_counter = data.get("ticket_counter", 0)
                followup_counter = data.get("followup_counter", 0)
                return tickets, followups, ticket_counter, followup_counter
        except (json.JSONDecodeError, FileNotFoundError):
            return {}, {}, 0, 0
    
    elif STORAGE_TYPE == "sqlite":
        _init_sqlite_db()
        tickets = {}
        followups = {}
        
        with _get_db_connection() as conn:
            # Load tickets
            for row in conn.execute("SELECT * FROM tickets"):
                tickets[row["ticket_id"]] = {
                    "ticket_id": row["ticket_id"],
                    "title": row["title"],
                    "body": row["body"],
                    "priority": row["priority"],
                    "status": row["status"],
                    "author": row["author"],
                    "created_at": row["created_at"]
                }
            
            # Load follow-ups
            for row in conn.execute("SELECT * FROM followups"):
                followups[row["followup_id"]] = {
                    "followup_id": row["followup_id"],
                    "datetime_iso": row["datetime_iso"],
                    "contact": row["contact"],
                    "channel": row["channel"],
                    "scheduled": bool(row["scheduled"]),
                    "created_at": row["created_at"]
                }
            
            # Load counters
            ticket_counter_row = conn.execute(
                "SELECT counter_value FROM storage_counters WHERE counter_name = 'ticket_counter'"
            ).fetchone()
            followup_counter_row = conn.execute(
                "SELECT counter_value FROM storage_counters WHERE counter_name = 'followup_counter'"
            ).fetchone()
            
            ticket_counter = ticket_counter_row["counter_value"] if ticket_counter_row else 0
            followup_counter = followup_counter_row["counter_value"] if followup_counter_row else 0
        
        return tickets, followups, ticket_counter, followup_counter
    
    # Memory storage - return current in-memory state
    return _tickets, _followups, _ticket_counter, _followup_counter


def _save_storage(tickets: Dict[str, Dict[str, Any]], followups: Dict[str, Dict[str, Any]], 
                  ticket_counter: int, followup_counter: int):
    """
    Save ticket/follow-up counters to persistent storage.
    
    Note: Individual tickets and follow-ups are saved immediately when created.
    This function only persists the counter values to maintain ID sequence.
    
    Args:
        tickets: Dictionary of all tickets (used for file storage only)
        followups: Dictionary of all follow-ups (used for file storage only)
        ticket_counter: Current ticket counter value
        followup_counter: Current follow-up counter value
    
    Storage behavior:
        - file: Saves entire state (tickets + followups + counters) to JSON
        - sqlite: Updates only counter values in storage_counters table
        - memory: No-op (counters remain in memory)
    
    Usage:
        Called automatically after creating tickets or follow-ups.
        Errors are logged but don't raise exceptions (graceful degradation).
    """
    if STORAGE_TYPE == "file":
        data = {
            "tickets": tickets,
            "followups": followups,
            "ticket_counter": ticket_counter,
            "followup_counter": followup_counter
        }
        
        try:
            with open(STORAGE_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            import logging
            logging.warning(f"Failed to save to file storage: {e}")
    
    elif STORAGE_TYPE == "sqlite":
        _init_sqlite_db()
        try:
            with _get_db_connection() as conn:
                # Update counters
                conn.execute("""
                    UPDATE storage_counters 
                    SET counter_value = ? 
                    WHERE counter_name = 'ticket_counter'
                """, (ticket_counter,))
                conn.execute("""
                    UPDATE storage_counters 
                    SET counter_value = ? 
                    WHERE counter_name = 'followup_counter'
                """, (followup_counter,))
        except Exception as e:
            import logging
            logging.warning(f"Failed to save counters to SQLite: {e}")


def _save_ticket_to_sqlite(ticket: Dict[str, Any]):
    """
    Save a single ticket to SQLite database.
    
    Uses INSERT OR REPLACE to handle both new tickets and updates.
    Only executes if STORAGE_TYPE is "sqlite".
    
    Args:
        ticket: Dictionary containing ticket data with keys:
            - ticket_id: Unique ticket identifier
            - title: Ticket title
            - body: Ticket description
            - priority: "low", "medium", or "high"
            - status: Ticket status (default: "created")
            - author: Optional author/customer identifier
            - created_at: ISO 8601 timestamp
    
    Usage:
        Called automatically by create_ticket() when using SQLite storage.
        Errors are logged but don't raise exceptions (graceful degradation).
    """
    if STORAGE_TYPE != "sqlite":
        return
    
    _init_sqlite_db()
    try:
        with _get_db_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO tickets 
                (ticket_id, title, body, priority, status, author, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                ticket["ticket_id"],
                ticket["title"],
                ticket["body"],
                ticket["priority"],
                ticket["status"],
                ticket.get("author"),
                ticket["created_at"]
            ))
    except Exception as e:
        import logging
        logging.warning(f"Failed to save ticket to SQLite: {e}")


def _save_followup_to_sqlite(followup: Dict[str, Any]):
    """
    Save a single follow-up to SQLite database.
    
    Uses INSERT OR REPLACE to handle both new follow-ups and updates.
    Only executes if STORAGE_TYPE is "sqlite".
    
    Args:
        followup: Dictionary containing follow-up data with keys:
            - followup_id: Unique follow-up identifier
            - datetime_iso: ISO 8601 datetime string
            - contact: Contact information (email/phone/name)
            - channel: "email", "phone", or "whatsapp"
            - scheduled: Boolean flag (stored as INTEGER in SQLite)
            - created_at: ISO 8601 timestamp
    
    Usage:
        Called automatically by schedule_followup() when using SQLite storage.
        Errors are logged but don't raise exceptions (graceful degradation).
    """
    if STORAGE_TYPE != "sqlite":
        return
    
    _init_sqlite_db()
    try:
        with _get_db_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO followups 
                (followup_id, datetime_iso, contact, channel, scheduled, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                followup["followup_id"],
                followup["datetime_iso"],
                followup["contact"],
                followup["channel"],
                1 if followup.get("scheduled", True) else 0,
                followup["created_at"]
            ))
    except Exception as e:
        import logging
        logging.warning(f"Failed to save follow-up to SQLite: {e}")


# ============================================================================
# Storage Initialization
# ============================================================================
# Load existing data from persistent storage on module import
# This ensures tickets and follow-ups persist across application restarts
if STORAGE_TYPE in ("file", "sqlite"):
    _tickets, _followups, _ticket_counter, _followup_counter = _load_storage()


# ============================================================================
# Knowledge Base Caching
# ============================================================================
# Cache KB data to avoid reloading on every search
# Only reloads if the file modification time has changed
_kb_cache: Optional[List[Dict[str, Any]]] = None
_kb_cache_mtime: Optional[float] = None


def load_kb() -> List[Dict[str, Any]]:
    """
    Load the knowledge base from kb.json file with intelligent caching.
    
    The knowledge base contains product information, FAQs, troubleshooting guides,
    and other documentation used by the search_kb tool.
    
    This function implements caching to avoid reloading the file on every search.
    The cache is automatically invalidated when the file modification time changes,
    ensuring data freshness while maintaining optimal performance.
    
    Returns:
        List of knowledge base entries, each containing:
        - id: Unique identifier
        - title: Entry title
        - content: Full content text
        - tags: List of relevant tags
        - audience: Target audience ("customer" or "internal")
        - last_updated: Last update timestamp
    
    Usage:
        Called automatically by search_kb() - uses cached data if available.
        Cache is automatically refreshed when kb.json is modified.
    
    Raises:
        FileNotFoundError: If kb.json doesn't exist
        json.JSONDecodeError: If kb.json contains invalid JSON
    
    Performance:
        First call: Loads from disk and caches result
        Subsequent calls: Returns cached data (fast)
        After file update: Automatically reloads and updates cache
    """
    global _kb_cache, _kb_cache_mtime
    
    kb_path = os.path.join(os.path.dirname(__file__), "..", "kb.json")
    
    # Get current file modification time
    try:
        current_mtime = os.path.getmtime(kb_path)
    except OSError:
        # File doesn't exist or can't be accessed - clear cache and raise
        _kb_cache = None
        _kb_cache_mtime = None
        raise FileNotFoundError(f"Knowledge base file not found: {kb_path}")
    
    # Check if cache is valid (exists and file hasn't been modified)
    if _kb_cache is not None and _kb_cache_mtime == current_mtime:
        return _kb_cache  # Return cached data
    
    # Cache miss or file updated - reload from disk
    with open(kb_path, "r", encoding="utf-8") as f:
        _kb_cache = json.load(f)
        _kb_cache_mtime = current_mtime  # Store modification time
    
    return _kb_cache


def search_kb(query: str, top_k: int = 5, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Search the knowledge base and return relevant entries with relevance scores.
    
    Uses a scoring algorithm that weights matches by location:
    - Title matches: 3.0 points per word
    - Tag matches: 2.0 points per word
    - Content matches: 1.0 points per word
    - Exact phrase match: 5.0 bonus points
    
    Results are sorted by score (descending) and limited to top_k entries.
    
    Args:
        query: Search query string (case-insensitive)
        top_k: Maximum number of results to return (default 5, max 10)
        filters: Optional filters to narrow results:
            - tags: List of tags to filter by (e.g., ["pricing", "product"])
            - audience: Filter by audience ("customer" or "internal")
    
    Returns:
        Dictionary with "results" list, each entry containing:
        - id: Knowledge base entry ID
        - title: Entry title
        - score: Relevance score (higher = more relevant)
        - snippet: First 150 characters of content
        - tags: List of associated tags
    
    Usage:
        # Basic search
        results = search_kb("pricing plans")
        
        # Search with filters
        results = search_kb("API", top_k=3, filters={"tags": ["technical"], "audience": "customer"})
        
        # Access results
        for entry in results["results"]:
            print(f"{entry['title']} (score: {entry['score']})")
    
    Example:
        >>> search_kb("refund policy", top_k=3)
        {
            "results": [
                {
                    "id": "kb-001",
                    "title": "Refund Policy",
                    "score": 8.5,
                    "snippet": "Customers can request refunds within 30 days...",
                    "tags": ["pricing", "policy"]
                }
            ]
        }
    """
    # Enforce maximum results limit (API requirement: max 10)
    if top_k > 10:
        top_k = 10
    
    # Load knowledge base data (cached - only reloads if file was modified)
    kb_data = load_kb()
    
    # Normalize query for case-insensitive matching
    query_lower = query.lower()
    query_words = set(query_lower.split())  # Convert to set for efficient intersection
    
    scored_results = []
    
    # Score each knowledge base entry
    for entry in kb_data:
        score = 0.0
        
        # Apply filters first (skip entries that don't match)
        if filters:
            # Filter by tags: entry must have at least one matching tag
            if "tags" in filters:
                entry_tags = [tag.lower() for tag in entry.get("tags", [])]
                filter_tags = [tag.lower() for tag in filters["tags"]]
                if not any(tag in entry_tags for tag in filter_tags):
                    continue  # Skip this entry if no tag matches
            
            # Filter by audience: exact match required
            if "audience" in filters:
                if entry.get("audience", "").lower() != filters["audience"].lower():
                    continue  # Skip if audience doesn't match
        
        # Scoring algorithm: Weight matches by location importance
        # Title matches are most important (3.0x weight)
        title_lower = entry.get("title", "").lower()
        title_words = set(title_lower.split())
        title_matches = len(query_words.intersection(title_words))
        score += title_matches * 3.0
        
        # Content matches are least important (1.0x weight)
        content_lower = entry.get("content", "").lower()
        content_words = set(content_lower.split())
        content_matches = len(query_words.intersection(content_words))
        score += content_matches * 1.0
        
        # Tag matches are moderately important (2.0x weight)
        entry_tags = [tag.lower() for tag in entry.get("tags", [])]
        tag_matches = len(query_words.intersection(entry_tags))
        score += tag_matches * 2.0
        
        # Exact phrase match bonus (high priority for precise queries)
        if query_lower in title_lower or query_lower in content_lower:
            score += 5.0
        
        # Only include entries with positive scores
        if score > 0:
            # Create content snippet (first 150 chars for preview)
            snippet = entry.get("content", "")[:150]
            if len(entry.get("content", "")) > 150:
                snippet += "..."
            
            scored_results.append({
                "id": entry.get("id", ""),
                "title": entry.get("title", ""),
                "score": round(score, 2),  # Round to 2 decimal places
                "snippet": snippet,
                "tags": entry.get("tags", [])
            })
    
    # Sort by relevance score (descending) and limit to top_k results
    scored_results.sort(key=lambda x: x["score"], reverse=True)
    results = scored_results[:top_k]
    
    return {"results": results}


def create_ticket(title: str, body: str, priority: str, author: Optional[str] = None) -> Dict[str, Any]:
    """
    Create a support ticket with automatic persistence.
    
    Generates a unique ticket ID (format: TICK-000001) and stores the ticket
    in the configured storage backend (memory, file, or SQLite).
    
    Args:
        title: Clear, concise title for the ticket (required)
        body: Detailed description of the issue or request (required)
        priority: Priority level - must be one of:
            - "low": Non-urgent issues
            - "medium": Standard priority
            - "high": Urgent issues requiring immediate attention
        author: Optional identifier of who created the ticket (e.g., customer_id).
                If not provided, defaults to None. Automatically set from request
                customer_id when called via the API.
    
    Returns:
        Dictionary containing:
        - ticket_id: Unique ticket identifier (e.g., "TICK-000123")
        - status: Ticket status (always "created" for new tickets)
    
    Raises:
        ValueError: If priority is not one of the allowed values
    
    Usage:
        # Basic ticket creation
        result = create_ticket(
            title="Login issue",
            body="Cannot log in with email",
            priority="high"
        )
        print(result["ticket_id"])  # TICK-000001
        
        # With author tracking
        result = create_ticket(
            title="Feature request",
            body="Add dark mode",
            priority="low",
            author="customer-12345"
        )
    
    Storage:
        Ticket is automatically saved to:
        - In-memory: Stored in _tickets dictionary
        - File: Saved to JSON file (STORAGE_FILE)
        - SQLite: Inserted into tickets table (DB_PATH)
    """
    global _ticket_counter
    
    if priority not in ["low", "medium", "high"]:
        raise ValueError(f"Invalid priority: {priority}. Must be one of: low, medium, high")
    
    _ticket_counter += 1
    ticket_id = f"TICK-{_ticket_counter:06d}"
    
    ticket = {
        "ticket_id": ticket_id,
        "title": title,
        "body": body,
        "priority": priority,
        "status": "created",  # Initial status (required for storage)
        "author": author,  # Track who created the ticket
        "created_at": datetime.now(timezone.utc).isoformat()
    }
    
    _tickets[ticket_id] = ticket
    
    # Save to persistent storage
    if STORAGE_TYPE == "sqlite":
        _save_ticket_to_sqlite(ticket)
        _save_storage(_tickets, _followups, _ticket_counter, _followup_counter)
    elif STORAGE_TYPE == "file":
        _save_storage(_tickets, _followups, _ticket_counter, _followup_counter)
    
    return {
        "ticket_id": ticket_id,
        "status": "created"
    }


def schedule_followup(datetime_iso: str, contact: str, channel: str) -> Dict[str, Any]:
    """
    Schedule a follow-up contact with automatic persistence.
    
    Generates a unique follow-up ID (format: FUP-000001) and stores the follow-up
    in the configured storage backend (memory, file, or SQLite).
    
    Args:
        datetime_iso: ISO 8601 datetime string for when the follow-up should occur.
                     Supports multiple formats:
                     - "2025-12-15T10:30:00+01:00" (with timezone)
                     - "2025-12-15T10:30:00Z" (UTC, Z suffix)
                     - "2025-12-15T10:30:00" (local time)
        contact: Contact information for the follow-up:
                - Email address (e.g., "user@example.com")
                - Phone number (e.g., "+1234567890")
                - Name or identifier (e.g., "John Doe")
        channel: Communication channel - must be one of:
                - "email": Email communication
                - "phone": Phone call
                - "whatsapp": WhatsApp message
    
    Returns:
        Dictionary containing:
        - scheduled: Boolean flag (always True for new follow-ups)
        - followup_id: Unique follow-up identifier (e.g., "FUP-000123")
    
    Raises:
        ValueError: If channel is invalid or datetime_iso format is invalid
    
    Usage:
        # Schedule email follow-up
        result = schedule_followup(
            datetime_iso="2025-12-15T10:30:00+01:00",
            contact="customer@example.com",
            channel="email"
        )
        print(result["followup_id"])  # FUP-000001
        
        # Schedule phone call
        result = schedule_followup(
            datetime_iso="2025-12-20T14:00:00Z",
            contact="+1234567890",
            channel="phone"
        )
    
    Storage:
        Follow-up is automatically saved to:
        - In-memory: Stored in _followups dictionary
        - File: Saved to JSON file (STORAGE_FILE)
        - SQLite: Inserted into followups table (DB_PATH)
    """
    global _followup_counter
    
    # Validate channel value (required by API specification)
    if channel not in ["email", "phone", "whatsapp"]:
        raise ValueError(f"Invalid channel: {channel}. Must be one of: email, phone, whatsapp")
    
    # Validate ISO 8601 datetime format (supports multiple variants)
    try:
        # Handle UTC timezone indicator (Z suffix)
        if datetime_iso.endswith("Z"):
            dt_str = datetime_iso.replace("Z", "+00:00")  # Convert Z to +00:00 for parsing
            datetime.fromisoformat(dt_str)
        else:
            # Try parsing (handles both with and without timezone)
            datetime.fromisoformat(datetime_iso)
    except (ValueError, AttributeError):
        raise ValueError(f"Invalid datetime format: {datetime_iso}. Must be ISO 8601 format")
    
    # Generate unique follow-up ID (format: FUP-000001, FUP-000002, ...)
    _followup_counter += 1
    followup_id = f"FUP-{_followup_counter:06d}"  # Zero-padded 6-digit counter
    
    # Create follow-up object with all metadata
    followup = {
        "followup_id": followup_id,
        "datetime_iso": datetime_iso,  # Store original ISO string
        "contact": contact,
        "channel": channel,
        "scheduled": True,  # Initial status
        "created_at": datetime.now(timezone.utc).isoformat()  # ISO 8601 timestamp
    }
    
    # Store in memory (always maintained for fast access)
    _followups[followup_id] = followup
    
    # Persist to storage backend (if not using in-memory storage)
    if STORAGE_TYPE == "sqlite":
        _save_followup_to_sqlite(followup)  # Save individual follow-up
        _save_storage(_tickets, _followups, _ticket_counter, _followup_counter)  # Update counters
    elif STORAGE_TYPE == "file":
        _save_storage(_tickets, _followups, _ticket_counter, _followup_counter)  # Save entire state
    
    return {
        "scheduled": True,
        "followup_id": followup_id
    }


def get_tool_definitions() -> List[Dict[str, Any]]:
    """
    Return OpenAI tool definitions for function calling.
    
    Provides the schema for all available tools that the AI agent can call.
    These definitions are used by OpenAI's function calling API to enable
    the model to decide when and how to use each tool.
    
    Behavioral Requirement 4 (No external web browsing): Only local tools are provided.
    All tools operate on local data (KB file, in-memory/file/SQLite storage).
    No web browsing, external APIs, or internet access capabilities.
    
    Returns:
        List of tool definition dictionaries, each containing:
        - type: Always "function"
        - function: Function schema with name, description, and parameters
    
    Tools included (all local):
        1. search_kb: Search local knowledge base (kb.json file)
        2. create_ticket: Create tickets in local storage (memory/file/SQLite)
        3. schedule_followup: Schedule follow-ups in local storage (memory/file/SQLite)
    
    Usage:
        # Used internally by OpenAI client wrapper
        tools = get_tool_definitions()
        ## Pass to OpenAI API for function calling
    
    Note:
        This function is called automatically by the OpenAI client wrapper.
        Manual invocation is typically not needed.
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "search_kb",
                "description": "Search the knowledge base for relevant information. Use this when you need to find information about products, services, policies, or troubleshooting guides.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query to find relevant knowledge base entries"
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Maximum number of results to return (default 5, max 10)",
                            "default": 5,
                            "minimum": 1,
                            "maximum": 10
                        },
                        "filters": {
                            "type": "object",
                            "description": "Optional filters to narrow down results",
                            "properties": {
                                "tags": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Filter by tags (e.g., ['pricing', 'product'])"
                                },
                                "audience": {
                                    "type": "string",
                                    "description": "Filter by audience: 'customer' or 'internal'",
                                    "enum": ["customer", "internal"]
                                }
                            }
                        }
                    },
                    "required": ["query"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "create_ticket",
                "description": "Create a support ticket for operations or technical issues. Use this when the user explicitly asks to create a ticket, or when an issue requires escalation.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "title": {
                            "type": "string",
                            "description": "A clear, concise title for the ticket"
                        },
                        "body": {
                            "type": "string",
                            "description": "Detailed description of the issue or request"
                        },
                        "priority": {
                            "type": "string",
                            "description": "Priority level of the ticket",
                            "enum": ["low", "medium", "high"]
                        },
                        "author": {
                            "type": "string",
                            "description": "Optional author/customer identifier who created the ticket"
                        }
                    },
                    "required": ["title", "body", "priority"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "schedule_followup",
                "description": "Schedule a follow-up call, meeting, or contact. Use this when the user explicitly asks to schedule something or set a reminder.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "datetime_iso": {
                            "type": "string",
                            "description": "ISO 8601 datetime string for when the follow-up should occur (e.g., '2025-12-15T10:30:00+01:00' for 10:30 CET)"
                        },
                        "contact": {
                            "type": "string",
                            "description": "Contact information: email address, phone number, or name"
                        },
                        "channel": {
                            "type": "string",
                            "description": "Communication channel for the follow-up",
                            "enum": ["email", "phone", "whatsapp"]
                        }
                    },
                    "required": ["datetime_iso", "contact", "channel"]
                }
            }
        }
    ]

