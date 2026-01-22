# AI Agent Router API

A production-ready HTTP API that uses OpenAI's function calling API to route user requests to appropriate server-side tools (knowledge base search, ticket creation, follow-up scheduling) and return clean final answers with execution traces.

This project demonstrates:
- **OpenAI Function Calling**: Intelligent tool orchestration using OpenAI's tool/function calling API
- **FastAPI**: Modern Python web framework with automatic API documentation
- **Tool Orchestration**: Multi-iteration agent loop with proper error handling
- **Observability**: Comprehensive logging, trace IDs, and metrics
- **Storage Flexibility**: Support for in-memory, file-based, and SQLite storage backends

## Development Tool

This project was built using **Cursor** (an AI-powered code editor) to assist with code generation and implementation.

## Features

- **Tool-using agent**: OpenAI function calling for server-side tool orchestration
- **Three tools**: `search_kb`, `create_ticket`, `schedule_followup`
- **Observability**: Trace IDs, latency metrics, tool call logging
- **Error handling**: Proper HTTP status codes (502, 504, 500) with trace IDs
- **Multi-language support**: Responds in requested language
- **Flexible storage**: Choose between in-memory, file-based, or SQLite storage

## Setup

### Prerequisites

- Python 3.8+
- OpenAI API key

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set environment variables:

**Option 1: Using `.env` file (recommended)**
```bash
# Create .env file
echo "OPENAI_API_KEY=sk-your-actual-key-here" > .env
```

**Option 2: Environment variables**

Linux/Mac:
```bash
export OPENAI_API_KEY="your-api-key-here"
export OPENAI_MODEL="gpt-4o"  # Optional, defaults to gpt-4o
export STORAGE_TYPE="memory"  # Optional: "memory" (default), "file", or "sqlite"
```

Windows PowerShell:
```powershell
$env:OPENAI_API_KEY="your-api-key-here"
$env:OPENAI_MODEL="gpt-4o"
$env:STORAGE_TYPE="memory"
```

### Running the Server

```bash
cd app
python main.py
```

API available at `http://127.0.0.1:8000`
Interactive docs at `http://127.0.0.1:8000/docs`

## API Endpoints

### POST /v1/agent/run

Process a user task using the agent with tool calling.

**Request:**
```json
{
  "task": "Give me the pricing model at a high level and what can change the quote.",
  "customer_id": "optional-customer-id",
  "language": "en"
}
```

**Response:**
```json
{
  "trace_id": "550e8400-e29b-41d4-a716-446655440000",
  "final_answer": "Based on the knowledge base...",
  "tool_calls": [
    {
      "name": "search_kb",
      "arguments": {"query": "pricing", "top_k": 5},
      "result": {"results": [...]}
    }
  ],
  "metrics": {
    "latency_ms": 1234,
    "model": "gpt-4o",
    "openai_calls": 2
  }
}
```

### GET /health

Health check endpoint.

**Response:**
```json
{
  "status": "ok"
}
```

## Testing

⚠️ **All tests make REAL calls to OpenAI and will use your API credits!**

### Prerequisites

1. Set your OpenAI API key (see Setup)
2. Start the server (for integration tests):
   ```bash
   cd app
   python main.py
   ```

### Running Tests

**Test tools (no server, no API key needed):**
```bash
pytest tests/test_tools.py -v
```

**Test agent orchestration (requires API key, no server needed):**
```bash
pytest tests/test_agent.py -v
```

**Test full integration (requires API key + running server):**
```bash
pytest tests/test_integration.py -v
```

Tests automatically skip if `OPENAI_API_KEY` is not set or is "test-key".

## Interactive Testing

Use the interactive test script:

```bash
python test_interactive.py
```

This allows you to ask questions interactively and see responses with tool calls and metrics.

## Architecture

- **app/main.py**: FastAPI application with endpoints
- **app/models.py**: Pydantic models for request/response validation
- **app/tools.py**: Tool implementations (search_kb, create_ticket, schedule_followup)
- **app/openai_client.py**: OpenAI client wrapper with tool calling orchestration
- **kb.json**: Knowledge base dataset

## Implementation Details

- **Tool loop orchestration**: Maximum 6 iterations to prevent infinite loops
- **Traceability**: Each request gets a unique trace_id, all tool calls and metrics are logged
- **Error handling**: 
  - OpenAI errors return 502
  - Timeout errors return 504
  - Other errors return 500
  - All errors include trace_id and latency
- **Storage options**: 
  - **In-memory** (default): Fast, but data lost on restart
  - **File-based**: Persistent storage in JSON file (set `STORAGE_TYPE=file`)
  - **SQLite**: Database storage with ACID guarantees (set `STORAGE_TYPE=sqlite`)

## Security Note

**IMPORTANT**: Your API key is private and should never be committed to Git. The `.env` file is already in `.gitignore` to protect your credentials.

## License

This project is open source and available for portfolio purposes.
