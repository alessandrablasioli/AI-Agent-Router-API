import time 
import uuid #for generating unique identifiers
import logging #for logging messages
from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv #for loading environment variables more securely
from openai import APIError, OpenAIError, APITimeoutError #for handling errors from the OpenAI API

from models import AgentRunRequest, AgentRunResponse, HealthResponse, ToolCall, Metrics #for defining the request and response models
from openai_client import OpenAIClientWrapper

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
) 
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Agent Router API",
    version="1.0.0",
    description="AI Agent Router API with OpenAI integration and tool calling"
)


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return { 
        "name": "AI Agent Router API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "agent": "/v1/agent/run",
            "docs": "/docs",
            "redoc": "/redoc"
        },
        "usage": {
            "health_check": "GET /health", 
            "run_agent": "POST /v1/agent/run",
            "example_request": {
                "task": "What is the pricing model?",
                "language": "en"
            }
        }
    } 

# Initialize OpenAI client wrapper
try:
    openai_client = OpenAIClientWrapper() 
except ValueError as e:
    logger.error(f"Failed to initialize OpenAI client: {e}")
    openai_client = None 


@app.get("/health")
async def health() -> HealthResponse:
    """Health check endpoint"""
    return HealthResponse(status="ok")


@app.get("/v1/agent/run")
async def agent_run_get():
    """GET endpoint - provides information on how to use the POST endpoint"""
    return {
        "error": "This endpoint requires POST method",
        "method": "POST",
        "endpoint": "/v1/agent/run",
        "required_fields": {
            "task": "string - The task/question for the agent",
            "language": "string (optional) - Response language (e.g., 'en', 'Spanish')",
            "customer_id": "string (optional) - Customer ID for ticket creation"
        },
        "example": {
            "task": "What is the pricing model?",
            "language": "en"
        },
        "how_to_use": [
            "Use the interactive docs at /docs to test the API",
            "Or use curl: curl -X POST http://127.0.0.1:8000/v1/agent/run -H 'Content-Type: application/json' -d '{\"task\": \"...\", \"language\": \"en\"}'",
            "Or use Python requests: requests.post('http://127.0.0.1:8000/v1/agent/run', json={'task': '...', 'language': 'en'})"
        ]
    }


@app.post("/v1/agent/run", response_model=AgentRunResponse)
async def agent_run(request: AgentRunRequest) -> AgentRunResponse:
    """
    Main agent endpoint that processes tasks using OpenAI with tool calling.
    Validates input, orchestrates tool calls, and returns final answer with trace.
    
    Behavioral Requirements Implemented:
    - API hygiene (3): Input validation via Pydantic models (4xx on invalid JSON)
    - Traceability (2): trace_id generation and logging
    """
    if openai_client is None:
        raise HTTPException(
            status_code=503,
            detail="OpenAI client not initialized. Check OPENAI_API_KEY environment variable."
        )
    
    trace_id = str(uuid.uuid4())
    start_time = time.time()
    
    # Log trace_id
    logger.info(f"trace_id={trace_id} - Received request: task='{request.task[:100]}...'")
    
    try:
        final_answer, tool_calls_list, openai_calls = openai_client.run_agent( 
            task=request.task,
            language=request.language,
            max_iterations=6,
            customer_id=request.customer_id
        )
        
        latency_ms = int((time.time() - start_time) * 1000)
        
        # Behavioral Requirement 2 (Traceability): Log each tool call name + duration
        for tool_call in tool_calls_list:
            duration_ms = tool_call.get('duration_ms', 0)
            logger.info(
                f"trace_id={trace_id} - tool_call: name={tool_call['name']}, "
                f"duration_ms={duration_ms}, arguments={tool_call['arguments']}"
            )
        
        # Behavioral Requirement 2 (Traceability): Log trace_id, latency, and OpenAI call count
        logger.info(
            f"trace_id={trace_id} - latency_ms={latency_ms}, "
            f"openai_calls={openai_calls}, model={openai_client.model}"
        )
        tool_calls_response = [
            ToolCall(
                name=tc["name"],
                arguments=tc["arguments"],
                result=tc["result"]
            )
            for tc in tool_calls_list
        ]
        
        return AgentRunResponse(
            trace_id=trace_id,
            final_answer=final_answer,
            tool_calls=tool_calls_response,
            metrics=Metrics(
                latency_ms=latency_ms,
                model=openai_client.model,
                openai_calls=openai_calls
            )
        )
     
    # Behavioral Requirement 3 (API hygiene): Handle API timeouts cleanly (return 504 with trace_id)
    except APITimeoutError as e:
        latency_ms = int((time.time() - start_time) * 1000)
        logger.error(f"trace_id={trace_id} - Timeout Error: {str(e)}", exc_info=True)
        
        raise HTTPException(
            status_code=504,
            detail={
                "trace_id": trace_id,
                "error": f"Request timeout: {str(e)}",
                "latency_ms": latency_ms
            }
        )
    # Behavioral Requirement 3 (API hygiene): Handle OpenAI errors cleanly (return 502 with trace_id)
    except (APIError, OpenAIError) as e:
        latency_ms = int((time.time() - start_time) * 1000)
        logger.error(f"trace_id={trace_id} - OpenAI Error: {str(e)}", exc_info=True)
        
        raise HTTPException(
            status_code=502,
            detail={
                "trace_id": trace_id,
                "error": str(e),
                "latency_ms": latency_ms
            }
        )
    except Exception as e: #for handling other exceptions
        latency_ms = int((time.time() - start_time) * 1000)
        logger.error(f"trace_id={trace_id} - Error: {str(e)}", exc_info=True)
        
        raise HTTPException(
            status_code=500, 
            detail={
                "trace_id": trace_id,
                "error": str(e),
                "latency_ms": latency_ms
            }
        )


if __name__ == "__main__":
    import uvicorn #for running the FastAPI application
    uvicorn.run(app, host="127.0.0.1", port=8000)

