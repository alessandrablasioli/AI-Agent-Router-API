from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, field_validator #for defining the request and response models

 #for defining the request model
class AgentRunRequest(BaseModel):
    task: str = Field(..., min_length=1, max_length=5000, description="The task or question for the agent")
    customer_id: Optional[str] = Field(None, max_length=100, description="Optional customer identifier")
    language: Optional[str] = Field(None, max_length=10, description="Optional language code (e.g., en, es, pt)")
    
    @field_validator('task')
    @classmethod
    def validate_task(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Task cannot be empty")
        return v.strip()


#for defining the tool call model
class ToolCall(BaseModel):
    name: str
    arguments: Dict[str, Any]
    result: Dict[str, Any]


#for defining the metrics model
class Metrics(BaseModel):
    latency_ms: int
    model: str
    openai_calls: int


#for defining the response model
class AgentRunResponse(BaseModel):
    trace_id: str
    final_answer: str
    tool_calls: List[ToolCall]
    metrics: Metrics


#for defining the health response model
class HealthResponse(BaseModel):
    status: str

