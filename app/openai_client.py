import os
import json
import time
from typing import List, Dict, Any, Optional
from openai import OpenAI
from tools import search_kb, create_ticket, schedule_followup, get_tool_definitions


class OpenAIClientWrapper:
    """
    Wrapper for OpenAI client with tool calling support.
    
    Uses OpenAI Responses API (Chat Completions API) with tool/function calling
    to enable the model to decide when to call server-side tools.
    
    OpenAI Requirements Compliance:
    - Uses OpenAI Responses API (chat.completions.create)
    - Implements tool/function calling for search_kb, create_ticket, schedule_followup
    - No web browsing - only local tools and knowledge base
    """
    
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        # Set timeout for API calls (60 seconds default, configurable via env)
        timeout_seconds = int(os.getenv("OPENAI_TIMEOUT_SECONDS", "60"))
        
        self.client = OpenAI(
            api_key=api_key,
            timeout=timeout_seconds
        )
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o")
        self.tool_definitions = get_tool_definitions()
        self.call_count = 0
        self.timeout_seconds = timeout_seconds
    
    def execute_tool(self, tool_name: str, arguments: Dict[str, Any], customer_id: Optional[str] = None) -> Dict[str, Any]:
        """Execute a tool by name with given arguments"""
        if tool_name == "search_kb":
            return search_kb(**arguments)
        elif tool_name == "create_ticket":
            # Inject customer_id as author if not already provided
            if customer_id and "author" not in arguments:
                arguments["author"] = customer_id
            return create_ticket(**arguments)
        elif tool_name == "schedule_followup":
            return schedule_followup(**arguments)
        else:
            raise ValueError(f"Unknown tool: {tool_name}")
    
    def run_agent(
        self,
        task: str,
        language: Optional[str] = None,
        max_iterations: int = 6, #for the number of iterations the agent will run
        customer_id: Optional[str] = None 
    ) -> tuple[str, List[Dict[str, Any]], int]: 
        """
        Run the agent with tool calling loop.
        
        Returns:
            tuple of (final_answer, tool_calls_list, openai_calls_count)
        """
        self.call_count = 0
        tool_calls_history = []
        
        system_message = (
            "You are a helpful assistant for a company that builds AI voice agents. "
            "You have access to a knowledge base and tools to help customers and internal teams. "
            "IMPORTANT: When the user mentions specific topics (like pricing, CRM writeback, custom SLA, onboarding, "
            "troubleshooting, integrations, etc.), you MUST use the search_kb tool to find relevant information "
            "before providing your final answer. This ensures accurate, up-to-date information. "
            "CRITICAL: After receiving tool results, you MUST provide your final answer immediately in the next response. "
            "Do NOT make additional tool calls unless the user explicitly requests another action. "
            "Only create tickets or schedule follow-ups when explicitly requested by the user. "
            "When scheduling follow-ups about specific topics, search the KB first to include relevant information in your response. "
            "Provide clear, accurate answers based on the knowledge base. "
            "If you don't know something, say so and offer to create a ticket if appropriate."
            
        ) 
        
        if language:
            system_message += f" Respond in {language}."
            
        system_message += f" It's {time.strftime('%d of %B %Y')}"
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": task}
        ]
        
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            self.call_count += 1
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self.tool_definitions,
                tool_choice="auto"
            )
            
            assistant_message = response.choices[0].message 
            messages.append({
                "role": "assistant",
                "content": assistant_message.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }
                    for tc in (assistant_message.tool_calls or [])
                ]
            })
            
            # If no tool calls, the model is providing the final answer
            if not assistant_message.tool_calls:
                # Use the content from the assistant message as final answer
                final_answer = assistant_message.content or "I apologize, but I couldn't generate a response."
                return final_answer, tool_calls_history, self.call_count
            
            tool_responses = []
            for tool_call in assistant_message.tool_calls:
                tool_name = tool_call.function.name
                tool_start_time = time.time()
                
                try:
                    arguments = json.loads(tool_call.function.arguments) 
                    result = self.execute_tool(tool_name, arguments, customer_id=customer_id)
                    tool_duration_ms = int((time.time() - tool_start_time) * 1000)
                    
                    tool_calls_history.append({
                        "name": tool_name,
                        "arguments": arguments,
                        "result": result,
                        "duration_ms": tool_duration_ms
                    })
                    
                    tool_responses.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(result)
                    })
                    #for handling tool execution errors
                except Exception as e:
                    # Handle tool execution errors
                    tool_duration_ms = int((time.time() - tool_start_time) * 1000)
                    error_result = {"error": str(e)}
                    try:
                        arguments = json.loads(tool_call.function.arguments)
                        # Preserve author if it was set
                        if customer_id and tool_name == "create_ticket" and "author" not in arguments:
                            arguments["author"] = customer_id
                    except:
                        arguments = {}
                    
                    tool_calls_history.append({
                        "name": tool_name,
                        "arguments": arguments,
                        "result": error_result,
                        "duration_ms": tool_duration_ms  # Track duration even for errors
                    })
                    
                    tool_responses.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(error_result)
                    })
            
            messages.extend(tool_responses)
        
        final_answer = (
            "I apologize, but I reached the maximum number of tool call iterations. "
            "Please try rephrasing your request or contact support."
        )
        return final_answer, tool_calls_history, self.call_count

