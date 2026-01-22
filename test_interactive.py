#!/usr/bin/env python3
"""
Interactive test script for AI Agent Router API.

Allows you to ask questions interactively and see the agent's responses.
"""

import requests
import json
import sys
from typing import Optional

API_URL = "http://127.0.0.1:8000/v1/agent/run"

def print_response(data: dict):
    """Pretty print the API response"""
    print("\n" + "="*70)
    print("RESPONSE")
    print("="*70)
    print(f"Trace ID: {data.get('trace_id')}")
    print(f"\nFinal Answer:")
    print("-" * 70)
    print(data.get('final_answer', 'No answer'))
    print("-" * 70)
    
    tool_calls = data.get('tool_calls', [])
    if tool_calls:
        print(f"\nTool Calls ({len(tool_calls)}):")
        for i, tc in enumerate(tool_calls, 1):
            print(f"\n  {i}. {tc.get('name', 'unknown')}")
            print(f"     Arguments: {json.dumps(tc.get('arguments', {}), indent=6)}")
            result = tc.get('result', {})
            if 'results' in result:
                kb_ids = [r.get('id') for r in result['results']]
                print(f"     KB Results: {', '.join(kb_ids)}")
            elif 'ticket_id' in result:
                print(f"     Ticket ID: {result['ticket_id']}")
            elif 'followup_id' in result:
                print(f"     Follow-up ID: {result['followup_id']}")
    
    metrics = data.get('metrics', {})
    if metrics:
        print(f"\nMetrics:")
        print(f"  Latency: {metrics.get('latency_ms', 0)}ms")
        print(f"  OpenAI Calls: {metrics.get('openai_calls', 0)}")
        print(f"  Model: {metrics.get('model', 'unknown')}")
    
    print("="*70 + "\n")

def test_agent(task: str, language: Optional[str] = None, customer_id: Optional[str] = None):
    """Send a request to the agent API"""
    payload = {"task": task}
    if language:
        payload["language"] = language
    if customer_id:
        payload["customer_id"] = customer_id
    
    try:
        print(f"\nSending request...")
        print(f"Task: {task}")
        if language:
            print(f"Language: {language}")
        if customer_id:
            print(f"Customer ID: {customer_id}")
        
        response = requests.post(API_URL, json=payload, timeout=120)
        
        if response.status_code == 200:
            data = response.json()
            print_response(data)
            return True
        else:
            print(f"\nERROR: Status {response.status_code}")
            print(response.text)
            return False
            
    except requests.exceptions.ConnectionError:
        print("\nERROR: Cannot connect to server. Is it running on http://127.0.0.1:8000?")
        return False
    except requests.exceptions.Timeout:
        print("\nERROR: Request timeout (120s). The request took too long.")
        return False
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        return False

def interactive_mode():
    """Run in interactive mode"""
    print("\n" + "="*70)
    print("AI AGENT ROUTER - INTERACTIVE TEST MODE")
    print("="*70)
    print("\nCommands:")
    print("  - Type your question and press Enter")
    print("  - Type 'lang:<language>' to set language (e.g., 'lang:Spanish')")
    print("  - Type 'customer:<id>' to set customer ID (e.g., 'customer:123')")
    print("  - Type 'clear' to clear language/customer settings")
    print("  - Type 'examples' to see example questions")
    print("  - Type 'quit' or 'exit' to exit")
    print("\n" + "="*70 + "\n")
    
    language = None
    customer_id = None
    
    while True:
        try:
            # Show current settings
            settings = []
            if language:
                settings.append(f"lang:{language}")
            if customer_id:
                settings.append(f"customer:{customer_id}")
            if settings:
                prompt = f"[{' '.join(settings)}] > "
            else:
                prompt = "> "
            
            user_input = input(prompt).strip()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!\n")
                break
            
            if user_input.lower() == 'clear':
                language = None
                customer_id = None
                print("Settings cleared.")
                continue
            
            if user_input.lower() in ['examples', 'help', 'h']:
                print("\nExample questions:")
                print("  1. What is the pricing model?")
                print("  2. How does CRM writeback work?")
                print("  3. Create a high priority ticket for ops")
                print("  4. Schedule a follow-up with Marta tomorrow at 10:30 CET via WhatsApp")
                print("  5. ¿En qué idiomas funciona el sistema?")
                print("\nCommands:")
                print("  lang:<language>  - Set response language")
                print("  customer:<id>   - Set customer ID")
                print("  clear           - Clear settings")
                print("  examples/help   - Show this help")
                print("  quit/exit       - Exit")
                continue
            
            if user_input.startswith('lang:'):
                language = user_input[5:].strip()
                print(f"Language set to: {language}")
                continue
            
            if user_input.startswith('customer:'):
                customer_id = user_input[9:].strip()
                print(f"Customer ID set to: {customer_id}")
                continue
            
            # Send request
            test_agent(user_input, language, customer_id)
            
        except KeyboardInterrupt:
            print("\n\nInterrupted. Type 'quit' to exit or continue asking questions.")
        except EOFError:
            print("\n\nGoodbye!\n")
            break

def main():
    """Main entry point"""
    # Check if server is running
    try:
        health = requests.get("http://127.0.0.1:8000/health", timeout=2)
        if health.status_code != 200:
            print("ERROR: Server health check failed.")
            sys.exit(1)
    except:
        print("ERROR: Cannot connect to server. Please start it first:")
        print("  cd app && python main.py")
        sys.exit(1)
    
    # Check command line arguments
    if len(sys.argv) > 1:
        # Non-interactive mode: single question
        task = " ".join(sys.argv[1:])
        test_agent(task)
    else:
        # Interactive mode
        interactive_mode()

if __name__ == "__main__":
    main()

