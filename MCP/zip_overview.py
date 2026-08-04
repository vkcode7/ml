"""
Zip Code Overview Client
=========================
Uses OpenAI GPT-5.4 (via the Responses API) to provide:
  - Place overview: city, county, state, population, timezone, area code
  - Live weather: via our MCP weather server tools (called as OpenAI functions)

The MCP weather server tools are registered as OpenAI function tools.
GPT-5.4 decides when to call them and assembles the final answer.

Architecture:
  User → This script → OpenAI GPT-5.4 (Responses API)
                              ↓ tool_call: get_weather
                       weather_server.py tools (called locally)
                              ↓ weather data
                       GPT-5.4 assembles final response
                              ↓
                       Printed to terminal

Setup:
  pip install openai httpx
  export OPENAI_API_KEY=sk-...

Usage:
  python zip_overview.py 10001
  python zip_overview.py 90210
  python zip_overview.py            # prompts for input
"""

import asyncio
import json
import os
import sys

from openai import OpenAI

# Import our MCP weather server tools directly
# (They are plain async Python functions — no MCP transport needed here)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from weather_server import get_weather, get_weather_bulk

# ---------------------------------------------------------------------------
# OpenAI client
# ---------------------------------------------------------------------------

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
MODEL = "gpt-5.4"

# ---------------------------------------------------------------------------
# Tool definitions — registered with OpenAI so GPT-5.4 can call them
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "name": "get_weather",
        "description": (
            "Get current weather conditions for a US zip code. "
            "Returns temperature (°F and °C), humidity, wind speed and direction, "
            "weather condition, UV index, and the city/state name."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "zip_code": {
                    "type": "string",
                    "description": "A 5-digit US zip code, e.g. '10001'",
                }
            },
            "required": ["zip_code"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "get_weather_bulk",
        "description": (
            "Get current weather for multiple US zip codes at once (max 10). "
            "Use this when comparing weather across several locations."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "zip_codes": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of 5-digit US zip codes",
                }
            },
            "required": ["zip_codes"],
            "additionalProperties": False,
        },
        "strict": True,
    },
]

# ---------------------------------------------------------------------------
# Tool dispatcher — runs the actual MCP server function when OpenAI calls it
# ---------------------------------------------------------------------------

async def dispatch_tool(name: str, arguments: dict) -> str:
    """
    Called when GPT-5.4 requests a tool call.
    Runs the corresponding weather server function and returns JSON result.
    """
    try:
        if name == "get_weather":
            result = await get_weather(arguments["zip_code"])
            return result.model_dump_json(indent=2)

        elif name == "get_weather_bulk":
            results = await get_weather_bulk(arguments["zip_codes"])
            return json.dumps([r.model_dump() for r in results], indent=2)

        else:
            return json.dumps({"error": f"Unknown tool: {name}"})

    except Exception as e:
        return json.dumps({"error": str(e)})


# ---------------------------------------------------------------------------
# Main agent loop — uses OpenAI Responses API with tool calling
# ---------------------------------------------------------------------------

async def get_zip_overview(zip_code: str) -> str:
    """
    Uses GPT-5.4 to produce a complete zip code overview.
    GPT-5.4 autonomously decides to call get_weather for live data,
    then combines it with its own knowledge of the location.
    """
    system_prompt = """You are a helpful local information assistant.

When given a US zip code, you must:
1. Call the get_weather tool to retrieve live weather data for that zip code
   (the tool also returns the city and state name)
2. Combine the live weather data with your knowledge to provide a complete overview

Your response must include these sections, clearly formatted:

📍 LOCATION
  - City, County, State
  - Zip code, Time zone
  - Area code(s)
  - Brief description of the area (what it's known for)

👥 DEMOGRAPHICS (from your knowledge)
  - Approximate population
  - Population density
  - Median household income (approximate)
  - Notable facts

🌤 CURRENT WEATHER (from the weather tool — always call it)
  - Condition and temperature (°F)
  - Feels like temperature
  - Humidity and UV index
  - Wind speed and direction

📌 QUICK FACTS
  - 2-3 interesting or notable things about this location

Keep the response concise but informative. Always call the weather tool first."""

    user_message = f"Give me a complete overview for US zip code: {zip_code}"

    # --- Turn 1: send to GPT-5.4 with tools available ---
    messages = [
        {"role": "system",  "content": system_prompt},
        {"role": "user",    "content": user_message},
    ]

    print(f"\n🔍 Looking up zip code {zip_code}...\n")

    # Agentic loop — GPT-5.4 may call tools multiple times before finishing
    max_iterations = 5
    iteration = 0

    # Convert to Responses API input format
    input_messages = messages[1:]  # user message (system goes in instructions)

    response = client.responses.create(
        model=MODEL,
        instructions=system_prompt,
        input=user_message,
        tools=TOOLS,
    )

    while iteration < max_iterations:
        iteration += 1

        # Check if model wants to call tools
        tool_calls = [
            item for item in response.output
            if item.type == "function_call"
        ]

        if not tool_calls:
            # No tool calls — model is done, return text output
            text_items = [
                item for item in response.output
                if item.type == "message"
            ]
            if text_items:
                # Extract text from message output
                for msg in text_items:
                    for content in msg.content:
                        if hasattr(content, "text"):
                            return content.text
            return response.output_text  # fallback

        # Execute all requested tool calls
        tool_results = []
        for tool_call in tool_calls:
            tool_name = tool_call.name
            tool_args = json.loads(tool_call.arguments)

            print(f"  🔧 Calling tool: {tool_name}({json.dumps(tool_args)})")

            result_str = await dispatch_tool(tool_name, tool_args)
            tool_results.append({
                "type":        "function_call_output",
                "call_id":     tool_call.call_id,
                "output":      result_str,
            })

            # Show brief confirmation
            try:
                result_data = json.loads(result_str)
                if "city" in result_data:
                    print(f"  ✅ Got weather for {result_data['city']}, "
                          f"{result_data['state']}: "
                          f"{result_data['temperature_f']}°F, "
                          f"{result_data['condition']}")
            except Exception:
                pass

        print()

        # Send tool results back to GPT-5.4 for next turn
        # Build the input with prior output + tool results
        prior_output = response.output  # list of output items

        response = client.responses.create(
            model=MODEL,
            instructions=system_prompt,
            input=[
                {"role": "user", "content": user_message},
                *[item.model_dump() for item in prior_output],
                *tool_results,
            ],
            tools=TOOLS,
        )

    return response.output_text


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main():
    # Get zip code from command line or prompt
    if len(sys.argv) > 1:
        zip_code = sys.argv[1].strip()
    else:
        zip_code = input("Enter a US zip code: ").strip()

    # Basic validation
    if not zip_code.isdigit() or len(zip_code) != 5:
        print(f"❌ '{zip_code}' is not a valid 5-digit US zip code.")
        sys.exit(1)

    # Check API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY environment variable not set.")
        print("   export OPENAI_API_KEY=sk-...")
        sys.exit(1)

    result = await get_zip_overview(zip_code)
    print("=" * 60)
    print(result)
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
