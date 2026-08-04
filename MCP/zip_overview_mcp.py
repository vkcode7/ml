"""
Zip Code Overview Client — MCP Transport Version
==================================================
Same as zip_overview.py but calls the weather MCP server as a real
subprocess over stdio transport instead of importing it directly.

Flow:
  1. Spawns weather_server.py as a child process
  2. Communicates via JSON-RPC over stdin/stdout (MCP stdio transport)
  3. Calls get_weather tool through the MCP protocol
  4. Passes result to GPT-5.4 which assembles the final overview

This is the "real" MCP pattern — server and client are separate processes.

Setup:
  pip install "mcp[cli]" openai httpx
  export OPENAI_API_KEY=sk-...

Usage:
  python zip_overview_mcp.py 10001
  python zip_overview_mcp.py 90210
  python zip_overview_mcp.py          # prompts for input
"""

import asyncio
import json
import os
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import OpenAI

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL = "gpt-5.4"

# Absolute path to our MCP weather server script
WEATHER_SERVER_PATH = str(
    Path(__file__).parent / "weather_server.py"
)

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# ---------------------------------------------------------------------------
# System prompt for GPT-5.4
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a helpful local information assistant.

When given a US zip code, you must:
1. Call the get_weather tool to retrieve live weather data for that zip code
   (the tool also returns the city and state name from the location lookup)
2. Combine the live weather data with your own knowledge to produce a complete overview

Your response must include these sections:

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


# ---------------------------------------------------------------------------
# MCP client session — discovers tools and calls them over stdio transport
# ---------------------------------------------------------------------------

async def call_mcp_weather(zip_code: str) -> dict:
    """
    Starts weather_server.py as a subprocess and calls get_weather via MCP
    stdio transport. Returns the parsed weather result dict.
    """
    server_params = StdioServerParameters(
        command=sys.executable,          # use the same Python interpreter
        args=[WEATHER_SERVER_PATH],      # run our MCP server script
        env=None,                        # inherit current environment
    )

    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:

            # Step 1 — initialise MCP handshake
            await session.initialize()

            # Step 2 — discover available tools (optional but good practice)
            tools_response = await session.list_tools()
            available = [t.name for t in tools_response.tools]
            print(f"  📡 MCP server tools: {available}")

            # Step 3 — call get_weather
            print(f"  🔧 Calling get_weather via MCP for zip: {zip_code}")
            result = await session.call_tool(
                name="get_weather",
                arguments={"zip_code": zip_code},
            )

            # Step 4 — extract the result
            # MCP returns content blocks; structured output is in structuredContent
            # Fall back to parsing the text content block if needed
            if result.structuredContent:
                return result.structuredContent

            # Parse from text content block
            for block in result.content:
                if hasattr(block, "text"):
                    return json.loads(block.text)

            raise ValueError("No usable content in MCP tool response")


# ---------------------------------------------------------------------------
# OpenAI tool definitions — same schema as the MCP tool
# ---------------------------------------------------------------------------

OPENAI_TOOLS = [
    {
        "type": "function",
        "name": "get_weather",
        "description": (
            "Get current weather conditions for a US zip code. "
            "Returns temperature (°F and °C), humidity, wind speed, "
            "wind direction, UV index, and the resolved city/state name."
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
]


# ---------------------------------------------------------------------------
# Main overview function — GPT-5.4 + MCP tool calling loop
# ---------------------------------------------------------------------------

async def get_zip_overview(zip_code: str) -> str:
    """
    Sends the zip code query to GPT-5.4.
    When GPT-5.4 calls get_weather, we forward that call to the
    MCP server subprocess over stdio transport and return the result.
    """
    print(f"\n🔍 Looking up zip code {zip_code}...\n")

    user_message = f"Give me a complete overview for US zip code: {zip_code}"

    # Turn 1 — initial request to GPT-5.4
    response = client.responses.create(
        model=MODEL,
        instructions=SYSTEM_PROMPT,
        input=user_message,
        tools=OPENAI_TOOLS,
    )

    # Agentic loop — handle tool calls until GPT-5.4 is done
    max_iterations = 5

    for iteration in range(max_iterations):

        # Collect any function call requests from this response
        tool_calls = [
            item for item in response.output
            if item.type == "function_call"
        ]

        if not tool_calls:
            # GPT-5.4 is done — extract and return the text
            for item in response.output:
                if item.type == "message":
                    for block in item.content:
                        if hasattr(block, "text"):
                            return block.text
            return response.output_text   # fallback

        # Execute each tool call via MCP subprocess
        tool_results = []
        for tool_call in tool_calls:
            name = tool_call.name
            args = json.loads(tool_call.arguments)

            print(f"  🌐 GPT-5.4 requested tool: {name}({args})")

            if name == "get_weather":
                # ← HERE: the call goes to MCP server subprocess, not imported fn
                weather_data = await call_mcp_weather(args["zip_code"])
                result_str   = json.dumps(weather_data, indent=2)

                # Show brief confirmation
                city  = weather_data.get("city", "?")
                state = weather_data.get("state", "?")
                temp  = weather_data.get("temperature_f", "?")
                cond  = weather_data.get("condition", "?")
                print(f"  ✅ MCP returned: {city}, {state} — {temp}°F, {cond}\n")
            else:
                result_str = json.dumps({"error": f"Unknown tool: {name}"})

            tool_results.append({
                "type":    "function_call_output",
                "call_id": tool_call.call_id,
                "output":  result_str,
            })

        # Send tool results back to GPT-5.4
        prior_output = response.output

        response = client.responses.create(
            model=MODEL,
            instructions=SYSTEM_PROMPT,
            input=[
                {"role": "user", "content": user_message},
                *[item.model_dump() for item in prior_output],
                *tool_results,
            ],
            tools=OPENAI_TOOLS,
        )

    return response.output_text


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main():
    if len(sys.argv) > 1:
        zip_code = sys.argv[1].strip()
    else:
        zip_code = input("Enter a US zip code: ").strip()

    if not zip_code.isdigit() or len(zip_code) != 5:
        print(f"❌ '{zip_code}' is not a valid 5-digit US zip code.")
        sys.exit(1)

    if not os.environ.get("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not set.")
        print("   export OPENAI_API_KEY=sk-...")
        sys.exit(1)

    if not Path(WEATHER_SERVER_PATH).exists():
        print(f"❌ weather_server.py not found at: {WEATHER_SERVER_PATH}")
        print("   Both files must be in the same directory.")
        sys.exit(1)

    result = await get_zip_overview(zip_code)
    print("=" * 60)
    print(result)
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
