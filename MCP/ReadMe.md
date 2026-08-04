# MCP Weather Server

Returns current weather for any US zip code using **free APIs — no API key needed**.

## How it works

```
Claude / MCP Client
      │
      │  get_weather("10001")
      ▼
MCP Weather Server
      │
      ├─ Nominatim API  (zip → lat/lon + city name)   — free, no key
      └─ Open-Meteo API (lat/lon → weather data)       — free, no key
```

## Setup

```bash
# Python 3.10+ required
pip install "mcp[cli]" httpx
```

## Running

### Option 1 — Claude Desktop integration (stdio)

Add to your Claude Desktop config (see below), then restart Claude Desktop.
Claude can then call `get_weather` and `get_weather_bulk` directly in conversation.

### Option 2 — HTTP server (for testing / MCP Inspector)

```bash
python weather_server.py --http
# Server starts on http://localhost:8000/mcp

# Test with MCP Inspector in another terminal:
npx @modelcontextprotocol/inspector
# Connect to: http://localhost:8000/mcp
```

### Option 3 — stdio directly (for Claude Code)

```bash
python weather_server.py
# Reads JSON-RPC from stdin, writes to stdout
```

## Claude Desktop Configuration

Add this block to your Claude Desktop `claude_desktop_config.json`:

**Mac:**  `~/Library/Application Support/Claude/claude_desktop_config.json`
**Windows:** `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "weather": {
      "command": "python",
      "args": ["/absolute/path/to/weather_server.py"],
      "env": {}
    }
  }
}
```

Replace `/absolute/path/to/weather_server.py` with the actual path.

Then restart Claude Desktop. You can now ask:
- *"What's the weather in zip code 10001?"*
- *"Compare weather for 90210 and 33101"*
- *"Is it raining in 98101?"*

## Claude Code Configuration

```bash
claude mcp add weather -- python /absolute/path/to/weather_server.py
```

## Tools exposed

### `get_weather(zip_code: str) → WeatherResult`

Returns full weather for a single US zip code.

**Input:** `zip_code` — 5-digit US zip code (e.g. `"10001"`)

**Output:**
```json
{
  "zip_code":       "10001",
  "city":           "New York City",
  "state":          "New York",
  "country":        "US",
  "latitude":       40.7484,
  "longitude":      -73.9967,
  "temperature_f":  72.3,
  "temperature_c":  22.4,
  "feels_like_f":   71.1,
  "feels_like_c":   21.7,
  "humidity_pct":   58,
  "wind_speed_mph": 8.2,
  "wind_direction": "SW",
  "condition":      "Partly cloudy",
  "description":    "Partly cloudy in New York City, New York. Currently 72.3°F (feels like 71.1°F), humidity 58%, wind 8.2 mph SW.",
  "uv_index":       3.2
}
```

### `get_weather_bulk(zip_codes: list[str]) → list[WeatherResult]`

Returns weather for multiple zip codes at once (max 10). All requests run concurrently.

**Input:** `zip_codes` — list of 5-digit US zip codes

**Example:**
```json
["10001", "90210", "60601", "98101"]
```

## APIs used

| API | Purpose | Cost | Key required |
|-----|---------|------|-------------|
| [Nominatim (OpenStreetMap)](https://nominatim.org) | Zip → lat/lon + city | Free | No |
| [zippopotam.us](https://www.zippopotam.us) | Fallback zip lookup | Free | No |
| [Open-Meteo](https://open-meteo.com) | Weather data | Free | No |

## Rate limits

- **Nominatim:** 1 request/second max. For high volume use, consider self-hosting.
- **Open-Meteo:** 10,000 requests/day on free tier.
- **zippopotam.us:** No documented limit, used as fallback only.

## Project structure

```
mcp-weather/
├── weather_server.py    ← the MCP server (this file)
└── README.md
```

## To run the client
```
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...

python zip_overview.py 10001   # New York
python zip_overview.py 90210   # Beverly Hills
python zip_overview.py 60601   # Chicago
python zip_overview.py         # prompts for input
```

## How it works
```
User: python zip_overview.py 10001
           │
           ▼
    GPT-5.4 (Responses API)
    "I need weather data — calling get_weather tool"
           │
           ▼ tool_call: get_weather("10001")
    dispatch_tool() in zip_overview.py
           │
           ▼ calls directly
    get_weather() from weather_server.py
           │
           ├── Nominatim API → lat/lon + city name
           └── Open-Meteo   → live weather data
           │
           ▼ WeatherResult JSON returned to GPT-5.4
    GPT-5.4 combines live weather + its own knowledge
           │
           ▼
    Printed output with 4 sections:
      📍 Location (city, county, state, timezone, area code)
      👥 Demographics (population, income — from GPT knowledge)
      🌤 Current Weather (live from MCP server)
      📌 Quick Facts
```

**Key design point:** The get_weather and get_weather_bulk functions from weather_server.py are plain async Python functions — so zip_overview.py imports and calls them directly without needing to run the MCP server as a separate process. This is the right pattern when both files are in the same project.


## Running it as separate MCP server
That requires running the MCP server as a separate process and communicating with it over stdio transport using the MCP client SDK. Here's the updated version:Here's exactly what changed and why:

---

### The core difference — `call_mcp_weather()`

The original `zip_overview.py` did this:
```python
# Direct Python import — no MCP involved at all
from weather_server import get_weather
result = await get_weather("10001")
```

The new `zip_overview_mcp.py` does this instead:
```python
async with stdio_client(server_params) as (read, write):
    async with ClientSession(read, write) as session:
        await session.initialize()           # MCP handshake
        result = await session.call_tool(    # JSON-RPC call
            name="get_weather",
            arguments={"zip_code": zip_code}
        )
```

`stdio_client` spawns `weather_server.py` as a real child process and pipes JSON-RPC messages through its stdin/stdout. The MCP protocol handshake happens, tools are discovered, and the call is made — exactly as Claude Desktop would do it.

---

### What happens under the hood when `call_mcp_weather()` runs

```
zip_overview_mcp.py (parent process)
  │
  │  spawns as subprocess
  ▼
weather_server.py (child process)
  │
  ├── stdin  ← JSON-RPC requests from client
  │     {"jsonrpc":"2.0","method":"tools/call",
  │      "params":{"name":"get_weather",
  │               "arguments":{"zip_code":"10001"}}}
  │
  └── stdout → JSON-RPC responses back to client
        {"jsonrpc":"2.0","result":{"content":[...]}}
```

---

### The new file structure

```
mcp-weather/
├── weather_server.py       ← MCP server (runs as subprocess)
├── zip_overview.py         ← original (imports directly)
├── zip_overview_mcp.py     ← new (calls via MCP transport)
└── requirements.txt
```

---

### To run it

```bash
export OPENAI_API_KEY=sk-...

# Both files must be in the same directory
python zip_overview_mcp.py 10001
```

The output will show the MCP interaction happening live:
```
🔍 Looking up zip code 10001...

  📡 MCP server tools: ['get_weather', 'get_weather_bulk']
  🌐 GPT-5.4 requested tool: get_weather({'zip_code': '10001'})
  🔧 Calling get_weather via MCP for zip: 10001
  ✅ MCP returned: New York City, New York — 72.3°F, Partly cloudy

============================================================
📍 LOCATION
  ...
```


