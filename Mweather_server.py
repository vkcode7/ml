"""
MCP Weather Server
==================
Takes a zip code and returns weather information.

Uses:
  - MCP Python SDK v2 (MCPServer / @tool decorator)
  - Open-Meteo API  — free, no API key needed for weather data
  - Nominatim API   — free, no API key needed for zip → lat/lon

Transport options:
  stdio            — for Claude Desktop / Claude Code integration
  streamable-http  — for remote/network access

Setup:
  pip install "mcp[cli]" httpx

Run (stdio — for Claude Desktop):
  python weather_server.py

Run (HTTP — for testing in browser or curl):
  python weather_server.py --http
  curl http://localhost:8000/mcp  (then use MCP Inspector)

Test with MCP Inspector:
  npx @modelcontextprotocol/inspector
  Connect to: http://localhost:8000/mcp
"""

import argparse
import asyncio
import json
from typing import Optional

import httpx
from mcp.server import MCPServer
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Pydantic model for structured output — MCP v2 returns this as JSON schema
# ---------------------------------------------------------------------------

class WeatherResult(BaseModel):
    zip_code:        str
    city:            str
    state:           str
    country:         str
    latitude:        float
    longitude:       float
    temperature_f:   float
    temperature_c:   float
    feels_like_f:    float
    feels_like_c:    float
    humidity_pct:    int
    wind_speed_mph:  float
    wind_direction:  str
    condition:       str
    description:     str
    uv_index:        float


# ---------------------------------------------------------------------------
# WMO weather code → human-readable condition
# Open-Meteo uses WMO codes: https://open-meteo.com/en/docs
# ---------------------------------------------------------------------------

WMO_CODES: dict[int, str] = {
    0:  "Clear sky",
    1:  "Mainly clear",
    2:  "Partly cloudy",
    3:  "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    71: "Slight snow",
    73: "Moderate snow",
    75: "Heavy snow",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}

WIND_DIRECTIONS = [
    "N", "NNE", "NE", "ENE",
    "E", "ESE", "SE", "SSE",
    "S", "SSW", "SW", "WSW",
    "W", "WNW", "NW", "NNW",
]


def degrees_to_compass(degrees: float) -> str:
    idx = round(degrees / 22.5) % 16
    return WIND_DIRECTIONS[idx]


def celsius_to_fahrenheit(c: float) -> float:
    return round(c * 9 / 5 + 32, 1)


def ms_to_mph(ms: float) -> float:
    return round(ms * 2.23694, 1)


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

async def zip_to_location(zip_code: str) -> dict:
    """
    Convert US zip code → city, state, lat, lon.

    Primary:  Nominatim (OpenStreetMap) — free, no key required.
    Fallback: zippopotam.us             — free, no key required.

    Both APIs are free and require no registration. Nominatim asks that
    you set a descriptive User-Agent string and avoid more than 1 req/sec.
    """
    # --- Primary: Nominatim --------------------------------------------------
    try:
        url = "https://nominatim.openstreetmap.org/search"
        params = {
            "postalcode":   zip_code,
            "country":      "US",
            "format":       "json",
            "limit":        1,
            "addressdetails": 1,
        }
        headers = {
            # Nominatim policy: identify your application in User-Agent
            "User-Agent": "MCP-Weather-Server/1.0 (educational example)"
        }

        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url, params=params, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        if data:
            result  = data[0]
            address = result.get("address", {})
            return {
                "city":      (
                    address.get("city")
                    or address.get("town")
                    or address.get("village")
                    or address.get("county")
                    or "Unknown"
                ),
                "state":     address.get("state", "Unknown"),
                "country":   address.get("country_code", "us").upper(),
                "latitude":  float(result["lat"]),
                "longitude": float(result["lon"]),
            }
    except Exception:
        pass  # fall through to backup

    # --- Fallback: zippopotam.us --------------------------------------------
    url = f"https://api.zippopotam.us/us/{zip_code}"
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(url)
        if resp.status_code == 404:
            raise ValueError(
                f"Zip code '{zip_code}' not found. "
                "Please enter a valid US zip code."
            )
        resp.raise_for_status()
        data = resp.json()

    place = data["places"][0]
    return {
        "city":      place["place name"],
        "state":     place["state"],
        "country":   data["country abbreviation"],
        "latitude":  float(place["latitude"]),
        "longitude": float(place["longitude"]),
    }


async def fetch_weather(latitude: float, longitude: float) -> dict:
    """
    Fetch current weather from Open-Meteo — free, no API key needed.
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude":            latitude,
        "longitude":           longitude,
        "current":             [
            "temperature_2m",
            "apparent_temperature",
            "relative_humidity_2m",
            "wind_speed_10m",
            "wind_direction_10m",
            "weather_code",
            "uv_index",
        ],
        "temperature_unit":    "celsius",
        "wind_speed_unit":     "ms",    # metres per second — convert to mph ourselves
        "timezone":            "auto",
    }

    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        return resp.json()


# ---------------------------------------------------------------------------
# MCP Server
# ---------------------------------------------------------------------------

mcp = MCPServer(
    name="weather-server",
)


@mcp.tool(
    title="Get Weather by Zip Code",
    description=(
        "Returns current weather conditions for a US zip code. "
        "Includes temperature (°F and °C), humidity, wind speed and direction, "
        "weather condition, UV index, and location details."
    ),
)
async def get_weather(zip_code: str) -> WeatherResult:
    """
    Get current weather for a US zip code.

    Args:
        zip_code: A 5-digit US zip code (e.g. '10001' for New York City)

    Returns:
        WeatherResult with full weather details and location info
    """
    # Validate basic format
    zip_code = zip_code.strip()
    if not zip_code.isdigit() or len(zip_code) != 5:
        raise ValueError(
            f"'{zip_code}' is not a valid US zip code. "
            "Please provide a 5-digit number like '10001'."
        )

    # Step 1 — zip → lat/lon + city/state
    location = await zip_to_location(zip_code)

    # Step 2 — lat/lon → weather
    weather_data = await fetch_weather(location["latitude"], location["longitude"])
    current      = weather_data["current"]

    temp_c       = current["temperature_2m"]
    feels_c      = current["apparent_temperature"]
    humidity     = current["relative_humidity_2m"]
    wind_ms      = current["wind_speed_10m"]
    wind_deg     = current["wind_direction_10m"]
    wmo_code     = current["weather_code"]
    uv           = current.get("uv_index", 0.0)

    condition    = WMO_CODES.get(wmo_code, "Unknown")

    return WeatherResult(
        zip_code        = zip_code,
        city            = location["city"],
        state           = location["state"],
        country         = location["country"],
        latitude        = location["latitude"],
        longitude       = location["longitude"],
        temperature_f   = celsius_to_fahrenheit(temp_c),
        temperature_c   = round(temp_c, 1),
        feels_like_f    = celsius_to_fahrenheit(feels_c),
        feels_like_c    = round(feels_c, 1),
        humidity_pct    = int(humidity),
        wind_speed_mph  = ms_to_mph(wind_ms),
        wind_direction  = degrees_to_compass(wind_deg),
        condition       = condition,
        description     = (
            f"{condition} in {location['city']}, {location['state']}. "
            f"Currently {celsius_to_fahrenheit(temp_c)}°F "
            f"(feels like {celsius_to_fahrenheit(feels_c)}°F), "
            f"humidity {int(humidity)}%, "
            f"wind {ms_to_mph(wind_ms)} mph {degrees_to_compass(wind_deg)}."
        ),
        uv_index        = round(uv, 1),
    )


@mcp.tool(
    title="Get Weather for Multiple Zip Codes",
    description="Returns weather for several US zip codes at once.",
)
async def get_weather_bulk(zip_codes: list[str]) -> list[WeatherResult]:
    """
    Get weather for multiple US zip codes in one call.

    Args:
        zip_codes: List of 5-digit US zip codes (max 10)

    Returns:
        List of WeatherResult objects, one per zip code
    """
    if len(zip_codes) > 10:
        raise ValueError("Maximum 10 zip codes per request.")

    # Fetch all concurrently
    results = await asyncio.gather(
        *[get_weather(z) for z in zip_codes],
        return_exceptions=True,
    )

    output = []
    for z, r in zip(zip_codes, results):
        if isinstance(r, Exception):
            # Return a partial result indicating the error
            raise ValueError(f"Failed for zip {z}: {r}")
        output.append(r)

    return output


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MCP Weather Server")
    parser.add_argument(
        "--http",
        action="store_true",
        help="Run as HTTP server on port 8000 (default: stdio for Claude Desktop)",
    )
    args = parser.parse_args()

    if args.http:
        # HTTP mode — accessible via browser, curl, MCP Inspector
        print("Starting MCP Weather Server on http://localhost:8000/mcp")
        print("Test with: npx @modelcontextprotocol/inspector")
        mcp.run(transport="streamable-http", host="0.0.0.0", port=8000)
    else:
        # stdio mode — for Claude Desktop / Claude Code
        mcp.run(transport="stdio")
