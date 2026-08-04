# Hosting in Azure

When you move from stdio (local subprocess) to a public deployment, the transport changes from stdin/stdout to HTTP. Here are your options ranked by simplicity:

---

### The Transport Shift First

```
Local (current):
  Client spawns server as subprocess
  Communication: stdin/stdout (stdio transport)

Public (what you need):
  Server runs independently on Azure
  Communication: HTTP (Streamable HTTP or SSE transport)
  Anyone can hit: https://your-server.com/mcp
```

Our `weather_server.py` already supports this — it has `--http` mode built in.

---

### Option 1 — Azure Container Apps (Recommended)

The easiest path. Serverless containers — no VM management, scales to zero when idle, pays per request.

```
User → https://weather-mcp.yourapp.azurecontainerapps.io/mcp
              ↓
         Azure Container Apps
              ↓
         weather_server.py (Streamable HTTP mode)
              ↓
         Nominatim + Open-Meteo APIs
```

**Dockerfile:**
```dockerfile
FROM python:3.12-slim

WORKDIR /app
COPY weather_server.py .
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD ["python", "weather_server.py", "--http"]
```

**Deploy:**
```bash
# Build and push to Azure Container Registry
az acr create --name weathermcpregistry --resource-group myRG --sku Basic
az acr login --name weathermcpregistry
docker build -t weather-mcp .
docker tag weather-mcp weathermcpregistry.azurecr.io/weather-mcp:latest
docker push weathermcpregistry.azurecr.io/weather-mcp:latest

# Deploy to Container Apps
az containerapp create \
  --name weather-mcp-server \
  --resource-group myRG \
  --image weathermcpregistry.azurecr.io/weather-mcp:latest \
  --target-port 8000 \
  --ingress external \
  --min-replicas 0 \
  --max-replicas 5
```

**Cost:** ~$0 when idle (scales to zero), pay only for requests. Good for low-to-medium traffic.

---

### Option 2 — Azure App Service (Web App)

More traditional, always-on, easier to manage for steady traffic.

```bash
# Create App Service plan
az appservice plan create \
  --name weather-mcp-plan \
  --resource-group myRG \
  --is-linux \
  --sku B1            # ~$13/month — cheapest always-on

# Create web app from container
az webapp create \
  --name weather-mcp-server \
  --resource-group myRG \
  --plan weather-mcp-plan \
  --deployment-container-image-name \
    weathermcpregistry.azurecr.io/weather-mcp:latest

# Set the port
az webapp config appsettings set \
  --name weather-mcp-server \
  --resource-group myRG \
  --settings WEBSITES_PORT=8000
```

Your server is then at: `https://weather-mcp-server.azurewebsites.net/mcp`

**Cost:** ~$13/month (B1 plan), always running.

---

### Option 3 — Azure Container Instance (Quick and Simple)

Single container, no orchestration. Good for testing or low-traffic use.

```bash
az container create \
  --name weather-mcp \
  --resource-group myRG \
  --image weathermcpregistry.azurecr.io/weather-mcp:latest \
  --ports 8000 \
  --dns-name-label weather-mcp-public \
  --os-type Linux \
  --cpu 1 \
  --memory 1
```

Your server is at: `http://weather-mcp-public.<region>.azurecontainer.io:8000/mcp`

**Cost:** ~$0.0015/vCPU/hour — roughly $1/month if running 24/7.

---

### Option 4 — Your Existing RHEL VM on Azure

The simplest if you already have it. Just run the server on the VM and open the port.

```bash
# On your Azure RHEL VM:
cd ~/projects/mcp-weather
pip install "mcp[cli]" httpx --break-system-packages

# Run server (use a systemd service or tmux for persistence)
python weather_server.py --http

# Open port 8000 in Azure Network Security Group:
az network nsg rule create \
  --nsg-name your-vm-nsg \
  --resource-group myRG \
  --name allow-mcp-8000 \
  --priority 1001 \
  --destination-port-ranges 8000 \
  --access Allow \
  --protocol Tcp
```

**But** — raw port 8000 with no TLS is not suitable for public use. You'd need to add nginx as a reverse proxy with HTTPS, which adds complexity. Use Container Apps instead unless you already want to practice nginx.

---

### What needs to change in `weather_server.py` for HTTP deployment

Almost nothing — the `--http` flag already uses `streamable-http` transport. But two small changes improve production readiness:

```python
# In weather_server.py — update the run section:

if args.http:
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))  # Azure sets PORT env var
    print(f"Starting MCP Weather Server on http://{host}:{port}/mcp")
    mcp.run(transport="streamable-http", host=host, port=port)
```

And update the `Dockerfile` CMD to pass `--http`:
```dockerfile
CMD ["python", "weather_server.py", "--http"]
```

---

### How the client changes for HTTP transport

Once deployed, `zip_overview_mcp.py` needs to switch from `stdio_client` to `streamable_http_client`:

```python
# Old (local subprocess):
from mcp.client.stdio import stdio_client
from mcp import StdioServerParameters

server_params = StdioServerParameters(
    command=sys.executable,
    args=[WEATHER_SERVER_PATH],
)
async with stdio_client(server_params) as (read, write):
    async with ClientSession(read, write) as session:
        ...

# New (remote HTTP server):
from mcp.client.streamable_http import streamable_http_client

MCP_SERVER_URL = "https://weather-mcp-server.azurewebsites.net/mcp"

async with streamable_http_client(MCP_SERVER_URL) as (read, write, _):
    async with ClientSession(read, write) as session:
        ...
```

Everything else in `zip_overview_mcp.py` stays identical — same `session.call_tool()`, same tool names, same result handling.

---

### Security considerations for public deployment

A public MCP server with no auth means anyone can hit it. A few things to add:

**API key authentication:**
```python
# In weather_server.py — check a header on every request
# The MCP SDK supports this via auth settings
from mcp.server.auth.settings import AuthSettings
```

Or simpler — put Azure API Management in front, which handles:
- API keys per consumer
- Rate limiting (prevent abuse of free weather APIs)
- Usage analytics
- HTTPS termination

**Rate limiting matters here** because Nominatim has a 1 req/sec limit and Open-Meteo has 10,000 req/day on the free tier. Without rate limiting a single abusive caller could exhaust your quota.

---

### Recommendation summary

| Option | Best for | Cost | Complexity |
|---|---|---|---|
| Container Apps | Public production, variable traffic | Pay per use | Low |
| App Service | Steady traffic, simpler ops | ~$13/mo | Low |
| Container Instance | Quick demo, low traffic | ~$1/mo | Very low |
| Existing RHEL VM | If you already have it + nginx | Already paying | Medium |

**Start with Azure Container Apps** — it's the most cost-effective for a public MCP server since it scales to zero when nobody is using it, and you get a proper HTTPS URL automatically with no certificate management needed.
