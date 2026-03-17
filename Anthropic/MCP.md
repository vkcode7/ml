# MCP
## Introducing MCP

Model Context Protocol (MCP) is a communication layer that provides Claude with context and tools without requiring you to write a bunch of tedious integration code. Think of it as a way to shift the burden of tool definitions and execution away from your server to specialized MCP servers.

### Understanding MCP Through a Real Example
Let's say you're building a chat interface where users can ask Claude about their GitHub data. A user might ask "What open pull requests are there across all my repositories?" To answer this, Claude needs tools to access GitHub's API.

Without MCP, you'd need to create all the GitHub integration tools yourself. This means writing schemas and functions for every piece of GitHub functionality you want to support.

### How MCP Solves This
MCP shifts the burden of tool definitions and execution from your server to MCP servers. Instead of you writing all those GitHub tools, they're authored and executed inside a dedicated MCP server.

The MCP server acts as a wrapper around GitHub's functionality, providing pre-built tools that you can use without having to implement them yourself.

MCP servers provide access to data or functionality implemented by outside services. They package up complex integrations into reusable components that any application can connect to.

### Isn't MCP Just Tool Use?
This is a common misconception. MCP servers and tool use are complementary but different concepts. MCP is about who does the work of creating and maintaining the tools. With MCP, someone else has already written the tool functions and schemas for you - they're packaged inside the MCP server.

The key insight is that MCP servers provide tool schemas and functions already defined for you, eliminating the need to build and maintain complex integrations yourself.

MCP = Model Context Protocol, communication layer providing Claude with context and tools without requiring developers to write tedious code.

Architecture: MCP client connects to MCP server. Server contains tools, resources, and prompts as internal components.

Problem solved: Eliminates burden of authoring/maintaining numerous tool schemas and functions for service integrations. Example: GitHub chatbot would require implementing tools for repositories, pull requests, issues, projects - significant developer effort.

Solution: MCP server handles tool definition and execution instead of your application server. MCP servers = interfaces to outside services, wrapping functionality into ready-to-use tools.

Key benefits: Developers avoid writing tool schemas and function implementations themselves.

Common questions:
- Who creates MCP servers? Anyone, often service providers make official implementations (AWS, etc.)
- vs direct API calls? MCP eliminates need to author tool schemas/functions yourself
- vs tool use? MCP and tool use are complementary - MCP handles WHO does the work (server vs developer), both still involve tools

Core value: Shifts integration burden from application developers to MCP server maintainers.

## MCP Clients
The MCP client serves as the communication bridge between your server and MCP servers. Think of it as your access point to all the tools that an MCP server provides. When you need to use external tools or services, the client handles all the message passing and protocol details for you.

MCP Client = communication interface between your server and MCP server, provides access to server's tools

Transport agnostic = client/server can communicate via multiple protocols (stdio, HTTP, WebSockets)

Common setup = client and server on same machine using standard input/output

Communication = message exchange defined by MCP spec

Key message types:
- list tools request = client asks server for available tools
- list tools result = server responds with tool list  
- call tool request = client asks server to run tool with arguments
- call tool result = server responds with tool execution result

Typical flow:
1. User queries server
2. Server requests tool list from MCP client
3. MCP client sends list tools request to MCP server
4. MCP server responds with list tools result
5. Server sends query + tools to Claude
6. Claude requests tool execution
7. Server asks MCP client to run tool
8. MCP client sends call tool request to MCP server
9. MCP server executes tool (e.g. GitHub API call)
10. Results flow back through chain: MCP server → MCP client → server → Claude → user

Purpose = enables servers to delegate tool execution to specialized MCP servers while maintaining Claude integration


## Project Setup
CLI-based chatbot project = teaches MCP client-server interaction through hands-on implementation

Project components:
- MCP client = connects to custom MCP server
- MCP server = provides 2 tools (read document, update document)
- Document collection = fake documents stored in memory only

Key distinction: Normal projects implement either client OR server, not both. This project implements both for educational purposes.

Setup process:
1. Download CLI_project.zip starter code
2. Extract and open in code editor
3. Follow readme.md setup directions
4. Add API key to .env file
5. Install dependencies (with/without UV)
6. Run project: "uv run main.py" or "python main.py"
7. Test with chat prompt

Expected outcome = working chat interface that responds to basic queries, ready for MCP feature additions.


## Defining Tools with MCP
MCP server implementation using Python SDK creates tools through decorators rather than manual JSON schemas.

Building an MCP server becomes much simpler when you use the official Python SDK. Instead of manually writing complex JSON schemas for tools, the SDK handles all that complexity for you with decorators and type hints.

MCP Python SDK = Official package that auto-generates tool JSON schemas from Python function definitions using @mcp.tool decorator.

The Python MCP SDK makes server creation incredibly straightforward. You can initialize a complete MCP server with just one line:
```py
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("DocumentMCP", log_level="ERROR")
```

Tool definition syntax = @mcp.tool(name="tool_name", description="description") + function with typed parameters using Field() for argument descriptions.

Two tools implemented:
1. read_doc_contents = Takes doc_id string, returns document content from in-memory docs dictionary
2. edit_document = Takes doc_id, old_string, new_string parameters, performs find/replace on document content

Error handling = Check if doc_id exists in docs dictionary, raise ValueError if not found.

Key advantage = SDK eliminates manual JSON schema writing, generates schemas automatically from Python function signatures and decorators.

Required imports = Field from pydantic for parameter descriptions, mcp package for server and tool decorators.

Implementation pattern = Decorator defines tool metadata, function parameters define tool arguments with types and descriptions, function body contains tool logic.

The first tool allows Claude to read any document by its ID. Here's the complete implementation:
```py
@mcp.tool(
    name="read_doc_contents",
    description="Read the contents of a document and return it as a string."
)
def read_document(
    doc_id: str = Field(description="Id of the document to read")
):
    if doc_id not in docs:
        raise ValueError(f"Doc with id {doc_id} not found")
    
    return docs[doc_id]
```
The @mcp.tool decorator automatically generates the JSON schema that Claude needs. The Field class from Pydantic provides parameter descriptions that help Claude understand what each argument expects.

## The Server Inspector
MCP Inspector = in-browser debugger for testing MCP servers without connecting to applications

Access: Run \`mcp dev [server_file.py]\` in terminal → opens server on port → navigate to provided URL in browser

_mcp dev mcp_server.py_

This starts a development server on port 6277 and gives you a local URL to open in your browser. The inspector interface will load, showing the MCP Inspector dashboard.

Click the "Connect" button on the left side to start your MCP server. Once connected, you'll see a navigation bar with sections for Resources, Prompts, Tools, and other features.

Interface: Left sidebar has connect button → top menu shows resources/prompts/tools sections → tools section lists available tools → click tool to open right panel for manual testing

Testing workflow: Connect to server → navigate to tools → select specific tool → input required parameters → click run tool → verify output

For example, to test a document reading tool, you'd enter a document ID (like "deposition.md") and run the tool. The inspector shows the result, including any returned content or success messages.

Key features: Live development testing, manual tool invocation, parameter input forms, success/failure feedback, no need for full application integration

Note: UI actively changing during development, core functionality remains similar

Example usage: Test document tools by inputting document IDs, verify read operations, test edit operations, chain operations to verify changes

Primary benefit: Debug MCP server implementations efficiently during development phase

### Development Workflow
The inspector creates an efficient development loop:

- Make changes to your MCP server code
- Test individual tools through the inspector
- Verify results without needing a full application setup
- Debug issues in isolation
This tool becomes essential as you build more complex MCP servers. It eliminates the need to wire up your server to Claude or another application just to test basic functionality, making development much faster and more focused.

## Implementing a Client
The client is what allows our application to communicate with the MCP server and access its functionality.

MCP Client Implementation:

MCP Client = wrapper class around client session for resource cleanup and connection management to MCP server

Client Session = actual connection to MCP server from MCP Python SDK, requires resource cleanup on close

Client Purpose = exposes MCP server functionality to rest of codebase, enables reaching out to server for tool lists and tool execution

Key Functions:
- list_tools() = await self.session.list_tools(), return result.tools
- call_tool() = await self.session.call_tool(tool_name, tool_input)

Usage Flow = client gets tool definitions to send to Claude, then executes tools when Claude requests them

Common Pattern = wrap client session in larger class for resource management rather than use session directly

Testing = can run client file directly with testing harness to verify server connection and tool retrieval

Integration = other code in project calls client functions to interact with MCP server, enabling Claude to inspect/edit documents through defined tools

## Defining Resources
MCP Resources = mechanism allowing MCP servers to expose data to clients for read operations

Resource Types = 2 types: direct (static URI like "docs://documents") and templated (parameterized URI like "docs://documents/{doc_id}")

URI = address/identifier for accessing specific resource, defined when creating resource

Resource Flow = client sends read resource request with URI → server matches URI to function → server executes function → returns data in read resource result

Implementation = use @mcp.resource decorator with URI and MIME type parameters

MIME Types = hint to client about returned data format (application/json for structured data, text/plain for plain text)

Templated Resources = URI parameters automatically parsed by SDK and passed as keyword arguments to handler function

Resource vs Tools = resources provide data proactively (fetch document contents when @ mentioned), tools perform actions reactively (when Claude decides to call them)

Data Return = SDK automatically serializes returned data to strings, client responsible for deserialization

Testing = MCP inspector can list direct resources separately from templated resources, allows testing individual resource calls


## Accessing Resources
MCP Resource Access Implementation:

Resource Reading Function = client-side function to request and parse resources from MCP server

Function Parameters = URI (resource identifier)

Implementation Steps:
- Import json module + AnyURL from pydantic
- Call await self.session.read_resource(AnyURL(uri))
- Extract first element from result.contents[0]
- Check resource.mime_type for parsing strategy

Content Parsing Logic:
- If mime_type == "application/json" → return json.loads(resource.text)
- Otherwise → return resource.text (plain text)

Server Response Structure = result.contents list with first element containing type/mime_type metadata

Resource Integration = MCP client functions called by other application components to fetch document contents for prompts

End Result = Document contents automatically included in Claude prompts without requiring tool calls

Key Point = Resources expose server information directly to clients through structured request/response pattern


## Defining Prompts
MCP Prompts = Pre-defined, tested prompt templates that MCP servers expose to client applications for specialized tasks.

Purpose = Instead of users writing ad-hoc prompts, server authors create high-quality, evaluated prompts tailored to their server's domain.

Implementation = Use @mcpserver.prompt decorator with name/description, define function that returns list of messages (user/assistant messages that can be sent directly to Claude).

Example Use Case = Document formatting prompt that takes document ID, instructs Claude to read document using tools, reformat to markdown, and save changes.

Key Benefits = Server-specific expertise, pre-tested quality, reusable across client applications, better results than user-generated prompts.

Message Structure = Returns base.UserMessage objects containing the formatted prompt text with interpolated parameters.

Client Integration = Prompts appear as autocomplete options (slash commands) in client applications, prompt user for required parameters, then execute the pre-built prompt workflow.


## Prompts in the Client
MCP Client Prompt Implementation:

List prompts = await self.session.list_prompts(), return result.prompts
Get prompt = await self.session.get_prompt(prompt_name, arguments), return result.messages

Prompt workflow:
1. Define prompt in MCP server with expected arguments (e.g., document_id)
2. Client calls get_prompt with prompt name + arguments dictionary
3. Arguments passed as keyword arguments to prompt function
4. Function interpolates arguments into prompt text
5. Returns messages array for direct feeding to LLM

Key concept: Prompts are server-defined templates that clients can invoke with specific arguments to generate contextualized instructions for LLMs. Arguments flow from client call → prompt function → interpolated prompt text → LLM consumption.
