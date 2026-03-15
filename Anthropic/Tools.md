# Tool use with Claude

## Introducing Tool Use
Tool use = method for Claude to access external information beyond training data.

Default limitation: Claude only knows information from training data, lacks current/real-time information.

Tool use flow:
1. Send initial request to Claude + instructions for external data access
2. Claude evaluates if external data needed, requests specific information
3. Server runs code to fetch requested data from external sources
4. Send follow-up request to Claude with retrieved data
5. Claude generates final response using original prompt + external data

Weather example: User asks current weather → Claude requests weather data → Server calls weather API → Claude receives weather data → Claude provides informed weather response.

Key concept: Tools enable Claude to augment responses with live/current information by orchestrating external data retrieval between Claude's requests.


## Project Overview

Goal = Teach Claude to set time-based reminders through tool implementation in Jupyter notebook

Target interaction = User: "Set reminder for doctor's appointment, week from Thursday" → Claude: "I will remind you at that point in time"

**Three core problems requiring tools:**

1. Time knowledge gap = Claude knows current date but not exact time
2. Time calculation errors = Claude sometimes miscalculates time-based addition (e.g., 379 days from January 13th, 1973)
3. No reminder mechanism = Claude understands reminder concept but lacks implementation capability

**Three corresponding tools to build:**

1. Current datetime tool = Gets current date + time
2. Duration addition tool = Adds time duration to datetime (e.g., current date + 20 days)
3. Reminder setting tool = Actually sets the reminder

Implementation approach = One tool at a time, building toward multi-tool coordination


## Tool Functions
Tool Functions = Python functions executed automatically when Claude needs extra information to help users.

Key characteristics:
- Plain Python functions called by Claude when it determines additional data is needed
- Must use descriptive function names and argument names
- Should validate inputs and raise errors with meaningful messages
- Error messages are visible to Claude, allowing it to retry with corrected parameters

Best practices:
1. Well-named functions and arguments
2. Input validation with immediate error raising for invalid inputs
3. Meaningful error messages that guide correction

Example implementation pattern:
```py
def get_current_datetime(date_format="%Y%m%d %H:%M:%S"):
    if not date_format:
        raise ValueError("date format cannot be empty")
    return datetime.now().strftime(date_format)
```

Tool function workflow: Claude identifies need for information → calls tool function → receives result or error → may retry with corrections if error occurred.

Purpose: Extend Claude's capabilities beyond its training data by providing access to real-time information like current datetime, weather, etc.


## Tool Schemas
Tool Schemas = JSON schema specifications that describe tool functions and their parameters for language models

JSON Schema = data validation specification (not ML-specific) used to validate JSON data, adopted by ML community for tool calling

Tool Schema Structure:
- name: tool identifier 
- description: 3-4 sentences explaining what tool does, when to use, what data it returns
- input_schema: actual JSON schema describing function arguments with types and descriptions

Schema Generation Trick:
1. Take tool function to Claude.ai
2. Prompt: "write valid JSON schema spec for tool calling for this function, follow best practices in attached documentation"
3. Attach Anthropic API documentation tool use page
4. Copy generated schema

Implementation Pattern:
- Name functions descriptively
- Name schemas as [function_name]_schema
- Import ToolParam from anthropic.types
- Wrap schema dictionary with ToolParam() to prevent type errors

Purpose = inform Claude about available tools, required arguments, and usage context through standardized JSON validation format


## Handling Message Blocks
**Tool-Enabled Claude Requests**

Step 3: Making requests to Claude with tools = include tool schema in request alongside user message using \`tools\` keyword argument containing JSON schema specs.

**Multi-Block Messages**

Content structure change = messages now contain multiple blocks instead of just text blocks.

Tool response format = assistant message with:
- Text block = user-facing explanation 
- Tool use block = contains function name + arguments for tool execution

**Message History Management**

Critical requirement = manually maintain conversation history since Claude stores nothing.

Multi-block handling = append entire response.content (all blocks) to messages list, not just text.

Helper function updates needed = add_user_message and add_assistant_message functions must support multiple blocks instead of single text blocks only.

Conversation flow = user message → assistant response with tool use block → execute tool → respond back to Claude with full history.


## Sending Tool Results
Tool Results = Results from executed tool functions sent back to Claude in follow-up requests.

Process: Execute tool function requested by Claude → Create tool result block → Send follow-up request with full conversation history.

Tool Result Block Structure:
- tool_use_id = Matches ID from original tool use block to pair requests with results
- content = Tool function output converted to string (usually JSON)
- is_error = Boolean flag for function execution errors (default false)

Tool Use ID Purpose = Links multiple tool requests to correct results when Claude makes simultaneous tool calls. Each tool use gets unique ID, tool results must reference matching IDs.

Follow-up Request Requirements:
- Include complete message history (original user message + assistant tool use message + new user message with tool result)
- Must include original tool schemas even if not using tools again
- Tool result block goes in user message, not assistant message

Conversation Flow: User request → Claude assistant response (text + tool use blocks) → Server executes tool → User message with tool result block → Claude final response with integrated results.


## Multi-Turn Conversations with Tools
Multi-Turn Tool Conversations = conversations where Claude uses multiple tools sequentially to answer a single user query.

Tool Chaining Process = user asks question → Claude requests first tool → tool executed → result returned → Claude requests second tool → tool executed → result returned → Claude provides final answer.

Example Flow = user asks "what day is 103 days from today" → Claude calls get_current_datetime → Claude calls add_duration_to_datetime → Claude provides answer.

Implementation Pattern = while loop that continues calling Claude until no more tool requests, checking each response for tool_use blocks.

run_conversation Function = takes initial messages, loops through Claude calls, executes requested tools, adds results to conversation, continues until final response.

Required Refactors:
- add_user_message/add_assistant_message = updated to handle multiple message blocks instead of just plain text
- chat function = accepts tools parameter, returns entire message instead of just first text block
- text_from_message helper = extracts all text blocks from a message with multiple content blocks

Key Insight = can't predict how many tools user queries will require, so system must handle arbitrary chains of tool calls automatically.

## Implementing Multiple Turns
**Multiple Turns Implementation = continuously calling Claude until it stops requesting tools**

**Stop Reason Field = indicates why Claude stopped generating text**
- stop_reason = "tool_use" means Claude wants to call a tool
- Other values exist but tool_use is most commonly checked

**run_conversation Function = main loop that:**
1. Calls Claude with messages + available tools
2. Adds assistant response to conversation history
3. Checks stop_reason - if not "tool_use", breaks loop
4. If tool_use, calls run_tools function
5. Adds tool results as user message
6. Repeats until no more tool requests

**run_tools Function = processes multiple tool use blocks:**
1. Filters message.content for blocks with type="tool_use"
2. Iterates through each tool request
3. Runs appropriate tool function via run_tool helper
4. Creates tool_result blocks with: type="tool_result", tool_use_id=original_id, content=JSON_encoded_output, is_error=boolean
5. Returns list of all tool result blocks

**run_tool Function = dispatcher that:**
- Takes tool_name and tool_input
- Uses if statements to match tool names to functions
- Executes appropriate tool function
- Scalable for adding multiple tools

**Error Handling = try/except blocks around tool execution:**
- Success: is_error=false, content=tool_output
- Failure: is_error=true, content=error_message

**Key Architecture Points:**
- Assistant messages can contain multiple blocks (text + multiple tool_use)
- Each tool_use block gets separate tool_result response
- Tool results sent back as user message containing all results
- Process repeats until Claude provides final text-only response


## Using Multiple Tools
Multiple Tools Implementation = Adding additional tools to an existing tool system after initial framework setup.

Process = 3 steps: (1) Add tool schemas to RunConversation function's tools list, (2) Add conditional cases in RunTool function to handle new tool names, (3) Implement actual tool functions.

Key Components:
- RunConversation function = Contains tools list that makes Claude aware of available tools
- RunTool function = Routes tool calls to appropriate functions based on tool name
- Tool schemas = Define tool structure for the AI model
- Tool functions = Actual implementation code

Example Tools Added:
- AddDurationToDateTime = Calculates date/time with duration offset
- SetReminder = Creates reminder (mock implementation that prints confirmation)

Tool Chaining = AI can use multiple tools sequentially in single conversation (e.g., calculate date first, then set reminder with result).

Message Structure = Assistant responses can contain multiple blocks: text blocks + tool use blocks in same message.

Scalability = After initial framework setup, adding new tools becomes simple pattern of schema + routing + implementation.


## The Batch Tool
Batch Tool = tool that enables Claude to run multiple tools in parallel within a single Assistant message instead of making separate sequential requests.

Problem: Claude can technically send multiple tool use blocks in one message but rarely does so in practice, leading to unnecessary sequential tool calls.

Solution: Create batch tool schema that takes list of invocations (each containing tool name + arguments). Instead of calling tools directly, Claude calls batch tool with array of desired tool executions.

Implementation:
- Add batch tool to schema with invocations parameter
- Create run_batch function that iterates through invocations list
- Extract tool name and JSON-parsed arguments from each invocation
- Call run_tool function for each requested tool
- Return batch_output list containing results from all tool executions

Mechanism: Tricks Claude into parallel tool execution by providing higher-level abstraction that manually handles what multiple tool use blocks would accomplish automatically.

Result: Single request-response cycle instead of multiple sequential rounds for parallel-executable tasks.


## Tools for Structured Data
Tools for Structured Data = alternative method to extract structured JSON from data sources using Claude's tool system instead of message pre-fill and stop sequences.

Key differences from prompt-based extraction:
- More reliable output
- More complex setup
- Requires JSON schema specification

Core Process:
1. Define JSON schema for tool where inputs = desired data structure
2. Send prompt + schema to Claude
3. Claude calls tool with structured arguments matching schema
4. Extract JSON from tool use block (no tool result needed)

Critical requirement = Force tool calling using tool_choice parameter:
- tool_choice = {"type": "tool", "name": "your_tool_name"}
- Ensures Claude always calls specified tool

Implementation steps:
1. Create schema definition for extraction tool
2. Update chat function to accept tool_choice parameter
3. Pass tool_choice to client.messages.create()
4. Access structured data from response.content[0].input

Use cases = When reliability more important than simplicity. Prompt-based methods better for quick/simple extractions, tools better for complex/reliable extractions.


## Fine Grained Tool Calling
Tool Streaming = streaming API responses while using tools with Claude

Key Components:
- Standard streaming returns content_block_delta events
- Tool streaming adds input_json_delta events with partial_json (chunk) and snapshot (cumulative sum)
- Implementation requires handling additional event type in streaming pipeline

Fine-Grained Tool Calling = feature that disables JSON validation for faster streaming

Default Behavior:
- Claude generates JSON chunks for tool arguments
- API buffers chunks until complete top-level key-value pair is generated
- Validates JSON against schema before sending chunks to server
- Results in delays followed by burst of chunks arriving simultaneously

Fine-Grained Mode (fine_grained: true):
- Disables API-side JSON validation
- Sends chunks immediately as generated
- Provides traditional streaming experience
- Requires client-side error handling for invalid JSON

Trade-offs:
- Default = slower but validated JSON
- Fine-grained = faster streaming but potential invalid JSON (like "undefined" instead of null)
- Invalid JSON in default mode gets wrapped as string rather than proper object structure

Use Cases:
- Fine-grained useful for immediate UI updates or early processing of tool arguments
- Default sufficient when validation delays acceptable


## The Text Edit Tool
Text Editor Tool = built-in Claude tool for file/text operations (read, write, create, replace, undo files/directories)

Key characteristics:
- Only JSON schema built into Claude, implementation must be custom-coded
- Schema stub sent to Claude gets auto-expanded to full schema
- Schema type string varies by Claude model version (3.5 vs 3.7 have different dates)
- Enables Claude to act as software engineer out-of-the-box

Required implementation:
- Custom class/functions to handle Claude's tool use requests
- Functions for: view files, string replace, create files, etc.
- Actual file system operations not provided by Claude

Workflow:
1. Send minimal schema stub to Claude (name + type with version-specific date)
2. Claude expands to full schema internally
3. Claude sends tool use requests
4. Custom implementation executes actual file operations
5. Results sent back to Claude

Use cases:
- Replicate AI code editor functionality
- File system operations where native editors unavailable
- Automated code generation/refactoring
- Multi-file project manipulation

Benefits = approximates fancy code editor capabilities through API calls rather than GUI interaction.


## The Web Search Tool
Web Search Tool = built-in Claude tool for searching web to find up-to-date/specialized information for user questions

Implementation = no custom code needed, Claude handles search execution automatically

Schema Requirements:
- type: "web_search_20250305"  
- name: "web_search"
- max_uses: number (limits total searches, default 5)
- allowed_domains: optional list to restrict search to specific domains

Response Structure:
- Text blocks = Claude's explanatory text
- Tool use blocks = search queries Claude executed  
- Web search result blocks = found pages (title, URL)
- Citation blocks = specific text supporting Claude's statements

Key Features:
- Multiple searches possible per request (up to max_uses limit)
- Domain restriction available for quality control
- Citation system links statements to source material

UI Rendering Pattern:
- Display text blocks as normal text
- Show search results as reference list
- Highlight citations with source attribution (domain, title, URL, quoted text)

Use Case Example: Restricting to NIH.gov for medical/exercise advice ensures scientifically-backed information vs generic web content.
