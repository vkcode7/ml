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

Tool use transforms Claude from a static knowledge base into a dynamic assistant that can work with live data. This opens up possibilities for building applications that need current information, whether that's weather data, stock prices, database queries, or any other real-time information your users might need.

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

This project demonstrates a key principle of working with AI: when the model has limitations, we extend its capabilities through tools rather than trying to work around those limitations in our prompts.

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

Creating the function is just the first step. Next, you'll need to write a JSON schema that describes the function to Claude, then integrate it into your chat system. This tool function approach gives Claude powerful capabilities while keeping your code organized and maintainable.

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

Once Claude generates your schema, copy it into your code file. Here's a good naming pattern to follow:
```py
def get_current_datetime(date_format="%Y-%m-%d %H:%M:%S"):
    if not date_format:
        raise ValueError("date_format cannot be empty")
    return datetime.now().strftime(date_format)

get_current_datetime_schema = {
    "name": "get_current_datetime",
    "description": "Returns the current date and time formatted according to the specified format",
    "input_schema": {
        "type": "object",
        "properties": {
            "date_format": {
                "type": "string",
                "description": "A string specifying the format of the returned datetime. Uses Python's strftime format codes.",
                "default": "%Y-%m-%d %H:%M:%S"
            }
        },
        "required": []
    }
}
```

**Adding Type Safety** <br>
For better type checking, import and use the ToolParam type from the Anthropic library:
```
from anthropic.types import ToolParam

get_current_datetime_schema = ToolParam({
    "name": "get_current_datetime",
    "description": "Returns the current date and time formatted according to the specified format",
    # ... rest of schema
})
```
Use the pattern of function_name followed by function_name_schema to keep your schemas organized and easy to match with their corresponding functions.

## Handling Message Blocks
**Tool-Enabled Claude Requests**

Step 3: Making requests to Claude with tools = include tool schema in request alongside user message using \`tools\` keyword argument containing JSON schema specs.

**Making Tool-Enabled API Calls**

To enable Claude to use tools, you need to include a tools parameter in your API call. Here's how to structure the request:
```py
messages = []
messages.append({
    "role": "user",
    "content": "What is the exact time, formatted as HH:MM:SS?"
})

response = client.messages.create(
    model=model,
    max_tokens=1000,
    messages=messages,
    tools=[get_current_datetime_schema],
)
```
The tools parameter takes a list of JSON schemas that describe the available functions Claude can call.

**Multi-Block Messages**

Content structure change = messages now contain multiple blocks instead of just text blocks.

A multi-block message typically contains:

- Text Block - Human-readable text explaining what Claude is doing (like "I can help you find out the current time. Let me find that information for you")
- ToolUse Block - Instructions for your code about which tool to call and what parameters to use

The ToolUse block includes:

- An ID for tracking the tool call
- The name of the function to call (like "get_current_datetime")
- Input parameters formatted as a dictionary
- The type designation "tool_use"

**Message History Management**

Critical requirement = manually maintain conversation history since Claude stores nothing.

Multi-block handling = append entire response.content (all blocks) to messages list, not just text.

Helper function updates needed = add_user_message and add_assistant_message functions must support multiple blocks instead of single text blocks only.

Conversation flow = user message → assistant response with tool use block → execute tool → respond back to Claude with full history.

Here's how to properly append a multi-block assistant message to your conversation history:
```py
messages.append({
    "role": "assistant",
    "content": response.content
})
```
This preserves both the text block and the tool use block, which is crucial for maintaining the conversation context when you make subsequent API calls.

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

The tool usage process follows this pattern:

- Send user message with tool schema to Claude
- Receive assistant message with text block and tool use block
- Extract tool information and execute the actual function
- Send tool result back to Claude along with complete conversation history
- Receive final response from Claude

Each step requires careful handling of the message structure to ensure Claude has the full context it needs to provide accurate responses.

## Sending tool results
After Claude requests a tool call, you need to execute the function and send the results back. This completes the tool use workflow by providing Claude with the information it requested.

When Claude responds with a tool use block, you extract the input parameters and call your function. Here's how to access the tool parameters:
```py
response.content[1].input
```
This gives you a dictionary of the arguments Claude wants to pass to your function. Since your function expects keyword arguments rather than a dictionary, you use Python's 

unpacking syntax:
```
result = get_current_datetime(**response.content[1].input)
```

**Building the Follow-up Request:** <br>
Your follow-up request to Claude must include the complete conversation history plus the new tool result. Here's the structure:
```py
messages.append({
    "role": "user",
    "content": [{
        "type": "tool_result",
        "tool_use_id": response.content[1].id,
        "content": result,
        "is_error": False
    }]
})
```
The complete message history now contains:

- Original user message
- Assistant message with tool use block
- User message with tool result block

**Making the Final Request:** <br>
When sending the follow-up request, you must still include the tool schema even though you're not expecting Claude to make another tool call. Claude needs the schema to understand the tool references in your conversation history.
```py
client.messages.create(
    model=model,
    max_tokens=1000,
    messages=messages,
    tools=[get_current_datetime_schema]
)
```
Claude will then respond with a final message that incorporates the tool results into a natural response for the user. The tool use workflow is now complete - you've successfully enabled Claude to access real-time information through your custom function.

## Example using OpenAI
```py
import json
from datetime import datetime
from openai import OpenAI

client = OpenAI(api_key="your-api-key")

# --- Your function ---
def get_current_datetime(date_format="%Y-%m-%d %H:%M:%S"):
    if not date_format:
        raise ValueError("date_format cannot be empty")
    return datetime.now().strftime(date_format)

# --- Tool definition for OpenAI ---
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_datetime",
            "description": "Returns the current date and time in the specified format.",
            "parameters": {
                "type": "object",
                "properties": {
                    "date_format": {
                        "type": "string",
                        "description": "A strftime format string. Defaults to '%Y-%m-%d %H:%M:%S'.",
                    }
                },
                "required": [],  # date_format is optional
            },
        },
    }
]

# --- Step 1: Send user message + tools to OpenAI ---
messages = [{"role": "user", "content": "What is the current date and time?"}]

response = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    tools=tools,
    tool_choice="auto",
)

response_message = response.choices[0].message

# --- Step 2: Check if OpenAI wants to call a tool ---
if response_message.tool_calls:
    for tool_call in response_message.tool_calls:
        if tool_call.function.name == "get_current_datetime":
            # Parse arguments
            args = json.loads(tool_call.function.arguments)
            date_format = args.get("date_format", "%Y-%m-%d %H:%M:%S")

            # Call your actual function
            result = get_current_datetime(date_format)
            print(f"Function returned: {result}")

            # --- Step 3: Send result back to OpenAI ---
            messages.append(response_message)  # append assistant's response
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            })

    # --- Step 4: Get final response from OpenAI ---
    final_response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
    )
    print("Assistant:", final_response.choices[0].message.content)
```

**Sample Output:**
```
Function returned: 2026-03-15 10:45:30
Assistant: The current date and time is March 15, 2026, at 10:45:30 AM.
```

## Multi-Turn Conversations with Tools
Multi-Turn Tool Conversations = conversations where Claude uses multiple tools sequentially to answer a single user query.

Tool Chaining Process = user asks question → Claude requests first tool → tool executed → result returned → Claude requests second tool → tool executed → result returned → Claude provides final answer.

Example Flow = user asks "what day is 103 days from today" → Claude calls get_current_datetime → Claude calls add_duration_to_datetime → Claude provides answer.

Implementation Pattern = while loop that continues calling Claude until no more tool requests, checking each response for tool_use blocks.

**Building a Conversation Loop:**<br>
To handle this pattern, you need a conversation loop that continues until Claude stops requesting tools:
```py
def run_conversation(messages):
    while True:
        response = chat(messages)
        
        add_user_message(messages, response)
        
        # Pseudo code
        if response isn't asking for a tool:
            break
            
        tool_result_blocks = run_tools(response)
        add_user_message(tool_result_blocks)
        
    return messages
```
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

**Detecting Tool Requests:**<br>
The key to knowing whether Claude wants to use a tool lies in the stop_reason field of the response message. When Claude decides it needs to call a tool, this field gets set to "tool_use". This gives us a clean way to check if we need to continue the conversation loop:

The main conversation function follows a simple pattern:
```py
def run_conversation(messages):
    while True:
        response = chat(messages, tools=[get_current_datetime_schema])
        add_assistant_message(messages, response)
        print(text_from_message(response))
        
        if response.stop_reason != "tool_use":
            break
            
        tool_results = run_tools(response)
        add_user_message(messages, tool_results)
    
    return messages
```
This loop continues until Claude provides a final answer without requesting any tools.

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

**Scalable Tool Routing** <br>
To support multiple tools, create a routing function that maps tool names to their implementations:
```py
def run_tool(tool_name, tool_input):
    if tool_name == "get_current_datetime":
        return get_current_datetime(**tool_input)
    elif tool_name == "another_tool":
        return another_tool(**tool_input)
    # Add more tools as needed
```
This approach makes it easy to add new tools without modifying the core conversation logic.

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

**Check 001_tool_009.ipynb for code till this point**

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

When you need structured data from Claude, you have two main approaches: prompt-based techniques using message prefills and stop sequences, or a more robust method using tools. While the prompt-based approach is simpler to set up, tools provide more reliable output at the cost of additional complexity.

The tool-based approach works by creating a JSON schema that defines the exact structure of data you want to extract. Instead of hoping Claude formats its response correctly, you're essentially giving Claude a function to call with specific parameters that match your desired output structure.

Here's how the process works:

- Write a schema that describes the structure of data you're looking for
- Force Claude to use a tool with the tool_choice parameter
- Extract the structured data from the tool use response
- No need to provide a follow-up response - you're done once you get the data

Key differences from prompt-based extraction:
- More reliable output
- More complex setup
- Requires JSON schema specification

Core Process:
1. Define JSON schema for tool where inputs = desired data structure
2. Send prompt + schema to Claude
3. Claude calls tool with structured arguments matching schema
4. Extract JSON from tool use block (no tool result needed)

Critical requirement = A critical part of this technique is ensuring Claude actually calls your tool. You can control this behavior using the tool_choice parameter:
- tool_choice = {"type": "tool", "name": "your_tool_name"}
- Ensures Claude always calls specified tool

Choices:
- {"type": "auto"} - Model decides if it needs to use a tool (default)
- {"type": "any"} - Model must use a tool, but can choose which one
- {"type": "tool", "name": "TOOL_NAME"} - Model must use the specified tool
For structured data extraction, you'll typically want the third option to guarantee Claude calls your specific schema tool.

Implementation steps:
1. Create schema definition for extraction tool
2. Update chat function to accept tool_choice parameter
3. Pass tool_choice to client.messages.create()
4. Access structured data from response.content[0].input

Use cases = When reliability more important than simplicity. Prompt-based methods better for quick/simple extractions, tools better for complex/reliable extractions.

**Implementation Example**<br>
Let's say you want to extract a title, author, and key insights from an article. First, you'd create a tool schema:
```py
article_summary_schema = {
    "name": "article_summary",
    "description": "Extracts structured data from articles",
    "input_schema": {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "author": {"type": "string"}, 
            "key_insights": {
                "type": "array",
                "items": {"type": "string"}
            }
        }
    }
}
```
Then you'd call Claude with the tool and force its use:
```py
response = chat(
    messages,
    tools=[article_summary_schema],
    tool_choice={"type": "tool", "name": "article_summary"}
)
```
The response will contain a tool use block with your structured data in the input field. You can access it directly:
```py
structured_data = response.content[0].input
```

**When to Use Each Approach?**<br>
Choose prompt-based structured output when you need something quick and simple. Use tools when you need guaranteed reliability and can handle the extra setup complexity. Both techniques are valuable depending on your specific use case and requirements.

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
