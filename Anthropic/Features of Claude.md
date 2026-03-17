# Features of Claude

## Extended Thinking
Extended Thinking = Claude feature that allows reasoning time before generating final response

Key mechanics:
- Displays separate thinking process visible to users
- Increases accuracy for complex tasks but adds cost (charged for thinking tokens) and latency
- Thinking budget = minimum 1024 tokens allocated for thinking phase
- Max tokens must exceed thinking budget (e.g., budget 1024 requires max_tokens ≥ 1025)

When to use:
- Enable after prompt optimization fails to achieve desired accuracy
- Use prompt evals to determine necessity

Response structure:
- Thinking block = contains reasoning text + cryptographic signature
- Text block = final response
- Signature = prevents tampering with thinking text (safety measure)

Special cases:
- Redacted thinking blocks = encrypted thinking text flagged by safety systems
- Provided for conversation continuity without losing context
- Can force redacted blocks using test string: "entropic magic string triggered redacted thinking [special characters]"

Implementation:
- Set thinking=true and thinking_budget parameter
- Ensure max_tokens > thinking_budget for adequate response generation capacity

```py
 def chat(
    messages,
    system=None,
    temperature=1.0,
    stop_sequences=[],
    tools=None,
    thinking=False,
    thinking_budget=1024,
):
    params = {
        "model": model,
        "max_tokens": 4000,
        "messages": messages,
        "temperature": temperature,
        "stop_sequences": stop_sequences,
    }

    if thinking:
        params["thinking"] = {
            "type": "enabled",
            "budget_tokens": thinking_budget,
        }

    if tools:
        params["tools"] = tools

    if system:
        params["system"] = system

    message = client.messages.create(**params)
    return message
```

## Image Support
Claude Vision Capabilities = ability to process images within user messages for analysis, comparison, counting, and description tasks.

Image Limitations:
- Max 100 images per request
- Size/dimension restrictions apply
- Images consume tokens (charged based on pixel height/width calculation)

Image Block Structure = special block type within user messages that holds either raw image data (base64) or URL reference to online image. Multiple image blocks allowed per message.

Critical Success Factor = strong prompting techniques required for accurate results. Simple prompts often fail.

Prompting Techniques for Images:
- Step-by-step analysis instructions
- One-shot/multi-shot examples (alternating image and text pairs)
- Clear guidelines and verification steps
- Structured analysis frameworks

Example Use Case = automated fire risk assessment from satellite imagery analyzing tree density, property access, roof overhang, and assigning numerical risk scores.

Implementation = base64 encode image data, create message with image block (type: image, source: base64, media_type, data) followed by text block containing detailed prompt instructions.

Key Takeaway = image accuracy depends entirely on prompt sophistication, not just image quality.

```py
with open("image.png", "rb") as f:
    image_bytes = base64.standard_b64encode(f.read()).decode("utf-8")

add_user_message(messages, [
    # Image Block
    {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/png",
            "data": image_bytes,
        }
    },
    # Text Block
    {
        "type": "text",
        "text": "What do you see in this image?"
    }
])
```

## PDF Support
PDF Support in Claude:

Claude can read PDF files directly using similar code to image processing. 

Key implementation changes:
- File type = "document" instead of "image"
- Media type = "application/pdf" instead of "image/png"
- Variable naming = file_bytes instead of image_bytes

Claude PDF capabilities = read text + images + charts + tables + mixed content extraction

PDF processing = one-stop solution for comprehensive document analysis

Usage pattern = same as image input but with document-specific parameters

```py
with open("earth.pdf", "rb") as f:
    file_bytes = base64.standard_b64encode(f.read()).decode("utf-8")

messages = []

add_user_message(
    messages,
    [
        {
            "type": "document",
            "source": {
                "type": "base64",
                "media_type": "application/pdf",
                "data": file_bytes,
            },
        },
        {"type": "text", "text": "Summarize the document in one sentence"},
    ],
)

chat(messages)
```

## Citations
Citations = feature allowing Claude to reference source documents and show where information comes from

Citation types:
- citation_page_location = for PDF documents, shows document index/title/start page/end page/cited text
- citation_char_location = for plain text, shows character position in text block

Implementation:
- Add "citations": {"enabled": true} to request
- Add "title" field to identify source document
- Works with both PDF files and plain text sources

Response structure = content becomes list of text blocks, some containing citations arrays with location data

Purpose = transparency for users to verify Claude's information sources and check accuracy of interpretations

UI benefit = enables citation popups/overlays showing source document, page numbers, and exact cited text when users hover over referenced content

Key use case = ensuring users can investigate how Claude builds responses from source materials rather than appearing to speak from memory alone

To enable citations, you need to modify your document message structure. Add two new fields to your document block:
```py
{
    "type": "document",
    "source": {
        "type": "base64",
        "media_type": "application/pdf",
        "data": file_bytes,
    },
    "title": "earth.pdf",
    "citations": { "enabled": True }
}
```
The title field gives your document a readable name, while citations: {"enabled": True} tells Claude to track where it finds information

## Prompt Caching
Prompt Caching = feature that speeds up Claude's responses and reduces text generation costs by reusing computational work from previous requests.

Normal request flow: User sends message → Claude processes input (creates internal data structures, performs calculations) → Claude generates output → Claude discards all processing work → Ready for next request.

Problem: When follow-up requests contain identical input messages, Claude must repeat all the same computational work it just threw away, creating inefficiency.

Solution: Prompt caching stores the results of input message processing in temporary cache instead of discarding. When identical input appears in subsequent requests, Claude retrieves cached work rather than reprocessing, dramatically speeding response generation.

Key benefit: Reuses previous computational work to avoid redundant processing of repeated content.

Prompt caching offers several advantages:

- Faster responses: Requests using cached content execute more quickly
- Lower costs: You pay less for the cached portions of your requests
- Automatic optimization: The initial request writes to the cache, follow-up requests read from it

However, there are important limitations to keep in mind:
- Cache duration: Cached content only lives for one hour
- Limited use cases: Only beneficial when you're repeatedly sending the same content
- High frequency requirement: Most effective when the same content appears extremely frequently in your requests


## Rules of Prompt Caching
Prompt Caching = system that saves processing work from initial request to reuse in follow-up requests with identical content

Core mechanism: Initial request → Claude processes + saves work to cache → Follow-up requests with identical content → Claude retrieves cached work instead of reprocessing

The process is straightforward: your initial request writes processing work to the cache, and follow-up requests can read from that cache instead of reprocessing the same content. The cache lives for one hour, so this feature is only useful if you're repeatedly sending the same content within that timeframe.

Cache duration = 1 hour maximum

Cache activation requires manual cache breakpoint addition to message blocks

Text block formats:
- Shorthand: content = "text string" (cannot add cache control)
- Longhand: content = [{"type": "text", "text": "content", "cache_control": {...}}] (required for caching)

Cache scope = all content up to and including breakpoint gets cached

Cache invalidation = any change in content before breakpoint invalidates entire cache

Content processing order = tools → system prompt → messages (joined together)

Cache breakpoint placement options:
- Tool schemas
- System prompts  
- Message blocks (text, image, tool use, tool result)

Maximum breakpoints = 4 per request

Multiple breakpoints = create multiple cache layers, partial cache hits possible if only later content changes

Minimum cache threshold = 1024 tokens required for content to be cached

Best use cases = repeated identical content (system prompts, tool definitions, static message prefixes)

Cross-Message Caching:

Cache breakpoints can span across multiple messages and message types. If you place a breakpoint in a later message, all previous messages (user, assistant, etc.) will be included in the cached content.

System Prompts and Tools: You're not limited to text blocks - cache breakpoints can be added to:
- System prompts
- Tool definitions
- Image blocks
- Tool use and tool result blocks

You can add up to four cache breakpoints total. For example, you might cache your tools, then add another breakpoint partway through your conversation history. This gives you flexibility in what gets cached when different parts of your request change.

## Prompt Caching in Action
Prompt Caching Implementation = automatically caches tool schemas and system prompts to reduce token usage

Setup = modify chat function to enable caching by default for tools and system prompts

Tool Schema Caching = add cache_control field with type "ephemeral" to last tool in list. Best practice: create copy of tools list, clone last tool schema, add cache control, then overwrite to avoid modifying original schemas

System Prompt Caching = wrap system prompt in text block dictionary with cache_control type "ephemeral"

Multiple Cache Breakpoints = can set cache points for both tools and system prompt in single request

Cache Order = tools → system prompt → messages

Token Usage Patterns:
- cache_creation_input_tokens = tokens written to cache on first use
- cache_read_input_tokens = tokens retrieved from cache on subsequent identical requests
- Partial cache reads possible when some content matches cached data

Cache Invalidation = any change to cached content (tools or system prompt) invalidates cache, forces new cache creation

Use Cases = identical content across requests - same tool schemas, system prompts, or message sequences

**Refer: 003_caching notebook**

## Code Execution and the Files API
Files API = allows uploading files ahead of time and referencing them later via file ID instead of including raw file data in each request. Upload file → get file metadata object with ID → use ID in future requests.

Code Execution = server-based tool where Claude executes Python code in isolated Docker containers. No implementation needed, just include predefined tool schema. Claude can run code multiple times, interpret results, generate final response.

Key constraints: Docker containers have no network access. Data input/output relies on Files API integration.

Combined workflow: Upload file via Files API → get file ID → include ID in container upload block → ask Claude to analyze → Claude writes/executes code with access to uploaded file → returns analysis and results.

Claude can generate files (plots, reports) inside container that can be downloaded using file IDs returned in response.

Use cases: Data analysis, file processing, automated code generation for complex tasks. Response contains code blocks, execution results, and final analysis.

Implementation: Use container upload block with file ID, include analysis prompt, Claude handles code execution automatically.

Here's how it works:
- Upload your file (image, PDF, text, etc.) to Claude using a separate API call
- Receive a file metadata object containing a unique file ID
- Reference that file ID in future messages instead of including raw file data

First, upload the file using a helper function:
```py
file_metadata = upload('streaming.csv')
```

Then create a message that includes both the uploaded file and a request for analysis:
```py
messages = []
add_user_message(
    messages,
    [
        {
            "type": "text",
            "text": """Run a detailed analysis to determine major drivers of churn.
            Your final output should include at least one detailed plot summarizing your findings."""
        },
        {"type": "container_upload", "file_id": file_metadata.id},
    ],
)

chat(
    messages,
    tools=[{"type": "code_execution_20250522", "name": "code_execution"}]
)
```

### Beyond Data Analysis
While data analysis is a natural fit, the combination of Files API and code execution opens up many possibilities:

- Image processing and manipulation
- Document parsing and transformation
- Mathematical computations and modeling
- Report generation with custom formatting
The key is that you can delegate complex, computational tasks to Claude while maintaining control over the inputs and outputs through the Files API. This creates a powerful workflow where Claude becomes your coding assistant that can actually execute and iterate on solutions.

**Refer: 005_code_execution notebook**
