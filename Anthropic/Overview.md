# Overview of Claude Models
https://github.com/anthropics/courses/tree/master/anthropic_api_fundamentals

Claude has three model families optimized for different priorities:

Opus = highest intelligence model for complex, multi-step tasks requiring deep reasoning and planning. Trade-off: higher cost and latency.

Sonnet = balanced model with good intelligence, speed, and cost efficiency. Strong coding abilities and precise code editing. Best for most practical use cases.

Haiku = fastest model optimized for speed and cost efficiency. No reasoning capabilities like Opus/Sonnet. Best for real-time user interactions and high-volume processing.

Selection framework: Intelligence priority → Opus. Speed priority → Haiku. Balanced requirements → Sonnet.

Common approach = use multiple models in same application based on specific task requirements rather than single model selection.

All models share core capabilities: text generation, coding, image analysis. Main difference is optimization focus.

# Accessing Claude with the API

## Accessing the API
API Access Flow = 5-step process from user input to response display

Step 1: Client sends user text to developer's server (never access Anthropic API directly from client apps to keep API key secret)

Step 2: Server makes request to Anthropic API using SDK (Python, TypeScript, JavaScript, Go, Ruby) or plain HTTP. Required parameters = API key + model name + messages list + max_tokens limit

Step 3: Text generation process has 4 stages:
- Tokenization = breaking input into tokens (words/word parts/symbols/spaces)
- Embedding = converting tokens to number lists representing all possible word meanings
- Contextualization = adjusting embeddings based on neighboring tokens to determine precise meaning
- Generation = output layer produces probabilities for next word, model selects using probability + randomness, adds selected word, repeats process

Step 4: Model stops when max_tokens reached or special end_of_sequence token generated

Step 5: API returns response with generated text + usage counts + stop_reason to server, server sends to client for display

- Token = text chunk (word/part/symbol)
- Embedding = numerical representation of word meanings
- Contextualization = meaning refinement using neighboring words
- Max_tokens = generation length limit
- Stop_reason = why model stopped generating


## Making a Request
Making API Request to Anthropic = Process involving 4 setup steps and understanding message structure

Setup Steps:
1. Install packages = pip install anthropic python-dotenv in Jupyter notebook
2. Store API key = Create .env file with ANTHROPIC_API_KEY="your_key" (ignore in version control)
3. Load environment variable = Use python-dotenv to securely load API key
4. Create client = Initialize anthropic client and define model variable (claude-3-sonnet)

API Request Structure:
- Function = client.messages.create()
- Required arguments = model, max_tokens, messages
- Model = Name of Claude model to use
- Max_tokens = Safety limit for generation length (not target length)
- Messages = List containing conversation exchanges

Message Types:
- User message = {"role": "user", "content": "your text"} (human-authored content)
- Assistant message = Contains model-generated responses

Response Access:
- Full response = Contains metadata and nested structure
- Text only = message.content[0].text extracts just generated text

Example request structure: 
```py
client.messages.create(model=model, max_tokens=1000, messages=[{"role": "user", "content": "What is quantum computing?"}])
```

## Multi-Turn Conversations
Multi-Turn Conversations = conversations with multiple back-and-forth exchanges that maintain context.

Key limitation: Anthropic API stores no messages. Each request is independent with no memory of previous exchanges.

Solution requires two steps:
1. Manually maintain message list in code
2. Send entire conversation history with every follow-up request

Message structure = list of dictionaries with "role" (user/assistant) and "content" fields.

Conversation flow:
- Send initial user message
- Receive assistant response
- Append assistant response to message history
- Add new user message to history
- Send complete history for context-aware follow-up

Helper functions needed:
- add_user_message(messages, text) = appends user message to history
- add_assistant_message(messages, text) = appends assistant response to history  
- chat(messages) = sends message history to API and returns response

```py
def add_user_message(messages, text):
    user_message = {"role": "user", "content": text}
    messages.append(user_message)

def add_assistant_message(messages, text):
    assistant_message = {"role": "assistant", "content": text}
    messages.append(assistant_message)

def chat(messages):
    message = client.messages.create(
        model=model,
        max_tokens=1000,
        messages=messages,
    )
    return message.content[0].text
```

Putting it all together:
```py
# Start with an empty message list
messages = []

# Add the initial user question
add_user_message(messages, "Define quantum computing in one sentence")

# Get Claude's response
answer = chat(messages)

# Add Claude's response to the conversation history
add_assistant_message(messages, answer)

# Add a follow-up question
add_user_message(messages, "Write another sentence")

# Get the follow-up response with full context
final_answer = chat(messages)
```

Without message history = responses lack context and continuity. With complete history = Claude maintains conversation context and provides relevant follow-ups.


## System Prompts
System Prompts = technique to customize Claude's response style and tone by assigning it a specific role or behavior pattern.

Implementation = pass system prompt as plain string to create function using system keyword argument.

Purpose = control how Claude responds rather than what it responds. Example: math tutor role makes Claude give hints instead of direct answers.

Structure = first line typically assigns role ("You are a patient math tutor"), followed by specific behavioral instructions.

Key principle = system prompts guide response approach, not content. Same question gets different treatment based on assigned role.

Technical implementation = create params dictionary, conditionally add system key if prompt provided, pass params to create function with ** unpacking. Handle None case by excluding system parameter entirely.

Use case example = Math tutor that gives guidance/hints rather than complete solutions, encouraging student thinking over direct answers.


## Temperature
Temperature = parameter (0-1) that controls randomness in Claude's text generation by influencing token selection probabilities.

Text generation process: Input text → tokenization → probability assignment to possible next tokens → token selection based on probabilities → repeat.

Temperature effects:
- Temperature 0 = deterministic output, always selects highest probability token
- Higher temperature = increases chance of selecting lower probability tokens, more creative/unexpected outputs

Usage guidelines:
- Low temperature (near 0) = data extraction, factual tasks requiring consistency
- High temperature (near 1) = creative tasks like brainstorming, writing, jokes, marketing

Implementation: Add temperature parameter to model API calls. Higher values don't guarantee different outputs, just increase probability of variation.

Key insight: Temperature directly manipulates the probability distribution of next token selection, making high-probability tokens more/less dominant in the selection process.

```py
def chat(messages, system=None, temperature=1.0):
    params = {
        "model": model,
        "max_tokens": 1000,
        "messages": messages,
        "temperature": temperature
    }
    
    if system:
        params["system"] = system
    
    message = client.messages.create(**params)
    return message.content[0].text
```

## Response Streaming
Response Streaming = technique to display AI responses chunk-by-chunk as they're generated instead of waiting for complete response.

Problem solved: AI responses can take 10-30 seconds. Users expect immediate feedback, not just spinners.

How it works:
1. Server sends user message to Claude
2. Claude immediately sends initial response (no text, just acknowledgment)
3. Stream of events follows, each containing text chunks
4. Server forwards chunks to frontend for real-time display

Event types:
- message_start = initial acknowledgment
- content_block_start = text generation begins
- content_block_delta = contains actual text chunks (most important)
- content_block_stop/message_stop = generation complete

Implementation:
- Basic: client.messages.create(stream=True) returns event iterator
- Simplified: client.messages.stream() with text_stream property extracts just text
- Final message: stream.get_final_message() assembles all chunks for storage

Key benefits: Better UX through immediate response visibility, complete message capture for database storage.

```py
with client.messages.stream(
    model=model,
    max_tokens=1000,
    messages=messages
) as stream:
    for text in stream.text_stream:
        # Send each chunk to your client or print(text, end=" ")
        pass
    
    # Get the complete message for database storage or logs
    final_message = stream.get_final_message()
```

## Controlling Model Output
**Controlling Model Output = Two key techniques beyond prompt modification**

**Pre-filling Assistant Messages = Manually adding assistant message at end of conversation to steer response direction**

How it works:
- Assemble messages list with user prompt + manual assistant message
- Claude sees assistant message as already authored content
- Claude continues response from exact end of pre-filled text
- Response gets steered toward pre-filled direction

Key point: Claude continues from exact endpoint of pre-fill, not complete sentences. Must stitch together pre-fill + generated response.

Example: Pre-fill "Coffee is better because" → Claude continues with justification for coffee

The key thing to understand is that Claude continues from exactly where your prefilled text ends. If you write "Coffee is better because", Claude won't repeat that text - it will pick up right after "because" and complete the thought.

Here's the code structure:
```py
messages = []
add_user_message(messages, "Is tea or coffee better at breakfast?")
add_assistant_message(messages, "Coffee is better because")
answer = chat(messages)
```
You can steer Claude in any direction using this technique:

- Favor coffee: "Coffee is better because"
- Favor tea: "Tea is better because"
- Take a contrarian stance: "Neither is very good because"

**Stop Sequences = Force Claude to halt generation when specific string appears**

How it works:
- Provide stop sequence string in chat function
- When Claude generates that exact string, response immediately stops
- Generated stop sequence text not included in final output

Example: Prompt "count 1 to 10" + stop sequence "five" → Output stops at "four, " (five not included)

Refinement: Stop sequence ", five" → Clean output "one, two, three, four"

The concept is straightforward: you provide a list of strings, and as soon as Claude generates any of those strings, it stops responding immediately. The stop sequence itself is not included in the final response.

```py
def chat(messages, system=None, temperature=1.0, stop_sequences=[]):
    params = {
        "model": model,
        "max_tokens": 1000,
        "messages": messages,
        "temperature": temperature,
        "stop_sequences": stop_sequences
    }
    
    if system:
        params["system"] = system
    
    message = client.messages.create(**params)
    return message.content[0].text
```

```
def chat(messages, stop_sequences=[]):
    # Add stop_sequences to your API call parameters
```
Then use it like this:

messages = []
add_user_message(messages, "Count from 1 to 10")
answer = chat(messages, stop_sequences=["5"])
You can fine-tune exactly where the stopping occurs. If you want to avoid trailing punctuation, use a more specific stop sequence like ", 5" instead of just "5".

Both techniques provide precise control over response direction and length without changing core prompts.


## Structured Data
Structured Data Generation = technique using assistant message prefilling + stop sequences to get raw output without Claude's natural explanatory headers/footers.

Problem = Claude automatically adds markdown formatting, headers, commentary when generating JSON/code/structured content such as ```json .... ```. Users often want just the raw data for copy/paste functionality.

Solution Pattern:
1. User message = request for structured data
2. Assistant message prefill = opening delimiter (e.g., "\`\`\`json")  
3. Stop sequence = closing delimiter (e.g., "\`\`\`")

How it works = Claude sees prefilled message, assumes it already started response, generates only the requested content, stops when hitting delimiter.

Result = Raw structured data output with no extra formatting or commentary.

Application = Works for any structured data type (JSON, Python code, lists, etc.), not just JSON. Use whenever you need clean, parseable output without explanatory text.

Key benefit = Output can be directly used/copied without manual selection or parsing of unwanted text.

The Solution: Assistant Message Prefilling + Stop Sequences

You can combine assistant message prefilling with stop sequences to get exactly the content you want. Here's how it works:
```py
messages = []

add_user_message(messages, "Generate a very short event bridge rule as json")
add_assistant_message(messages, "```json")

text = chat(messages, stop_sequences=["```"])
```
This technique works by:

- The user message tells Claude what to generate
- The prefilled assistant message makes Claude think it already started a markdown code block
- Claude continues by writing just the JSON content
- When Claude tries to close the code block with ```, the stop sequence immediately ends generation

The result is clean JSON with no extra formatting
```py
import json

# Clean up and parse the JSON
clean_json = json.loads(text.strip()) #no need to check for json markdown tag
```

# Prompt Engineering vs Prompt Evaluation
Prompt engineering is your toolkit for crafting effective prompts. It includes techniques like:

Prompt evaluation takes a different approach. Instead of focusing on how to write prompts, it's about measuring their effectiveness through automated testing.

# Prompt Evaluation

## Prompt Evaluation
Prompt Engineering = techniques for writing/editing prompts to help Claude understand requests and desired responses.

Prompt Evaluation = automated testing of prompts using objective metrics to measure effectiveness.

Three paths after writing a prompt:
1. Test once/twice, deploy to production (trap)
2. Test with custom inputs, minor tweaks for corner cases (trap)  
3. Run through evaluation pipeline for objective scoring (recommended)

Key takeaway: Engineers commonly under-test prompts. Use evaluation pipelines to get objective performance scores before iterating and deploying prompts.

## A Typical Eval Workflow
Typical Eval Workflow = 6-step iterative process for prompt improvement

Step 1: Write initial prompt draft - create baseline prompt to optimize

Step 2: Create evaluation dataset - collection of test inputs (can be 3 examples or thousands, hand-written or LLM-generated)

Step 3: Generate prompt variations - interpolate each dataset input into prompt template

Step 4: Get LLM responses - feed each prompt variation to Claude, collect outputs

Step 5: Grade responses - use grader system to score each response (e.g. 1-10 scale), average scores for overall prompt performance

Step 6: Iterate - modify prompt based on scores, repeat entire process, compare versions

Key points: No standard methodology exists. Many open-source/paid tools available. Can start simple with custom implementation. Grading complexity varies. Objective scoring enables systematic prompt improvement through A/B comparison.


## Generating Test Datasets
Custom prompt evaluation workflow = build prompt + generate test dataset + evaluate performance

Goal = AWS code assistance prompt that outputs only Python, JSON config, or regex without explanations

Dataset generation approaches = manual assembly or automated with Claude (use faster models like Haiku for generation)

Dataset structure = array of JSON objects with task property describing user requests

Generation process = prompt Claude to create test cases → use pre-filling with assistant message "\`\`\`json" → set stop sequence "\`\`\`" → parse response as JSON → save to file

Key implementation = generate_dataset() function that sends prompt to Claude, gets structured JSON response of test tasks, saves to dataset.json file for later evaluation use

Test dataset enables systematic evaluation by running prompt against multiple input scenarios to measure performance consistency.


## Running the Eval
Eval execution process = merging test cases with prompts, running through LLM, and grading outputs.

Test case = individual record from dataset (JSON object).

Three core functions:
- run_prompt = merges test case with prompt, sends to Claude, returns output
- run_test_case = calls run_prompt, grades result, returns summary dictionary 
- run_eval = loops through dataset, calls run_test_case for each, assembles results

Basic prompt structure = "Please solve the following task: [test_case_task]" (v1 starting point).

Current limitations = no output formatting instructions, hardcoded scoring (score=10), verbose Claude responses.

Runtime = ~31 seconds with Haiku model for full dataset execution.

Output format = array of objects containing Claude output, original test case, and score.

Next step = implement proper grading system to replace hardcoded scores.

Eval pipeline core = dataset + prompt + LLM + grader, with minimal code complexity.


Check 001_prompt_evals_complete.ipynb for implementation level details


## Model Based Grading
Check Check 001_prompt_evals_complete.ipynb for implementation level details

Model Based Grading = evaluation system that takes model outputs and assigns objective scores (typically 1-10 scale, 10 = highest quality)

Three grader types:
- Code graders = programmatic checks (length, word presence, syntax validation, readability scores)
- Model graders = additional API call to evaluate original model output, highly flexible for quality/instruction-following assessment
- Human graders = person evaluates responses, most flexible but time-consuming and tedious

Key requirements: Must return objective signal (usually numerical score). Define evaluation criteria upfront.

Implementation pattern for model graders:
- Create detailed prompt requesting strengths/weaknesses/reasoning/score (not just score alone to avoid default middling scores)
- Use JSON response format with pre-filled assistant message and stop sequences
- Parse returned JSON for score and reasoning
- Calculate average scores across test cases for final metric

Model graders offer high flexibility but may be inconsistent. Still provides objective baseline for prompt optimization.


## Code Based Grading
Code Based Grading = automated validation system for LLM outputs containing code, JSON, or regex

Core Implementation:
- validate_json() = attempts JSON parsing, returns 10 if valid, 0 if error
- validate_python() = attempts AST parsing, returns 10 if valid, 0 if error  
- validate_regex() = attempts regex compilation, returns 10 if valid, 0 if error

Dataset Requirements:
- Must include "format" key specifying expected output type (JSON/Python/RegEx)
- Updated via prompt template modification for automated dataset generation

Prompt Engineering:
- Instruct model to respond only with raw code/JSON/regex
- No comments, explanations, or commentary
- Use pre-filled Assistant message with \`\`\`code\`\`\` blocks
- Add stop sequences to extract clean output

Scoring System:
- Final score = (model_score + syntax_score) / 2
- Combines semantic evaluation with syntax validation
- Enables measurement of both correctness and technical validity

Key Limitation = requires known expected format for proper validator selection

```py
def validate_json(text):
    try:
        json.loads(text.strip())
        return 10
    except json.JSONDecodeError:
        return 0

def validate_python(text):
    try:
        ast.parse(text.strip())
        return 10
    except SyntaxError:
        return 0

def validate_regex(text):
    try:
        re.compile(text.strip())
        return 10
    except re.error:
        return 0
```

Combine Scores:
```py
model_grade = grade_by_model(test_case, output)
model_score = model_grade["score"]
syntax_score = grade_syntax(output, test_case)

score = (model_score + syntax_score) / 2
```

# Prompt Engineering Techniques

## Prompt Engineering
Prompt Engineering = improving prompts to get more reliable, higher-quality outputs from language models.

Module Structure: Start with initial poor prompt → Apply prompt engineering techniques step-by-step → Evaluate improvements after each technique → Observe performance gains over time.

Example Goal: Generate one-day meal plan for athletes based on height, weight, physical goal, dietary restrictions.

Technical Setup:
- Updated eval pipeline with flexible prompt evaluator class
- Supports concurrency (adjust max_concurrent_tasks based on rate limits)
- generate_dataset() method creates test cases with specified inputs
- run_prompt() function processes each test case individually

Key Components:
- prompt_input_spec = dictionary defining required prompt inputs
- extra_criteria = additional validation requirements for model grading
- output.html = formatted evaluation report showing test case results and scores

Process: Write initial prompt → Interpolate test case inputs → Run evaluation → Apply engineering techniques → Re-evaluate → Repeat until satisfactory performance.

Initial Results: Expect poor scores (example: 2.32) with basic prompts, especially when using less capable models. Scores improve as techniques are applied.


## Being Clear and Direct
Being Clear and Direct = Use simple, direct language with action verbs in the first line of prompts to specify the exact task.

First line importance = Most critical part of prompt that sets the foundation for AI response.

Structure = Action verb + clear task description + output specifications.

Examples:
- "Write three paragraphs about how solar panels work"
- "Identify three countries that use geothermal energy and for each include generation stats"
- "Generate a one day meal plan for an athlete that meets their dietary restrictions"

Key components = Action verb at start + direct task statement + expected output details.

Result = Improved prompt performance (example showed score increase from 2.32 to 3.92).


## Being Specific
Being Specific = adding guidelines or steps to direct model output in particular direction

Two types of guidelines:
Type A (Attributes) = list qualities/attributes desired in output (length, structure, format)
Type B (Steps) = provide specific steps for model to follow in reasoning process

Type A controls output characteristics. Type B controls how model arrives at answer.

Both techniques often combined in professional prompts.

When to use:
- Type A (attributes): recommended for almost all prompts
- Type B (steps): use for complex problems where you want model to consider broader perspective or additional viewpoints it might not naturally consider

Example improvement: meal planning prompt score jumped from 3.92 to 7.86 when guidelines added, demonstrating significant quality improvement through specificity.

## Structure with XML Tags
XML Tags for Prompt Structure = Using XML tags to organize and delineate different content sections within prompts to improve AI comprehension.

Purpose = When interpolating large amounts of content into prompts, XML tags help AI models distinguish between different types of information and understand text grouping.

Implementation = Wrap content sections in descriptive XML tags like <sales_records></sales_records> or <my_code></my_code> rather than dumping unstructured text.

Tag naming = Use descriptive, specific tag names (e.g., "sales_records" better than "data") to provide context about content nature.

Example use case = Debugging prompt with mixed code and documentation becomes clearer when separated into <my_code> and <docs> tags.

Benefits = Makes prompt structure obvious to AI, reduces confusion about content boundaries, improves output quality even for smaller content blocks.

Application = Can wrap any interpolated content like <athlete_information> even when content is short, to clarify it's external input requiring consideration.

XML tags are most useful when:

- Including large amounts of context or data
- Mixing different types of content (code, documentation, data)
- You want to be extra clear about content boundaries
- Working with complex prompts that interpolate multiple variables

Even for shorter content, XML tags can help serve as delimiters that make your prompt structure more obvious to Claude.

In practice, you might structure a prompt like this:
```xml
<athlete_information>
- Height: 6'2"
- Weight: 180 lbs
- Goal: Build muscle
- Dietary restrictions: Vegetarian
</athlete_information>
```
Generate a meal plan based on the athlete information above.

## Providing Examples
One-shot/Multi-shot prompting = providing examples in prompts to guide model behavior. One-shot = single example, multi-shot = multiple examples.

Implementation: Structure examples with XML tags containing sample input and ideal output. Always wrap examples clearly to distinguish from actual prompt content.

Key applications:
- Corner case handling (sarcasm detection, edge scenarios)
- Complex output formatting (JSON structures, specific formats)
- Clarifying expected response quality/style

Best practices:
- Add context for corner cases ("be especially careful with sarcasm")
- Include reasoning explaining why output is ideal
- Use highest-scoring examples from prompt evaluations as templates
- Place examples after main instructions/guidelines

Effectiveness boost: Combine examples with explanations of what makes them ideal to reinforce desired output characteristics.

**Adding Context to Examples**
Don't just provide the input/output pair - explain why the output is good:
```xml
<ideal_output>
[Your example output here]
</ideal_output>

This example is well-structured, provides detailed information 
on food choices and quantities, and aligns with the athlete's 
goals and restrictions.
```
This additional context helps Claude understand the reasoning behind good responses, not just the format.

Examples are especially powerful because they show rather than tell. Instead of trying to describe exactly what you want in words, you demonstrate it directly. This makes your prompts much more reliable and helps Claude understand subtle requirements that might be hard to express in instructions alone.



Agents
<note title="Agents and Workflows">
Workflows and agents = strategies for handling user tasks that can't be completed by Claude in a single request.

Decision rule: Use workflows when you have precise task understanding and know exact steps sequence. Use agents when task details are unclear.

Workflow = series of calls to Claude for specific problems where steps are predetermined.

Example workflow: Image to 3D model converter
- Step 1: Claude describes uploaded image in detail
- Step 2: Claude uses CADQuery Python library to model object from description
- Step 3: Create rendering of model
- Step 4: Claude compares rendering to original image
- Step 5: If inaccurate, repeat from step 2 with feedback

This follows evaluator-optimizer pattern:
- Producer = generates output (Claude + CADQuery modeling)
- Evaluator = assesses output quality (comparison step)
- Loop continues until evaluator accepts output

Key point: Workflows are implementation patterns that other engineers have successfully used. Identifying workflow patterns doesn't automatically implement them - you still need to write the actual code.
</note>

<note title="Parallelization Workflows">
Parallelization Workflows = breaking one complex task into multiple simultaneous subtasks, then aggregating results.

Example: Material selection for parts
- Instead of: One large prompt asking Claude to choose between metal/polymer/ceramic/composite with all criteria
- Use: Separate parallel requests, each evaluating one material's suitability, then final aggregation step to compare results

Structure: Input → Multiple parallel subtasks → Aggregator → Final output

Benefits:
- Focus = Each subtask handles one specific analysis instead of juggling multiple considerations
- Modularity = Individual prompts can be improved/evaluated separately  
- Scalability = Easy to add new subtasks without affecting existing ones
- Quality = Reduces confusion from overly complex single prompts

Key principle: Decompose complex decisions into specialized parallel analyses, then synthesize results.
</note>

<note title="Chaining Workflows">
Chaining Workflows = breaking large tasks into series of distinct sequential steps rather than single complex prompt

Core concept: Instead of one massive prompt with multiple requirements, split into separate calls where each focuses on one specific subtask.

Example workflow: User enters topic → search trending topics → Claude selects most interesting → Claude researches topic → Claude writes script → generate video → post to social media

Key benefit: Allows AI to focus on individual tasks rather than juggling multiple constraints simultaneously

Primary use case: When Claude consistently ignores constraints in complex prompts despite repetition. Common with long prompts containing many "don't do X" requirements.

Problem scenario: Long prompt with constraints (don't mention AI, no emojis, professional tone) → Claude violates some constraints regardless of repetition

Solution: Step 1 - Send initial prompt, accept imperfect output. Step 2 - Follow-up prompt asking Claude to rewrite based on specific violations found.

Critical insight: Even simple-seeming workflow becomes essential when dealing with constraint-heavy prompts that AI struggles to follow completely in single pass.
</note>

<note title="Routing Workflows">
Routing Workflows = workflow pattern that categorizes user input to determine appropriate processing pipeline

Key mechanism: Initial request to Claude categorizes user input into predefined genres/categories. Based on categorization response, system routes to specialized processing pipeline with customized prompts/tools.

Example flow:
1. User enters topic (e.g., "Python functions")
2. Claude categorizes topic (e.g., "educational")
3. System uses educational-specific prompt template
4. Claude generates script with educational tone/structure

Benefits: Ensures output matches topic nature. Programming topics get educational treatment with definitions/explanations. Entertainment topics get trendy language/engaging hooks.

Structure: One routing step → Multiple specialized processing pipelines → Each pipeline has customized prompts/tools for specific category

Use case: Social media video script generation where different topics require different tones and approaches.
</note>

<note title="Agents and Tools">
Agents = AI systems that create plans to complete tasks using provided tools, effective when exact steps are unknown. Workflows = better when precise steps are known.

Key differences: Workflows require predetermined steps, agents dynamically plan using available tools.

Agent advantages: Flexibility to solve variety of tasks with same toolset, can combine tools in unexpected ways.

Tool abstraction principle: Provide generic/abstract tools rather than hyper-specialized ones. Example - Claude code uses bash, web_fetch, file_write (abstract) rather than refactor_tool, install_dependencies (specialized).

Tool combination examples: get_current_datetime + add_duration + set_reminder can solve various time-related tasks through different combinations.

Agent behavior: Can request additional information when needed, combines tools creatively to achieve goals, works best with small set of flexible tools.

Design approach: Give agent abstract tools that can be pieced together rather than single-purpose specialized tools. This enables dynamic problem-solving and unexpected use cases.
</note>

<note title="Environment Inspection">
Environment Inspection = agents evaluating their environment and action results to understand progress and handle errors.

Core concept: After each action, agents need feedback mechanisms beyond basic tool returns to understand new environment state.

Computer use example: Claude takes screenshot after every action (typing, clicking) to see how environment changed, since it cannot predict exact results of actions like button clicks.

Code editing example: Before modifying files, agents must read current file contents to understand existing state.

Social media video agent applications:
- Use Whisper CPP via bash to generate timestamped captions, verify dialogue placement
- Use FFmpeg to extract video screenshots at intervals, inspect visual results
- Validate video creation meets expectations before posting

Key benefit: Environment inspection enables agents to gauge task progress, detect errors, and adapt to unexpected results rather than operating blindly.
</note>

<note title="Workflows vs Agents">
Workflows = pre-defined series of calls to Claude with known exact steps. Agents = flexible approach using basic tools that Claude combines to complete unknown tasks.

Key differences:

Task division: Workflows break big tasks into smaller, specific subtasks enabling higher focus and accuracy. Agents handle varied challenges creatively without predetermined steps.

Testing/evaluation: Workflows easier to test due to known execution sequence. Agents harder to test since execution path unpredictable.

User experience: Workflows require specific inputs. Agents create own inputs from user queries and can request additional input when needed.

Success rates: Workflows = higher task completion rates due to structured approach. Agents = lower completion rates due to delegated complexity.

Recommendation: Prioritize workflows for reliability. Use agents only when flexibility truly required. Users want 100% working products over fancy agents.

Core principle: Solve problems reliably first, innovation second.
</note>
</notes>
