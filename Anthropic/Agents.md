
# Agents
## Agents and Workflows
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


## Parallelization Workflows
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


## Chaining Workflows
Chaining Workflows = breaking large tasks into series of distinct sequential steps rather than single complex prompt

Core concept: Instead of one massive prompt with multiple requirements, split into separate calls where each focuses on one specific subtask.

Example workflow: User enters topic → search trending topics → Claude selects most interesting → Claude researches topic → Claude writes script → generate video → post to social media

Key benefit: Allows AI to focus on individual tasks rather than juggling multiple constraints simultaneously

Primary use case: When Claude consistently ignores constraints in complex prompts despite repetition. Common with long prompts containing many "don't do X" requirements.

Problem scenario: Long prompt with constraints (don't mention AI, no emojis, professional tone) → Claude violates some constraints regardless of repetition

Solution: Step 1 - Send initial prompt, accept imperfect output. Step 2 - Follow-up prompt asking Claude to rewrite based on specific violations found.

Critical insight: Even simple-seeming workflow becomes essential when dealing with constraint-heavy prompts that AI struggles to follow completely in single pass.


## Routing Workflows
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


## Agents and Tools
Agents = AI systems that create plans to complete tasks using provided tools, effective when exact steps are unknown. Workflows = better when precise steps are known.

Key differences: Workflows require predetermined steps, agents dynamically plan using available tools.

Agent advantages: Flexibility to solve variety of tasks with same toolset, can combine tools in unexpected ways.

Tool abstraction principle: Provide generic/abstract tools rather than hyper-specialized ones. Example - Claude code uses bash, web_fetch, file_write (abstract) rather than refactor_tool, install_dependencies (specialized).

Tool combination examples: get_current_datetime + add_duration + set_reminder can solve various time-related tasks through different combinations.

Agent behavior: Can request additional information when needed, combines tools creatively to achieve goals, works best with small set of flexible tools.

Design approach: Give agent abstract tools that can be pieced together rather than single-purpose specialized tools. This enables dynamic problem-solving and unexpected use cases.


## Environment Inspection
Environment Inspection = agents evaluating their environment and action results to understand progress and handle errors.

Core concept: After each action, agents need feedback mechanisms beyond basic tool returns to understand new environment state.

Computer use example: Claude takes screenshot after every action (typing, clicking) to see how environment changed, since it cannot predict exact results of actions like button clicks.

Code editing example: Before modifying files, agents must read current file contents to understand existing state.

Social media video agent applications:
- Use Whisper CPP via bash to generate timestamped captions, verify dialogue placement
- Use FFmpeg to extract video screenshots at intervals, inspect visual results
- Validate video creation meets expectations before posting

Key benefit: Environment inspection enables agents to gauge task progress, detect errors, and adapt to unexpected results rather than operating blindly.


## Workflows vs Agents
Workflows = pre-defined series of calls to Claude with known exact steps. Agents = flexible approach using basic tools that Claude combines to complete unknown tasks.

Key differences:

Task division: Workflows break big tasks into smaller, specific subtasks enabling higher focus and accuracy. Agents handle varied challenges creatively without predetermined steps.

Testing/evaluation: Workflows easier to test due to known execution sequence. Agents harder to test since execution path unpredictable.

User experience: Workflows require specific inputs. Agents create own inputs from user queries and can request additional input when needed.

Success rates: Workflows = higher task completion rates due to structured approach. Agents = lower completion rates due to delegated complexity.

Recommendation: Prioritize workflows for reliability. Use agents only when flexibility truly required. Users want 100% working products over fancy agents.

Core principle: Solve problems reliably first, innovation second.
