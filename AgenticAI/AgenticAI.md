Generative AI

Generative AI systems are reactive; they wait for user prompts and generate content such as text, images, code, or audio based on learned patterns from large datasets.
They function as sophisticated pattern-matching machines, predicting what comes next but do not take further actions without human input.

Agentic AI

Agentic AI systems are proactive, using user prompts to pursue goals through a cycle of perceiving the environment, deciding actions, executing them, and learning from outcomes with minimal human intervention.
These systems excel in managing multi-step processes, such as personal shopping agents that autonomously handle tasks like price monitoring and checkout.

Common Foundations and Reasoning

Both AI types often rely on large language models (LLMs), which power chatbots and provide reasoning capabilities for agentic systems.
Agentic AI uses chain of thought reasoning, breaking complex tasks into smaller steps, enabling internal dialogue and decision-making, exemplified by planning a conference.

Future Outlook

The most advanced AI systems will likely combine generative and agentic capabilities, intelligently deciding when to generate content and when to take autonomous actions, enhancing collaboration and efficiency.

### What is chain of thought reasoning in agentic AI?

Chain of thought reasoning in agentic AI is a process where the AI breaks down a complex task into smaller, logical steps to solve problems more effectively. It mimics how humans think through difficult problems by generating an internal dialogue or step-by-step reasoning before taking action.

For example, when planning a conference, an agent might:
```text
First understand the requirements (size, duration, budget).
Then research suitable venues.
Next, check availability for those venues.
And continue breaking down the task until it can act.
```
This reasoning ability is powered by large language models (LLMs) and helps the agent make informed decisions and plan multi-step actions autonomously.


## How can agentic AI be applied to automate multi-step tasks?

Agentic AI can automate multi-step tasks by proactively managing a sequence of actions with minimal human input. It perceives the environment, decides on the next action, executes it, learns from the outcome, and repeats this cycle.

## What are AI Agents?
AI Agents can be characterized as autonomous software entities designed for goal-directed task execution within specific digital environments. Their operation typically involves three capabilities:

Autonomy: The ability to function with minimal human intervention after initial deployment, capable of perceiving environmental inputs, reasoning over contextual data, and executing actions in real-time.

Task-Specificity: Each agent is optimized for narrow, well-defined tasks such as email filtering, database querying, and so on.

Reactivity: Agents respond to inputs from users, APIs, or other software environments in real time.

## What Is Agentic AI?
Agentic AI takes things a step further. Instead of just one agent doing one job, Agentic AI brings multiple agents together into a team. These agents coordinate tasks, exchange information, adapt roles dynamically, and share memory. Key features include:

- Task Decomposition: Goals are split into subtasks automatically.
- Inter-Agent Communication: Agents share updates and results via messaging or shared memory.
- Memory and Reflection: Agents remember past steps and learn from outcomes.
- Orchestration: A lead agent or system coordinates the team.

Example: Planning a vacation—one agent books the flight, another finds hotels and a third checks visa requirements. A coordinator agent makes sure everything matches your preferences.

## Summary
Generative AI is a reactive system that creates content like text or images based on prompts. Agentic AI, on the other hand, is proactive and uses prompts to pursue goals. 

LangGraph is an advanced framework designed for building stateful, multiagent applications. 

Nodes are functions that do the actual computation. Edges define how the execution flows from one step to the next. State is a shared memory that remembers everything across nodes. 

LangGraph's unique capabilities include looping and branching for making dynamic decisions, state persistence to maintain context over long interactions, human-in-the-loop functionality for timely human interventions, and time travel   to facilitate convenient debugging. 

LangGraph offers state management, allowing the workflow to maintain and modify context across different nodes. It also offers conditional transitions, enabling the workflow to make decisions at runtime and branch accordingly.

A LangGraph workflow can branch, loop, pause for human input, and resume execution, all while preserving full conversational memory.

LangGraph graphs can be visualized using Mermaid diagrams with core primitives such as nodes and edges clearly represented.

LangChain helps developers build LLM-powered applications using modular components like prompts, memory, and tools. LangGraph, on the other hand, extends LangChain's capabilities by enabling stateful, multiagent workflows

State in LangGraph is a complex, evolving memory that contains all inputs, intermediate values, and outputs.

Nodes are functions that process the current state. Some nodes modify the state, whereas others are used for side effects.

Edges define how the execution flows between nodes, passing the updated state from one step to the next.

Conditional edges allow the workflow to make dynamic decisions, routing the state to different nodes.

Building a LangGraph application involves creating a StateGraph object, incorporating nodes, connecting them, setting an entry point, and then compiling the graph into a runnable application.

Running a LangGraph workflow is done by invoking the compiled application with an initial state.

Workflow visualization helps to understand the execution flow and how the state progresses through different nodes.


## Reflection Agents and Their Roles

Reflection agents use a generator to produce content and a reflector to critique and improve it iteratively.
The process involves multiple cycles where the generator refines its output based on the reflector’s feedback, enhancing the quality of responses.

Using LangChain and LangGraph for Agent Development

LangChain is used to create structured prompts and guide the generator and reflector LLMs with chat prompt templates and system messages.

Understanding Reflexion Agents: A Simple Explanation

Reflexion agents are like smart helpers that get better by learning from their own mistakes. Imagine you ask a friend for advice on eating more minerals. They give you an answer, then think about it, check if it’s right using a quick internet search, and improve their advice with facts and sources. They keep doing this back-and-forth until the advice is clear, accurate, and trustworthy. This process is called Reflexion, where the agent generates a response, critiques it, uses tools like web searches to find new information, and revises the answer with references.

LangGraph manages the conversational workflow by tracking message states and connecting nodes for generation and reflection in a loop.

## What are the key components of a React agent's reasoning process?

The key components of a React agent's reasoning process are:

- Thought: The agent thinks about what to do next or what information it needs.
- Action: The agent decides which tool to use to get that information.
- Action Input: The specific input or query given to the chosen tool.
- Observation: The result or response returned by the tool after performing the action.
- Final Answer: The complete response the agent provides based on the observations and reasoning.
  
These components work together in a structured sequence to enable step-by-step reasoning and tool use.

# Agentic RAG pipeline.

Retrieval Augmented Generation (RAG) Basics

RAG enhances large language model (LLM) responses by incorporating relevant data retrieved from a vector database as context in the prompt.
This grounding in concrete information improves the quality and reliability of the LLM's generated output.

Agentic RAG Pipeline

- Unlike typical RAG that calls the LLM once for response generation, agentic RAG uses the LLM as an agent to make intelligent decisions.
- The agent can decide which vector database to query among multiple sources and determine the type of response (text, chart, code snippet) based on query context.

Multi-Source Data and Intelligent Query Routing

- The pipeline can include multiple vector databases, such as internal documentation and general industry knowledge.
- The agent interprets the query context to route it to the most relevant database or a fail-safe response if the query is unrelated to available data.
- This approach improves relevance, accuracy, and adaptability, with applications in customer support, legal tech, healthcare, and more.


## Essential design patterns for AI systems using LangGraph: sequential (prompt chaining), routing, and parallelization.

### Sequential (Prompt Chaining)

Passes output from one LLM as input to the next, breaking complex tasks into manageable steps.

Example: generating a resume summary from a job description, then using that summary to create a cover letter.

### Routing Pattern

Uses a router agent to analyze input and decide which specialized agent to invoke (e.g., summarize or translate).

Routes the workflow based on the task type, storing results in state variables.

### Parallelization Pattern

Runs multiple independent LLM tasks simultaneously to improve speed and throughput.

Example: translating text into multiple languages in parallel, then aggregating the results into a combined output.

### LangGraph Implementation
Each pattern uses state variables, nodes, and directed edges to structure workflows for clear execution flow and outputs.

## The Orchestrator Design Pattern

Imagine you're organizing a big party with many different types of food, like Italian pasta, Mexican tacos, and Indian curry. Instead of cooking everything yourself, you act like a party planner who assigns each dish to a specialist chef. The planner (orchestrator) listens to the guest requests, breaks down the big task into smaller parts (dishes), and sends each part to the right chef. These chefs work at the same time, cooking their dishes in parallel. Once all the dishes are ready, another helper (the synthesizer) combines all the recipes into one complete dinner guide.

In this pattern, the orchestrator manages the whole process dynamically, meaning it adapts to whatever the guests ask for, even if the menu changes daily. It keeps track of shared information (like the overall menu) and also lets each chef keep their own notes for their specific dish. This way, the workflow is flexible and can handle many tasks at once, making it perfect for complex and changing requests.

The orchestrator pattern improves workflow flexibility by allowing dynamic task assignment and parallel execution based on real-time input, unlike static workflows that require knowing all steps in advance.

Key points:
- Dynamic task breakdown: The orchestrator analyzes incoming requests and breaks them into tasks on the fly, adapting to changing complexity.
- Parallel worker coordination: It assigns tasks to multiple specialized workers simultaneously, speeding up processing.
- State management: It keeps shared context and individual worker states separate but accessible, enabling coordinated yet independent task handling.
- Scalability: The number of workers can increase or decrease based on task complexity, making the system flexible to varying workloads.

This flexibility means the workflow can handle unpredictable or evolving problems efficiently, unlike static workflows that are rigid and predefined.

## Evaluator-Optimizer design pattern used to iteratively refine AI-generated outputs until they meet target criteria.

Evaluator-Optimizer Pattern Overview

- A query is sent to a generator LLM, whose output is evaluated by an evaluator LLM.
- If the output is rejected, feedback is provided and the process repeats until acceptance.

Multi-Agent Investment Advisor Example

- Investor profile input leads to a target risk grade generated by an LLM.
- The generator has two personas: Kathy Wood creates an initial investment plan; Warren Buffett evaluates it.
- Based on feedback, Ray Dalio refines the plan iteratively with Warren Buffett’s evaluation until the plan meets the target risk grade or iteration limit.

State Management and Workflow Construction

- State variables track investor profile, investment plan, risk grades, feedback, and iteration count.
- Nodes are created for grading, generation (Kathy Wood and Ray Dalio), and evaluation (Warren Buffett).
- A graph connects these nodes with conditional loops to implement the reflection loop for plan refinement.

Key Takeaways

- The pattern uses iterative feedback loops to optimize LLM outputs.
- Multiple LLM personas simulate different investment strategies and evaluations.
- The workflow graph orchestrates the entire investment planning process from input to final plan.

The state variable in the Evaluator-Optimizer pattern plays a crucial role by holding and tracking key data throughout the iterative process. Specifically, it:

- Stores the investor profile and investment plan strings.
- Keeps the target risk grade and the current risk grade.
- Holds evaluator feedback for improving the plan.
- Maintains an iteration counter to track how many refinement cycles have occurred.

This centralized state management enables the system to pass relevant information between the generator and evaluator nodes, manage feedback loops, and decide when to stop iterations based on criteria like matching risk grades or reaching iteration limits.
