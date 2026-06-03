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
