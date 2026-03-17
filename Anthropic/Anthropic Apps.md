# Anthropic Apps - Claude Code and Computer Use

## Anthropic Apps
Anthropic Apps = two deployed applications by Anthropic: Claude Code and Computer Use.

Claude Code = terminal-based coding assistant that serves as example of agent architecture.

Computer Use = toolset that expands Claude's capabilities beyond text generation.

Key purpose = these apps demonstrate agent concepts and provide practical examples for understanding agent design and implementation.

Setup process = involves terminal configuration for Claude Code usage on sample projects.

Agent connection = both applications exemplify how agents work, serving as learning models for building effective agents.


## Claude Code Setup
Claude Code = terminal-based coding assistant program that helps with code-related tasks

Core capabilities = search/read/edit files + advanced tools (web fetching, terminal access) + MCP client support for expanded functionality via MCP servers

Setup process:
1. Install Node.js (check with "npm help" command)
2. Run npm install to install Claude Code
3. Execute "claude" command in terminal to login to Anthropic account

Full setup guide = docs.anthropic.com

MCP client functionality = can consume tools from MCP servers to extend capabilities beyond basic file operations

Claude Code comes with a comprehensive set of tools to help with your development workflow:

## What Claude Code can do?

- File operations - Search, read, and edit files in your project
- Terminal access - Run commands directly from the conversation
- Web access - Search documentation, fetch code examples, and more
- MCP Server support - Add additional tools by connecting MCP servers

The MCP integration is particularly powerful because it means you can extend Claude Code's capabilities by adding specialized tools for databases, APIs, or any other services you work with.

## Claude Code in Action
Claude Code = AI coding assistant that functions as a collaborative engineer on projects, not just a code generator.

Key capabilities: project setup, feature design, code writing, testing, deployment, error fixing in production.

Setup workflow:
- Download project, open in editor
- Run \`claude\` command to launch
- Ask Claude to read README and execute setup directions
- Run \`init\` command = Claude scans codebase for architecture/coding style, creates claude.md file
- claude.md = automatically included context for future requests

Memory types: Project (shared), Local, User memory files.

Context management:
- Use # symbol to add specific notes to memory
- Can manually edit claude.md or rerun init to update
- Claude can handle Git operations (staging, committing)

Effective prompting strategies:

Method 1 - Three-step workflow:
1. Identify relevant files, ask Claude to analyze them
2. Describe feature, ask Claude to plan solution (no code yet)
3. Ask Claude to implement the plan

Method 2 - Test-driven development:
1. Provide relevant context
2. Ask Claude to suggest tests for the feature
3. Select and implement chosen tests
4. Ask Claude to write code until tests pass

Core principle: Claude Code = effort multiplier. More detailed instructions = significantly better results. Treat as collaborative engineer, not just code generator.

## The /init Command
When you start working with Claude Code on a project, the first thing you'll want to do is run the /init command. This tells Claude to scan your entire codebase and understand your project's structure, dependencies, coding style, and architecture.

Claude summarizes everything it learns in a special file called CLAUDE.md. This file automatically gets included as context in all future conversations, so Claude remembers important details about your project.

You can have multiple CLAUDE.md files for different scopes:

- Project - Shared between all engineers working on the project
- Local - Your personal notes that aren't checked into git
- User - Used across all your projects

A sample claud.md file:
```text
# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## IMPORTANT: Docs-First Requirement

**Before generating any code, Claude Code MUST first read and reference the relevant documentation files in the `/docs` directory.** Always check `/docs` for existing specs, designs, or guidelines that apply to the task at hand, and ensure all generated code aligns with those documents:

- /docs/ui.md
- /docs/data-fetching.md
- /docs/auth.md
- /docs/data-mutations.md
- /docs/server-components.md
- /docs/routing.md

## Commands

'''bash
npm run dev      # Start development server
npm run build    # Production build
npm run lint     # Run ESLint
'''

No test framework is configured yet.

## Architecture

This is a **Next.js App Router** project using TypeScript and Tailwind CSS v4.

- `src/app/` — All routes and layouts using file-based routing
- `src/app/layout.tsx` — Root layout (Geist fonts, metadata, HTML shell)
- `src/app/page.tsx` — Home page (`/`)
- `src/app/globals.css` — Global styles; uses Tailwind v4 `@import` syntax and CSS custom properties for light/dark theming

Path alias `@/*` maps to `src/*`.

The project is a fresh Create Next App scaffold — no database, auth, or API routes exist yet.
```

## Enhancements with MCP Servers

Claude Code = AI assistant with embedded MCP (Model Context Protocol) client that can connect to MCP servers to expand functionality.

MCP Server Integration = Connect external tools/services to Claude Code via command: \`claude mcp add [server-name] [startup-command]\`

Example Implementation = Document processing server exposing "Document Path to Markdown" tool, allowing Claude Code to read PDF/Word documents by running \`uv run main.py\`

Dynamic Capability Expansion = MCP servers add new functions to Claude Code in real-time without core modifications.

Common Use Cases = Production monitoring (Sentry), project management (Jira), communication (Slack), custom development workflow tools.

Key Benefit = Significant flexibility increase for development workflows through modular server connections.

Setup Process = 1) Create MCP server with tools, 2) Add server to Claude Code with name and startup command, 3) Restart Claude Code to access new capabilities.

Setting Up an MCP Server

Adding an MCP server to Claude Code is straightforward. You use the command line to register your server:
```bash
claude mcp add [server-name] [command-to-start-server]
```
For example, if you have a document processing server that starts with uv run main.py, you'd run:
```bash
claude mcp add documents uv run main.py
```
Once registered, Claude Code will automatically connect to your server when it starts up.

## Parallelizing Claude Code
Parallelizing Claude Code = running multiple Claude instances simultaneously to complete different tasks in parallel

Core Problem = multiple Claude instances modifying same files simultaneously creates conflicts and invalid code

Solution = Git work trees providing isolated workspaces per Claude instance

Git Work Trees = feature creating complete project copies in separate directories, each corresponding to different Git branches

Workflow = create work tree → assign task to Claude instance → work in isolation → commit changes → merge back to main branch

Custom Commands = automating work tree creation/management through .claude/commands directory with markdown files containing prompts

Command Structure = .claude/commands/filename.md with $ARGUMENTS placeholder for dynamic values

Parallel Execution Benefits = single developer commanding virtual team of software engineers, major productivity scaling limited only by engineer's management capacity

Merge Conflicts = Claude automatically resolves conflicts during branch merging process

Cleanup = Claude handles work tree removal after feature completion

Key Advantage = scales to unlimited parallel instances based on developer's capacity to manage simultaneous tasks


## Automated Debugging
Automated Debugging = using AI (Claude) to automatically detect, analyze, and fix production errors without manual intervention.

Core Workflow:
1. GitHub Action runs daily to check production environment
2. Fetches CloudWatch logs from last 24 hours
3. Claude identifies errors, deduplicates them
4. Claude analyzes each error and generates fixes
5. Creates pull request with proposed solutions

Key Components:
- GitHub Actions for scheduling/automation
- AWS CLI for log retrieval
- Claude Code for error analysis and code fixes
- CloudWatch for production error monitoring

Benefits:
- Catches production-only errors (issues not present in development)
- Reduces manual log hunting and debugging time
- Provides context-aware fixes with explanations
- Creates reviewable pull requests for changes

Common Use Case: Configuration errors between environments (invalid model IDs, API keys, etc. that work locally but fail in production)

Implementation Requirements: Repository access, cloud logging service, AI coding assistant, CI/CD pipeline integration.

## Computer Use
Computer Use = Claude's ability to interact with computer interfaces through visual observation and control actions.

Key capabilities:
- Takes screenshots of applications/browsers
- Clicks buttons, types text, navigates interfaces
- Follows multi-step instructions autonomously
- Performs QA testing and automation tasks

How it works:
- Runs in isolated Docker container environment
- User provides instructions via chat interface
- Claude observes screen visually and executes actions
- Generates reports on task completion/results

Primary use cases:
- Automated QA testing of web applications
- UI interaction testing across different scenarios
- Time-saving for repetitive computer tasks
- Bug identification through systematic testing

Setup requirement = Reference implementation available for local testing

Example workflow: User describes testing requirements → Claude navigates to application → Executes test cases → Reports pass/fail results with detailed findings

## How Computer Use Works
Computer use = tool system implementation allowing Claude to interact with computing environments

Tool use flow: User sends message + tool schema → Claude responds with tool use request (ID, name, input) → Server executes code → Result sent back to Claude as tool result

Computer use follows identical flow:
- Special tool schema sent to Claude (small schema expands to larger structure behind scenes)
- Expanded schema includes action function with arguments: mouse move, left click, screenshot, etc.
- Claude sends tool use request
- Developers must fulfill request via computing environment (typically Docker container)
- Container executes programmatic key presses/mouse movements
- Response sent back to Claude

Key points:
- Claude doesn't directly manipulate computers
- Computer use = tool system + developer-provided computing environment
- Anthropic provides reference implementation (Docker container with pre-built mouse/keyboard execution code)
- Setup requires Docker + simple command execution
- Enables direct chat interface for testing Claude's computer use functionality

Computer use = abstraction layer where tool system handles Claude communication while Docker container handles actual computer interactions.

