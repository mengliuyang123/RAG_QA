> ## Documentation Index
> Fetch the complete documentation index at: https://docs.langchain.com/llms.txt
> Use this file to discover all available pages before exploring further.

# Quickstart

This quickstart demonstrates how to build a calculator agent with the LangGraph Graph API.

Already have an agent and want to deploy it? See [LangSmith Deployment](https://docs.langchain.com/langsmith/deployment.md).

> [!TIP]
>
> **Using an AI coding assistant?**
>
> * Install the [LangChain Docs MCP servers](https://docs.langchain.com/use-these-docs.md) to give your agent access to up-to-date LangChain documentation and examples.
>
>   Connect LangChain docs MCP servers
>
>   Connect both LangChain documentation MCP servers to my coding agent so it can look up current LangChain, LangGraph, and LangSmith docs and API reference.
>
>   Servers to add:
>
>   * `docs-langchain`: [https://docs.langchain.com/mcp](https://docs.langchain.com/mcp)
>   * `reference-langchain`: [https://reference.langchain.com/mcp](https://reference.langchain.com/mcp)
>
>   Detect which agent or editor I am using (Claude Code, Cursor, Codex CLI, Claude Desktop, Deep Agents Code, VS Code, Antigravity, or another MCP-compatible client). Use the matching setup from [https://docs.langchain.com/use-these-docs.md](https://docs.langchain.com/use-these-docs.md):
>
>   * Claude Code: `claude mcp add --transport http` for each server (project scope by default; use `--scope user` only if I ask for global access).
>   * Codex CLI: `codex mcp add` with each server URL.
>   * Cursor, Deep Agents Code, VS Code, or Antigravity: merge both entries into the MCP settings JSON using the field names shown on that page for my client.
>   * Claude Desktop: add both URLs under Settings > Connectors.
>
>   Do not invent alternate MCP URLs. After configuring, confirm both servers are listed and reachable.
> * Install [LangChain Skills](https://github.com/langchain-ai/langchain-skills) to improve your agent's performance on LangChain ecosystem tasks.
>
>   Install LangChain Skills
>
>   Install LangChain Skills for my coding agent so it can perform better on LangChain, LangGraph, and Deep Agents tasks.
>
>   Use the Agent Skills installer from [https://github.com/langchain-ai/langchain-skills](https://github.com/langchain-ai/langchain-skills):
>
>   ```bash theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
>   npx skills add langchain-ai/langchain-skills --skill '*' --yes
>   ```
>
>   If I ask for a global install instead, use:
>
>   ```bash theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
>   npx skills add langchain-ai/langchain-skills --skill '*' --yes --global
>   ```
>
>   Detect which agent or editor I am using. If I use Claude Code and prefer the plugin path, follow the marketplace install from that repository README (`/plugin marketplace add` then `/plugin install`). Do not invent alternate skill package names or install URLs. After installing, confirm the skills are available to the agent.

## Prerequisites

Before you begin, make sure you have an API key from a model provider (e.g., Gemini, Anthropic, OpenAI).
See [chat model integrations](https://docs.langchain.com/oss/python/integrations/chat.md) for all available providers.

If you use [LangSmith Gateway](https://docs.langchain.com/langsmith/llm-gateway.md), you can [bring your own provider keys](https://docs.langchain.com/langsmith/llm-gateway-quickstart.md) or use [Gateway Credits](https://docs.langchain.com/langsmith/llm-gateway-credits.md) to access models without a provider key.

## Setup

Install `langgraph`, `langchain`, and `langchain-anthropic`:

```bash pip theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
pip install -U langgraph langchain langchain-anthropic
```

```bash uv theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
uv add langgraph langchain langchain-anthropic
```

This example uses Anthropic. Set your API key in the shell or a `.env` file:

```bash Shell theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
export ANTHROPIC_API_KEY="your-api-key"
```

```bash .env theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
ANTHROPIC_API_KEY=your-api-key
```

To use another provider, see [chat model integrations](https://docs.langchain.com/oss/python/integrations/chat.md) and adapt the model initialization in the next steps.

## Build the agent

The model can call tools, then the agent calls the model again until it returns a final answer.

Create one file. Add the code from each step below the previous step.

For conceptual information, see the [Graph API overview](https://docs.langchain.com/oss/python/langgraph/graph-api.md).

Build the LangGraph Graph API calculator quickstart

Build a LangGraph calculator agent with the Graph API in this working directory by following the LangGraph quickstart.

## Step 1: Read the guide

Detect whether this project uses Python or TypeScript/JavaScript. Fetch and follow the matching page; treat it as the source of truth for package names, model strings, and code:

* Python: [https://docs.langchain.com/oss/python/langgraph/quickstart.md](https://docs.langchain.com/oss/python/langgraph/quickstart.md)
* TypeScript: [https://docs.langchain.com/oss/javascript/langgraph/quickstart.md](https://docs.langchain.com/oss/javascript/langgraph/quickstart.md)

## Step 2: Install dependencies

Install the packages required by this quickstart with the package manager already used in this project.

## Step 3: Configure model credentials

This quickstart uses Anthropic by default. Check whether `ANTHROPIC_API_KEY` is set. If it is not, ask the user to create a key and set it in the shell or a `.env` file, then stop and wait for confirmation. Do not invent, hardcode, or commit API keys. If the user prefers another chat model provider from the integrations docs, adapt the example accordingly after confirming.

## Step 4: Implement the calculator agent

Define tools and model, then Graph API state, nodes, edges, and the compiled graph from the guide.

## Step 5: Optional LangSmith tracing

Ask the user if they want tracing. If yes, ask them to set `LANGSMITH_TRACING=true` and `LANGSMITH_API_KEY` themselves. Do not invent a LangSmith key.

## Step 6: Run the agent

Invoke the agent on a sample arithmetic query from the guide and print the final result.

## Rules

* Stay scoped to this quickstart. Do not add deployment, evals, or unrelated frameworks unless the user asks.
* Prefer the structure shown on the fetched guide over inventing a different agent pattern.
* Ask rather than guess when a secret, provider choice, or project convention is unclear.

## 1. Define tools and model

This example uses the Claude Sonnet model and defines tools for addition, multiplication, and division.

Create a file for the agent (`agent.py` or `agent.ts`). Add the following:

```python theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
from langchain.tools import tool
from langchain.chat_models import init_chat_model

model = init_chat_model(
    "claude-sonnet-4-6",
    temperature=0
)

# Define tools
@tool
def multiply(a: int, b: int) -> int:
    """Multiply `a` and `b`.

    Args:
        a: First int
        b: Second int
    """
    return a * b

@tool
def add(a: int, b: int) -> int:
    """Adds `a` and `b`.

    Args:
        a: First int
        b: Second int
    """
    return a + b

@tool
def divide(a: int, b: int) -> float:
    """Divide `a` and `b`.

    Args:
        a: First int
        b: Second int
    """
    return a / b

# Augment the LLM with tools
tools = [add, multiply, divide]
tools_by_name = {tool.name: tool for tool in tools}
model_with_tools = model.bind_tools(tools)
```

## 2. Define state

The graph's state stores the messages and the number of LLM calls.

Add the following below the tools and model:

> [!TIP]
>
> State in LangGraph persists throughout the agent's execution.
>
> The `Annotated` type with `operator.add` appends new messages to the list instead of replacing it.

```python theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
from langchain.messages import AnyMessage
from typing_extensions import TypedDict, Annotated
import operator

class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int
```

## 3. Define model node

The model node calls the LLM and decides whether to call a tool.

Add the following below the state schema:

```python theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
from langchain.messages import SystemMessage

def llm_call(state: dict):
    """LLM decides whether to call a tool or not"""

    return {
        "messages": [
            model_with_tools.invoke(
                [
                    SystemMessage(
                        content="You are a helpful assistant tasked with performing arithmetic on a set of inputs."
                    )
                ]
                + state["messages"]
            )
        ],
        "llm_calls": state.get('llm_calls', 0) + 1
    }
```

## 4. Define tool node

The tool node calls the tools and returns the results.

Add the following below the model node:

```python theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
from langchain.messages import ToolMessage

def tool_node(state: dict):
    """Performs the tool call"""

    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
    return {"messages": result}
```

## 5. Define end logic

The conditional edge function routes to the tool node or to the end, depending on whether the LLM made a tool call.

Add the following below the tool node:

```python theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
from typing import Literal
from langgraph.graph import StateGraph, START, END

def should_continue(state: MessagesState) -> Literal["tool_node", END]:
    """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""

    messages = state["messages"]
    last_message = messages[-1]

    # If the LLM makes a tool call, then perform an action
    if last_message.tool_calls:
        return "tool_node"

    # Otherwise, we stop (reply to the user)
    return END
```

## 6. Build and compile the agent

Add the following below the routing function. Build the graph with [`StateGraph`](https://reference.langchain.com/python/langgraph/graph/state/StateGraph) and compile it with [`compile`](https://reference.langchain.com/python/langgraph/graph/state/StateGraph/compile):

```python theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
# Build workflow
agent_builder = StateGraph(MessagesState)

# Add nodes
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)

# Add edges to connect nodes
agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges(
    "llm_call",
    should_continue,
    ["tool_node", END]
)
agent_builder.add_edge("tool_node", "llm_call")

# Compile the agent
agent = agent_builder.compile()
```

## 7. Set up LangSmith tracing

[LangSmith](https://smith.langchain.com?utm_source=docs\&utm_medium=cta\&utm_campaign=langsmith-signup\&utm_content=oss-langgraph-quickstart) shows tool calls and model responses as the agent runs.

This step is optional. Sign up at [smith.langchain.com](https://smith.langchain.com?utm_source=docs\&utm_medium=cta\&utm_campaign=langsmith-signup\&utm_content=oss-langgraph-quickstart), create an API key, and set these environment variables in the shell before you run the agent:

```bash theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
export LANGSMITH_TRACING=true
export LANGSMITH_API_KEY="your-langsmith-api-key"
```

## 8. Run the agent

Add the following below the compiled graph:

```python theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
from langchain.messages import HumanMessage

messages = [HumanMessage(content="Add 3 and 4.")]
messages = agent.invoke({"messages": messages})
for m in messages["messages"]:
    m.pretty_print()
```

Run the file:

```bash theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
python agent.py
```

You should see the human message, a tool call to `add`, the tool result `7`, and a final assistant reply. The exact text varies by model, but the tool result is `7`.

To check your file, compare it with the full example:

> [accordion] Full code example
>
> ```python theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
> # Step 2: Define tools and model
> 
> from langchain.tools import tool
> from langchain.chat_models import init_chat_model
> 
> 
> model = init_chat_model(
>  "claude-sonnet-4-6",
>  temperature=0
> )
> 
> 
> # Define tools
> @tool
> def multiply(a: int, b: int) -> int:
>  """Multiply `a` and `b`.
> 
>  Args:
>      a: First int
>      b: Second int
>  """
>  return a * b
> 
> 
> @tool
> def add(a: int, b: int) -> int:
>  """Adds `a` and `b`.
> 
>  Args:
>      a: First int
>      b: Second int
>  """
>  return a + b
> 
> 
> @tool
> def divide(a: int, b: int) -> float:
>  """Divide `a` and `b`.
> 
>  Args:
>      a: First int
>      b: Second int
>  """
>  return a / b
> 
> 
> # Augment the LLM with tools
> tools = [add, multiply, divide]
> tools_by_name = {tool.name: tool for tool in tools}
> model_with_tools = model.bind_tools(tools)
> 
> # Step 3: Define state
> 
> from langchain.messages import AnyMessage
> from typing_extensions import TypedDict, Annotated
> import operator
> 
> 
> class MessagesState(TypedDict):
>  messages: Annotated[list[AnyMessage], operator.add]
>  llm_calls: int
> 
> # Step 4: Define model node
> from langchain.messages import SystemMessage
> 
> 
> def llm_call(state: MessagesState):
>  """LLM decides whether to call a tool or not"""
> 
>  return {
>      "messages": [
>          model_with_tools.invoke(
>              [
>                  SystemMessage(
>                      content="You are a helpful assistant tasked with performing arithmetic on a set of inputs."
>                  )
>              ]
>                 + state["messages"]
>             )
>         ],
>         "llm_calls": state.get('llm_calls', 0) + 1
>     }
> 
> 
> # Step 5: Define tool node
> 
> from langchain.messages import ToolMessage
> 
> 
> def tool_node(state: MessagesState):
>     """Performs the tool call"""
> 
>     result = []
>     for tool_call in state["messages"][-1].tool_calls:
>         tool = tools_by_name[tool_call["name"]]
>         observation = tool.invoke(tool_call["args"])
>         result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
>     return {"messages": result}
> 
> # Step 6: Define end logic
> 
> from typing import Literal
> from langgraph.graph import StateGraph, START, END
> 
> 
> # Conditional edge function to route to the tool node or end based upon whether the LLM made a tool call
> def should_continue(state: MessagesState) -> Literal["tool_node", END]:
>     """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""
> 
>     messages = state["messages"]
>     last_message = messages[-1]
> 
>     # If the LLM makes a tool call, then perform an action
>     if last_message.tool_calls:
>         return "tool_node"
> 
>     # Otherwise, we stop (reply to the user)
>     return END
> 
> # Step 7: Build and compile the agent
> 
> # Build workflow
> agent_builder = StateGraph(MessagesState)
> 
> # Add nodes
> agent_builder.add_node("llm_call", llm_call)
> agent_builder.add_node("tool_node", tool_node)
> 
> # Add edges to connect nodes
> agent_builder.add_edge(START, "llm_call")
> agent_builder.add_conditional_edges(
>     "llm_call",
>     should_continue,
>     ["tool_node", END]
> )
> agent_builder.add_edge("tool_node", "llm_call")
> 
> # Compile the agent
> agent = agent_builder.compile()
> 
> 
> # Step 8: Set up LangSmith tracing (optional)
> # export LANGSMITH_TRACING=true
> # export LANGSMITH_API_KEY="your-langsmith-api-key"
> 
> # Step 9: Run the agent
> from langchain.messages import HumanMessage
> messages = [HumanMessage(content="Add 3 and 4.")]
> messages = agent.invoke({"messages": messages})
> for m in messages["messages"]:
>     m.pretty_print()
> ```

## Next steps

You now have a working calculator agent. Continue with one of the following subjects:

* **Build with the Graph API**: [Build checklist for the Graph API](https://docs.langchain.com/oss/python/langgraph/use-graph-api.md) and [Thinking in the Graph API](https://docs.langchain.com/oss/python/langgraph/thinking-in-graph-api.md).
* **Add persistence**: Use a [checkpointer](https://docs.langchain.com/oss/python/langgraph/persistence.md) so runs can pause and resume across sessions.
* **Add human-in-the-loop**: Pause with [interrupts](https://docs.langchain.com/oss/python/langgraph/interrupts.md) for approval or edits.
* **Stream results**: Follow [streaming](https://docs.langchain.com/oss/python/langgraph/streaming.md) for tokens and state updates.
* **Go deeper**: [Workflows and agents](https://docs.langchain.com/oss/python/langgraph/workflows-agents.md).
* **Deploy**: Host the agent with [LangSmith Deployment](https://docs.langchain.com/langsmith/deployment.md).
* **Evaluate**: Test behavior with [LangSmith evaluation](https://docs.langchain.com/langsmith/evaluation-quickstart.md).

***

[Connect these docs](https://docs.langchain.com/use-these-docs.md) to your agent of choice via MCP for real-time answers.

[Edit this page on GitHub](https://github.com/langchain-ai/docs/edit/main/src/oss/langgraph/quickstart.mdx) or [file an issue](https://github.com/langchain-ai/docs/issues/new/choose).