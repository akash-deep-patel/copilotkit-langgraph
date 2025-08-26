"""
This is the main entry point for the agent.
It defines the workflow graph, state, tools, nodes and edges.
"""

from typing import Any, List
from typing_extensions import Literal
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, BaseMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.types import Command
from langgraph.graph import MessagesState
# from langgraph.prebuilt import ToolNode  # Removed to avoid version mismatch

class AgentState(MessagesState):
    """
    Here we define the state of the agent

    In this instance, we're inheriting from CopilotKitState, which will bring in
    the CopilotKitState fields. We're also adding a custom field, `language`,
    which will be used to set the language of the agent.
    """
    proverbs: List[str] = []
    tools: List[Any]
    # your_custom_agent_state: str = ""

@tool
def get_weather(location: str):
    """
    Get the weather for a given location.
    """
    return f"The weather for {location} is 80 degrees."

@tool
def evaluate_math_expr(expr: str):
    """Evaluate a mathematical expression."""
    print(f"Evaluating expression: {expr}")
    try:
        result = eval(expr)
        return f"The result of {expr} is {result}."
    except Exception as e:
        return f"Error evaluating {expr}: {e}"

backend_tools = [
    get_weather,
    evaluate_math_expr,
    # your_tool_here

]

# Extract tool names from backend_tools for comparison
backend_tool_names = [tool.name for tool in backend_tools]


async def chat_node(state: AgentState, config: RunnableConfig) -> Command[Literal["tool_node", "__end__"]]:
    """
    Standard chat node based on the ReAct design pattern. It handles:
    - The model to use (and binds in CopilotKit actions and the tools defined above)
    - The system prompt
    - Getting a response from the model
    - Handling tool calls

    For more about the ReAct design pattern, see:
    https://www.perplexity.ai/search/react-agents-NcXLQhreS0WDzpVaS4m9Cg
    """

    # 1. Define the model
    model = ChatOpenAI(model="gpt-4o")

    # 2. Bind the tools to the model
    model_with_tools = model.bind_tools(
        [
            *state.get("tools", []), # bind tools defined by ag-ui
            *backend_tools,
            # your_tool_here
        ],

        # 2.1 Disable parallel tool calls to avoid race conditions,
        #     enable this for faster performance if you want to manage
        #     the complexity of running tool calls in parallel.
        parallel_tool_calls=False,
    )

    # 3. Define the system message by which the chat model will be run
    system_message = SystemMessage(
        content=f"You are a helpful assistant. The current proverbs are {state.get('proverbs', [])}."
    )

    # 4. Run the model to generate a response
    response = await model_with_tools.ainvoke([
        system_message,
        *state["messages"],
    ], config)

    # only route to tool node if tool is not in the tools list
    if route_to_tool_node(response):
        print("routing to tool node")
        return Command(
            goto="tool_node",
            update={
                "messages": [response],
            }
        )

    # 5. We've handled all tool calls, so we can end the graph.
    return Command(
        goto=END,
        update={
            "messages": [response],
        }
    )

def route_to_tool_node(response: BaseMessage):
    """
    Route to tool node if any tool call in the response matches a backend tool name.
    """
    tool_calls = getattr(response, "tool_calls", None)
    if not tool_calls:
        return False

    for tool_call in tool_calls:
        if tool_call.get("name") in backend_tool_names:
            return True
    return False

async def tool_node(state: AgentState, config: RunnableConfig) -> Command[Literal["chat_node"]]:
    """
    Execute tool calls emitted by the last AI message for backend-defined tools.
    Appends resulting ToolMessage(s) and routes back to chat.
    """
    last_message = state["messages"][-1] if state.get("messages") else None
    tool_calls = getattr(last_message, "tool_calls", None) if last_message else None

    if not tool_calls:
        return Command(
            goto="chat_node",
            update={}
        )

    name_to_tool = {t.name: t for t in backend_tools}
    new_messages = []

    for call in tool_calls:
        tool_name = call.get("name")
        tool_args = call.get("args") or {}
        tool_call_id = call.get("id") or ""

        if tool_name not in name_to_tool:
            continue

        tool_fn = name_to_tool[tool_name]
        try:
            # LangChain tool objects are callable; many also expose .invoke
            if hasattr(tool_fn, "invoke"):
                result = tool_fn.invoke(tool_args, config=config)
            else:
                # Fallback to direct call with kwargs
                result = tool_fn(**tool_args)
        except Exception as e:
            result = f"Tool '{tool_name}' errored: {e}"

        new_messages.append(
            ToolMessage(
                content=str(result),
                name=tool_name,
                tool_call_id=tool_call_id,
            )
        )

    return Command(
        goto="chat_node",
        update={
            "messages": new_messages,
        }
    )

# Define the workflow graph
workflow = StateGraph(AgentState)
workflow.add_node("chat_node", chat_node)
workflow.add_node("tool_node", tool_node)
workflow.add_edge("tool_node", "chat_node")
workflow.set_entry_point("chat_node")

graph = workflow.compile()
