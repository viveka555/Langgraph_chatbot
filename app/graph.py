# =====================================================
# app/graph.py
# LangGraph Workflow (Production Version)
# =====================================================

from typing import TypedDict, Annotated

from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver

from app.config import SQLITE_DB, PLANNER_PROMPT
from app.llm import llm
from app.tools import tools
from app.schema import PlannerDecision
# =====================================================
# Graph State
# Shared memory available to every node
# =====================================================

class ChatState(TypedDict):
    # Complete conversation history
    messages: Annotated[list[BaseMessage], add_messages]

    # Sidebar chat title
    title: str

    # Planner decision
    route: str


# =====================================================
# LLM with Tool Calling
# =====================================================

llm_with_tools = llm.bind_tools(tools)


# =====================================================
# Planner Node
# Responsibility:
# Decide whether the query requires tools
# =====================================================

def planner_node(state: ChatState):

    planner = llm.with_structured_output(
        PlannerDecision,
        method="json_mode"
    )

    decision = planner.invoke([
        ("system", PLANNER_PROMPT),
        ("human", state["messages"][-1].content)
    ])

    route = decision.route

    # Defensive normalization
    if isinstance(route, list):
        route = route[0]

    return {
        "route": route
    }
# =====================================================
# Assistant Node
# Responsibility:
# Generate responses and call tools when needed
# =====================================================

def assistant_node(state: ChatState):

    response = llm_with_tools.invoke(state["messages"])

    return {
        "messages": [response],
        "title": state["title"]
    }


# =====================================================
# Router
# Planner → Assistant or End
# =====================================================

def route_decision(state: ChatState):

    if state["route"] == "tool":
        return "assistant"

    return "assistant"

#Actually, in production you can even remove this routing and always go to Assistant.
# The planner exists for observability (logging, analytics, tracing).

# =====================================================
# Build Graph
# =====================================================

builder = StateGraph(ChatState)

# Nodes
builder.add_node("planner", planner_node)
builder.add_node("assistant", assistant_node)
builder.add_node("tools", ToolNode(tools))

# Start
builder.add_edge(START, "planner")

# Planner routing
builder.add_conditional_edges(
    "planner",
    route_decision,
    {
        "assistant": "assistant",
        
    },
)

# Assistant → ToolNode (only if tool calls exist)
builder.add_conditional_edges(
    "assistant",
    tools_condition,
)
builder.add_edge("assistant", END)
# Tool execution → Assistant
builder.add_edge("tools", "assistant")


# =====================================================
# SQLite Checkpointer
# =====================================================

checkpointer_cm = SqliteSaver.from_conn_string(
    str(SQLITE_DB)
)

checkpointer = checkpointer_cm.__enter__()

chatbot = builder.compile(
    checkpointer=checkpointer
)