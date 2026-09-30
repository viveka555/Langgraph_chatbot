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

from app.config import SQLITE_DB, PLANNER_PROMPT, MAX_CONTEXT_TOKENS
from app.llm import llm
from app.tools import tools
from app.schema import PlannerDecision
from app.memory import summarize_conversation, trim_messages, build_context
from app.semantic_memory import (
    extract_memories,
    store_memories,
    retrieve_memories,
)

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

    #summary 
    summary: str

    memories: list[str]


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
# Memory Node
# Responsibility:
# 
# =====================================================
def memory_node(state: ChatState, config):

    messages = state["messages"]

    thread_id = config["configurable"]["thread_id"]

    summary = state.get("summary", "")

    # Summarize long conversations
    if len(messages) > 40:

        summary = summarize_conversation(messages)

        facts = extract_memories(messages)

        store_memories(
            memories=facts,
            thread_id=thread_id,
        )

    # Retrieve only relevant memories
    query = messages[-1].content

    memories = retrieve_memories(
        query=query,
        thread_id=thread_id,
        top_k=3,
    )

    return {
        "summary": summary,
        "memories": memories,
    }

# =====================================================
# Assistant Node
# Responsibility:
# Generate responses and call tools when needed
# =====================================================

def assistant_node(state: ChatState):

    summary = state.get("summary", "")

    memories = state.get("memories", [])

    memory_text = "\n".join(
        f"- {item}" for item in memories
    )

    recent = trim_messages(
        messages=state["messages"],
        max_tokens=MAX_CONTEXT_TOKENS,
    )

    context = build_context(
        summary=summary,
        memories=memory_text,
        recent_messages=recent,
    )

    response = llm_with_tools.invoke(context)

    return {
        "messages": [response],
        "title": state["title"],
    }


# =====================================================
# Router
# Planner → Assistant or End
# =====================================================

def route_decision(state: ChatState):

    if state["route"] == "tool":
        return "assistant"

    return "assistant"

#Actually, in prorecent = trim_messages(messages=state["messges"],max_tokens=2380,)

    context = build_context(
    summary,
    recent_messages,
)
#production you can even remove this routing and always go to Assistant.
# The planner exists for observability (logging, analytics, tracing).

# =====================================================
# Build Graph
# =====================================================

builder = StateGraph(ChatState)

# =====================================================
# Nodes
# =====================================================

builder.add_node("planner", planner_node)
builder.add_node("memory", memory_node)
builder.add_node("assistant", assistant_node)
builder.add_node("tools", ToolNode(tools))

# =====================================================
# Start
# =====================================================

builder.add_edge(START, "planner")

# =====================================================
# Planner → Memory
# =====================================================

builder.add_conditional_edges(
    "planner",
    route_decision,
    {
        "assistant": "memory",
        END: END,
    },
)

# =====================================================
# Memory → Assistant
# =====================================================

builder.add_edge("memory", "assistant")

# =====================================================
# Assistant → ToolNode (only if tool call exists)
# =====================================================

builder.add_conditional_edges(
    "assistant",
    tools_condition,
)

# =====================================================
# ToolNode → Assistant
# =====================================================

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