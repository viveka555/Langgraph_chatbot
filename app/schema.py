# =====================================================
# app/schema.py
# Structured outputs used by LangGraph
# =====================================================

from typing import Literal
from pydantic import BaseModel, Field


class PlannerDecision(BaseModel):
    """
    Structured output produced by the Planner node.
    """

    # Should the workflow use tools?
    route: Literal["tool", "direct"]

    # Why this routing decision was made
    reason: str | None = None