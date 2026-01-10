"""
Base agent class and agent registry for AgentArmy.
"""

from .base import BaseAgent, AgentRegistry
from .planner import PlannerAgent
from .research import ResearchAgent
from .implementation import ImplementationAgent
from .testing import TestingAgent
from .analysis import AnalysisAgent

__all__ = [
    "BaseAgent",
    "AgentRegistry",
    "PlannerAgent",
    "ResearchAgent",
    "ImplementationAgent",
    "TestingAgent",
    "AnalysisAgent",
]

# Auto-register all agents
def register_default_agents():
    """Register all default agents."""
    AgentRegistry.register("planner", PlannerAgent)
    AgentRegistry.register("research", ResearchAgent)
    AgentRegistry.register("implementation", ImplementationAgent)
    AgentRegistry.register("testing", TestingAgent)
    AgentRegistry.register("analysis", AnalysisAgent)

register_default_agents()
