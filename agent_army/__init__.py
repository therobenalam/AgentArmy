"""
AgentArmy - Standalone AI Agent Orchestration System

A powerful, modular orchestration system with an army of specialized AI agents
for general-purpose task execution.

Usage:
    # CLI
    agent-army execute "Build a REST API with authentication"
    
    # Python
    from agent_army import Orchestrator
    orchestrator = Orchestrator()
    result = await orchestrator.execute("Your task here")
"""

__version__ = "1.0.0"
__author__ = "AgentArmy"

from .orchestrator import Orchestrator
from .state import State, ExecutionState
from .config import Config

__all__ = [
    "Orchestrator",
    "State",
    "ExecutionState", 
    "Config",
    "__version__",
]
