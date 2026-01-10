"""
State management for AgentArmy orchestration.
"""

import json
import uuid
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, List
from enum import Enum


class ExecutionState(Enum):
    """Execution state of a task."""
    PENDING = "pending"
    PLANNING = "planning"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class AgentExecution:
    """Record of a single agent execution."""
    agent_name: str
    started_at: datetime
    ended_at: Optional[datetime] = None
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0
    success: bool = True
    error: Optional[str] = None
    result: Optional[str] = None


@dataclass
class TaskStep:
    """A single step in the execution plan."""
    step_id: str
    description: str
    agent: str
    status: str = "pending"  # pending, running, completed, failed
    dependencies: List[str] = field(default_factory=list)
    result: Optional[str] = None
    error: Optional[str] = None


@dataclass
class State:
    """
    Main state container for an orchestration execution.
    
    Tracks the entire lifecycle of a task from request to completion.
    """
    
    # Identification
    execution_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    
    # Request
    user_request: str = ""
    project_context: Dict[str, Any] = field(default_factory=dict)
    
    # Execution state
    status: ExecutionState = ExecutionState.PENDING
    current_agent: Optional[str] = None
    
    # Plan
    plan: List[TaskStep] = field(default_factory=list)
    current_step_index: int = 0
    
    # Results
    final_result: Optional[str] = None
    artifacts: Dict[str, Any] = field(default_factory=dict)  # files, code, etc.
    
    # Agent executions
    agent_history: List[AgentExecution] = field(default_factory=list)
    
    # Metrics
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    total_tokens: int = 0
    total_cost: float = 0.0
    
    # Error tracking
    error: Optional[str] = None
    retry_count: int = 0
    
    # Conversation history (for context)
    messages: List[Dict[str, str]] = field(default_factory=list)
    
    def start(self):
        """Mark execution as started."""
        self.status = ExecutionState.PLANNING
        self.started_at = datetime.now()
    
    def complete(self, result: str):
        """Mark execution as completed."""
        self.status = ExecutionState.COMPLETED
        self.final_result = result
        self.ended_at = datetime.now()
    
    def fail(self, error: str):
        """Mark execution as failed."""
        self.status = ExecutionState.FAILED
        self.error = error
        self.ended_at = datetime.now()
    
    def add_agent_execution(self, execution: AgentExecution):
        """Record an agent execution."""
        self.agent_history.append(execution)
        self.total_tokens += execution.input_tokens + execution.output_tokens
        self.total_cost += execution.cost
    
    def add_message(self, role: str, content: str):
        """Add a message to conversation history."""
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
    
    def get_context_for_agent(self, max_messages: int = 10) -> List[Dict[str, str]]:
        """Get conversation context for agent calls."""
        # Get recent messages
        recent = self.messages[-max_messages:] if len(self.messages) > max_messages else self.messages
        
        # Add any completed step results as context
        completed_steps = [s for s in self.plan if s.status == "completed" and s.result]
        if completed_steps:
            context_msg = "Previous step results:\n"
            for step in completed_steps[-3:]:  # Last 3 steps
                context_msg += f"- {step.description}: {step.result[:500]}...\n" if len(step.result or "") > 500 else f"- {step.description}: {step.result}\n"
            recent.insert(0, {"role": "system", "content": context_msg})
        
        return recent
    
    @property
    def duration_seconds(self) -> float:
        """Get execution duration in seconds."""
        if not self.started_at:
            return 0.0
        end = self.ended_at or datetime.now()
        return (end - self.started_at).total_seconds()
    
    @property
    def agents_used(self) -> List[str]:
        """Get list of unique agents used."""
        return list(set(e.agent_name for e in self.agent_history))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for serialization."""
        data = {
            "execution_id": self.execution_id,
            "user_request": self.user_request,
            "project_context": self.project_context,
            "status": self.status.value,
            "current_agent": self.current_agent,
            "plan": [asdict(s) for s in self.plan] if self.plan else [],
            "current_step_index": self.current_step_index,
            "final_result": self.final_result,
            "artifacts": self.artifacts,
            "agent_history": [
                {
                    "agent_name": e.agent_name,
                    "started_at": e.started_at.isoformat() if e.started_at else None,
                    "ended_at": e.ended_at.isoformat() if e.ended_at else None,
                    "input_tokens": e.input_tokens,
                    "output_tokens": e.output_tokens,
                    "cost": e.cost,
                    "success": e.success,
                    "error": e.error,
                }
                for e in self.agent_history
            ],
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost,
            "error": self.error,
            "messages": self.messages,
        }
        return data
    
    def save(self, state_dir: str = "~/.agent_army/state"):
        """Save state to disk."""
        dir_path = Path(state_dir).expanduser()
        dir_path.mkdir(parents=True, exist_ok=True)
        
        file_path = dir_path / f"{self.execution_id}.json"
        with open(file_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, execution_id: str, state_dir: str = "~/.agent_army/state") -> Optional["State"]:
        """Load state from disk."""
        file_path = Path(state_dir).expanduser() / f"{execution_id}.json"
        
        if not file_path.exists():
            return None
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        state = cls(
            execution_id=data["execution_id"],
            user_request=data["user_request"],
            project_context=data.get("project_context", {}),
            status=ExecutionState(data["status"]),
            current_agent=data.get("current_agent"),
            current_step_index=data.get("current_step_index", 0),
            final_result=data.get("final_result"),
            artifacts=data.get("artifacts", {}),
            total_tokens=data.get("total_tokens", 0),
            total_cost=data.get("total_cost", 0.0),
            error=data.get("error"),
            messages=data.get("messages", []),
        )
        
        # Restore plan
        for step_data in data.get("plan", []):
            state.plan.append(TaskStep(**step_data))
        
        # Restore timestamps
        if data.get("started_at"):
            state.started_at = datetime.fromisoformat(data["started_at"])
        if data.get("ended_at"):
            state.ended_at = datetime.fromisoformat(data["ended_at"])
        
        return state
