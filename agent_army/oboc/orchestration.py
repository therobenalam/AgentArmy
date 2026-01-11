"""
OBOC Orchestration Patterns

Implements the five orchestration patterns from OBOC best practices:

1. SEQUENTIAL: [Input] → [Agent A] → [Agent B] → [Agent C] → [Output]
   Best for: Predetermined pipelines, progressive refinement

2. CONCURRENT (Fan-Out/Fan-In): Input → Router → [A, B, C parallel] → Aggregator → Output
   Best for: Parallel analysis, multi-source research
   Performance: 90.2% improvement over single-agent for breadth-first queries

3. GROUP CHAT: Chat Manager coordinates turn-taking between agents
   Best for: Collaborative ideation, quality validation, maker-checker workflows

4. DYNAMIC ROUTING: Query → Router → [best matching agent]
   Routing Methods: Rule-based, Semantic, LLM-based, Auction-based

5. ORCHESTRATOR/MANAGER: Manager builds dynamic task ledger, iterates, backtracks
   Best for: Open-ended problems, incident response, research tasks
"""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional, List, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from .tiers import BaseOBOCAgent, AgentContext


class OrchestrationPattern(Enum):
    """Available orchestration patterns."""
    SEQUENTIAL = "sequential"
    CONCURRENT = "concurrent"
    GROUP_CHAT = "group_chat"
    DYNAMIC_ROUTING = "dynamic_routing"
    MANAGER = "manager"


@dataclass
class AgentResult:
    """Result from an agent execution."""
    agent_name: str
    success: bool
    output: Any
    execution_time: float = 0.0
    tokens_used: int = 0
    cost: float = 0.0
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OrchestrationResult:
    """Result from an orchestration pattern execution."""
    pattern: OrchestrationPattern
    success: bool
    results: List[AgentResult]
    final_output: Any
    total_time: float = 0.0
    total_cost: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseOrchestrationPattern(ABC):
    """Base class for orchestration patterns."""
    
    def __init__(self, name: str, pattern: OrchestrationPattern):
        self.name = name
        self.pattern = pattern
    
    @abstractmethod
    async def execute(
        self,
        agents: List["BaseOBOCAgent"],
        task: str,
        context: "AgentContext",
    ) -> OrchestrationResult:
        """Execute the orchestration pattern."""
        pass


class SequentialOrchestration(BaseOrchestrationPattern):
    """
    SEQUENTIAL ORCHESTRATION
    
    [Input] → [Agent A] → [Agent B] → [Agent C] → [Output]
    
    Best for:
    - Predetermined pipelines
    - Progressive refinement
    - Contract generation (Draft → Review → Legal Check → Finalize)
    """
    
    def __init__(self):
        super().__init__("sequential", OrchestrationPattern.SEQUENTIAL)
    
    async def execute(
        self,
        agents: List["BaseOBOCAgent"],
        task: str,
        context: "AgentContext",
    ) -> OrchestrationResult:
        """Execute agents sequentially, passing output to next agent."""
        start_time = datetime.now()
        results = []
        current_input = task
        final_output = None
        success = True
        
        for agent in agents:
            agent_start = datetime.now()
            try:
                result = await agent.execute(current_input, context)
                execution_time = (datetime.now() - agent_start).total_seconds()
                
                agent_result = AgentResult(
                    agent_name=agent.name,
                    success=result.get("success", True),
                    output=result.get("result", result),
                    execution_time=execution_time,
                )
                results.append(agent_result)
                
                # Pass output to next agent
                current_input = str(result.get("result", result))
                final_output = result
                
                if not result.get("success", True):
                    success = False
                    break
                    
            except Exception as e:
                results.append(AgentResult(
                    agent_name=agent.name,
                    success=False,
                    output=None,
                    error=str(e),
                    execution_time=(datetime.now() - agent_start).total_seconds(),
                ))
                success = False
                break
        
        total_time = (datetime.now() - start_time).total_seconds()
        total_cost = sum(r.cost for r in results)
        
        return OrchestrationResult(
            pattern=self.pattern,
            success=success,
            results=results,
            final_output=final_output,
            total_time=total_time,
            total_cost=total_cost,
        )


class ConcurrentOrchestration(BaseOrchestrationPattern):
    """
    CONCURRENT ORCHESTRATION (Fan-Out/Fan-In)
    
                    ┌──► [Agent A] ──┐
                    │                │
    [Input] ──► [Router]├──► [Agent B] ──├──► [Aggregator] ──► [Output]
                    │                │
                    └──► [Agent C] ──┘
    
    Best for:
    - Parallel analysis
    - Multi-source research
    
    Performance: 90.2% improvement over single-agent for breadth-first queries
    """
    
    def __init__(self, aggregator: Optional[Callable] = None):
        super().__init__("concurrent", OrchestrationPattern.CONCURRENT)
        self.aggregator = aggregator or self._default_aggregator
    
    def _default_aggregator(self, results: List[AgentResult]) -> Any:
        """Default aggregator: combine all outputs."""
        outputs = []
        for result in results:
            if result.success and result.output:
                outputs.append({
                    "agent": result.agent_name,
                    "output": result.output,
                })
        return {
            "combined_results": outputs,
            "total_agents": len(results),
            "successful": sum(1 for r in results if r.success),
        }
    
    async def execute(
        self,
        agents: List["BaseOBOCAgent"],
        task: str,
        context: "AgentContext",
    ) -> OrchestrationResult:
        """Execute all agents concurrently and aggregate results."""
        start_time = datetime.now()
        
        # Create tasks for all agents
        async def run_agent(agent):
            agent_start = datetime.now()
            try:
                result = await agent.execute(task, context)
                return AgentResult(
                    agent_name=agent.name,
                    success=result.get("success", True),
                    output=result.get("result", result),
                    execution_time=(datetime.now() - agent_start).total_seconds(),
                )
            except Exception as e:
                return AgentResult(
                    agent_name=agent.name,
                    success=False,
                    output=None,
                    error=str(e),
                    execution_time=(datetime.now() - agent_start).total_seconds(),
                )
        
        # Run all agents concurrently
        tasks = [run_agent(agent) for agent in agents]
        results = await asyncio.gather(*tasks)
        
        # Aggregate results
        final_output = self.aggregator(list(results))
        
        total_time = (datetime.now() - start_time).total_seconds()
        total_cost = sum(r.cost for r in results)
        success = any(r.success for r in results)
        
        return OrchestrationResult(
            pattern=self.pattern,
            success=success,
            results=list(results),
            final_output=final_output,
            total_time=total_time,
            total_cost=total_cost,
        )


@dataclass
class ChatMessage:
    """A message in the group chat."""
    agent_name: str
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    message_type: str = "contribution"  # contribution, critique, validation, summary


class GroupChatOrchestration(BaseOrchestrationPattern):
    """
    GROUP CHAT ORCHESTRATION
    
             ┌─────────────────────────────────────┐
             │           Chat Manager              │
             │    (Coordinates turn-taking)        │
             └─────────────────────────────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        ▼                    ▼                    ▼
    [Agent A]           [Agent B]            [Agent C]
    "Proposes"          "Critiques"          "Validates"
    
    Best for:
    - Collaborative ideation
    - Quality validation
    - Maker-checker workflows
    """
    
    def __init__(
        self,
        max_rounds: int = 5,
        turn_order: Optional[List[str]] = None,
        consensus_required: bool = False,
    ):
        super().__init__("group_chat", OrchestrationPattern.GROUP_CHAT)
        self.max_rounds = max_rounds
        self.turn_order = turn_order
        self.consensus_required = consensus_required
    
    async def execute(
        self,
        agents: List["BaseOBOCAgent"],
        task: str,
        context: "AgentContext",
    ) -> OrchestrationResult:
        """Execute group chat with coordinated turn-taking."""
        start_time = datetime.now()
        results = []
        chat_history: List[ChatMessage] = []
        
        # Initialize with the task
        chat_history.append(ChatMessage(
            agent_name="user",
            content=task,
            message_type="task",
        ))
        
        # Determine turn order
        if self.turn_order:
            agent_order = [a for a in agents if a.name in self.turn_order]
        else:
            agent_order = agents
        
        # Execute rounds
        for round_num in range(self.max_rounds):
            round_contributions = []
            
            for agent in agent_order:
                # Build context with chat history
                chat_context = self._build_chat_context(chat_history)
                full_task = f"{task}\n\nDiscussion so far:\n{chat_context}"
                
                agent_start = datetime.now()
                try:
                    result = await agent.execute(full_task, context)
                    
                    agent_result = AgentResult(
                        agent_name=agent.name,
                        success=result.get("success", True),
                        output=result.get("result", result),
                        execution_time=(datetime.now() - agent_start).total_seconds(),
                        metadata={"round": round_num},
                    )
                    results.append(agent_result)
                    round_contributions.append(agent_result)
                    
                    # Add to chat history
                    chat_history.append(ChatMessage(
                        agent_name=agent.name,
                        content=str(result.get("result", result)),
                        message_type="contribution",
                    ))
                    
                except Exception as e:
                    results.append(AgentResult(
                        agent_name=agent.name,
                        success=False,
                        output=None,
                        error=str(e),
                        execution_time=(datetime.now() - agent_start).total_seconds(),
                    ))
            
            # Check for consensus if required
            if self.consensus_required and self._check_consensus(round_contributions):
                break
        
        # Generate final summary
        final_output = self._generate_summary(chat_history)
        
        total_time = (datetime.now() - start_time).total_seconds()
        total_cost = sum(r.cost for r in results)
        success = any(r.success for r in results)
        
        return OrchestrationResult(
            pattern=self.pattern,
            success=success,
            results=results,
            final_output=final_output,
            total_time=total_time,
            total_cost=total_cost,
            metadata={"chat_history": [m.__dict__ for m in chat_history]},
        )
    
    def _build_chat_context(self, history: List[ChatMessage]) -> str:
        """Build context string from chat history."""
        lines = []
        for msg in history:
            lines.append(f"[{msg.agent_name}]: {msg.content}")
        return "\n".join(lines)
    
    def _check_consensus(self, contributions: List[AgentResult]) -> bool:
        """Check if agents have reached consensus."""
        # Simple heuristic: all agents agree
        # In production, this would use semantic similarity
        return len(contributions) > 0 and all(c.success for c in contributions)
    
    def _generate_summary(self, history: List[ChatMessage]) -> Dict[str, Any]:
        """Generate a summary of the discussion."""
        return {
            "total_messages": len(history),
            "participants": list(set(m.agent_name for m in history)),
            "final_contributions": [
                {"agent": m.agent_name, "content": m.content[:200]}
                for m in history[-3:]
            ],
        }


class DynamicRoutingOrchestration(BaseOrchestrationPattern):
    """
    DYNAMIC ROUTING
    
                    ┌──► [GitHub Agent]
                    │
    [Query] ──► [Router]├──► [Slack Agent]
                    │
                    └──► [Knowledge Base Agent]
    
    Routing Methods:
    - Rule-based (keyword matching)
    - Semantic (embedding similarity)
    - LLM-based (contextual reasoning)
    - Auction-based (agent bidding)
    """
    
    def __init__(
        self,
        routing_method: str = "rule_based",
        routing_rules: Optional[Dict[str, List[str]]] = None,
        fallback_agent: Optional[str] = None,
    ):
        super().__init__("dynamic_routing", OrchestrationPattern.DYNAMIC_ROUTING)
        self.routing_method = routing_method
        self.routing_rules = routing_rules or {}
        self.fallback_agent = fallback_agent
    
    def _route_rule_based(
        self,
        task: str,
        agents: List["BaseOBOCAgent"],
    ) -> Optional["BaseOBOCAgent"]:
        """Route using keyword matching rules."""
        task_lower = task.lower()
        
        for agent in agents:
            keywords = self.routing_rules.get(agent.name, [])
            for keyword in keywords:
                if keyword.lower() in task_lower:
                    return agent
        
        # Return fallback or first agent
        if self.fallback_agent:
            for agent in agents:
                if agent.name == self.fallback_agent:
                    return agent
        return agents[0] if agents else None
    
    def _route_semantic(
        self,
        task: str,
        agents: List["BaseOBOCAgent"],
    ) -> Optional["BaseOBOCAgent"]:
        """Route using embedding similarity."""
        # In production, this would use vector similarity
        # For now, fallback to rule-based
        return self._route_rule_based(task, agents)
    
    def _route_auction(
        self,
        task: str,
        agents: List["BaseOBOCAgent"],
    ) -> Optional["BaseOBOCAgent"]:
        """Route using agent bidding."""
        # In production, agents would bid on confidence
        # For now, fallback to rule-based
        return self._route_rule_based(task, agents)
    
    async def execute(
        self,
        agents: List["BaseOBOCAgent"],
        task: str,
        context: "AgentContext",
    ) -> OrchestrationResult:
        """Route task to best matching agent."""
        start_time = datetime.now()
        
        # Select routing method
        if self.routing_method == "semantic":
            selected_agent = self._route_semantic(task, agents)
        elif self.routing_method == "auction":
            selected_agent = self._route_auction(task, agents)
        else:
            selected_agent = self._route_rule_based(task, agents)
        
        if not selected_agent:
            return OrchestrationResult(
                pattern=self.pattern,
                success=False,
                results=[],
                final_output={"error": "No agent available for routing"},
                total_time=(datetime.now() - start_time).total_seconds(),
            )
        
        # Execute the selected agent
        agent_start = datetime.now()
        try:
            result = await selected_agent.execute(task, context)
            
            agent_result = AgentResult(
                agent_name=selected_agent.name,
                success=result.get("success", True),
                output=result.get("result", result),
                execution_time=(datetime.now() - agent_start).total_seconds(),
                metadata={"routing_method": self.routing_method},
            )
            
            return OrchestrationResult(
                pattern=self.pattern,
                success=agent_result.success,
                results=[agent_result],
                final_output=agent_result.output,
                total_time=(datetime.now() - start_time).total_seconds(),
                metadata={"routed_to": selected_agent.name},
            )
            
        except Exception as e:
            return OrchestrationResult(
                pattern=self.pattern,
                success=False,
                results=[AgentResult(
                    agent_name=selected_agent.name,
                    success=False,
                    output=None,
                    error=str(e),
                )],
                final_output={"error": str(e)},
                total_time=(datetime.now() - start_time).total_seconds(),
            )


@dataclass
class TaskLedgerEntry:
    """An entry in the manager's task ledger."""
    task_id: str
    description: str
    status: str = "pending"  # pending, in_progress, completed, failed, backtracked
    assigned_agent: Optional[str] = None
    result: Optional[Any] = None
    attempts: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


class ManagerOrchestration(BaseOrchestrationPattern):
    """
    ORCHESTRATOR/MANAGER PATTERN
    
             ┌─────────────────────────────────────┐
             │         Manager Agent               │
             │   • Builds dynamic task ledger      │
             │   • Iterates and backtracks         │
             │   • Delegates and monitors          │
             └─────────────────────────────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        ▼                    ▼                    ▼
    [Executor A]        [Executor B]         [Executor C]
    
    Best for:
    - Open-ended problems
    - Incident response
    - Research tasks
    """
    
    def __init__(
        self,
        max_iterations: int = 10,
        allow_backtracking: bool = True,
        max_retries_per_task: int = 2,
    ):
        super().__init__("manager", OrchestrationPattern.MANAGER)
        self.max_iterations = max_iterations
        self.allow_backtracking = allow_backtracking
        self.max_retries_per_task = max_retries_per_task
        self.task_ledger: List[TaskLedgerEntry] = []
    
    async def execute(
        self,
        agents: List["BaseOBOCAgent"],
        task: str,
        context: "AgentContext",
    ) -> OrchestrationResult:
        """Execute with dynamic task management and backtracking."""
        start_time = datetime.now()
        results = []
        self.task_ledger = []
        
        # Initialize task ledger with the main task
        self.task_ledger.append(TaskLedgerEntry(
            task_id="main",
            description=task,
        ))
        
        iteration = 0
        while iteration < self.max_iterations:
            iteration += 1
            
            # Get next pending task
            pending_task = self._get_next_task()
            if not pending_task:
                break
            
            # Select best agent for this task
            selected_agent = self._select_agent(pending_task, agents)
            if not selected_agent:
                pending_task.status = "failed"
                pending_task.result = "No suitable agent found"
                continue
            
            # Execute task
            pending_task.status = "in_progress"
            pending_task.assigned_agent = selected_agent.name
            pending_task.attempts += 1
            
            agent_start = datetime.now()
            try:
                result = await selected_agent.execute(pending_task.description, context)
                
                agent_result = AgentResult(
                    agent_name=selected_agent.name,
                    success=result.get("success", True),
                    output=result.get("result", result),
                    execution_time=(datetime.now() - agent_start).total_seconds(),
                    metadata={"task_id": pending_task.task_id, "iteration": iteration},
                )
                results.append(agent_result)
                
                if result.get("success", True):
                    pending_task.status = "completed"
                    pending_task.result = result
                    
                    # Check for new subtasks
                    new_tasks = result.get("subtasks", [])
                    for subtask in new_tasks:
                        self.task_ledger.append(TaskLedgerEntry(
                            task_id=f"{pending_task.task_id}_{len(self.task_ledger)}",
                            description=subtask,
                        ))
                else:
                    # Handle failure
                    if self.allow_backtracking and pending_task.attempts < self.max_retries_per_task:
                        pending_task.status = "backtracked"
                    else:
                        pending_task.status = "failed"
                        pending_task.result = result.get("error", "Unknown error")
                        
            except Exception as e:
                results.append(AgentResult(
                    agent_name=selected_agent.name,
                    success=False,
                    output=None,
                    error=str(e),
                    execution_time=(datetime.now() - agent_start).total_seconds(),
                ))
                
                if self.allow_backtracking and pending_task.attempts < self.max_retries_per_task:
                    pending_task.status = "backtracked"
                else:
                    pending_task.status = "failed"
                    pending_task.result = str(e)
            
            pending_task.updated_at = datetime.now()
        
        # Generate final output
        final_output = self._generate_final_output()
        
        total_time = (datetime.now() - start_time).total_seconds()
        total_cost = sum(r.cost for r in results)
        success = any(r.success for r in results)
        
        return OrchestrationResult(
            pattern=self.pattern,
            success=success,
            results=results,
            final_output=final_output,
            total_time=total_time,
            total_cost=total_cost,
            metadata={"task_ledger": [t.__dict__ for t in self.task_ledger]},
        )
    
    def _get_next_task(self) -> Optional[TaskLedgerEntry]:
        """Get the next pending or backtracked task."""
        for task in self.task_ledger:
            if task.status in ("pending", "backtracked"):
                return task
        return None
    
    def _select_agent(
        self,
        task: TaskLedgerEntry,
        agents: List["BaseOBOCAgent"],
    ) -> Optional["BaseOBOCAgent"]:
        """Select the best agent for a task."""
        # Simple selection: use first agent that hasn't failed this task
        task_lower = task.description.lower()
        
        for agent in agents:
            # Check capabilities (simplified)
            for cap in agent.capabilities.values():
                if any(kw in task_lower for kw in [cap.name, agent.name]):
                    return agent
        
        # Return first agent as fallback
        return agents[0] if agents else None
    
    def _generate_final_output(self) -> Dict[str, Any]:
        """Generate final output from task ledger."""
        completed = [t for t in self.task_ledger if t.status == "completed"]
        failed = [t for t in self.task_ledger if t.status == "failed"]
        
        return {
            "total_tasks": len(self.task_ledger),
            "completed": len(completed),
            "failed": len(failed),
            "results": [
                {"task": t.task_id, "result": t.result}
                for t in completed
            ],
        }


# Factory for creating orchestration patterns
class OrchestrationFactory:
    """Factory for creating orchestration pattern instances."""
    
    _patterns = {
        OrchestrationPattern.SEQUENTIAL: SequentialOrchestration,
        OrchestrationPattern.CONCURRENT: ConcurrentOrchestration,
        OrchestrationPattern.GROUP_CHAT: GroupChatOrchestration,
        OrchestrationPattern.DYNAMIC_ROUTING: DynamicRoutingOrchestration,
        OrchestrationPattern.MANAGER: ManagerOrchestration,
    }
    
    @classmethod
    def create(
        cls,
        pattern: OrchestrationPattern,
        **kwargs,
    ) -> BaseOrchestrationPattern:
        """Create an orchestration pattern instance."""
        pattern_class = cls._patterns.get(pattern)
        if not pattern_class:
            raise ValueError(f"Unknown pattern: {pattern}")
        return pattern_class(**kwargs)
    
    @classmethod
    def list_patterns(cls) -> List[str]:
        """List available patterns."""
        return [p.value for p in OrchestrationPattern]
