"""
OBOC Orchestrator

The main unified orchestrator that brings together all OBOC components:
- Three-Tier Agent Hierarchy (Strategic → Tactical → Operational)
- Memory Management (Short-term, Working, Long-term)
- Knowledge Graph Integration
- Multiple Orchestration Patterns
- Resilience (Circuit Breakers, Retry, Degradation)
- Observability (Tracing, Metrics)

This orchestrator replaces the original AgentArmy orchestrator with
OBOC (One Brain, One Context) principles.
"""

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional, List, TYPE_CHECKING

from .tiers import (
    TierLevel,
    BaseOBOCAgent,
    StrategicAgent,
    TacticalAgent,
    OperationalAgent,
    AgentContext,
)
from .memory import (
    MemoryManager,
    MemoryEntry,
    MemoryType,
)
from .knowledge import (
    KnowledgeGraph,
    Entity,
    EntityType,
    RelationType,
)
from .orchestration import (
    OrchestrationPattern,
    OrchestrationFactory,
    OrchestrationResult,
    BaseOrchestrationPattern,
)
from .resilience import (
    ResilienceManager,
    CircuitBreakerConfig,
)


@dataclass
class OBOCConfig:
    """Configuration for OBOC Orchestrator."""
    # Orchestration
    default_pattern: OrchestrationPattern = OrchestrationPattern.MANAGER
    max_iterations: int = 10
    enable_parallel: bool = True
    
    # Memory
    enable_memory: bool = True
    memory_storage_path: Optional[str] = None
    
    # Knowledge Graph
    enable_knowledge_graph: bool = True
    knowledge_storage_path: Optional[str] = None
    
    # Resilience
    enable_circuit_breakers: bool = True
    enable_retry: bool = True
    circuit_failure_threshold: int = 5
    circuit_timeout_seconds: float = 30.0
    retry_max_attempts: int = 3
    
    # Observability
    enable_tracing: bool = True
    trace_all_operations: bool = False
    
    # Tiers
    strategic_agent_model: str = "us.anthropic.claude-3-5-sonnet-20241022-v2:0"
    tactical_agent_model: str = "us.anthropic.claude-3-5-sonnet-20241022-v2:0"
    operational_agent_model: str = "us.anthropic.claude-3-5-haiku-20241022-v1:0"


@dataclass
class ExecutionTrace:
    """Trace of an execution for observability."""
    trace_id: str
    task: str
    pattern: OrchestrationPattern
    started_at: datetime
    ended_at: Optional[datetime] = None
    success: bool = False
    agents_used: List[str] = field(default_factory=list)
    tier_transitions: List[Dict[str, str]] = field(default_factory=list)
    memory_operations: int = 0
    knowledge_operations: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    error: Optional[str] = None


class OBOCOrchestrator:
    """
    OBOC (One Brain, One Context) Orchestrator
    
    The main orchestration engine that coordinates:
    
    ┌─────────────────────────────────────────────────────────────┐
    │                    OBOC ORCHESTRATOR                         │
    ├─────────────────────────────────────────────────────────────┤
    │                                                              │
    │   ┌─────────────────────────────────────────────────────┐   │
    │   │              STRATEGIC TIER                          │   │
    │   │         Coordinator / Project Manager                │   │
    │   └─────────────────────────┬───────────────────────────┘   │
    │                             │                               │
    │   ┌─────────────────────────▼───────────────────────────┐   │
    │   │              TACTICAL TIER                           │   │
    │   │   Research │ Implementation │ Testing │ Analysis     │   │
    │   └─────────────────────────┬───────────────────────────┘   │
    │                             │                               │
    │   ┌─────────────────────────▼───────────────────────────┐   │
    │   │              OPERATIONAL TIER                        │   │
    │   │      File │ API │ Database │ Compute │ Search        │   │
    │   └─────────────────────────────────────────────────────┘   │
    │                                                              │
    │   ╔═══════════════════════════════════════════════════════╗ │
    │   ║  Memory Manager │ Knowledge Graph │ Resilience        ║ │
    │   ╚═══════════════════════════════════════════════════════╝ │
    │                                                              │
    └─────────────────────────────────────────────────────────────┘
    """
    
    def __init__(self, config: Optional[OBOCConfig] = None):
        self.config = config or OBOCConfig()
        
        # Initialize components
        self._init_memory()
        self._init_knowledge_graph()
        self._init_resilience()
        self._init_agents()
        
        # Execution tracking
        self._traces: Dict[str, ExecutionTrace] = {}
        self._current_trace: Optional[ExecutionTrace] = None
    
    def _init_memory(self):
        """Initialize memory management."""
        if self.config.enable_memory:
            from pathlib import Path
            storage_path = Path(self.config.memory_storage_path) if self.config.memory_storage_path else None
            self.memory = MemoryManager(storage_path)
        else:
            self.memory = None
    
    def _init_knowledge_graph(self):
        """Initialize knowledge graph."""
        if self.config.enable_knowledge_graph:
            from pathlib import Path
            storage_path = Path(self.config.knowledge_storage_path) if self.config.knowledge_storage_path else None
            self.knowledge = KnowledgeGraph(storage_path)
        else:
            self.knowledge = None
    
    def _init_resilience(self):
        """Initialize resilience manager."""
        self.resilience = ResilienceManager()
    
    def _init_agents(self):
        """Initialize the three-tier agent hierarchy."""
        # Strategic tier
        self.strategic_agents: Dict[str, StrategicAgent] = {
            "coordinator": StrategicAgent(
                name="coordinator",
                memory_store=self.memory.short_term if self.memory else None,
                knowledge_graph=self.knowledge,
            ),
        }
        
        # Tactical tier
        self.tactical_agents: Dict[str, TacticalAgent] = {}
        for role in ["research", "implementation", "testing", "analysis", "memory"]:
            self.tactical_agents[role] = TacticalAgent(
                name=role,
                role=role,
                memory_store=self.memory.working if self.memory else None,
                knowledge_graph=self.knowledge,
            )
        
        # Operational tier
        self.operational_agents: Dict[str, OperationalAgent] = {}
        for category in ["file", "api", "database", "compute", "search"]:
            self.operational_agents[category] = OperationalAgent(
                name=f"{category}_executor",
                tool_category=category,
                memory_store=self.memory.short_term if self.memory else None,
                knowledge_graph=self.knowledge,
            )
        
        # Wire up the hierarchy
        self._wire_hierarchy()
    
    def _wire_hierarchy(self):
        """Wire up the agent hierarchy for delegation."""
        # Strategic agents can delegate to tactical
        for strategic in self.strategic_agents.values():
            for tactical in self.tactical_agents.values():
                strategic.register_subordinate(tactical)
        
        # Tactical agents can delegate to operational
        for tactical in self.tactical_agents.values():
            for operational in self.operational_agents.values():
                tactical.register_subordinate(operational)
    
    async def execute(
        self,
        task: str,
        pattern: Optional[OrchestrationPattern] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Execute a task using the OBOC architecture.
        
        Args:
            task: The task to execute
            pattern: Orchestration pattern to use (defaults to config)
            context: Optional execution context
            
        Returns:
            Execution result
        """
        trace_id = str(uuid.uuid4())[:12]
        pattern = pattern or self.config.default_pattern
        
        # Create execution trace
        trace = ExecutionTrace(
            trace_id=trace_id,
            task=task,
            pattern=pattern,
            started_at=datetime.now(),
        )
        self._traces[trace_id] = trace
        self._current_trace = trace
        
        print(f"\n{'='*60}")
        print(f"🧠 OBOC Orchestrator Starting")
        print(f"{'='*60}")
        print(f"Trace ID: {trace_id}")
        print(f"Pattern: {pattern.value}")
        print(f"Task: {task[:100]}{'...' if len(task) > 100 else ''}")
        print(f"{'='*60}\n")
        
        try:
            # Add to short-term memory
            if self.memory:
                self.memory.short_term.add_message("user", task)
                trace.memory_operations += 1
            
            # Record in knowledge graph
            if self.knowledge:
                task_entity = self.knowledge.create_entity(
                    name=task[:50],
                    entity_type=EntityType.TASK,
                    properties={"full_task": task, "trace_id": trace_id},
                )
                trace.knowledge_operations += 1
            
            # Create agent context
            agent_context = AgentContext(
                task_id=trace_id,
                tier=TierLevel.STRATEGIC,
                goal=task,
                trace_id=trace_id,
            )
            
            # Select orchestration pattern
            orchestration = self._select_orchestration(pattern, task)
            
            # Get agents to use based on pattern
            agents = self._select_agents(pattern, task)
            trace.agents_used = [a.name for a in agents]
            
            # Execute with resilience
            result = await self.resilience.execute_with_resilience(
                f"orchestration_{pattern.value}",
                orchestration.execute,
                agents,
                task,
                agent_context,
                use_retry=self.config.enable_retry,
                use_circuit_breaker=self.config.enable_circuit_breakers,
            )
            
            # Process result
            trace.success = result.success
            trace.total_cost = result.total_cost
            trace.ended_at = datetime.now()
            
            # Store result in memory
            if self.memory and result.success:
                self.memory.short_term.add_message(
                    "assistant",
                    str(result.final_output),
                    agent_name="orchestrator",
                )
            
            # Update knowledge graph
            if self.knowledge and result.success:
                self.knowledge.update_entity(
                    task_entity.id,
                    properties={"status": "completed", "success": True},
                )
            
            print(f"\n{'='*60}")
            print(f"✅ OBOC Execution Complete")
            print(f"{'='*60}")
            print(f"Success: {result.success}")
            print(f"Agents Used: {', '.join(trace.agents_used)}")
            print(f"Duration: {(trace.ended_at - trace.started_at).total_seconds():.2f}s")
            print(f"{'='*60}\n")
            
            return {
                "success": result.success,
                "trace_id": trace_id,
                "pattern": pattern.value,
                "output": result.final_output,
                "agents_used": trace.agents_used,
                "duration_seconds": (trace.ended_at - trace.started_at).total_seconds(),
                "total_cost": result.total_cost,
            }
            
        except Exception as e:
            trace.success = False
            trace.error = str(e)
            trace.ended_at = datetime.now()
            
            print(f"\n❌ OBOC Execution Failed: {e}")
            
            return {
                "success": False,
                "trace_id": trace_id,
                "error": str(e),
                "duration_seconds": (trace.ended_at - trace.started_at).total_seconds(),
            }
    
    def _select_orchestration(
        self,
        pattern: OrchestrationPattern,
        task: str,
    ) -> BaseOrchestrationPattern:
        """Select the orchestration pattern instance."""
        if pattern == OrchestrationPattern.DYNAMIC_ROUTING:
            # Configure routing rules based on keywords
            routing_rules = {
                "research": ["research", "find", "search", "look up", "documentation"],
                "implementation": ["implement", "build", "create", "code", "write"],
                "testing": ["test", "validate", "check", "verify"],
                "analysis": ["analyze", "review", "evaluate", "assess"],
            }
            return OrchestrationFactory.create(
                pattern,
                routing_rules=routing_rules,
                fallback_agent="research",
            )
        elif pattern == OrchestrationPattern.GROUP_CHAT:
            return OrchestrationFactory.create(
                pattern,
                max_rounds=3,
                turn_order=["research", "implementation", "testing"],
            )
        elif pattern == OrchestrationPattern.MANAGER:
            return OrchestrationFactory.create(
                pattern,
                max_iterations=self.config.max_iterations,
                allow_backtracking=True,
            )
        else:
            return OrchestrationFactory.create(pattern)
    
    def _select_agents(
        self,
        pattern: OrchestrationPattern,
        task: str,
    ) -> List[BaseOBOCAgent]:
        """Select agents based on pattern and task."""
        task_lower = task.lower()
        
        if pattern == OrchestrationPattern.SEQUENTIAL:
            # Typical sequence: research → implement → test
            return [
                self.tactical_agents["research"],
                self.tactical_agents["implementation"],
                self.tactical_agents["testing"],
            ]
        
        elif pattern == OrchestrationPattern.CONCURRENT:
            # Run multiple tactical agents in parallel
            return list(self.tactical_agents.values())[:4]
        
        elif pattern == OrchestrationPattern.GROUP_CHAT:
            return [
                self.tactical_agents["research"],
                self.tactical_agents["implementation"],
                self.tactical_agents["analysis"],
            ]
        
        elif pattern == OrchestrationPattern.DYNAMIC_ROUTING:
            # Return all tactical agents, router will select
            return list(self.tactical_agents.values())
        
        elif pattern == OrchestrationPattern.MANAGER:
            # Manager pattern uses strategic + tactical
            agents = [self.strategic_agents["coordinator"]]
            agents.extend(self.tactical_agents.values())
            return agents
        
        else:
            # Default: all tactical agents
            return list(self.tactical_agents.values())
    
    async def chat(
        self,
        message: str,
        agent: Optional[str] = None,
    ) -> str:
        """
        Chat with a specific agent or the default coordinator.
        
        Args:
            message: User message
            agent: Specific agent to chat with
            
        Returns:
            Agent response
        """
        # Add to short-term memory
        if self.memory:
            self.memory.short_term.add_message("user", message)
        
        # Select agent
        if agent and agent in self.tactical_agents:
            selected_agent = self.tactical_agents[agent]
        elif agent and agent in self.strategic_agents:
            selected_agent = self.strategic_agents[agent]
        else:
            selected_agent = self.strategic_agents["coordinator"]
        
        # Create context
        context = AgentContext(
            task_id=str(uuid.uuid4())[:8],
            tier=selected_agent.tier,
            goal=message,
        )
        
        # Execute
        result = await selected_agent.execute(message, context)
        
        # Add response to memory
        response = str(result.get("result", result))
        if self.memory:
            self.memory.short_term.add_message(
                "assistant",
                response,
                agent_name=selected_agent.name,
            )
        
        return response
    
    def list_agents(self) -> List[Dict[str, Any]]:
        """List all available agents."""
        agents = []
        
        for name, agent in self.strategic_agents.items():
            agents.append({
                "name": name,
                "tier": agent.tier.value,
                "type": "strategic",
                "capabilities": list(agent.capabilities.keys()),
            })
        
        for name, agent in self.tactical_agents.items():
            agents.append({
                "name": name,
                "tier": agent.tier.value,
                "type": "tactical",
                "role": agent.role,
                "capabilities": list(agent.capabilities.keys()),
            })
        
        for name, agent in self.operational_agents.items():
            agents.append({
                "name": name,
                "tier": agent.tier.value,
                "type": "operational",
                "tool_category": agent.tool_category,
            })
        
        return agents
    
    def get_status(self) -> Dict[str, Any]:
        """Get orchestrator status."""
        return {
            "agents": {
                "strategic": len(self.strategic_agents),
                "tactical": len(self.tactical_agents),
                "operational": len(self.operational_agents),
            },
            "memory": {
                "enabled": self.memory is not None,
                "session_id": self.memory.short_term.session_id if self.memory else None,
            },
            "knowledge": {
                "enabled": self.knowledge is not None,
                "stats": self.knowledge.get_stats() if self.knowledge else {},
            },
            "resilience": self.resilience.get_status(),
            "traces": len(self._traces),
        }
    
    async def session_end(self):
        """Handle session end - consolidate memory and knowledge."""
        if self.memory:
            await self.memory.session_end()
        
        if self.knowledge:
            self.knowledge.save()
    
    def get_memory_context(self, query: str) -> Dict[str, Any]:
        """Get relevant memory context for a query."""
        if not self.memory:
            return {"enabled": False}
        
        # This would be async in production
        return {
            "enabled": True,
            "session_messages": len(self.memory.short_term.get_messages()),
            "working_tasks": len(self.memory.working._notes),
        }
    
    def get_knowledge_context(self, query: str) -> Dict[str, Any]:
        """Get relevant knowledge context for a query."""
        if not self.knowledge:
            return {"enabled": False}
        
        entities = self.knowledge.search(query, limit=5)
        return {
            "enabled": True,
            "relevant_entities": [
                {"id": e.id, "name": e.name, "type": e.entity_type.value}
                for e in entities
            ],
        }
