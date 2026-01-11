"""
OBOC Three-Tier Hierarchy

Implements the Strategic → Tactical → Operational agent hierarchy:

STRATEGIC TIER:
    - Project Manager / Coordinator Agent
    - Interprets requests, formulates strategy
    - Decomposes goals, monitors progress

TACTICAL TIER:
    - Specialist / Domain Agents
    - Research, Implementation, QA, Memory agents

OPERATIONAL TIER:
    - Execution Agents
    - Database queries, API calls, file operations
    - Computations, tool execution
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional, List, TYPE_CHECKING

if TYPE_CHECKING:
    from .memory import MemoryStore
    from .knowledge import KnowledgeGraph


class TierLevel(Enum):
    """Agent tier levels in the OBOC hierarchy."""
    STRATEGIC = "strategic"   # High-level planning and coordination
    TACTICAL = "tactical"     # Domain-specific expertise
    OPERATIONAL = "operational"  # Tool execution and atomic operations


@dataclass
class AgentCapability:
    """Defines what an agent can do."""
    name: str
    description: str
    tools: List[str] = field(default_factory=list)
    decision_scope: str = "limited"  # limited, moderate, autonomous
    can_delegate: bool = False


@dataclass
class AgentContext:
    """Context passed between agents in the hierarchy."""
    task_id: str
    parent_agent: Optional[str] = None
    tier: TierLevel = TierLevel.OPERATIONAL
    goal: str = ""
    constraints: Dict[str, Any] = field(default_factory=dict)
    memory_refs: List[str] = field(default_factory=list)
    knowledge_refs: List[str] = field(default_factory=list)
    trace_id: str = ""
    created_at: datetime = field(default_factory=datetime.now)


class BaseOBOCAgent(ABC):
    """
    Base class for all OBOC-aligned agents.
    
    Every agent has:
    - A defined tier level
    - Clear capabilities
    - Access to memory hierarchy
    - Knowledge graph integration
    - Observability hooks
    """
    
    def __init__(
        self,
        name: str,
        tier: TierLevel,
        capabilities: List[AgentCapability],
        memory_store: Optional["MemoryStore"] = None,
        knowledge_graph: Optional["KnowledgeGraph"] = None,
    ):
        self.name = name
        self.tier = tier
        self.capabilities = {cap.name: cap for cap in capabilities}
        self.memory_store = memory_store
        self.knowledge_graph = knowledge_graph
        self._subordinates: Dict[str, "BaseOBOCAgent"] = {}
    
    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """Return the agent's system prompt."""
        pass
    
    @abstractmethod
    async def execute(
        self,
        task: str,
        context: AgentContext,
    ) -> Dict[str, Any]:
        """Execute a task with the given context."""
        pass
    
    def register_subordinate(self, agent: "BaseOBOCAgent"):
        """Register a subordinate agent (for delegation)."""
        if self.tier.value == "strategic" and agent.tier.value in ["tactical", "operational"]:
            self._subordinates[agent.name] = agent
        elif self.tier.value == "tactical" and agent.tier.value == "operational":
            self._subordinates[agent.name] = agent
        else:
            raise ValueError(f"Cannot register {agent.tier.value} agent as subordinate of {self.tier.value}")
    
    async def delegate(
        self,
        agent_name: str,
        task: str,
        context: AgentContext
    ) -> Dict[str, Any]:
        """Delegate a task to a subordinate agent."""
        if agent_name not in self._subordinates:
            raise ValueError(f"No subordinate agent named: {agent_name}")
        
        # Update context for delegation
        context.parent_agent = self.name
        
        return await self._subordinates[agent_name].execute(task, context)


class StrategicAgent(BaseOBOCAgent):
    """
    STRATEGIC TIER Agent (Project Manager / Coordinator)
    
    Responsibilities:
    - Interpret user requests
    - Formulate overall strategy
    - Decompose goals into tactical objectives
    - Monitor overall progress
    - Make high-level decisions
    """
    
    def __init__(
        self,
        name: str = "coordinator",
        memory_store: Optional["MemoryStore"] = None,
        knowledge_graph: Optional["KnowledgeGraph"] = None,
    ):
        capabilities = [
            AgentCapability(
                name="strategic_planning",
                description="Formulate overall strategy and decompose goals",
                decision_scope="autonomous",
                can_delegate=True,
            ),
            AgentCapability(
                name="progress_monitoring",
                description="Monitor and assess progress across all agents",
                decision_scope="autonomous",
                can_delegate=False,
            ),
            AgentCapability(
                name="resource_allocation",
                description="Allocate tasks to appropriate tactical agents",
                decision_scope="autonomous",
                can_delegate=True,
            ),
        ]
        super().__init__(
            name=name,
            tier=TierLevel.STRATEGIC,
            capabilities=capabilities,
            memory_store=memory_store,
            knowledge_graph=knowledge_graph,
        )
    
    @property
    def system_prompt(self) -> str:
        return """You are a Strategic Coordinator Agent operating at the highest tier of the OBOC hierarchy.

Your responsibilities:
1. INTERPRET user requests to understand true intent
2. FORMULATE overall strategy by breaking down complex goals
3. DECOMPOSE goals into tactical objectives for specialist agents
4. MONITOR progress across all subordinate agents
5. ADAPT strategy based on feedback and results

Decision Authority:
- Full autonomy over planning and delegation
- Can approve or reject tactical recommendations
- Responsible for final deliverables

Communication:
- Provide clear, actionable objectives to tactical agents
- Synthesize results from multiple sources
- Report progress in structured format

Always think strategically: What's the best approach to achieve the user's goal?"""

    async def execute(
        self,
        task: str,
        context: AgentContext,
    ) -> Dict[str, Any]:
        """Execute strategic planning and coordination."""
        # Strategic agents decompose and delegate
        plan = await self._create_strategic_plan(task, context)
        
        results = []
        for objective in plan.get("objectives", []):
            agent_name = objective.get("assigned_to")
            if agent_name and agent_name in self._subordinates:
                result = await self.delegate(
                    agent_name,
                    objective.get("task", ""),
                    context
                )
                results.append(result)
        
        return {
            "success": True,
            "tier": self.tier.value,
            "plan": plan,
            "results": results,
        }
    
    async def _create_strategic_plan(
        self,
        task: str,
        context: AgentContext
    ) -> Dict[str, Any]:
        """Create a strategic execution plan."""
        # This would call the LLM for planning
        # Placeholder for now
        return {
            "goal": task,
            "strategy": "Decompose and delegate",
            "objectives": [],
        }


class TacticalAgent(BaseOBOCAgent):
    """
    TACTICAL TIER Agent (Specialist / Domain Expert)
    
    Responsibilities:
    - Deep domain expertise
    - Research and analysis
    - Implementation planning
    - Quality assurance
    """
    
    TACTICAL_ROLES = {
        "research": {
            "description": "Information gathering, best practices, documentation",
            "tools": ["web_search", "semantic_search", "document_retrieval"],
        },
        "implementation": {
            "description": "Code generation, feature implementation",
            "tools": ["code_write", "file_create", "file_edit"],
        },
        "testing": {
            "description": "Test generation, validation, quality checks",
            "tools": ["test_run", "code_analyze", "coverage_check"],
        },
        "analysis": {
            "description": "Code review, architecture analysis, optimization",
            "tools": ["code_review", "performance_analyze", "security_scan"],
        },
        "memory": {
            "description": "Context management, knowledge extraction",
            "tools": ["memory_store", "memory_retrieve", "knowledge_update"],
        },
    }
    
    def __init__(
        self,
        name: str,
        role: str,
        memory_store: Optional["MemoryStore"] = None,
        knowledge_graph: Optional["KnowledgeGraph"] = None,
    ):
        if role not in self.TACTICAL_ROLES:
            raise ValueError(f"Unknown tactical role: {role}. Valid: {list(self.TACTICAL_ROLES.keys())}")
        
        role_config = self.TACTICAL_ROLES[role]
        capabilities = [
            AgentCapability(
                name=role,
                description=role_config["description"],
                tools=role_config["tools"],
                decision_scope="moderate",
                can_delegate=True,
            ),
        ]
        
        super().__init__(
            name=name,
            tier=TierLevel.TACTICAL,
            capabilities=capabilities,
            memory_store=memory_store,
            knowledge_graph=knowledge_graph,
        )
        self.role = role
    
    @property
    def system_prompt(self) -> str:
        role_config = self.TACTICAL_ROLES[self.role]
        return f"""You are a Tactical {self.role.title()} Agent in the OBOC hierarchy.

Your specialization: {role_config['description']}

Your responsibilities:
1. RECEIVE objectives from Strategic Coordinator
2. PLAN detailed approach for your domain
3. DELEGATE atomic tasks to Operational agents
4. VALIDATE results and report back

Available Tools: {', '.join(role_config['tools'])}

Decision Authority:
- Moderate autonomy within your domain
- Can make tactical decisions independently
- Escalate strategic conflicts to coordinator

Always focus on your specialization. Delegate tool execution to operational agents."""

    async def execute(
        self,
        task: str,
        context: AgentContext,
    ) -> Dict[str, Any]:
        """Execute domain-specific tactical work."""
        # Tactical agents plan and delegate to operational
        approach = await self._plan_approach(task, context)
        
        results = []
        for step in approach.get("steps", []):
            agent_name = step.get("assigned_to")
            if agent_name and agent_name in self._subordinates:
                result = await self.delegate(
                    agent_name,
                    step.get("task", ""),
                    context
                )
                results.append(result)
        
        return {
            "success": True,
            "tier": self.tier.value,
            "role": self.role,
            "approach": approach,
            "results": results,
        }
    
    async def _plan_approach(
        self,
        task: str,
        context: AgentContext
    ) -> Dict[str, Any]:
        """Plan tactical approach for the task."""
        return {
            "task": task,
            "approach": f"Apply {self.role} expertise",
            "steps": [],
        }


class OperationalAgent(BaseOBOCAgent):
    """
    OPERATIONAL TIER Agent (Executor)
    
    Responsibilities:
    - Execute atomic operations
    - Database queries
    - API calls
    - File operations
    - Tool execution
    """
    
    OPERATIONAL_TOOLS = {
        "database": ["query", "insert", "update", "delete"],
        "api": ["get", "post", "put", "patch", "delete"],
        "file": ["read", "write", "create", "delete", "move"],
        "compute": ["calculate", "transform", "aggregate"],
        "search": ["semantic", "keyword", "grep"],
    }
    
    def __init__(
        self,
        name: str,
        tool_category: str,
        memory_store: Optional["MemoryStore"] = None,
        knowledge_graph: Optional["KnowledgeGraph"] = None,
    ):
        if tool_category not in self.OPERATIONAL_TOOLS:
            raise ValueError(f"Unknown tool category: {tool_category}")
        
        tools = self.OPERATIONAL_TOOLS[tool_category]
        capabilities = [
            AgentCapability(
                name=f"{tool_category}_operations",
                description=f"Execute {tool_category} operations",
                tools=tools,
                decision_scope="limited",
                can_delegate=False,
            ),
        ]
        
        super().__init__(
            name=name,
            tier=TierLevel.OPERATIONAL,
            capabilities=capabilities,
            memory_store=memory_store,
            knowledge_graph=knowledge_graph,
        )
        self.tool_category = tool_category
    
    @property
    def system_prompt(self) -> str:
        tools = self.OPERATIONAL_TOOLS[self.tool_category]
        return f"""You are an Operational Executor Agent in the OBOC hierarchy.

Your function: Execute {self.tool_category} operations

Available Operations: {', '.join(tools)}

Your responsibilities:
1. RECEIVE specific tasks from Tactical agents
2. EXECUTE operations precisely as instructed
3. REPORT results and any errors

Decision Authority:
- Limited to operation parameters
- Cannot make strategic decisions
- Report all anomalies upward

Focus on precise execution. Report results clearly and completely."""

    async def execute(
        self,
        task: str,
        context: AgentContext,
    ) -> Dict[str, Any]:
        """Execute atomic operation."""
        # Operational agents execute directly
        result = await self._execute_operation(task, context)
        
        return {
            "success": result.get("success", False),
            "tier": self.tier.value,
            "tool_category": self.tool_category,
            "operation": task,
            "result": result,
        }
    
    async def _execute_operation(
        self,
        task: str,
        context: AgentContext
    ) -> Dict[str, Any]:
        """Execute the actual operation."""
        # This would call the actual tool
        return {
            "success": True,
            "output": f"Executed: {task}",
        }
