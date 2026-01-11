"""
OBOC - One Brain, One Context

Multi-Agent Orchestration ontology following the OBOC architecture:
- Three-Tier Hierarchy (Strategic, Tactical, Operational)
- Memory Hierarchy (Short-term, Working, Long-term)
- Knowledge Graph Integration
- Multiple Orchestration Patterns
- Resilience & Observability
"""

from .tiers import (
    TierLevel,
    BaseOBOCAgent,
    StrategicAgent,
    TacticalAgent,
    OperationalAgent,
    AgentContext,
    AgentCapability,
)
from .memory import (
    MemoryType,
    MemoryStore,
    MemoryEntry,
    MemoryManager,
    ShortTermMemory,
    WorkingMemory,
    LongTermMemory,
    TaskNote,
)
from .knowledge import (
    Entity,
    EntityType,
    Relationship,
    RelationType,
    KnowledgeGraph,
    GraphQuery,
)
from .orchestration import (
    OrchestrationPattern,
    OrchestrationFactory,
    OrchestrationResult,
    AgentResult,
    SequentialOrchestration,
    ConcurrentOrchestration,
    GroupChatOrchestration,
    DynamicRoutingOrchestration,
    ManagerOrchestration,
)
from .resilience import (
    CircuitBreaker,
    CircuitState,
    CircuitBreakerConfig,
    RetryPolicy,
    RetryConfig,
    GracefulDegradation,
    StatePreservation,
    ResilienceManager,
)
from .oboc_orchestrator import (
    OBOCOrchestrator,
    OBOCConfig,
    ExecutionTrace,
)
from .observability import (
    ObservabilityManager,
    Tracer,
    Trace,
    Span,
    SpanType,
    MetricsCollector,
    Evaluator,
)

__all__ = [
    # Tiers
    "TierLevel",
    "BaseOBOCAgent",
    "StrategicAgent",
    "TacticalAgent",
    "OperationalAgent",
    "AgentContext",
    "AgentCapability",
    # Memory
    "MemoryType",
    "MemoryStore",
    "MemoryEntry",
    "MemoryManager",
    "ShortTermMemory",
    "WorkingMemory",
    "LongTermMemory",
    "TaskNote",
    # Knowledge Graph
    "Entity",
    "EntityType",
    "Relationship",
    "RelationType",
    "KnowledgeGraph",
    "GraphQuery",
    # Orchestration
    "OrchestrationPattern",
    "OrchestrationFactory",
    "OrchestrationResult",
    "AgentResult",
    "SequentialOrchestration",
    "ConcurrentOrchestration",
    "GroupChatOrchestration",
    "DynamicRoutingOrchestration",
    "ManagerOrchestration",
    # Resilience
    "CircuitBreaker",
    "CircuitState",
    "CircuitBreakerConfig",
    "RetryPolicy",
    "RetryConfig",
    "GracefulDegradation",
    "StatePreservation",
    "ResilienceManager",
    # Main Orchestrator
    "OBOCOrchestrator",
    "OBOCConfig",
    "ExecutionTrace",
    # Observability
    "ObservabilityManager",
    "Tracer",
    "Trace",
    "Span",
    "SpanType",
    "MetricsCollector",
    "Evaluator",
]

__all__ = [
    # Tiers
    "TierLevel",
    "StrategicAgent",
    "TacticalAgent",
    "OperationalAgent",
    # Memory
    "MemoryType",
    "MemoryStore",
    "ShortTermMemory",
    "WorkingMemory",
    "LongTermMemory",
    # Knowledge Graph
    "Entity",
    "Relationship",
    "KnowledgeGraph",
    # Orchestration
    "OrchestrationPattern",
    "OBOCOrchestrator",
    # Resilience
    "CircuitBreaker",
    "CircuitState",
]
