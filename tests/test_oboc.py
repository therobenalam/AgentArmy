"""
OBOC Integration Tests

Tests for the OBOC (One Brain, One Context) architecture:
- Three-Tier Hierarchy
- Memory Management
- Knowledge Graph
- Orchestration Patterns
- Resilience
"""

import asyncio
import pytest
from datetime import datetime
from pathlib import Path
import tempfile
import shutil

# Import OBOC components
from agent_army.oboc import (
    # Tiers
    TierLevel,
    StrategicAgent,
    TacticalAgent,
    OperationalAgent,
    AgentContext,
    # Memory
    MemoryManager,
    MemoryEntry,
    MemoryType,
    ShortTermMemory,
    WorkingMemory,
    LongTermMemory,
    # Knowledge Graph
    KnowledgeGraph,
    Entity,
    EntityType,
    RelationType,
    # Orchestration
    OrchestrationPattern,
    OrchestrationFactory,
    SequentialOrchestration,
    ConcurrentOrchestration,
    # Resilience
    CircuitBreaker,
    CircuitState,
    CircuitBreakerConfig,
    RetryPolicy,
    ResilienceManager,
    # Orchestrator
    OBOCOrchestrator,
    OBOCConfig,
)


# ============================================================
# Tier Tests
# ============================================================

class TestTiers:
    """Test the three-tier agent hierarchy."""
    
    def test_tier_levels(self):
        """Test tier level enum."""
        assert TierLevel.STRATEGIC.value == "strategic"
        assert TierLevel.TACTICAL.value == "tactical"
        assert TierLevel.OPERATIONAL.value == "operational"
    
    def test_strategic_agent_creation(self):
        """Test creating a strategic agent."""
        agent = StrategicAgent(name="test_coordinator")
        
        assert agent.name == "test_coordinator"
        assert agent.tier == TierLevel.STRATEGIC
        assert "strategic_planning" in agent.capabilities
    
    def test_tactical_agent_creation(self):
        """Test creating tactical agents for each role."""
        roles = ["research", "implementation", "testing", "analysis", "memory"]
        
        for role in roles:
            agent = TacticalAgent(name=f"test_{role}", role=role)
            assert agent.tier == TierLevel.TACTICAL
            assert agent.role == role
    
    def test_tactical_agent_invalid_role(self):
        """Test that invalid role raises error."""
        with pytest.raises(ValueError):
            TacticalAgent(name="test", role="invalid_role")
    
    def test_operational_agent_creation(self):
        """Test creating operational agents for each category."""
        categories = ["file", "api", "database", "compute", "search"]
        
        for category in categories:
            agent = OperationalAgent(name=f"test_{category}", tool_category=category)
            assert agent.tier == TierLevel.OPERATIONAL
            assert agent.tool_category == category
    
    def test_hierarchy_delegation(self):
        """Test that delegation follows tier hierarchy."""
        strategic = StrategicAgent(name="coordinator")
        tactical = TacticalAgent(name="research", role="research")
        operational = OperationalAgent(name="file_exec", tool_category="file")
        
        # Strategic can register tactical
        strategic.register_subordinate(tactical)
        assert "research" in strategic._subordinates
        
        # Tactical can register operational
        tactical.register_subordinate(operational)
        assert "file_exec" in tactical._subordinates
    
    def test_invalid_delegation(self):
        """Test that invalid delegation raises error."""
        tactical = TacticalAgent(name="research", role="research")
        strategic = StrategicAgent(name="coordinator")
        
        # Tactical cannot register strategic
        with pytest.raises(ValueError):
            tactical.register_subordinate(strategic)


# ============================================================
# Memory Tests
# ============================================================

class TestMemory:
    """Test the memory hierarchy."""
    
    def test_short_term_memory(self):
        """Test short-term memory operations."""
        memory = ShortTermMemory(max_messages=10)
        
        # Add messages
        memory.add_message("user", "Hello")
        memory.add_message("assistant", "Hi there!")
        
        messages = memory.get_messages()
        assert len(messages) == 2
        assert messages[0].role == "user"
        assert messages[1].role == "assistant"
    
    def test_short_term_memory_limit(self):
        """Test that short-term memory respects limits."""
        memory = ShortTermMemory(max_messages=5)
        
        for i in range(10):
            memory.add_message("user", f"Message {i}")
        
        messages = memory.get_messages()
        assert len(messages) == 5
        assert messages[0].content == "Message 5"  # Oldest kept
    
    def test_working_memory_notes(self):
        """Test working memory task notes."""
        memory = WorkingMemory()
        
        # Create notes
        notes = memory.create_task_notes("task_1")
        assert notes.task_id == "task_1"
        
        # Update notes
        memory.update_task_notes(
            "task_1",
            current_phase="Implementation",
            completed_item="Research completed",
            key_decision={"decision": "Use Python", "reason": "Team expertise"},
        )
        
        notes = memory.get_task_notes("task_1")
        assert notes.current_phase == "Implementation"
        assert len(notes.completed) == 1
        assert len(notes.key_decisions) == 1
    
    def test_working_memory_artifacts(self):
        """Test working memory artifact storage."""
        memory = WorkingMemory()
        
        memory.store_artifact("task_1", "code", "print('hello')")
        artifact = memory.get_artifact("task_1", "code")
        
        assert artifact == "print('hello')"
    
    def test_memory_manager(self):
        """Test unified memory manager."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = MemoryManager(storage_path=Path(tmpdir))
            
            # Short-term
            manager.short_term.add_message("user", "Test message")
            
            # Working
            manager.working.create_task_notes("task_1")
            
            assert len(manager.short_term.get_messages()) == 1
            assert manager.working.get_task_notes("task_1") is not None


# ============================================================
# Knowledge Graph Tests
# ============================================================

class TestKnowledgeGraph:
    """Test the knowledge graph."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.tmpdir = tempfile.mkdtemp()
        self.graph = KnowledgeGraph(storage_path=Path(self.tmpdir))
    
    def teardown_method(self):
        """Clean up."""
        shutil.rmtree(self.tmpdir, ignore_errors=True)
    
    def test_create_entity(self):
        """Test creating entities."""
        entity = self.graph.create_entity(
            name="Test Task",
            entity_type=EntityType.TASK,
            properties={"status": "pending"},
        )
        
        assert entity.name == "Test Task"
        assert entity.entity_type == EntityType.TASK
        assert entity.properties["status"] == "pending"
    
    def test_create_relationship(self):
        """Test creating relationships."""
        task1 = self.graph.create_entity("Task 1", EntityType.TASK)
        task2 = self.graph.create_entity("Task 2", EntityType.TASK)
        
        rel = self.graph.create_relationship(
            task1.id,
            task2.id,
            RelationType.DEPENDS_ON,
        )
        
        assert rel is not None
        assert rel.source_id == task1.id
        assert rel.target_id == task2.id
    
    def test_get_neighbors(self):
        """Test getting neighboring entities."""
        task = self.graph.create_entity("Main Task", EntityType.TASK)
        dep1 = self.graph.create_entity("Dependency 1", EntityType.TASK)
        dep2 = self.graph.create_entity("Dependency 2", EntityType.TASK)
        
        self.graph.create_relationship(task.id, dep1.id, RelationType.DEPENDS_ON)
        self.graph.create_relationship(task.id, dep2.id, RelationType.DEPENDS_ON)
        
        neighbors = self.graph.get_neighbors(task.id, direction="outgoing")
        assert len(neighbors) == 2
    
    def test_find_path(self):
        """Test finding paths between entities."""
        a = self.graph.create_entity("A", EntityType.TASK)
        b = self.graph.create_entity("B", EntityType.TASK)
        c = self.graph.create_entity("C", EntityType.TASK)
        
        self.graph.create_relationship(a.id, b.id, RelationType.PRECEDES)
        self.graph.create_relationship(b.id, c.id, RelationType.PRECEDES)
        
        path = self.graph.find_path(a.id, c.id)
        
        assert path is not None
        assert len(path) == 3
    
    def test_search(self):
        """Test searching entities."""
        self.graph.create_entity("Python implementation", EntityType.TASK)
        self.graph.create_entity("JavaScript testing", EntityType.TASK)
        
        results = self.graph.search("Python")
        assert len(results) == 1
        assert "Python" in results[0].name
    
    def test_persistence(self):
        """Test graph persistence."""
        self.graph.create_entity("Persistent Task", EntityType.TASK)
        self.graph.save()
        
        # Create new graph instance
        graph2 = KnowledgeGraph(storage_path=Path(self.tmpdir))
        
        results = graph2.search("Persistent")
        assert len(results) == 1


# ============================================================
# Orchestration Pattern Tests
# ============================================================

class TestOrchestrationPatterns:
    """Test orchestration patterns."""
    
    def test_pattern_enum(self):
        """Test orchestration pattern enum."""
        assert OrchestrationPattern.SEQUENTIAL.value == "sequential"
        assert OrchestrationPattern.CONCURRENT.value == "concurrent"
        assert OrchestrationPattern.GROUP_CHAT.value == "group_chat"
        assert OrchestrationPattern.DYNAMIC_ROUTING.value == "dynamic_routing"
        assert OrchestrationPattern.MANAGER.value == "manager"
    
    def test_factory_creation(self):
        """Test pattern factory."""
        for pattern in OrchestrationPattern:
            instance = OrchestrationFactory.create(pattern)
            assert instance is not None
            assert instance.pattern == pattern
    
    def test_factory_list_patterns(self):
        """Test listing available patterns."""
        patterns = OrchestrationFactory.list_patterns()
        
        assert "sequential" in patterns
        assert "concurrent" in patterns
        assert "manager" in patterns


# ============================================================
# Resilience Tests
# ============================================================

class TestResilience:
    """Test resilience patterns."""
    
    def test_circuit_breaker_initial_state(self):
        """Test circuit breaker starts closed."""
        cb = CircuitBreaker("test")
        assert cb.state == CircuitState.CLOSED
    
    def test_circuit_breaker_opens_on_failures(self):
        """Test circuit breaker opens after threshold failures."""
        config = CircuitBreakerConfig(failure_threshold=3)
        cb = CircuitBreaker("test", config)
        
        for _ in range(3):
            cb.record_failure()
        
        assert cb.state == CircuitState.OPEN
    
    def test_circuit_breaker_resets_on_success(self):
        """Test circuit breaker resets failure count on success."""
        config = CircuitBreakerConfig(failure_threshold=3)
        cb = CircuitBreaker("test", config)
        
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        
        assert cb._failure_count == 0
    
    def test_circuit_breaker_blocks_when_open(self):
        """Test circuit breaker blocks calls when open."""
        config = CircuitBreakerConfig(failure_threshold=1)
        cb = CircuitBreaker("test", config)
        
        cb.record_failure()
        
        assert cb.state == CircuitState.OPEN
        assert cb.can_execute() == False
    
    def test_retry_policy_backoff(self):
        """Test retry policy calculates backoff."""
        policy = RetryPolicy()
        
        delay0 = policy.calculate_delay(0)
        delay1 = policy.calculate_delay(1)
        delay2 = policy.calculate_delay(2)
        
        # Delays should increase (exponential)
        assert delay1 > delay0
        assert delay2 > delay1
    
    def test_resilience_manager(self):
        """Test unified resilience manager."""
        manager = ResilienceManager()
        
        # Get circuit breaker
        cb = manager.get_circuit_breaker("test_service")
        assert cb is not None
        
        # Same name returns same instance
        cb2 = manager.get_circuit_breaker("test_service")
        assert cb is cb2


# ============================================================
# OBOC Orchestrator Tests
# ============================================================

class TestOBOCOrchestrator:
    """Test the main OBOC orchestrator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.tmpdir = tempfile.mkdtemp()
        self.config = OBOCConfig(
            enable_memory=True,
            enable_knowledge_graph=True,
            memory_storage_path=self.tmpdir,
            knowledge_storage_path=self.tmpdir,
        )
        self.orchestrator = OBOCOrchestrator(self.config)
    
    def teardown_method(self):
        """Clean up."""
        shutil.rmtree(self.tmpdir, ignore_errors=True)
    
    def test_orchestrator_initialization(self):
        """Test orchestrator initializes all components."""
        assert self.orchestrator.memory is not None
        assert self.orchestrator.knowledge is not None
        assert self.orchestrator.resilience is not None
    
    def test_orchestrator_agents(self):
        """Test orchestrator has all agent tiers."""
        assert len(self.orchestrator.strategic_agents) > 0
        assert len(self.orchestrator.tactical_agents) > 0
        assert len(self.orchestrator.operational_agents) > 0
    
    def test_list_agents(self):
        """Test listing all agents."""
        agents = self.orchestrator.list_agents()
        
        tiers = {a["tier"] for a in agents}
        assert "strategic" in tiers
        assert "tactical" in tiers
        assert "operational" in tiers
    
    def test_get_status(self):
        """Test getting orchestrator status."""
        status = self.orchestrator.get_status()
        
        assert "agents" in status
        assert "memory" in status
        assert "knowledge" in status
        assert "resilience" in status


# ============================================================
# Run Tests
# ============================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
