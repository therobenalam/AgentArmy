"""
Tests for AgentArmy Orchestrator
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from agent_army import Orchestrator, Config, State
from agent_army.state import ExecutionState, TaskStep
from agent_army.agents.base import BaseAgent, AgentRegistry


class TestConfig:
    """Tests for Config class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = Config()
        
        assert config.aws_region == "us-east-1"
        assert config.enable_parallel is True
        assert config.max_iterations == 10
        assert config.timeout_seconds == 300
    
    def test_default_agents_initialized(self):
        """Test that default agents are initialized."""
        config = Config()
        config._init_default_agents()
        
        expected_agents = ["planner", "research", "implementation", "testing", "analysis"]
        for agent in expected_agents:
            assert agent in config.agents
    
    def test_env_override(self):
        """Test environment variable overrides."""
        import os
        os.environ["AGENT_ARMY_AWS_REGION"] = "us-west-2"
        
        config = Config()
        config._load_from_env()
        
        assert config.aws_region == "us-west-2"
        
        # Cleanup
        del os.environ["AGENT_ARMY_AWS_REGION"]


class TestState:
    """Tests for State class."""
    
    def test_state_initialization(self):
        """Test state initialization."""
        state = State(user_request="Test task")
        
        assert state.user_request == "Test task"
        assert state.status == ExecutionState.PENDING
        assert len(state.execution_id) == 8
    
    def test_state_start(self):
        """Test starting execution."""
        state = State(user_request="Test")
        state.start()
        
        assert state.status == ExecutionState.PLANNING
        assert state.started_at is not None
    
    def test_state_complete(self):
        """Test completing execution."""
        state = State(user_request="Test")
        state.start()
        state.complete("Done!")
        
        assert state.status == ExecutionState.COMPLETED
        assert state.final_result == "Done!"
        assert state.ended_at is not None
    
    def test_state_fail(self):
        """Test failing execution."""
        state = State(user_request="Test")
        state.start()
        state.fail("Error occurred")
        
        assert state.status == ExecutionState.FAILED
        assert state.error == "Error occurred"
    
    def test_duration_calculation(self):
        """Test duration calculation."""
        state = State(user_request="Test")
        state.started_at = datetime.now()
        
        # Duration should be small but positive
        assert state.duration_seconds >= 0
    
    def test_add_message(self):
        """Test adding messages."""
        state = State(user_request="Test")
        state.add_message("user", "Hello")
        state.add_message("assistant", "Hi there!")
        
        assert len(state.messages) == 2
        assert state.messages[0]["role"] == "user"
        assert state.messages[1]["content"] == "Hi there!"


class TestTaskStep:
    """Tests for TaskStep class."""
    
    def test_task_step_creation(self):
        """Test creating a task step."""
        step = TaskStep(
            step_id="1",
            description="Research the topic",
            agent="research"
        )
        
        assert step.step_id == "1"
        assert step.agent == "research"
        assert step.status == "pending"
        assert step.dependencies == []
    
    def test_task_step_with_dependencies(self):
        """Test task step with dependencies."""
        step = TaskStep(
            step_id="2",
            description="Implement solution",
            agent="implementation",
            dependencies=["1"]
        )
        
        assert step.dependencies == ["1"]


class TestAgentRegistry:
    """Tests for AgentRegistry."""
    
    def test_list_agents(self):
        """Test listing registered agents."""
        agents = AgentRegistry.list_agents()
        
        expected = ["planner", "research", "implementation", "testing", "analysis"]
        for agent in expected:
            assert agent in agents
    
    def test_get_agent(self):
        """Test getting agent class."""
        from agent_army.agents import ResearchAgent
        
        agent_class = AgentRegistry.get("research")
        assert agent_class == ResearchAgent
    
    def test_get_unknown_agent(self):
        """Test getting unknown agent returns None."""
        agent_class = AgentRegistry.get("unknown_agent")
        assert agent_class is None


class TestOrchestrator:
    """Tests for Orchestrator."""
    
    def test_orchestrator_init(self):
        """Test orchestrator initialization."""
        with patch('boto3.client'):
            orchestrator = Orchestrator()
            
            assert orchestrator.config is not None
            assert "planner" in orchestrator.agents
    
    def test_list_agents(self):
        """Test listing agents from orchestrator."""
        with patch('boto3.client'):
            orchestrator = Orchestrator()
            agents = orchestrator.list_agents()
            
            assert len(agents) >= 5
            agent_names = [a["name"] for a in agents]
            assert "research" in agent_names


class TestOrchestratorExecution:
    """Integration-style tests for orchestration execution."""
    
    def test_execute_simple_task(self):
        """Test executing a simple task (sync wrapper)."""
        import asyncio
        
        async def run_test():
            with patch('boto3.client') as mock_boto:
                # Mock Bedrock response
                mock_client = MagicMock()
                mock_client.converse.return_value = {
                    "output": {
                        "message": {
                            "content": [{"text": '{"plan": [{"step_id": "1", "description": "Research", "agent": "research"}]}'}]
                        }
                    },
                    "usage": {"inputTokens": 100, "outputTokens": 50}
                }
                mock_boto.return_value = mock_client
                
                orchestrator = Orchestrator()
                
                # Mock agent execution
                for agent in orchestrator.agents.values():
                    agent._call_bedrock = AsyncMock(return_value={
                        "content": "Task completed successfully",
                        "input_tokens": 100,
                        "output_tokens": 50
                    })
                
                result = await orchestrator.execute("Test task")
                
                assert "success" in result
                assert "execution_id" in result
        
        asyncio.run(run_test())


# Run with: pytest tests/test_orchestrator.py -v
