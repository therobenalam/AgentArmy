"""
Base agent class for AgentArmy.

All specialized agents inherit from this class.
"""

import json
import boto3
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Any, Optional, List, Type

from ..config import AgentConfig
from ..state import State, AgentExecution


class AgentRegistry:
    """Registry for all available agents."""
    
    _agents: Dict[str, Type["BaseAgent"]] = {}
    
    @classmethod
    def register(cls, name: str, agent_class: Type["BaseAgent"]):
        """Register an agent class."""
        cls._agents[name] = agent_class
    
    @classmethod
    def get(cls, name: str) -> Optional[Type["BaseAgent"]]:
        """Get an agent class by name."""
        return cls._agents.get(name)
    
    @classmethod
    def list_agents(cls) -> List[str]:
        """List all registered agent names."""
        return list(cls._agents.keys())
    
    @classmethod
    def create(cls, name: str, config: AgentConfig, **kwargs) -> "BaseAgent":
        """Create an agent instance by name."""
        agent_class = cls.get(name)
        if not agent_class:
            raise ValueError(f"Unknown agent: {name}")
        return agent_class(config, **kwargs)


class BaseAgent(ABC):
    """
    Base class for all AgentArmy agents.
    
    Provides common functionality for:
    - AWS Bedrock integration
    - Token/cost tracking
    - Error handling
    - State management
    """
    
    # Cost per 1K tokens (Claude Sonnet)
    INPUT_COST_PER_1K = 0.003
    OUTPUT_COST_PER_1K = 0.015
    
    def __init__(self, config: AgentConfig, aws_region: str = "us-east-1"):
        self.config = config
        self.name = config.name
        self.model_id = config.model_id
        self.max_tokens = config.max_tokens
        self.temperature = config.temperature
        self.system_prompt = config.system_prompt
        
        # Initialize Bedrock client
        self.bedrock = boto3.client(
            "bedrock-runtime",
            region_name=aws_region
        )
    
    @property
    @abstractmethod
    def agent_type(self) -> str:
        """Return the agent type identifier."""
        pass
    
    @property
    def description(self) -> str:
        """Return agent description."""
        return f"{self.agent_type.title()} Agent"
    
    async def execute(
        self,
        task: str,
        state: State,
        context: Optional[Dict[str, Any]] = None
    ) -> AgentExecution:
        """
        Execute a task and return execution record.
        
        Args:
            task: The task description
            state: Current execution state
            context: Optional additional context
            
        Returns:
            AgentExecution with results
        """
        execution = AgentExecution(
            agent_name=self.name,
            started_at=datetime.now()
        )
        
        try:
            print(f"[{self.name.upper()}] 🚀 Starting execution...")
            print(f"[{self.name.upper()}] Task: {task[:100]}{'...' if len(task) > 100 else ''}")
            
            # Build messages
            messages = self._build_messages(task, state, context)
            
            # Call Bedrock
            response = await self._call_bedrock(messages)
            
            # Extract result
            result = response["content"]
            input_tokens = response["input_tokens"]
            output_tokens = response["output_tokens"]
            
            # Calculate cost
            cost = self._calculate_cost(input_tokens, output_tokens)
            
            # Update execution
            execution.ended_at = datetime.now()
            execution.input_tokens = input_tokens
            execution.output_tokens = output_tokens
            execution.cost = cost
            execution.result = result
            execution.success = True
            
            print(f"[{self.name.upper()}] ✅ Completed (tokens: {input_tokens + output_tokens}, cost: ${cost:.4f})")
            
        except Exception as e:
            execution.ended_at = datetime.now()
            execution.success = False
            execution.error = str(e)
            print(f"[{self.name.upper()}] ❌ Error: {e}")
        
        return execution
    
    def _build_messages(
        self,
        task: str,
        state: State,
        context: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, str]]:
        """Build messages for Bedrock call."""
        messages = []
        
        # Add system prompt if available
        if self.system_prompt:
            messages.append({
                "role": "user",
                "content": f"[SYSTEM CONTEXT]\n{self.system_prompt}\n\n[USER REQUEST]\n{task}"
            })
        else:
            messages.append({
                "role": "user",
                "content": task
            })
        
        # Add context from previous steps
        context_msgs = state.get_context_for_agent()
        for msg in context_msgs:
            if msg["role"] != "system":  # Skip system messages, already included
                messages.append(msg)
        
        return messages
    
    async def _call_bedrock(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Call AWS Bedrock with messages."""
        # Format for Bedrock Converse API
        formatted_messages = []
        for msg in messages:
            formatted_messages.append({
                "role": msg["role"],
                "content": [{"text": msg["content"]}]
            })
        
        print(f"[{self.name.upper()}] Calling Bedrock with {len(formatted_messages)} message(s)")
        print(f"[{self.name.upper()}] Model: {self.model_id}")
        
        response = self.bedrock.converse(
            modelId=self.model_id,
            messages=formatted_messages,
            inferenceConfig={
                "maxTokens": self.max_tokens,
                "temperature": self.temperature,
            }
        )
        
        # Extract response
        output = response["output"]["message"]["content"][0]["text"]
        usage = response.get("usage", {})
        
        return {
            "content": output,
            "input_tokens": usage.get("inputTokens", 0),
            "output_tokens": usage.get("outputTokens", 0),
        }
    
    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost based on token usage."""
        input_cost = (input_tokens / 1000) * self.INPUT_COST_PER_1K
        output_cost = (output_tokens / 1000) * self.OUTPUT_COST_PER_1K
        return input_cost + output_cost
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(name={self.name}, model={self.model_id})>"
