"""
Configuration management for AgentArmy.
"""

import os
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List


@dataclass
class AgentConfig:
    """Configuration for an individual agent."""
    name: str
    model_id: str = "us.anthropic.claude-3-5-sonnet-20241022-v2:0"
    max_tokens: int = 4096
    temperature: float = 0.7
    enabled: bool = True
    system_prompt: Optional[str] = None


@dataclass
class Config:
    """Main configuration for AgentArmy."""
    
    # AWS Bedrock settings
    aws_region: str = "us-east-1"
    default_model: str = "us.anthropic.claude-3-5-sonnet-20241022-v2:0"
    
    # Orchestration settings
    enable_parallel: bool = True
    max_parallel_agents: int = 3
    max_iterations: int = 10
    timeout_seconds: int = 300
    
    # Agent configurations
    agents: Dict[str, AgentConfig] = field(default_factory=dict)
    
    # Logging
    log_level: str = "INFO"
    log_to_file: bool = False
    log_file_path: str = "~/.agent_army/logs/agent_army.log"
    
    # State persistence
    persist_state: bool = True
    state_dir: str = "~/.agent_army/state"
    
    # Cost tracking
    track_costs: bool = True
    cost_limit: Optional[float] = None  # Max cost per execution
    
    @classmethod
    def load(cls, config_path: Optional[str] = None) -> "Config":
        """Load configuration from YAML file."""
        if config_path is None:
            # Look in standard locations
            locations = [
                Path.cwd() / "config.yaml",
                Path.cwd() / "agent_army.yaml",
                Path.home() / ".agent_army" / "config.yaml",
                Path(__file__).parent.parent / "config.yaml",
            ]
            for loc in locations:
                if loc.exists():
                    config_path = str(loc)
                    break
        
        config = cls()
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                data = yaml.safe_load(f) or {}
            
            # Update from file
            for key, value in data.items():
                if key == "agents" and isinstance(value, dict):
                    for agent_name, agent_data in value.items():
                        config.agents[agent_name] = AgentConfig(
                            name=agent_name,
                            **agent_data
                        )
                elif hasattr(config, key):
                    setattr(config, key, value)
        
        # Override from environment
        config._load_from_env()
        
        # Initialize default agents if not specified
        config._init_default_agents()
        
        return config
    
    def _load_from_env(self):
        """Load overrides from environment variables."""
        env_mappings = {
            "AGENT_ARMY_AWS_REGION": "aws_region",
            "AGENT_ARMY_MODEL": "default_model",
            "AGENT_ARMY_LOG_LEVEL": "log_level",
            "AGENT_ARMY_PARALLEL": ("enable_parallel", lambda x: x.lower() == "true"),
            "AGENT_ARMY_TIMEOUT": ("timeout_seconds", int),
            "AGENT_ARMY_COST_LIMIT": ("cost_limit", float),
        }
        
        for env_var, mapping in env_mappings.items():
            value = os.environ.get(env_var)
            if value:
                if isinstance(mapping, tuple):
                    attr, converter = mapping
                    setattr(self, attr, converter(value))
                else:
                    setattr(self, mapping, value)
    
    def _init_default_agents(self):
        """Initialize default agent configurations."""
        defaults = {
            "planner": AgentConfig(
                name="planner",
                model_id=self.default_model,
                system_prompt="You are a task planning expert. Break down complex tasks into actionable steps."
            ),
            "research": AgentConfig(
                name="research",
                model_id=self.default_model,
                system_prompt="You are a research specialist. Find information, best practices, and solutions."
            ),
            "implementation": AgentConfig(
                name="implementation",
                model_id=self.default_model,
                max_tokens=8192,
                system_prompt="You are an expert software engineer. Write clean, production-ready code."
            ),
            "testing": AgentConfig(
                name="testing",
                model_id=self.default_model,
                system_prompt="You are a QA engineer. Write comprehensive tests and validate implementations."
            ),
            "analysis": AgentConfig(
                name="analysis",
                model_id=self.default_model,
                system_prompt="You are a code analyst. Review code, identify issues, and suggest improvements."
            ),
        }
        
        for name, default_config in defaults.items():
            if name not in self.agents:
                self.agents[name] = default_config
    
    def save(self, config_path: str):
        """Save configuration to YAML file."""
        data = {
            "aws_region": self.aws_region,
            "default_model": self.default_model,
            "enable_parallel": self.enable_parallel,
            "max_parallel_agents": self.max_parallel_agents,
            "max_iterations": self.max_iterations,
            "timeout_seconds": self.timeout_seconds,
            "log_level": self.log_level,
            "persist_state": self.persist_state,
            "track_costs": self.track_costs,
            "agents": {
                name: {
                    "model_id": agent.model_id,
                    "max_tokens": agent.max_tokens,
                    "temperature": agent.temperature,
                    "enabled": agent.enabled,
                }
                for name, agent in self.agents.items()
            }
        }
        
        Path(config_path).parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)
