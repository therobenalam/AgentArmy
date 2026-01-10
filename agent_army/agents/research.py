"""
Research Agent - Information gathering and best practices.
"""

from typing import Dict, Any, Optional

from .base import BaseAgent
from ..state import State


class ResearchAgent(BaseAgent):
    """
    Research Agent for information gathering.
    
    Responsibilities:
    - Find best practices
    - Research solutions
    - Gather documentation
    - Identify approaches
    """
    
    @property
    def agent_type(self) -> str:
        return "research"
    
    @property
    def description(self) -> str:
        return "Research & Information Gathering Agent"
    
    async def research(
        self,
        topic: str,
        state: State,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Research a topic and return findings.
        
        Args:
            topic: What to research
            state: Current state
            context: Optional additional context
            
        Returns:
            Research findings as string
        """
        prompt = f"""You are an expert researcher. Research the following topic thoroughly.

TOPIC:
{topic}

PROJECT CONTEXT:
{state.project_context if state.project_context else "General research"}

INSTRUCTIONS:
1. Identify the key aspects of this topic
2. Find best practices and recommended approaches
3. Note any potential pitfalls or considerations
4. Provide actionable recommendations

OUTPUT FORMAT:
## Summary
Brief overview of findings

## Key Findings
- Finding 1
- Finding 2
...

## Best Practices
1. Practice 1
2. Practice 2
...

## Recommendations
Specific actionable recommendations for this project

## Resources
Any relevant documentation, patterns, or examples to consider

Provide thorough, practical research findings."""

        execution = await self.execute(prompt, state, context)
        state.add_agent_execution(execution)
        
        return execution.result or "Research completed but no detailed findings available."
