"""
Analysis Agent - Code review and architecture analysis.
"""

import json
from typing import Dict, Any, Optional

from .base import BaseAgent
from ..state import State


class AnalysisAgent(BaseAgent):
    """
    Analysis Agent for code review and optimization.
    
    Responsibilities:
    - Review code quality
    - Analyze architecture
    - Identify improvements
    - Suggest optimizations
    """
    
    @property
    def agent_type(self) -> str:
        return "analysis"
    
    @property
    def description(self) -> str:
        return "Code Analysis & Review Agent"
    
    async def analyze(
        self,
        content: str,
        state: State,
        analysis_type: str = "code_review"
    ) -> str:
        """
        Analyze code or architecture.
        
        Args:
            content: Content to analyze
            state: Current state
            analysis_type: Type of analysis (code_review, architecture, performance, security)
            
        Returns:
            Analysis results
        """
        analysis_prompts = {
            "code_review": self._code_review_prompt,
            "architecture": self._architecture_prompt,
            "performance": self._performance_prompt,
            "security": self._security_prompt,
        }
        
        prompt_builder = analysis_prompts.get(analysis_type, self._code_review_prompt)
        prompt = prompt_builder(content, state)
        
        execution = await self.execute(prompt, state)
        state.add_agent_execution(execution)
        
        return execution.result or "Analysis completed."
    
    def _code_review_prompt(self, content: str, state: State) -> str:
        return f"""You are a senior code reviewer. Review this code thoroughly.

CODE TO REVIEW:
{content[:5000]}

REVIEW CRITERIA:
1. Code quality and readability
2. Best practices adherence
3. Error handling
4. Potential bugs
5. Performance considerations
6. Maintainability

OUTPUT FORMAT:
## Summary
Overall assessment

## Strengths
- Good aspects of the code

## Issues Found
- Issue 1 (severity: high/medium/low)
- Issue 2 ...

## Recommendations
1. Specific improvement suggestions

## Rating
X/10 with justification

Provide actionable, constructive feedback."""
    
    def _architecture_prompt(self, content: str, state: State) -> str:
        return f"""You are a software architect. Analyze this architecture.

ARCHITECTURE/CODE:
{content[:5000]}

ANALYSIS AREAS:
1. Design patterns used
2. Separation of concerns
3. Scalability considerations
4. Extensibility
5. Coupling and cohesion

OUTPUT FORMAT:
## Architecture Overview
Summary of the architecture

## Design Patterns
Patterns identified and their appropriateness

## Strengths
Architectural positives

## Concerns
Areas needing attention

## Recommendations
Architectural improvements

Provide expert architectural analysis."""
    
    def _performance_prompt(self, content: str, state: State) -> str:
        return f"""You are a performance engineer. Analyze for performance.

CODE/SYSTEM:
{content[:5000]}

PERFORMANCE AREAS:
1. Time complexity
2. Space complexity
3. Database/IO operations
4. Memory usage
5. Bottlenecks

OUTPUT FORMAT:
## Performance Summary
Overall performance assessment

## Complexity Analysis
Time and space complexity of key operations

## Bottlenecks Identified
Potential performance issues

## Optimization Recommendations
Specific improvements with expected impact

Provide actionable performance insights."""
    
    def _security_prompt(self, content: str, state: State) -> str:
        return f"""You are a security engineer. Perform security analysis.

CODE TO ANALYZE:
{content[:5000]}

SECURITY CHECKLIST:
1. Input validation
2. Authentication/Authorization
3. Data protection
4. SQL/Command injection
5. XSS vulnerabilities
6. Secret management

OUTPUT FORMAT:
## Security Summary
Overall security posture

## Vulnerabilities Found
- Vulnerability (severity: critical/high/medium/low)

## Security Best Practices
What's done well

## Remediation Recommendations
Specific fixes for issues found

Provide thorough security analysis."""
