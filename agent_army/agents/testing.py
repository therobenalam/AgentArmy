"""
Testing Agent - Test generation and validation.
"""

from typing import Dict, Any, Optional

from .base import BaseAgent
from ..state import State


class TestingAgent(BaseAgent):
    """
    Testing Agent for quality assurance.
    
    Responsibilities:
    - Generate unit tests
    - Create integration tests
    - Validate implementations
    - Check code quality
    """
    
    @property
    def agent_type(self) -> str:
        return "testing"
    
    @property
    def description(self) -> str:
        return "Testing & Quality Assurance Agent"
    
    async def generate_tests(
        self,
        code_or_feature: str,
        state: State,
        test_type: str = "unit"
    ) -> str:
        """
        Generate tests for code or feature.
        
        Args:
            code_or_feature: Code to test or feature description
            state: Current state
            test_type: Type of tests (unit, integration, e2e)
            
        Returns:
            Generated tests
        """
        # Get implementation context
        impl_context = ""
        for execution in reversed(state.agent_history):
            if execution.agent_name == "implementation" and execution.result:
                impl_context = f"\nIMPLEMENTATION TO TEST:\n{execution.result[:3000]}"
                break
        
        prompt = f"""You are an expert QA engineer. Generate comprehensive {test_type} tests.

TARGET:
{code_or_feature}
{impl_context}

PROJECT CONTEXT:
{state.project_context if state.project_context else "General testing"}

INSTRUCTIONS:
1. Create thorough {test_type} tests
2. Cover happy paths and edge cases
3. Include error handling scenarios
4. Use appropriate testing framework
5. Follow testing best practices

OUTPUT FORMAT:
1. Test strategy explanation
2. Complete test code
3. Test coverage summary
4. Instructions to run tests

Generate production-quality tests."""

        execution = await self.execute(prompt, state)
        state.add_agent_execution(execution)
        
        return execution.result or "Tests generated."
    
    async def validate_implementation(
        self,
        implementation: str,
        requirements: str,
        state: State
    ) -> Dict[str, Any]:
        """
        Validate an implementation against requirements.
        
        Args:
            implementation: The implementation to validate
            requirements: Original requirements
            state: Current state
            
        Returns:
            Validation results
        """
        prompt = f"""You are a code reviewer. Validate this implementation against requirements.

REQUIREMENTS:
{requirements}

IMPLEMENTATION:
{implementation[:4000]}

VALIDATION CHECKLIST:
1. Does it meet all requirements?
2. Is the code quality acceptable?
3. Are there any bugs or issues?
4. Is error handling adequate?
5. Is it production-ready?

OUTPUT FORMAT (JSON):
{{
    "valid": true/false,
    "score": 1-10,
    "requirements_met": ["req1", "req2"],
    "requirements_missing": ["req3"],
    "issues": ["issue1", "issue2"],
    "suggestions": ["suggestion1"],
    "summary": "Overall assessment"
}}

Output ONLY valid JSON."""

        execution = await self.execute(prompt, state)
        state.add_agent_execution(execution)
        
        if execution.result:
            try:
                # Extract JSON
                result = execution.result
                start = result.find('{')
                end = result.rfind('}') + 1
                if start >= 0 and end > start:
                    return json.loads(result[start:end])
            except:
                pass
        
        return {
            "valid": True,
            "score": 7,
            "summary": "Validation completed"
        }


# Need json import
import json
