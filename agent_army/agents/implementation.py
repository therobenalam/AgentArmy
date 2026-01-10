"""
Implementation Agent - Code generation and file creation.
"""

import json
from typing import Dict, Any, Optional, List

from .base import BaseAgent
from ..state import State


class ImplementationAgent(BaseAgent):
    """
    Implementation Agent for code generation.
    
    Responsibilities:
    - Write production-ready code
    - Create file structures
    - Implement features
    - Apply best practices
    """
    
    @property
    def agent_type(self) -> str:
        return "implementation"
    
    @property
    def description(self) -> str:
        return "Code Implementation & Generation Agent"
    
    async def implement(
        self,
        task: str,
        state: State,
        research_context: Optional[str] = None
    ) -> str:
        """
        Implement a feature or write code.
        
        Args:
            task: What to implement
            state: Current state  
            research_context: Optional research findings to incorporate
            
        Returns:
            Implementation result (code, explanations, etc.)
        """
        # Build implementation prompt
        context_section = ""
        if research_context:
            context_section = f"""
RESEARCH FINDINGS:
{research_context}

"""
        
        project_info = ""
        if state.project_context:
            project_info = f"""
PROJECT CONTEXT:
- Workspace: {state.project_context.get('workspace', 'Not specified')}
- Framework: {state.project_context.get('framework', 'Not specified')}
- Language: {state.project_context.get('language', 'Not specified')}

"""
        
        prompt = f"""You are an expert software engineer. Implement the following task.

TASK:
{task}
{project_info}{context_section}
INSTRUCTIONS:
1. Write clean, production-ready code
2. Follow best practices for the language/framework
3. Include proper error handling
4. Add helpful comments where appropriate
5. Structure code for maintainability

OUTPUT FORMAT:
Provide your implementation with:
1. A brief explanation of your approach
2. The complete code with proper formatting
3. Any setup or usage instructions
4. Notes on potential improvements or considerations

Be thorough and provide working, complete code."""

        execution = await self.execute(prompt, state)
        state.add_agent_execution(execution)
        
        # Store any code artifacts
        if execution.result:
            self._extract_artifacts(execution.result, state)
        
        return execution.result or "Implementation completed."
    
    def _extract_artifacts(self, result: str, state: State):
        """Extract code blocks and file content from result."""
        import re
        
        # Find code blocks with file paths
        file_pattern = r'```(\w+)?\s*(?:#\s*)?([\/\w\-\.]+\.\w+)?\n(.*?)```'
        matches = re.findall(file_pattern, result, re.DOTALL)
        
        for lang, filepath, code in matches:
            if filepath:
                state.artifacts[filepath] = {
                    "type": "file",
                    "language": lang or "text",
                    "content": code.strip()
                }
        
        # Count code blocks
        code_blocks = re.findall(r'```[\s\S]*?```', result)
        state.artifacts["code_block_count"] = len(code_blocks)
