"""
Planner Agent - Task decomposition and workflow planning.
"""

import json
from typing import Dict, Any, Optional, List

from .base import BaseAgent
from ..state import State, TaskStep


class PlannerAgent(BaseAgent):
    """
    Planner Agent for task decomposition.
    
    Responsibilities:
    - Break down complex tasks into steps
    - Determine which agents to use
    - Create execution order
    - Handle dependencies
    """
    
    @property
    def agent_type(self) -> str:
        return "planner"
    
    @property
    def description(self) -> str:
        return "Task Planning & Decomposition Agent"
    
    async def create_plan(
        self,
        user_request: str,
        state: State,
        available_agents: List[str]
    ) -> List[TaskStep]:
        """
        Create an execution plan for the user request.
        
        Args:
            user_request: The user's request
            state: Current state
            available_agents: List of available agent names
            
        Returns:
            List of TaskStep objects
        """
        # Build planning prompt
        prompt = f"""You are a task planning expert. Analyze the following request and create an execution plan.

USER REQUEST:
{user_request}

PROJECT CONTEXT:
{json.dumps(state.project_context, indent=2) if state.project_context else "No specific context provided"}

AVAILABLE AGENTS:
{', '.join(available_agents)}

Agent Descriptions:
- research: Finds information, best practices, documentation, solutions
- implementation: Writes code, creates files, implements features
- testing: Creates tests, validates implementations, runs quality checks
- analysis: Reviews code, analyzes architecture, suggests improvements

INSTRUCTIONS:
1. Break down the task into logical steps
2. Assign the most appropriate agent to each step
3. Order steps by dependencies (what must happen first)
4. Keep steps focused and actionable

OUTPUT FORMAT (JSON):
{{
    "plan": [
        {{
            "step_id": "1",
            "description": "Clear description of what this step accomplishes",
            "agent": "agent_name",
            "dependencies": []
        }},
        {{
            "step_id": "2", 
            "description": "Next step description",
            "agent": "agent_name",
            "dependencies": ["1"]
        }}
    ],
    "reasoning": "Brief explanation of the plan"
}}

Create a practical plan with 2-5 steps. Output ONLY valid JSON."""

        # Execute planning
        execution = await self.execute(prompt, state)
        
        if not execution.success or not execution.result:
            # Fallback to simple plan
            return [
                TaskStep(
                    step_id="1",
                    description="Research approach and best practices",
                    agent="research"
                ),
                TaskStep(
                    step_id="2",
                    description="Implement the solution",
                    agent="implementation",
                    dependencies=["1"]
                )
            ]
        
        # Parse plan
        try:
            # Extract JSON from response
            result = execution.result
            # Find JSON in response
            start = result.find('{')
            end = result.rfind('}') + 1
            if start >= 0 and end > start:
                json_str = result[start:end]
                data = json.loads(json_str)
                
                steps = []
                for step_data in data.get("plan", []):
                    steps.append(TaskStep(
                        step_id=str(step_data.get("step_id", len(steps) + 1)),
                        description=step_data.get("description", "Execute step"),
                        agent=step_data.get("agent", "research"),
                        dependencies=step_data.get("dependencies", [])
                    ))
                
                if steps:
                    return steps
        except (json.JSONDecodeError, KeyError) as e:
            print(f"[PLANNER] Warning: Could not parse plan JSON: {e}")
        
        # Fallback
        return [
            TaskStep(
                step_id="1",
                description="Research approach and best practices",
                agent="research"
            ),
            TaskStep(
                step_id="2", 
                description="Implement the solution",
                agent="implementation",
                dependencies=["1"]
            )
        ]
    
    async def should_continue(
        self,
        state: State,
        max_iterations: int = 10
    ) -> bool:
        """
        Determine if execution should continue.
        
        Args:
            state: Current state
            max_iterations: Maximum allowed iterations
            
        Returns:
            True if should continue, False otherwise
        """
        # Check iteration limit
        if len(state.agent_history) >= max_iterations:
            print(f"[PLANNER] Max iterations ({max_iterations}) reached")
            return False
        
        # Check if all steps completed
        pending_steps = [s for s in state.plan if s.status in ("pending", "running")]
        if not pending_steps:
            print("[PLANNER] All steps completed")
            return False
        
        # Check for too many failures
        failed_steps = [s for s in state.plan if s.status == "failed"]
        if len(failed_steps) >= 3:
            print("[PLANNER] Too many failed steps")
            return False
        
        return True
    
    async def get_next_step(self, state: State) -> Optional[TaskStep]:
        """
        Get the next step to execute.
        
        Args:
            state: Current state
            
        Returns:
            Next TaskStep or None if done
        """
        for step in state.plan:
            if step.status == "pending":
                # Check dependencies
                deps_met = all(
                    any(s.step_id == dep and s.status == "completed" for s in state.plan)
                    for dep in step.dependencies
                )
                if deps_met:
                    return step
        
        return None
