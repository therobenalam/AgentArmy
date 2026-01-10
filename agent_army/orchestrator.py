"""
AgentArmy Orchestrator - Main orchestration engine.

Coordinates the army of AI agents to execute complex tasks.
"""

import asyncio
from datetime import datetime
from typing import Dict, Any, Optional, List

from .config import Config
from .state import State, ExecutionState, TaskStep, AgentExecution
from .agents import AgentRegistry, PlannerAgent
from .agents.base import BaseAgent


class Orchestrator:
    """
    Main orchestration engine for AgentArmy.
    
    Manages:
    - Task planning and decomposition
    - Agent selection and routing
    - Execution flow
    - State management
    - Cost tracking
    """
    
    def __init__(
        self,
        config: Optional[Config] = None,
        enable_parallel: Optional[bool] = None
    ):
        self.config = config or Config.load()
        self.enable_parallel = enable_parallel if enable_parallel is not None else self.config.enable_parallel
        
        # Initialize agents
        self.agents: Dict[str, BaseAgent] = {}
        self._init_agents()
        
        # Planner agent
        self.planner: PlannerAgent = self.agents.get("planner")
    
    def _init_agents(self):
        """Initialize all enabled agents."""
        for name, agent_config in self.config.agents.items():
            if agent_config.enabled:
                try:
                    agent = AgentRegistry.create(
                        name,
                        agent_config,
                        aws_region=self.config.aws_region
                    )
                    self.agents[name] = agent
                    print(f"[ORCHESTRATOR] Initialized agent: {name}")
                except ValueError as e:
                    print(f"[ORCHESTRATOR] Warning: Could not initialize {name}: {e}")
    
    async def execute(
        self,
        user_request: str,
        project_context: Optional[Dict[str, Any]] = None,
        state: Optional[State] = None
    ) -> Dict[str, Any]:
        """
        Execute a complete workflow from user request to completion.
        
        Args:
            user_request: The user's request
            project_context: Optional project context
            state: Optional pre-existing state
            
        Returns:
            Execution results
        """
        print(f"\n{'='*60}")
        print(f"🪖 AgentArmy Orchestrator Starting")
        print(f"{'='*60}")
        print(f"Request: {user_request[:100]}{'...' if len(user_request) > 100 else ''}")
        print(f"Parallel Execution: {self.enable_parallel}")
        print(f"Available Agents: {', '.join(self.agents.keys())}")
        print(f"{'='*60}\n")
        
        # Initialize state
        if state is None:
            state = State(
                user_request=user_request,
                project_context=project_context or {}
            )
        
        state.start()
        state.add_message("user", user_request)
        
        try:
            # Step 1: Planning
            print("[ORCHESTRATOR] 📋 Creating execution plan...")
            state.status = ExecutionState.PLANNING
            
            plan = await self.planner.create_plan(
                user_request,
                state,
                available_agents=list(self.agents.keys())
            )
            state.plan = plan
            
            print(f"[ORCHESTRATOR] Plan created with {len(plan)} steps:")
            for step in plan:
                print(f"  {step.step_id}. [{step.agent}] {step.description}")
            print()
            
            # Step 2: Execution
            state.status = ExecutionState.EXECUTING
            await self._execute_plan(state)
            
            # Step 3: Finalize
            state.complete(self._generate_summary(state))
            
            print(f"\n{'='*60}")
            print(f"✅ Execution Complete!")
            print(f"{'='*60}")
            
            # Save state if configured
            if self.config.persist_state:
                state.save()
            
            return self._generate_result(state)
            
        except Exception as e:
            print(f"\n❌ Error during execution: {e}")
            state.fail(str(e))
            
            if self.config.persist_state:
                state.save()
            
            return {
                "success": False,
                "error": str(e),
                "state": state.to_dict()
            }
    
    async def _execute_plan(self, state: State):
        """Execute the plan step by step."""
        max_iterations = self.config.max_iterations
        iteration = 0
        
        while await self.planner.should_continue(state, max_iterations):
            iteration += 1
            
            # Get next step
            step = await self.planner.get_next_step(state)
            if not step:
                print("[ORCHESTRATOR] No more steps to execute")
                break
            
            print(f"\n[ORCHESTRATOR] 🚀 Executing step {step.step_id}: {step.description}")
            
            # Get agent
            agent = self.agents.get(step.agent)
            if not agent:
                print(f"[ORCHESTRATOR] ⚠️ Agent '{step.agent}' not found, skipping")
                step.status = "failed"
                step.error = f"Agent '{step.agent}' not available"
                continue
            
            # Update state
            state.current_agent = step.agent
            step.status = "running"
            
            # Execute step
            try:
                execution = await self._execute_step(agent, step, state)
                
                if execution.success:
                    step.status = "completed"
                    step.result = execution.result
                    state.add_message("assistant", f"[{step.agent}] {execution.result[:500]}...")
                else:
                    step.status = "failed"
                    step.error = execution.error
                
                state.add_agent_execution(execution)
                
            except Exception as e:
                step.status = "failed"
                step.error = str(e)
                print(f"[ORCHESTRATOR] Step failed: {e}")
        
        state.current_agent = None
    
    async def _execute_step(
        self,
        agent: BaseAgent,
        step: TaskStep,
        state: State
    ) -> AgentExecution:
        """Execute a single step with an agent."""
        # Build task prompt based on step and previous results
        task = step.description
        
        # Add context from previous steps
        completed_steps = [s for s in state.plan if s.status == "completed" and s.result]
        if completed_steps:
            context = "\n\nPrevious step results:\n"
            for prev in completed_steps[-2:]:  # Last 2 completed steps
                context += f"- {prev.description}: {prev.result[:300]}...\n"
            task = f"{task}\n{context}"
        
        return await agent.execute(task, state)
    
    def _generate_summary(self, state: State) -> str:
        """Generate execution summary."""
        completed = [s for s in state.plan if s.status == "completed"]
        failed = [s for s in state.plan if s.status == "failed"]
        
        summary_parts = [
            f"Completed {len(completed)}/{len(state.plan)} steps",
            f"Agents used: {', '.join(state.agents_used)}",
            f"Total cost: ${state.total_cost:.4f}",
            f"Duration: {state.duration_seconds:.1f}s"
        ]
        
        if failed:
            summary_parts.append(f"Failed steps: {len(failed)}")
        
        return " | ".join(summary_parts)
    
    def _generate_result(self, state: State) -> Dict[str, Any]:
        """Generate final result dictionary."""
        # Get final output from last completed step
        completed_steps = [s for s in state.plan if s.status == "completed" and s.result]
        final_output = completed_steps[-1].result if completed_steps else "No output generated"
        
        return {
            "success": state.status == ExecutionState.COMPLETED,
            "execution_id": state.execution_id,
            "output": final_output,
            "summary": state.final_result,
            "agents_used": state.agents_used,
            "steps_completed": len([s for s in state.plan if s.status == "completed"]),
            "steps_total": len(state.plan),
            "total_tokens": state.total_tokens,
            "total_cost": state.total_cost,
            "duration_seconds": state.duration_seconds,
            "artifacts": state.artifacts,
            "state": state.to_dict()
        }
    
    async def chat(
        self,
        message: str,
        state: Optional[State] = None,
        agent: str = "research"
    ) -> str:
        """
        Simple chat interface with a specific agent.
        
        Args:
            message: User message
            state: Optional existing state for context
            agent: Which agent to chat with
            
        Returns:
            Agent response
        """
        if state is None:
            state = State(user_request=message)
        
        state.add_message("user", message)
        
        target_agent = self.agents.get(agent)
        if not target_agent:
            return f"Agent '{agent}' not available. Available: {', '.join(self.agents.keys())}"
        
        execution = await target_agent.execute(message, state)
        state.add_agent_execution(execution)
        
        if execution.success:
            state.add_message("assistant", execution.result)
            return execution.result
        else:
            return f"Error: {execution.error}"
    
    def list_agents(self) -> List[Dict[str, str]]:
        """List all available agents."""
        return [
            {
                "name": name,
                "type": agent.agent_type,
                "description": agent.description,
                "model": agent.model_id
            }
            for name, agent in self.agents.items()
        ]


async def quick_execute(task: str, **kwargs) -> Dict[str, Any]:
    """Quick execution helper."""
    orchestrator = Orchestrator()
    return await orchestrator.execute(task, **kwargs)
