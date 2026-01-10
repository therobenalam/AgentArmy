#!/usr/bin/env python3
"""
AgentArmy CLI - Command Line Interface

Access your army of AI agents from anywhere.
"""

import asyncio
import json
import sys
import functools
from pathlib import Path
from typing import Optional

import click

# Ensure agent_army is importable
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from agent_army import Orchestrator, Config


def async_command(f):
    """Decorator to run async commands."""
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        return asyncio.run(f(*args, **kwargs))
    return wrapper


@click.group()
@click.version_option(version="1.0.0", prog_name="AgentArmy")
def cli():
    """
    🪖 AgentArmy - Your AI Agent Orchestration System
    
    Execute complex tasks with an army of specialized AI agents.
    """
    pass


@cli.command()
@click.argument('task')
@click.option('--workspace', '-w', default=None, help='Project workspace path')
@click.option('--framework', '-f', default=None, help='Framework (e.g., React, FastAPI)')
@click.option('--language', '-l', default=None, help='Programming language')
@click.option('--parallel/--no-parallel', default=True, help='Enable parallel execution')
@click.option('--output', '-o', type=click.Choice(['text', 'json']), default='text', help='Output format')
@async_command
async def execute(task: str, workspace: Optional[str], framework: Optional[str], 
                  language: Optional[str], parallel: bool, output: str):
    """
    Execute a task with the agent army.
    
    Examples:
    
        agent-army execute "Build a REST API with authentication"
        
        agent-army execute "Refactor the auth module" -w /path/to/project -f FastAPI
    """
    click.echo("\n🪖 AgentArmy - Executing Task\n")
    click.echo(f"📋 Task: {task}\n")
    
    # Build context
    project_context = {}
    if workspace:
        project_context['workspace'] = workspace
    if framework:
        project_context['framework'] = framework
    if language:
        project_context['language'] = language
    
    if project_context:
        click.echo(f"🔧 Context: {json.dumps(project_context)}\n")
    
    # Create orchestrator and execute
    orchestrator = Orchestrator(enable_parallel=parallel)
    result = await orchestrator.execute(task, project_context)
    
    if output == 'json':
        # Clean result for JSON output
        json_result = {
            "success": result["success"],
            "execution_id": result["execution_id"],
            "output": result["output"],
            "summary": result["summary"],
            "agents_used": result["agents_used"],
            "cost": f"${result['total_cost']:.4f}",
            "duration": f"{result['duration_seconds']:.1f}s"
        }
        click.echo(json.dumps(json_result, indent=2))
    else:
        click.echo("\n" + "="*60)
        click.echo("📊 RESULTS")
        click.echo("="*60)
        click.echo(f"\n✅ Status: {'Success' if result['success'] else 'Failed'}")
        click.echo(f"🤖 Agents: {', '.join(result['agents_used'])}")
        click.echo(f"💰 Cost: ${result['total_cost']:.4f}")
        click.echo(f"⏱️  Duration: {result['duration_seconds']:.1f}s")
        click.echo(f"\n📝 Output:\n{result['output'][:2000]}{'...' if len(result['output']) > 2000 else ''}")


@cli.command()
@click.argument('message')
@click.option('--agent', '-a', default='research', help='Agent to chat with')
@async_command
async def chat(message: str, agent: str):
    """
    Chat with a specific agent.
    
    Examples:
    
        agent-army chat "What are best practices for REST API design?"
        
        agent-army chat "Review this code pattern" -a analysis
    """
    click.echo(f"\n🪖 AgentArmy - Chatting with {agent.title()} Agent\n")
    
    orchestrator = Orchestrator()
    response = await orchestrator.chat(message, agent=agent)
    
    click.echo(f"📝 Response:\n\n{response}")


@cli.command()
def agents():
    """List all available agents."""
    click.echo("\n🪖 AgentArmy - Available Agents\n")
    
    orchestrator = Orchestrator()
    agent_list = orchestrator.list_agents()
    
    for agent in agent_list:
        click.echo(f"  🤖 {agent['name'].upper()}")
        click.echo(f"     {agent['description']}")
        click.echo(f"     Model: {agent['model']}\n")


@cli.command()
@click.option('--output', '-o', default=None, help='Output file path')
def init(output: Optional[str]):
    """Initialize configuration file."""
    config = Config.load()
    
    output_path = output or "agent_army.yaml"
    config.save(output_path)
    
    click.echo(f"✅ Configuration saved to: {output_path}")
    click.echo("\nEdit this file to customize:")
    click.echo("  - AWS region and model")
    click.echo("  - Agent configurations")
    click.echo("  - Execution settings")


@cli.command()
@click.argument('execution_id')
def status(execution_id: str):
    """Check status of a previous execution."""
    from agent_army.state import State
    
    state = State.load(execution_id)
    
    if not state:
        click.echo(f"❌ Execution {execution_id} not found")
        return
    
    click.echo(f"\n🪖 AgentArmy - Execution Status\n")
    click.echo(f"ID: {state.execution_id}")
    click.echo(f"Status: {state.status.value}")
    click.echo(f"Request: {state.user_request[:100]}...")
    click.echo(f"Agents Used: {', '.join(state.agents_used)}")
    click.echo(f"Cost: ${state.total_cost:.4f}")
    click.echo(f"Duration: {state.duration_seconds:.1f}s")
    
    if state.error:
        click.echo(f"\n❌ Error: {state.error}")
    
    if state.final_result:
        click.echo(f"\n📝 Result:\n{state.final_result}")


def main():
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main()
