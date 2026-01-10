#!/usr/bin/env python3
"""
AgentArmy MCP Server

Model Context Protocol server for VSCode/Copilot integration.
Enables chatting with the agent army directly from your editor.
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

# MCP SDK imports
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Tool,
    TextContent,
    CallToolResult,
)

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from agent_army import Orchestrator, Config, State


# Initialize server
server = Server("agent-army")

# Global orchestrator instance
_orchestrator = None


def get_orchestrator() -> Orchestrator:
    """Get or create orchestrator instance."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = Orchestrator()
    return _orchestrator


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available MCP tools."""
    return [
        Tool(
            name="agent_army_execute",
            description="Execute a complex task with the agent army. Agents will plan, research, implement, and validate.",
            inputSchema={
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "The task to execute (e.g., 'Build a REST API with JWT authentication')"
                    },
                    "workspace": {
                        "type": "string",
                        "description": "Optional: Path to project workspace"
                    },
                    "framework": {
                        "type": "string",
                        "description": "Optional: Framework being used (e.g., 'FastAPI', 'React')"
                    },
                    "language": {
                        "type": "string",
                        "description": "Optional: Programming language (e.g., 'Python', 'TypeScript')"
                    }
                },
                "required": ["task"]
            }
        ),
        Tool(
            name="agent_army_chat",
            description="Chat with a specific agent for quick questions or analysis.",
            inputSchema={
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "Your message or question"
                    },
                    "agent": {
                        "type": "string",
                        "description": "Agent to chat with: research, implementation, testing, analysis, planner",
                        "default": "research"
                    }
                },
                "required": ["message"]
            }
        ),
        Tool(
            name="agent_army_agents",
            description="List all available agents and their capabilities.",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        Tool(
            name="agent_army_status",
            description="Check status of a previous execution by ID.",
            inputSchema={
                "type": "object",
                "properties": {
                    "execution_id": {
                        "type": "string",
                        "description": "The execution ID to check"
                    }
                },
                "required": ["execution_id"]
            }
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Handle tool calls."""
    
    if name == "agent_army_execute":
        return await handle_execute(arguments)
    
    elif name == "agent_army_chat":
        return await handle_chat(arguments)
    
    elif name == "agent_army_agents":
        return await handle_list_agents()
    
    elif name == "agent_army_status":
        return await handle_status(arguments)
    
    else:
        return [TextContent(type="text", text=f"Unknown tool: {name}")]


async def handle_execute(arguments: dict) -> list[TextContent]:
    """Handle execute tool call."""
    task = arguments.get("task", "")
    
    project_context = {}
    if arguments.get("workspace"):
        project_context["workspace"] = arguments["workspace"]
    if arguments.get("framework"):
        project_context["framework"] = arguments["framework"]
    if arguments.get("language"):
        project_context["language"] = arguments["language"]
    
    orchestrator = get_orchestrator()
    result = await orchestrator.execute(task, project_context)
    
    # Format response
    response = f"""🪖 AgentArmy Execution Complete

**Status:** {'✅ Success' if result['success'] else '❌ Failed'}
**Execution ID:** {result['execution_id']}
**Agents Used:** {', '.join(result['agents_used'])}
**Cost:** ${result['total_cost']:.4f}
**Duration:** {result['duration_seconds']:.1f}s

## Output

{result['output']}

---
*Summary: {result['summary']}*
"""
    
    return [TextContent(type="text", text=response)]


async def handle_chat(arguments: dict) -> list[TextContent]:
    """Handle chat tool call."""
    message = arguments.get("message", "")
    agent = arguments.get("agent", "research")
    
    orchestrator = get_orchestrator()
    response = await orchestrator.chat(message, agent=agent)
    
    return [TextContent(type="text", text=f"**{agent.title()} Agent:**\n\n{response}")]


async def handle_list_agents() -> list[TextContent]:
    """Handle list agents tool call."""
    orchestrator = get_orchestrator()
    agents = orchestrator.list_agents()
    
    response = "🪖 **AgentArmy - Available Agents**\n\n"
    for agent in agents:
        response += f"### {agent['name'].upper()}\n"
        response += f"- **Type:** {agent['type']}\n"
        response += f"- **Description:** {agent['description']}\n"
        response += f"- **Model:** {agent['model']}\n\n"
    
    return [TextContent(type="text", text=response)]


async def handle_status(arguments: dict) -> list[TextContent]:
    """Handle status tool call."""
    execution_id = arguments.get("execution_id", "")
    
    state = State.load(execution_id)
    
    if not state:
        return [TextContent(type="text", text=f"❌ Execution {execution_id} not found")]
    
    response = f"""🪖 **Execution Status**

**ID:** {state.execution_id}
**Status:** {state.status.value}
**Request:** {state.user_request[:100]}...
**Agents:** {', '.join(state.agents_used)}
**Cost:** ${state.total_cost:.4f}
**Duration:** {state.duration_seconds:.1f}s
"""
    
    if state.error:
        response += f"\n**Error:** {state.error}"
    
    if state.final_result:
        response += f"\n\n## Result\n{state.final_result}"
    
    return [TextContent(type="text", text=response)]


async def main():
    """Main entry point for MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
