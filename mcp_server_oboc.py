#!/usr/bin/env python3
"""
AgentArmy OBOC MCP Server

Model Context Protocol server for VSCode/Copilot integration
using the OBOC (One Brain, One Context) architecture.

Features:
- Three-Tier Agent Hierarchy
- Memory Management (Short-term, Working, Long-term)
- Knowledge Graph Integration
- Multiple Orchestration Patterns
- Resilience & Observability
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

from agent_army.oboc import (
    OBOCOrchestrator,
    OBOCConfig,
    OrchestrationPattern,
)


# Initialize server
server = Server("agent-army-oboc")

# Global OBOC orchestrator instance
_orchestrator: OBOCOrchestrator = None


def get_orchestrator() -> OBOCOrchestrator:
    """Get or create OBOC orchestrator instance."""
    global _orchestrator
    if _orchestrator is None:
        config = OBOCConfig(
            enable_memory=True,
            enable_knowledge_graph=True,
            enable_circuit_breakers=True,
            default_pattern=OrchestrationPattern.MANAGER,
        )
        _orchestrator = OBOCOrchestrator(config)
    return _orchestrator


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available MCP tools."""
    return [
        Tool(
            name="agent_army_execute",
            description="""Execute a complex task with the OBOC agent army.

Uses the Three-Tier Hierarchy:
- Strategic: High-level planning and coordination
- Tactical: Domain specialists (research, implementation, testing, analysis)
- Operational: Tool execution

Orchestration Patterns:
- sequential: Agents run one after another
- concurrent: Agents run in parallel
- group_chat: Collaborative discussion
- dynamic_routing: Route to best agent
- manager: Dynamic task management with backtracking""",
            inputSchema={
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "The task to execute"
                    },
                    "pattern": {
                        "type": "string",
                        "description": "Orchestration pattern: sequential, concurrent, group_chat, dynamic_routing, manager",
                        "enum": ["sequential", "concurrent", "group_chat", "dynamic_routing", "manager"],
                        "default": "manager"
                    },
                    "workspace": {
                        "type": "string",
                        "description": "Optional: Path to project workspace"
                    },
                    "framework": {
                        "type": "string",
                        "description": "Optional: Framework being used"
                    }
                },
                "required": ["task"]
            }
        ),
        Tool(
            name="agent_army_chat",
            description="""Chat with a specific agent in the OBOC hierarchy.

Available agents:
- coordinator (Strategic): High-level planning
- research (Tactical): Information gathering
- implementation (Tactical): Code generation
- testing (Tactical): Validation
- analysis (Tactical): Code review""",
            inputSchema={
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "Your message or question"
                    },
                    "agent": {
                        "type": "string",
                        "description": "Agent to chat with",
                        "enum": ["coordinator", "research", "implementation", "testing", "analysis"],
                        "default": "coordinator"
                    }
                },
                "required": ["message"]
            }
        ),
        Tool(
            name="agent_army_agents",
            description="List all agents in the OBOC hierarchy with their tiers and capabilities.",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        Tool(
            name="agent_army_status",
            description="Get OBOC orchestrator status including memory, knowledge graph, and resilience.",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        Tool(
            name="agent_army_memory",
            description="""Query the OBOC memory system.

Memory Hierarchy:
- short_term: Current session context
- working: Task-specific notes and artifacts
- long_term: Persistent learnings and preferences""",
            inputSchema={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "description": "Memory action",
                        "enum": ["get_context", "get_session", "clear_session"],
                        "default": "get_context"
                    },
                    "query": {
                        "type": "string",
                        "description": "Optional query for context retrieval"
                    }
                },
                "required": ["action"]
            }
        ),
        Tool(
            name="agent_army_knowledge",
            description="""Query the OBOC knowledge graph.

The knowledge graph stores entities and relationships:
- Tasks, decisions, learnings
- Code entities (files, functions, classes)
- Dependency tracking""",
            inputSchema={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "description": "Knowledge action",
                        "enum": ["search", "stats", "get_entity", "add_learning"],
                        "default": "search"
                    },
                    "query": {
                        "type": "string",
                        "description": "Search query or entity ID"
                    },
                    "content": {
                        "type": "string",
                        "description": "Content for add_learning action"
                    }
                },
                "required": ["action"]
            }
        ),
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
        return await handle_status()
    
    elif name == "agent_army_memory":
        return await handle_memory(arguments)
    
    elif name == "agent_army_knowledge":
        return await handle_knowledge(arguments)
    
    else:
        return [TextContent(type="text", text=f"Unknown tool: {name}")]


async def handle_execute(arguments: dict) -> list[TextContent]:
    """Handle execute tool call."""
    task = arguments.get("task", "")
    pattern_str = arguments.get("pattern", "manager")
    
    # Map pattern string to enum
    pattern_map = {
        "sequential": OrchestrationPattern.SEQUENTIAL,
        "concurrent": OrchestrationPattern.CONCURRENT,
        "group_chat": OrchestrationPattern.GROUP_CHAT,
        "dynamic_routing": OrchestrationPattern.DYNAMIC_ROUTING,
        "manager": OrchestrationPattern.MANAGER,
    }
    pattern = pattern_map.get(pattern_str, OrchestrationPattern.MANAGER)
    
    # Build context
    context = {}
    if arguments.get("workspace"):
        context["workspace"] = arguments["workspace"]
    if arguments.get("framework"):
        context["framework"] = arguments["framework"]
    
    orchestrator = get_orchestrator()
    result = await orchestrator.execute(task, pattern=pattern, context=context)
    
    # Format response
    status_icon = "✅" if result["success"] else "❌"
    response = f"""🧠 **OBOC Execution Complete**

**Status:** {status_icon} {'Success' if result['success'] else 'Failed'}
**Trace ID:** {result['trace_id']}
**Pattern:** {result['pattern']}
**Agents Used:** {', '.join(result.get('agents_used', []))}
**Duration:** {result.get('duration_seconds', 0):.2f}s
**Cost:** ${result.get('total_cost', 0):.4f}

## Output

{json.dumps(result.get('output', {}), indent=2) if isinstance(result.get('output'), dict) else result.get('output', 'No output')}
"""
    
    return [TextContent(type="text", text=response)]


async def handle_chat(arguments: dict) -> list[TextContent]:
    """Handle chat tool call."""
    message = arguments.get("message", "")
    agent = arguments.get("agent", "coordinator")
    
    orchestrator = get_orchestrator()
    response = await orchestrator.chat(message, agent=agent)
    
    # Get agent tier
    tier = "Strategic" if agent == "coordinator" else "Tactical"
    
    return [TextContent(type="text", text=f"**{agent.title()} Agent** ({tier}):\n\n{response}")]


async def handle_list_agents() -> list[TextContent]:
    """Handle list agents tool call."""
    orchestrator = get_orchestrator()
    agents = orchestrator.list_agents()
    
    response = """🧠 **OBOC Agent Hierarchy**

## Strategic Tier (High-Level Coordination)
"""
    
    for agent in agents:
        if agent["tier"] == "strategic":
            response += f"\n### {agent['name'].upper()}\n"
            response += f"- **Capabilities:** {', '.join(agent.get('capabilities', []))}\n"
    
    response += "\n## Tactical Tier (Domain Specialists)\n"
    
    for agent in agents:
        if agent["tier"] == "tactical":
            response += f"\n### {agent['name'].upper()}\n"
            response += f"- **Role:** {agent.get('role', 'specialist')}\n"
            response += f"- **Capabilities:** {', '.join(agent.get('capabilities', []))}\n"
    
    response += "\n## Operational Tier (Tool Executors)\n"
    
    for agent in agents:
        if agent["tier"] == "operational":
            response += f"\n### {agent['name'].upper()}\n"
            response += f"- **Category:** {agent.get('tool_category', 'general')}\n"
    
    return [TextContent(type="text", text=response)]


async def handle_status() -> list[TextContent]:
    """Handle status tool call."""
    orchestrator = get_orchestrator()
    status = orchestrator.get_status()
    
    response = f"""🧠 **OBOC Orchestrator Status**

## Agents
- **Strategic:** {status['agents']['strategic']}
- **Tactical:** {status['agents']['tactical']}
- **Operational:** {status['agents']['operational']}

## Memory System
- **Enabled:** {status['memory']['enabled']}
- **Session ID:** {status['memory'].get('session_id', 'N/A')}

## Knowledge Graph
- **Enabled:** {status['knowledge']['enabled']}
- **Entities:** {status['knowledge'].get('stats', {}).get('total_entities', 0)}
- **Relationships:** {status['knowledge'].get('stats', {}).get('total_relationships', 0)}

## Resilience
- **Circuit Breakers:** {len(status['resilience'].get('circuit_breakers', {}))}
- **Degradation Level:** {status['resilience'].get('degradation_level', 'full')}

## Traces
- **Total:** {status['traces']}
"""
    
    return [TextContent(type="text", text=response)]


async def handle_memory(arguments: dict) -> list[TextContent]:
    """Handle memory tool call."""
    action = arguments.get("action", "get_context")
    query = arguments.get("query", "")
    
    orchestrator = get_orchestrator()
    
    if action == "get_context":
        context = orchestrator.get_memory_context(query)
        response = f"""🧠 **Memory Context**

- **Enabled:** {context.get('enabled', False)}
- **Session Messages:** {context.get('session_messages', 0)}
- **Working Tasks:** {context.get('working_tasks', 0)}
"""
    
    elif action == "get_session":
        if orchestrator.memory:
            messages = orchestrator.memory.short_term.get_messages(limit=10)
            response = "🧠 **Recent Session Messages**\n\n"
            for msg in messages:
                agent_prefix = f"[{msg.agent_name}] " if msg.agent_name else ""
                response += f"**{agent_prefix}{msg.role.upper()}:** {msg.content[:200]}...\n\n"
        else:
            response = "Memory system is not enabled."
    
    elif action == "clear_session":
        if orchestrator.memory:
            orchestrator.memory.short_term.clear()
            response = "✅ Session memory cleared."
        else:
            response = "Memory system is not enabled."
    
    else:
        response = f"Unknown action: {action}"
    
    return [TextContent(type="text", text=response)]


async def handle_knowledge(arguments: dict) -> list[TextContent]:
    """Handle knowledge tool call."""
    action = arguments.get("action", "search")
    query = arguments.get("query", "")
    content = arguments.get("content", "")
    
    orchestrator = get_orchestrator()
    
    if not orchestrator.knowledge:
        return [TextContent(type="text", text="Knowledge graph is not enabled.")]
    
    if action == "search":
        entities = orchestrator.knowledge.search(query, limit=10)
        response = f"🧠 **Knowledge Graph Search: '{query}'**\n\n"
        for entity in entities:
            response += f"### {entity.name}\n"
            response += f"- **Type:** {entity.entity_type.value}\n"
            response += f"- **ID:** {entity.id}\n"
            if entity.observations:
                response += f"- **Observations:** {len(entity.observations)}\n"
            response += "\n"
        
        if not entities:
            response += "No entities found."
    
    elif action == "stats":
        stats = orchestrator.knowledge.get_stats()
        response = f"""🧠 **Knowledge Graph Statistics**

- **Total Entities:** {stats['total_entities']}
- **Total Relationships:** {stats['total_relationships']}

### Entities by Type
"""
        for entity_type, count in stats.get('entities_by_type', {}).items():
            response += f"- {entity_type}: {count}\n"
        
        response += "\n### Relationships by Type\n"
        for rel_type, count in stats.get('relationships_by_type', {}).items():
            response += f"- {rel_type}: {count}\n"
    
    elif action == "get_entity":
        entity = orchestrator.knowledge.get_entity(query)
        if entity:
            response = f"""🧠 **Entity: {entity.name}**

- **ID:** {entity.id}
- **Type:** {entity.entity_type.value}
- **Properties:** {json.dumps(entity.properties, indent=2)}
- **Observations:** {len(entity.observations)}
- **Importance:** {entity.importance}
"""
        else:
            response = f"Entity not found: {query}"
    
    elif action == "add_learning":
        if content:
            entity = orchestrator.knowledge.add_learning(content, source="mcp_user")
            response = f"✅ Learning added with ID: {entity.id}"
        else:
            response = "Content is required for add_learning action."
    
    else:
        response = f"Unknown action: {action}"
    
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
