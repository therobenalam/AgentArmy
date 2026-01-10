# 🪖 AgentArmy

A standalone AI agent orchestration system with an army of specialized agents for general-purpose task execution.

## ✨ Features

- **5 Specialized Agents**: Planner, Research, Implementation, Testing, Analysis
- **Smart Orchestration**: Automatic task decomposition and agent routing
- **CLI Access**: Run from anywhere with global command
- **VSCode Integration**: MCP server for Copilot integration
- **Cost Tracking**: Monitor API usage and costs
- **State Persistence**: Resume interrupted executions
- **Parallel Execution**: Run independent tasks concurrently

## 🚀 Quick Start

### Installation

```bash
cd /Users/robenalam/Documents/4.0/AgentArmy
pip install -e .
```

### CLI Usage

```bash
# Execute a task
agent-army execute "Build a REST API with JWT authentication"

# With project context
agent-army execute "Refactor the auth module" \
    --workspace /path/to/project \
    --framework FastAPI \
    --language Python

# Chat with an agent
agent-army chat "What are best practices for error handling?"

# List agents
agent-army agents
```

### Or use the launcher

```bash
./run.sh execute "Your task here"
```

### Global Alias (Run from Anywhere)

Add to `~/.zshrc`:

```bash
alias agent-army="/Users/robenalam/Documents/4.0/AgentArmy/run.sh"
```

Then:

```bash
source ~/.zshrc
agent-army execute "Build a feature" --workspace $(pwd)
```

## 🤖 Available Agents

| Agent | Description |
|-------|-------------|
| **Planner** | Task decomposition and workflow planning |
| **Research** | Information gathering, best practices |
| **Implementation** | Code generation, feature implementation |
| **Testing** | Test generation, validation |
| **Analysis** | Code review, architecture analysis |

## 🔧 VSCode Integration

### Add to MCP Settings

Edit `~/Library/Application Support/Code/User/settings.json`:

```json
{
  "mcp": {
    "servers": {
      "agent-army": {
        "command": "python3",
        "args": ["/Users/robenalam/Documents/4.0/AgentArmy/mcp_server.py"]
      }
    }
  }
}
```

### MCP Tools Available

- `agent_army_execute` - Execute complex tasks
- `agent_army_chat` - Chat with specific agents
- `agent_army_agents` - List available agents
- `agent_army_status` - Check execution status

## 📖 Python API

```python
import asyncio
from agent_army import Orchestrator

async def main():
    orchestrator = Orchestrator()
    
    # Execute a task
    result = await orchestrator.execute(
        "Build a user authentication system",
        project_context={
            "workspace": "/path/to/project",
            "framework": "FastAPI",
            "language": "Python"
        }
    )
    
    print(f"Success: {result['success']}")
    print(f"Output: {result['output']}")
    print(f"Cost: ${result['total_cost']:.4f}")

asyncio.run(main())
```

## ⚙️ Configuration

Create `agent_army.yaml` or `config.yaml`:

```yaml
aws_region: us-east-1
default_model: anthropic.claude-sonnet-4-20250514-v1:0
enable_parallel: true
max_iterations: 10

agents:
  implementation:
    max_tokens: 8192
    temperature: 0.5
```

Environment variables:

```bash
export AGENT_ARMY_AWS_REGION=us-east-1
export AGENT_ARMY_MODEL=anthropic.claude-sonnet-4-20250514-v1:0
```

## 📊 Execution Flow

```
User Request
    ↓
[Planner] → Creates execution plan
    ↓
[Research] → Gathers information (if needed)
    ↓
[Implementation] → Writes code
    ↓
[Testing] → Validates implementation
    ↓
[Analysis] → Reviews quality (optional)
    ↓
Final Result
```

## 🧪 Testing

```bash
# Run tests
pytest tests/

# With coverage
pytest tests/ --cov=agent_army
```

## 📁 Project Structure

```
AgentArmy/
├── agent_army/           # Main package
│   ├── __init__.py
│   ├── orchestrator.py   # Core orchestration engine
│   ├── state.py          # State management
│   ├── config.py         # Configuration
│   └── agents/           # Agent implementations
│       ├── base.py       # Base agent class
│       ├── planner.py
│       ├── research.py
│       ├── implementation.py
│       ├── testing.py
│       └── analysis.py
├── cli.py                # CLI interface
├── mcp_server.py         # MCP server for VSCode
├── run.sh                # Quick launcher
├── config.yaml           # Default configuration
├── requirements.txt
├── pyproject.toml
└── tests/
```

## 🔒 Requirements

- Python 3.10+
- AWS credentials configured (`~/.aws/credentials`)
- Access to AWS Bedrock with Claude models

## 📄 License

MIT License
