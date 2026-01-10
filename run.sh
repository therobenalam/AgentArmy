#!/usr/bin/env bash
#
# AgentArmy - Quick Launcher
#
# Usage:
#   ./run.sh execute "your task"
#   ./run.sh chat "your message"
#   ./run.sh agents
#

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR" || exit 1

# Activate virtual environment if it exists
if [ -d "../.venv" ]; then
    source "../.venv/bin/activate"
elif [ -d ".venv" ]; then
    source ".venv/bin/activate"
fi

# Run CLI with all arguments
python3 cli.py "$@"
