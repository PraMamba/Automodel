#!/usr/bin/env bash
# Post-tool-use hook: Remind to update expert agents when related code changes.
# Triggered after Write/Edit operations via .claude/settings.json.

# Get the file path from the tool use event
# The hook receives JSON on stdin with the tool use details
if ! command -v jq &>/dev/null; then
    exit 0  # Skip if jq not available
fi

INPUT=$(cat)
FILE_PATH=$(echo "$INPUT" | jq -r '.tool_input.file_path // .tool_input.filePath // empty' 2>/dev/null || true)

if [[ -z "$FILE_PATH" ]]; then
    exit 0
fi

AGENT_TO_UPDATE=""
AGENT_FILE=""

# Map code paths to expert agents
case "$FILE_PATH" in
    *nemo_automodel/components/distributed/*)
        AGENT_TO_UPDATE="distributed-expert"
        AGENT_FILE=".claude/agents/distributed-expert.md"
        ;;
    *nemo_automodel/components/models/* | *nemo_automodel/_transformers/*)
        AGENT_TO_UPDATE="model-expert"
        AGENT_FILE=".claude/agents/model-expert.md"
        ;;
    *nemo_automodel/components/checkpoint/*)
        AGENT_TO_UPDATE="checkpoint-expert"
        AGENT_FILE=".claude/agents/checkpoint-expert.md"
        ;;
    *nemo_automodel/components/moe/*)
        AGENT_TO_UPDATE="moe-expert"
        AGENT_FILE=".claude/agents/moe-expert.md"
        ;;
    *nemo_automodel/components/_peft/*)
        AGENT_TO_UPDATE="peft-expert"
        AGENT_FILE=".claude/agents/peft-expert.md"
        ;;
    *nemo_automodel/recipes/* | *nemo_automodel/components/training/*)
        AGENT_TO_UPDATE="recipe-expert"
        AGENT_FILE=".claude/agents/recipe-expert.md"
        ;;
    *nemo_automodel/_cli/* | *nemo_automodel/components/launcher/*)
        AGENT_TO_UPDATE="launcher-expert"
        AGENT_FILE=".claude/agents/launcher-expert.md"
        ;;
esac

if [[ -n "$AGENT_TO_UPDATE" ]]; then
    echo ""
    echo "┌─────────────────────────────────────────────────────────────┐"
    echo "│  💡 Consider updating: $AGENT_FILE"
    echo "│  Agent: $AGENT_TO_UPDATE"
    echo "│  Reason: You modified code this agent covers."
    echo "└─────────────────────────────────────────────────────────────┘"
    echo ""
fi

exit 0
