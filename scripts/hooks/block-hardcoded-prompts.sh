#!/bin/bash
INPUT=$(cat)
FILE_PATH=$(echo "$INPUT" | jq -r '.tool_input.file_path // empty')
if [[ "$FILE_PATH" == *.py ]] && [[ "$FILE_PATH" != *prompts* ]] && [[ "$FILE_PATH" != *test* ]]; then
  CONTENT=$(echo "$INPUT" | jq -r '.tool_input.content // .tool_input.new_str // empty')
  if echo "$CONTENT" | grep -qP '(You are a|Answer the following|Based on the context|Answer the user|system prompt)'; then
    echo "BLOCKED: Use YAML prompt files in src/citesage/prompts/ instead." >&2
    exit 2
  fi
fi
