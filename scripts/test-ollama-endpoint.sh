#!/usr/bin/env sh

# test-ollama-endpoint.sh

set -e

export OLLAMA_HOST OLLAMA_API_KEY

OLLAMA_HOST="${OLLAMA_HOST:-http://localhost:11434}"
OLLAMA_API_KEY="${OLLAMA_API_KEY}"

if [ "$OLLAMA_API_KEY" = '' ]; then
    echo "No OLLAMA_API_KEY was set in the environment."
    exit 1
fi

curl "${OLLAMA_HOST}/api/generate" \
     -H "Authorization: Bearer $OLLAMA_API_KEY" \
     -d '{
        "model": "gpt-oss:120b-cloud",
        "prompt": "Why is the sky blue?"
     }'
