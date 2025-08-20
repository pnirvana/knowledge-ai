#!/usr/bin/env bash
set -euo pipefail
cd /Users/pascal/Dev/ai-knowledgebase/graphrag
source ./graphrag_env/bin/activate
exec python mcp_server.py