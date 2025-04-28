#!/bin/bash

# Dominic's FastAPI dev server 자동 실행 스크립트

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")

source "$SCRIPT_DIR/activate_venv.sh"
cd "$PROJECT_ROOT/apps/hub_server"
PYTHONPATH="$PROJECT_ROOT" uvicorn main:app --reload --host 0.0.0.0 --port 8000 