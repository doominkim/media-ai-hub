#!/bin/bash

# Dominic's venv 자동화 스크립트

if [ ! -d ".venv" ]; then
  echo "[+] .venv 생성 중..."
  python3 -m venv .venv
fi

source .venv/bin/activate

if [ -f requirements.txt ]; then
  echo "[+] requirements.txt 설치 중..."
  pip install -r requirements.txt
else
  echo "[!] requirements.txt가 없습니다."
fi

echo "[+] venv 활성화 완료. (deactivate로 해제)" 