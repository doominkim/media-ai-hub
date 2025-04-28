# Media AI Hub

FastAPI 기반 Whisper 서버, Vision, LLM 연동 구조 설계

## 구조

```
├── apps/
│   └── hub-server/          # FastAPI 기반 메인 서버
├── libs/
│   ├── whisper-client/       # 오디오 → 텍스트 변환 모듈
│   ├── vision-client/        # 이미지 감정 분석 모듈
│   └── llm-client/           # 로컬 LLM (예: 텍스트 요약/감정 분석) 모듈
├── scripts/
│   ├── audio-worker/         # 백그라운드 오디오 변환 작업자
│   └── image-worker/         # 백그라운드 이미지 분석 작업자
├── docker-compose.yml        # 전체 서비스 띄우기
├── README.md
└── .gitignore
```

---

Dominic's AI Media Hub

## 소개

`media-ai-hub`는 오디오, 텍스트, 이미지 등 다양한 미디어 데이터를 AI 모델을 통해 분석하고, API 형태로 결과를 제공하는 서버입니다.

- Whisper 기반 오디오 텍스트 변환
- Vision 모델 기반 이미지 감정 분석
- 로컬 LLM 기반 텍스트 요약 및 감정 분석
- FastAPI 기반 경량 서버 구조
- NestJS, Frontend 서버와 손쉬운 연동 가능

## 기술 스택

- Python 3.10+
- FastAPI
- Whisper (openai/whisper, whisper.cpp)
- Huggingface Transformers (LLM, Vision 모델)
- Docker (옵션)

## 실행 방법

```bash
# FastAPI 서버 실행
uvicorn apps.hub_server.main:app --reload

# Docker로 전체 실행
docker-compose up --build
```

## venv 환경 고정 및 실행법

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cd apps/hub-server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

- requirements.txt로 패키지 버전 고정
- `.venv`는 .gitignore에 이미 포함
