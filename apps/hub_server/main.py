from fastapi import FastAPI, UploadFile, File
from apps.hub_server.routers import health
from libs.whisper_client import whisper as whisper_client
from libs.vision_client import vision as vision_client
import shutil
import os

app = FastAPI(
    title="Media AI Hub",
    description="멀티미디어 데이터 AI 분석 허브 서버",
    version="0.1.0",
)

# 라우터 등록
app.include_router(health.router)

@app.post("/whisper/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    temp_path = f"/tmp/{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    try:
        text = whisper_client.transcribe(temp_path)
    finally:
        os.remove(temp_path)
    return {"text": text}

@app.post("/vision/describe")
async def describe_image(file: UploadFile = File(...)):
    temp_path = f"/tmp/{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    try:
        label = vision_client.describe_image(temp_path)
    finally:
        os.remove(temp_path)
    return {"label": label}

@app.get("/")
async def root():
    return {"message": "Welcome to Media AI Hub"}
