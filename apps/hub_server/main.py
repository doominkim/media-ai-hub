# pip install asyncpg databases 필요
from fastapi import FastAPI, UploadFile, File, Query
from apps.hub_server.routers import health
from libs.whisper_client import whisper as whisper_client
from libs.vision_client import vision as vision_client
from apps.hub_server import queue
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

@app.post("/whisper/transcribe-minio")
async def transcribe_audio_minio(
    key: str = Query(..., description="MinIO 버킷 내 오디오 파일 경로")
):
    text = whisper_client.transcribe_from_minio(key)
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

@app.post("/vision/describe-minio")
async def describe_image_minio(
    key: str = Query(..., description="MinIO 버킷 내 이미지 파일 경로")
):
    job_id = queue.enqueue_vision(key)
    return {"job_id": job_id}

@app.get("/jobs/{queue_name}/{job_id}")
async def get_job_status(queue_name: str, job_id: str):
    return queue.get_job_status(queue_name, job_id)

@app.get("/")
async def root():
    return {"message": "Welcome to Media AI Hub"}
