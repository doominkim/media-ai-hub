# pip install asyncpg databases 필요
from fastapi import FastAPI, UploadFile, File, Query
from apps.hub_server.routers import health
from libs.whisper_client import whisper as whisper_client
from libs.vision_client import vision as vision_client
from apps.hub_server import db
import shutil
import os

app = FastAPI(
    title="Media AI Hub",
    description="멀티미디어 데이터 AI 분석 허브 서버",
    version="0.1.0",
)

@app.on_event("startup")
async def startup():
    try:
        await db.connect_db()
        print("[DB] 연결 성공!")
    except Exception as e:
        print(f"[DB] 연결 실패: {e}")

@app.on_event("shutdown")
async def shutdown():
    await db.disconnect_db()

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

@app.get("/pg/health")
async def pg_health():
    try:
        row = await db.database.fetch_one("SELECT 1 AS ok")
        return {"pg_status": row["ok"]}
    except Exception as e:
        return {"pg_status": "error", "detail": str(e)}

@app.get("/")
async def root():
    return {"message": "Welcome to Media AI Hub"}
