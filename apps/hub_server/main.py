# pip install asyncpg databases faster-whisper 필요
from fastapi import FastAPI, UploadFile, File, Query, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, List
from libs.fast_whisper_client import fast_whisper as whisper_client

# Vision client는 선택적 import (호환성 문제 방지)
try:
    from libs.vision_client import vision as vision_client
    VISION_AVAILABLE = True
    print("✅ Vision client 로드 성공")
except Exception as e:
    print(f"⚠️  Vision client 로드 실패: {e}")
    print("🎵 Audio processing만 사용 가능합니다.")
    vision_client = None
    VISION_AVAILABLE = False

from apps.hub_server.nest_client import nest_client
import shutil      
import os
import asyncio
import logging
import sys
from logging.handlers import RotatingFileHandler
from datetime import datetime, timezone, timedelta

# KST 시간대 설정
KST = timezone(timedelta(hours=9))

class KSTFormatter(logging.Formatter):
    """KST 시간대를 사용하는 로그 포맷터"""
    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created).astimezone(KST)
        if datefmt:
            return dt.strftime(datefmt)
        return dt.strftime("%Y-%m-%d %H:%M:%S")

def get_log_directory():
    """오늘 날짜의 로그 디렉토리 경로를 반환"""
    today = datetime.now(KST).strftime("%Y-%m-%d")
    log_dir = os.path.join("logs", today)
    os.makedirs(log_dir, exist_ok=True)
    return log_dir

# 로깅 설정
def setup_logger(worker_id):
    logger = logging.getLogger(f"worker_{worker_id}")
    logger.setLevel(logging.INFO)
    
    # 콘솔 출력 핸들러
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    formatter = KSTFormatter(f'[Worker-{worker_id}] %(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    
    # 파일 출력 핸들러
    log_dir = get_log_directory()
    file_handler = RotatingFileHandler(
        os.path.join(log_dir, f'worker_{worker_id}.log'),
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

# 로그 디렉토리 생성
os.makedirs('logs', exist_ok=True)

# worker ID를 환경 변수에서 가져오기
worker_id = int(os.getenv("WORKER_ID", "0"))
logger = setup_logger(worker_id)

logger.info(f"=== Worker {worker_id} 시작 ===")

app = FastAPI(
    title="Media AI Hub",
    description="멀티미디어 데이터 AI 분석 허브 서버",
    version="0.1.0",
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 작업 상태를 저장할 딕셔너리
active_jobs = {}

class JobStatus(BaseModel):
    job_id: str
    status: str
    start_time: datetime
    progress: float = 0.0
    error: str = None

async def process_audio_job():
    """오디오 처리 작업 실행"""
    while True:
        try:
            # Nest 서버에서 다음 작업 가져오기
            job = nest_client.get_next_job("audio-processing")
            logger.info(f"작업 정보: {job}")

            # 작업 시작 시간 기록
            job_start_time = asyncio.get_event_loop().time()
            job_id = job["id"]

            # 작업 상태 초기화
            active_jobs[job_id] = JobStatus(
                job_id=job_id,
                status="processing",
                start_time=datetime.now()
            )

            # 작업 데이터 추출
            job_data = job["data"]
            file_path = job_data["filePath"]
            channel_id = job_data["channelId"]
            live_id = job_data["liveId"]
            start_time = job_data["startTime"]
            end_time = job_data["endTime"]
            logger.info(f"작업 데이터: filePath={file_path}, channelId={channel_id}, liveId={live_id}")

            try:
                # Whisper 변환 시작 시간 기록
                whisper_start_time = asyncio.get_event_loop().time()
                logger.info(f"Fast-Whisper 변환 시작: {file_path}")
                
                # Fast-Whisper로 오디오 변환 (비동기로 실행)
                loop = asyncio.get_event_loop()
                text = await loop.run_in_executor(
                    None,  # 기본 스레드 풀 사용
                    lambda: whisper_client.transcribe_from_minio(file_path)
                )
                
                # Whisper 변환 종료 시간 기록
                whisper_end_time = asyncio.get_event_loop().time()
                whisper_duration = whisper_end_time - whisper_start_time
                logger.info(f"Fast-Whisper 변환 완료: {round(whisper_duration, 2)}초 소요")
                
                # 작업 완료 처리
                result = {
                    "status": "success",
                    "text": text,
                    "metadata": {
                        "channelId": channel_id,
                        "liveId": live_id,
                        "startTime": start_time,
                        "endTime": end_time,
                        "processingTime": f"{round(asyncio.get_event_loop().time() - job_start_time, 2)}초",
                        "transcriptionTime": f"{round(whisper_duration, 2)}초"
                    }
                }
                logger.info(f"작업 완료 처리: job_id={job_id}")
                nest_client.complete_job("audio-processing", job_id, result)

                # whisper-processing 큐에 결과 추가
                whisper_job_data = {
                    "filePath": file_path,
                    "channelId": channel_id,
                    "liveId": live_id,
                    "startTime": start_time,
                    "endTime": end_time,
                    "text": text,
                    "processingTime": f"{round(asyncio.get_event_loop().time() - job_start_time, 2)}초",
                    "transcriptionTime": f"{round(whisper_duration, 2)}초"
                }
                logger.info("whisper-processing 큐에 결과 추가")
                nest_client.add_job("whisper-processing", whisper_job_data)
                logger.info("=== 작업 처리 완료 ===\n")

                # 작업 상태 업데이트
                active_jobs[job_id].status = "completed"
                active_jobs[job_id].progress = 100.0

            except Exception as e:
                # 작업 실패 처리
                error_msg = str(e)
                logger.error(f"작업 처리 중 에러 발생: {error_msg}")
                nest_client.fail_job("audio-processing", job_id, error_msg)
                
                # 작업 상태 업데이트
                active_jobs[job_id].status = "failed"
                active_jobs[job_id].error = error_msg

        except HTTPException as e:
            if e.status_code == 404:
                # 작업이 없는 경우 잠시 대기
                logger.info("처리할 작업이 없습니다. 5초 후 재시도...")
                await asyncio.sleep(5)
                continue
            logger.error(f"HTTP 에러 발생: {str(e)}")
            await asyncio.sleep(5)
        except Exception as e:
            error_msg = str(e)
            if "No jobs available" in error_msg:
                # 작업이 없는 경우는 예상된 상황이므로 INFO 레벨로 로깅
                logger.info("처리할 작업이 없습니다. 5초 후 재시도...")
                await asyncio.sleep(5)
                continue
            logger.error(f"예상치 못한 에러 발생: {error_msg}")
            await asyncio.sleep(5)

@app.on_event("startup")
async def startup_event():
    """서버 시작 시 백그라운드 작업 시작"""
    asyncio.create_task(process_audio_job())

@app.get("/")
async def root():
    return {"message": "Welcome to Media AI Hub", "worker_id": worker_id}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "version": app.version,
        "service": "Media AI Hub",
        "worker_id": worker_id
    }

@app.get("/jobs")
async def get_jobs():
    """현재 처리 중인 작업 목록 조회"""
    return {
        "active_jobs": list(active_jobs.values()),
        "worker_id": worker_id
    }

@app.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    """특정 작업의 상태 조회"""
    if job_id not in active_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return active_jobs[job_id]

if __name__ == "__main__":
    import uvicorn
    import sys
    
    # worker ID를 포트 번호로 사용
    worker_id = int(os.getenv("WORKER_ID", "0"))
    port = 8000 + worker_id
    
    logger.info(f"Worker {worker_id} 시작 - 포트: {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
