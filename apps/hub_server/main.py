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

class JobLog:
    """작업별 로그를 수집하고 관리하는 클래스"""
    def __init__(self, job_id: str, job_data: dict):
        self.job_id = job_id
        self.job_data = job_data
        self.start_time = datetime.now(KST)
        self.logs = []
        self.timers = {}
        self.result = None
        self.error = None
        
    def add_log(self, event: str, data: dict = None):
        """로그 이벤트 추가"""
        log_entry = {
            "timestamp": datetime.now(KST).isoformat(),
            "event": event,
            "data": data or {}
        }
        self.logs.append(log_entry)
    
    def start_timer(self, name: str):
        """타이머 시작"""
        self.timers[name] = {
            "start": datetime.now(KST),
            "duration": None
        }
        
    def end_timer(self, name: str):
        """타이머 종료"""
        if name in self.timers:
            start_time = self.timers[name]["start"]
            duration = (datetime.now(KST) - start_time).total_seconds()
            self.timers[name]["duration"] = round(duration, 2)
            return duration
        return 0
    
    def set_result(self, result: dict):
        """작업 결과 설정"""
        self.result = result
        
    def set_error(self, error: str):
        """에러 설정"""
        self.error = error
        
    def get_summary(self) -> dict:
        """작업 요약 정보 반환"""
        end_time = datetime.now(KST)
        total_duration = (end_time - self.start_time).total_seconds()
        
        # 타이머 정보를 깔끔하게 정리
        clean_timers = {}
        for name, timer in self.timers.items():
            clean_timers[name] = timer.get("duration", 0)
        
        summary = {
            "job_id": self.job_id,
            "status": "success" if self.result else "failed",
            "duration": round(total_duration, 2),
            "file": self.job_data.get("filePath", "").split('/')[-1],  # 파일명만
            "channel": self.job_data.get("channelId", ""),
            "timers": clean_timers,
            "error": self.error[:100] + "..." if self.error and len(self.error) > 100 else self.error
        }
        
        return summary
    
    def print_final_log(self):
        """최종 로그 출력"""
        summary = self.get_summary()
        
        # 성공/실패에 따른 이모지와 색상
        status_emoji = "✅" if summary["status"] == "success" else "❌"
        
        # 한 줄로 요약된 로그 출력
        transcription_time = summary['timers'].get('transcription', 0)
        logger.info(f"{status_emoji} 작업 완료 | "
                   f"ID: {summary['job_id']} | "
                   f"파일: {summary['file']} | "
                   f"채널: {summary['channel']} | "
                   f"총: {summary['duration']}초 | "
                   f"변환: {transcription_time}초 | "
                   f"상태: {summary['status']}")
        
        # 성공한 경우 변환된 텍스트 출력
        if summary["status"] == "success" and self.result and "text" in self.result:
            text = self.result["text"]
            logger.info(f"변환된 텍스트: {text}")
        
        # 상세 JSON 로그 (실패하거나 10초 이상 걸린 경우만, 단 VAD analysis complete는 제외)
        is_vad_error = self.error and "VAD analysis complete" in self.error
        if (summary["status"] == "failed" or summary["duration"] > 10) and not is_vad_error:
            import json
            clean_json = json.dumps(summary, ensure_ascii=False, indent=2)
            logger.info(f"📋 상세 정보:\n{clean_json}")

# 작업별 로그 저장소
job_logs = {}

async def process_audio_job():
    """오디오 처리 작업 실행"""
    while True:
        try:
            # Nest 서버에서 다음 작업 가져오기
            job = nest_client.get_next_job("audio-processing")

            # 'No jobs available' 메시지 확인
            if isinstance(job, dict) and "message" in job and job.get("message") == "No jobs available":
                # 조용히 5초 대기 (로그 스팸 방지)
                await asyncio.sleep(5)
                continue

            # 유효한 작업인지 확인
            if not isinstance(job, dict) or "id" not in job:
                logger.warning(f"유효하지 않은 작업 형식: {job}")
                await asyncio.sleep(5)
                continue

            # 작업 ID 및 데이터 추출
            job_id = job["id"]
            job_data = job["data"]
            
            # JobLog 인스턴스 생성
            job_log = JobLog(job_id, job_data)
            job_logs[job_id] = job_log
            
            # 작업 상태 초기화
            active_jobs[job_id] = JobStatus(
                job_id=job_id,
                status="processing",
                start_time=datetime.now()
            )

            file_path = job_data["filePath"]
            channel_id = job_data["channelId"]
            live_id = job_data["liveId"]
            start_time = job_data["startTime"]
            end_time = job_data["endTime"]
            
            job_log.add_log("job_started", {
                "file_path": file_path,
                "channel_id": channel_id,
                "live_id": live_id
            })

            try:
                # Whisper 변환 시작
                job_log.start_timer("transcription")
                job_log.add_log("transcription_started")
                
                # Fast-Whisper로 오디오 변환 (비동기로 실행)
                loop = asyncio.get_event_loop()
                text = await loop.run_in_executor(
                    None,  # 기본 스레드 풀 사용
                    lambda: whisper_client.transcribe_from_minio(file_path)
                )
                
                # 변환 완료 시간 기록
                transcription_duration = job_log.end_timer("transcription")
                job_log.add_log("transcription_completed", {
                    "duration": transcription_duration,
                    "text_length": len(text)
                })
                
                # 작업 완료 처리
                job_log.start_timer("post_processing")
                
                result = {
                    "status": "success",
                    "text": text,
                    "metadata": {
                        "channelId": channel_id,
                        "liveId": live_id,
                        "startTime": start_time,
                        "endTime": end_time,
                        "transcriptionTime": f"{transcription_duration}초"
                    }
                }
                
                # Nest 서버에 완료 알림
                nest_client.complete_job("audio-processing", job_id, result)
                job_log.add_log("job_completed_notify")

                # whisper-processing 큐에 결과 추가
                whisper_job_data = {
                    "filePath": file_path,
                    "channelId": channel_id,
                    "liveId": live_id,
                    "startTime": start_time,
                    "endTime": end_time,
                    "text": text,
                    "transcriptionTime": f"{transcription_duration}초"
                }
                nest_client.add_job("whisper-processing", whisper_job_data)
                job_log.add_log("result_queued")
                
                job_log.end_timer("post_processing")
                job_log.set_result(result)

                # 작업 상태 업데이트
                active_jobs[job_id].status = "completed"
                active_jobs[job_id].progress = 100.0

            except Exception as e:
                # 작업 실패 처리
                error_msg = str(e)
                job_log.set_error(error_msg)
                job_log.add_log("job_failed", {"error": error_msg})
                
                nest_client.fail_job("audio-processing", job_id, error_msg)
                
                # 작업 상태 업데이트
                active_jobs[job_id].status = "failed"
                active_jobs[job_id].error = error_msg

            finally:
                # 최종 로그 출력 및 정리
                job_log.print_final_log()
                
                # 메모리 정리 (완료된 작업 로그 제거)
                if job_id in job_logs:
                    del job_logs[job_id]

        except HTTPException as e:
            if e.status_code == 404:
                # 작업이 없는 경우 조용히 대기
                await asyncio.sleep(5)
                continue
            logger.error(f"HTTP 에러 발생: {str(e)}")
            await asyncio.sleep(5)
        except Exception as e:
            logger.error(f"예상치 못한 에러 발생: {str(e)}")
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
