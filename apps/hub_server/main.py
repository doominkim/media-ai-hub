# pip install asyncpg databases 필요
from fastapi import FastAPI, UploadFile, File, Query, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any
from libs.whisper_client import whisper as whisper_client
from libs.vision_client import vision as vision_client
from apps.hub_server.nest_client import nest_client
import shutil
import os
import asyncio

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

async def process_audio_job():
    """오디오 처리 작업 실행"""
    while True:
        try:
            print("\n=== 새로운 작업 처리 시작 ===")
            # Nest 서버에서 다음 작업 가져오기
            job = nest_client.get_next_job("audio-processing")
            print(f"작업 정보: {job}")

            # 작업 시작 시간 기록
            job_start_time = asyncio.get_event_loop().time()

            # 작업 데이터 추출
            job_data = job["data"]
            file_path = job_data["filePath"]
            channel_id = job_data["channelId"]
            live_id = job_data["liveId"]
            start_time = job_data["startTime"]
            end_time = job_data["endTime"]
            print(f"작업 데이터: filePath={file_path}, channelId={channel_id}, liveId={live_id}")

            try:
                # Whisper 변환 시작 시간 기록
                whisper_start_time = asyncio.get_event_loop().time()
                print(f"Whisper 변환 시작: {file_path}")
                
                # Whisper로 오디오 변환
                text = whisper_client.transcribe_from_minio(file_path)
                
                # Whisper 변환 종료 시간 기록
                whisper_end_time = asyncio.get_event_loop().time()
                whisper_duration = whisper_end_time - whisper_start_time
                print(f"Whisper 변환 완료: {round(whisper_duration, 2)}초 소요")
                
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
                print(f"작업 완료 처리: job_id={job['id']}")
                nest_client.complete_job("audio-processing", job["id"], result)

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
                print("whisper-processing 큐에 결과 추가")
                nest_client.add_job("whisper-processing", whisper_job_data)
                print("=== 작업 처리 완료 ===\n")

            except Exception as e:
                # 작업 실패 처리
                print(f"작업 처리 중 에러 발생: {str(e)}")
                nest_client.fail_job("audio-processing", job["id"], str(e))

        except HTTPException as e:
            if e.status_code == 404:
                # 작업이 없는 경우 잠시 대기
                print("처리할 작업이 없습니다. 5초 후 재시도...")
                await asyncio.sleep(5)
                continue
            print(f"HTTP 에러 발생: {str(e)}")
            await asyncio.sleep(5)
        except Exception as e:
            print(f"예상치 못한 에러 발생: {str(e)}")
            await asyncio.sleep(5)

@app.on_event("startup")
async def startup_event():
    """서버 시작 시 백그라운드 작업 시작"""
    asyncio.create_task(process_audio_job())

@app.get("/")
async def root():
    return {"message": "Welcome to Media AI Hub"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
