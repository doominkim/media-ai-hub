# pip install asyncpg databases 필요
from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any
from libs.whisper_client import whisper as whisper_client
from libs.vision_client import vision as vision_client
from apps.hub_server.nest_client import nest_client
import shutil
import os

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

# 라우터 등록

@app.post("/process-audio")
async def process_audio():
    """오디오 처리 작업 실행"""
    try:
        # Nest 서버에서 다음 작업 가져오기
        job = nest_client.get_next_job("audio-processing")
        print(job)

        # 작업 데이터 추출
        job_data = job["data"]
        file_path = job_data["filePath"]
        channel_id = job_data["channelId"]
        live_id = job_data["liveId"]
        start_time = job_data["startTime"]
        end_time = job_data["endTime"]

        try:
            # Whisper로 오디오 변환
            text = whisper_client.transcribe_from_minio(file_path)
            
            # 작업 완료 처리
            result = {
                "status": "success",
                "text": text,
                "metadata": {
                    "channelId": channel_id,
                    "liveId": live_id,
                    "startTime": start_time,
                    "endTime": end_time
                }
            }
            nest_client.complete_job("audio-processing", job["id"], result)

            # whisper-processing 큐에 결과 추가
            whisper_job_data = {
                "filePath": file_path,
                "channelId": channel_id,
                "liveId": live_id,
                "startTime": start_time,
                "endTime": end_time,
                "text": text
            }
            nest_client.add_job("whisper-processing", whisper_job_data)

            return result

        except Exception as e:
            # 작업 실패 처리
            nest_client.fail_job("audio-processing", job["id"], str(e))
            raise HTTPException(status_code=500, detail=str(e))

    except HTTPException as e:
        if e.status_code == 404:
            raise HTTPException(status_code=404, detail="No jobs available")
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.get("/")
async def root():
    return {"message": "Welcome to Media AI Hub"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
