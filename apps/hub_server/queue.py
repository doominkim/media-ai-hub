import os
from redis import Redis
from rq import Queue
from typing import Dict, Any

# Redis 연결 설정
redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
redis_conn = Redis.from_url(redis_url)

# 작업 큐 생성
transcribe_queue = Queue("transcribe", connection=redis_conn)
vision_queue = Queue("vision", connection=redis_conn)
audio_processing_queue = Queue("audio-processing", connection=redis_conn)

def process_audio_job(job_data: Dict[str, Any]) -> Dict[str, Any]:
    """오디오 처리 작업 실행"""
    from libs.whisper_client import whisper as whisper_client
    
    try:
        # 작업 데이터 추출
        file_path = job_data["filePath"]
        channel_id = job_data["channelId"]
        live_id = job_data["liveId"]
        start_time = job_data["startTime"]
        end_time = job_data["endTime"]
        
        # Whisper로 오디오 변환
        text = whisper_client.transcribe_from_minio(file_path)
        

        print(text)
        return {
            "status": "success",
            "text": text,
            "metadata": {
                "channelId": channel_id,
                "liveId": live_id,
                "startTime": start_time,
                "endTime": end_time
            }
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "metadata": {
                "channelId": channel_id,
                "liveId": live_id
            }
        }

def enqueue_transcribe(key: str) -> str:
    """오디오 변환 작업을 큐에 추가"""
    from libs.whisper_client import whisper as whisper_client
    job = transcribe_queue.enqueue(
        whisper_client.transcribe_from_minio,
        key,
        job_timeout="10m"
    )
    return job.id

def enqueue_vision(key: str) -> str:
    """이미지 분석 작업을 큐에 추가"""
    from libs.vision_client import vision as vision_client
    job = vision_queue.enqueue(
        vision_client.describe_from_minio,
        key,
        job_timeout="5m"
    )
    return job.id

def get_job_status(queue_name: str, job_id: str) -> dict:
    """작업 상태 조회"""
    queue = Queue(queue_name, connection=redis_conn)
    job = queue.fetch_job(job_id)
    
    if not job:
        return {"status": "not_found"}
    
    return {
        "status": job.get_status(),
        "result": job.result if job.is_finished else None,
        "error": str(job.exc_info) if job.is_failed else None
    } 