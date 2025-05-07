import os
import sys
from rq import Worker, Queue, Connection
from redis import Redis

# 프로젝트 루트 디렉토리를 Python 경로에 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Redis 연결 설정
redis_url = os.getenv("REDIS_URL", "redis://121.167.129.36:6379")
redis_conn = Redis.from_url(redis_url)

# 작업자 설정
queues = [Queue("audio-processing", connection=redis_conn)]

def main():
    """오디오 처리 작업자 실행"""
    with Connection(redis_conn):
        worker = Worker(queues)
        print("🎧 오디오 처리 작업자를 시작합니다...")
        worker.work()

if __name__ == "__main__":
    main() 