#!/bin/bash

# 로그 디렉토리 생성
mkdir -p logs

# 각 worker 시작
for i in {0..3}
do
    echo "Worker $i 시작 중..."
    WORKER_ID=$i python main.py &
    sleep 2  # 각 worker 시작 사이에 2초 대기
done

echo "모든 worker가 시작되었습니다."
echo "포트: 8000-8003" 