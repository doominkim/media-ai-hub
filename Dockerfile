# PyTorch가 미리 설치된 이미지 사용
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# 작업 디렉토리 설정
WORKDIR /app

# 필요한 패키지 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드 복사
COPY . .

# 포트 설정
EXPOSE 8000

# 서버 실행
CMD ["uvicorn", "apps.hub_server.main:app", "--host", "0.0.0.0", "--port", "8000"]
