FROM python:3.10-slim

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
