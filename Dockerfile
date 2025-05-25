# 안정적인 PyTorch CUDA 이미지 사용 (cuDNN 8.x와 호환)
FROM cnstark/pytorch:2.1.0-py3.10.15-cuda12.1.0-ubuntu22.04

# cuDNN 호환성을 위한 환경 변수 설정
ENV CUDNN_ENABLE_SKIP_RUNTIME_CHECK=1
ENV CUDA_LAUNCH_BLOCKING=1
ENV TORCH_CUDNN_V8_API_ENABLED=1
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# 시스템 패키지 업데이트 및 오디오 처리 라이브러리 설치
RUN apt-get update && apt-get install -y \
    wget \
    gnupg \
    ffmpeg \
    libsndfile1-dev \
    libsox-dev \
    && rm -rf /var/lib/apt/lists/*

# NVIDIA GPG key를 더 안정적인 방법으로 추가
RUN wget -qO - https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub | apt-key add - || \
    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub || \
    echo "GPG key 추가 실패, 계속 진행"

# NVIDIA 레포지토리 추가 (시도만 하고 실패해도 계속 진행)
RUN echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64 /" > /etc/apt/sources.list.d/cuda.list || echo "CUDA repo 추가 실패"

# cuDNN 8.x 설치 시도 (실패해도 계속 진행)
RUN apt-get update && (apt-get install -y \
    libcudnn8 \
    libcudnn8-dev || echo "cuDNN 패키지 설치 실패, 기본 cuDNN 사용") \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리 설정
WORKDIR /app

# 필요한 패키지 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 오디오 백엔드 패키지 추가 설치
RUN pip install soundfile

# 애플리케이션 코드 복사
COPY . .

# 포트 설정
EXPOSE 8000

# 서버 실행
CMD ["uvicorn", "apps.hub_server.main:app", "--host", "0.0.0.0", "--port", "8000"]
