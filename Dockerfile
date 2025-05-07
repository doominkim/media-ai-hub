FROM python:3.10-slim

ARG REPO_URL=git@github.com:your-org/your-private-repo.git

RUN apt-get update && \
    apt-get install -y git openssh-client && \
    mkdir -p /root/.ssh && chmod 700 /root/.ssh

# known_hosts 등록 (github 예시)
RUN ssh-keyscan github.com >> /root/.ssh/known_hosts

# 실제 코드 clone
RUN git clone $REPO_URL /app

WORKDIR /app

RUN pip install -r requirements.txt

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
