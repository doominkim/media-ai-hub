# pip install openai-whisper 필요
import boto3
import whisper
from io import BytesIO
import tempfile

# 전역 변수로 모델을 한 번만 로드
WHISPER_MODEL = "medium"  # 또는 "small", "medium", "large" 중 선택
model = whisper.load_model(WHISPER_MODEL)

def transcribe(audio_path: str) -> str:
    result = model.transcribe(audio_path)
    return result["text"]

def transcribe_from_minio(
    key,
    bucket="stream-project-data",
    endpoint_url="http://121.167.129.36:9000",
    access_key="dominic",
    secret_key="gumdong1!530"
):
    s3 = boto3.client(
        's3',
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    )
    obj = s3.get_object(Bucket=bucket, Key=key)
    audio_bytes = obj['Body'].read()
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=True) as tmp:
        tmp.write(audio_bytes)
        tmp.flush()
        return transcribe(tmp.name)