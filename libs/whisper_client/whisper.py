# pip install openai-whisper 필요
import boto3
import whisper
from io import BytesIO
import tempfile
import torch
import torchaudio
import os

# 전역 변수로 모델을 한 번만 로드
WHISPER_MODEL = "medium"  # 또는 "small", "medium", "large" 중 선택
model = whisper.load_model(WHISPER_MODEL)

# Silero VAD 모델 로드
torch.set_num_threads(4)
vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                model='silero_vad',
                                force_reload=True)
(get_speech_timestamps,
 save_audio,
 read_audio,
 VADIterator,
 collect_chunks) = utils

def extract_voice_segments(audio_path: str, min_duration: float = 0.5) -> list:
    """
    오디오 파일에서 음성 구간을 추출합니다.
    
    Args:
        audio_path: 오디오 파일 경로
        min_duration: 최소 음성 구간 길이 (초)
        
    Returns:
        list: [(시작 시간, 종료 시간), ...] 형식의 음성 구간 리스트
    """
    # 오디오 로드
    wav = read_audio(audio_path, sampling_rate=16000)
    
    # 음성 구간 감지
    speech_timestamps = get_speech_timestamps(wav, vad_model, 
                                            threshold=0.5,
                                            sampling_rate=16000,
                                            min_speech_duration_ms=int(min_duration * 1000),
                                            min_silence_duration_ms=500)
    
    # 구간을 (시작 시간, 종료 시간) 형식으로 변환
    voice_segments = []
    for ts in speech_timestamps:
        start_time = ts['start'] / 16000  # 샘플링 레이트로 나누어 초 단위로 변환
        end_time = ts['end'] / 16000
        voice_segments.append((start_time, end_time))
    
    return voice_segments

def extract_audio_segment(audio_path: str, start_time: float, end_time: float) -> str:
    """
    오디오 파일에서 특정 구간을 추출합니다.
    
    Args:
        audio_path: 원본 오디오 파일 경로
        start_time: 시작 시간 (초)
        end_time: 종료 시간 (초)
        
    Returns:
        str: 추출된 오디오 파일의 임시 경로
    """
    # 오디오 로드
    waveform, sample_rate = torchaudio.load(audio_path)
    
    # 구간 추출
    start_sample = int(start_time * sample_rate)
    end_sample = int(end_time * sample_rate)
    segment = waveform[:, start_sample:end_sample]
    
    # 임시 파일로 저장
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        torchaudio.save(tmp.name, segment, sample_rate)
        return tmp.name

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
    
    # 임시 파일로 저장
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp.flush()
        
        # 음성 구간 추출
        voice_segments = extract_voice_segments(tmp.name)
        
        # 각 구간별로 텍스트 추출
        texts = []
        for start_time, end_time in voice_segments:
            # 구간 추출
            segment_path = extract_audio_segment(tmp.name, start_time, end_time)
            try:
                # 구간 텍스트 추출
                text = transcribe(segment_path)
                if text.strip():  # 빈 텍스트가 아닌 경우만 추가
                    texts.append(text)
            finally:
                # 임시 파일 삭제
                os.unlink(segment_path)
        
        # 원본 임시 파일 삭제
        os.unlink(tmp.name)
        
        return " ".join(texts)