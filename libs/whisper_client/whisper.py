# pip install openai-whisper 필요
import boto3
import whisper
from io import BytesIO
import tempfile
import torch
import torchaudio
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps

# 전역 변수로 모델을 한 번만 로드
WHISPER_MODEL = "medium"  # 또는 "small", "medium", "large" 중 선택
whisper_model = whisper.load_model(WHISPER_MODEL)

# Silero VAD 모델 로드
torch.set_num_threads(1)  # 공식 문서 권장사항
vad_model = load_silero_vad()

def extract_voice_segments(audio_path: str, min_duration: float = 0.5) -> list:
    """
    오디오 파일에서 음성 구간을 추출합니다.
    
    Args:
        audio_path: 오디오 파일 경로
        min_duration: 최소 음성 구간 길이 (초)
        
    Returns:
        list: [(시작 시간, 종료 시간), ...] 형식의 음성 구간 리스트
    """
    # 오디오 로드 (16kHz로 리샘플링)
    wav = read_audio(audio_path, sampling_rate=16000)
    
    # 음성 구간 감지
    speech_timestamps = get_speech_timestamps(wav, vad_model, 
                                            threshold=0.8,  # 더 엄격한 임계값 (게임 사운드 필터링)
                                            sampling_rate=16000,
                                            min_speech_duration_ms=500,  # 최소 0.5초 이상의 음성만 감지
                                            min_silence_duration_ms=400,  # 무음 구간 최소 길이 증가
                                            window_size_samples=2048,  # 더 큰 윈도우로 안정성 확보
                                            speech_pad_ms=50,  # 패딩 증가로 음성 손실 방지
                                            return_seconds=True)  # 초 단위로 반환
    
    # 구간을 (시작 시간, 종료 시간) 형식으로 변환하고 최소 길이 보장
    voice_segments = []
    for ts in speech_timestamps:
        start, end = ts['start'], ts['end']
        if end - start < min_duration:
            # 세그먼트가 너무 짧은 경우, 앞뒤로 확장
            center = (start + end) / 2
            start = max(0, center - min_duration/2)
            end = center + min_duration/2
        voice_segments.append((start, end))
    
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
    
    # 최소 길이 확인 (0.5초)
    min_duration = 0.5
    if end_time - start_time < min_duration:
        # 세그먼트가 너무 짧은 경우, 앞뒤로 확장
        center = (start_time + end_time) / 2
        start_time = max(0, center - min_duration/2)
        end_time = center + min_duration/2
    
    # 구간 추출
    start_sample = int(start_time * sample_rate)
    end_sample = int(end_time * sample_rate)
    segment = waveform[:, start_sample:end_sample]
    
    # 임시 파일로 저장
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        torchaudio.save(tmp.name, segment, sample_rate)
        return tmp.name

def transcribe(audio_path: str) -> str:
    result = whisper_model.transcribe(audio_path,   
                                      language="ko"  # 한국어로 번역
                                    )
    return result["text"]

def process_segment(args: Tuple[str, float, float]) -> str:
    """
    단일 오디오 구간을 처리합니다.
    
    Args:
        args: (오디오 파일 경로, 시작 시간, 종료 시간) 튜플
        
    Returns:
        str: 추출된 텍스트
        
    Raises:
        ValueError: 텍스트 추출 실패 시
    """
    audio_path, start_time, end_time = args
    
    # 최소 길이 확인 (0.5초)
    min_duration = 0.5
    if end_time - start_time < min_duration:
        # 세그먼트가 너무 짧은 경우, 앞뒤로 확장
        center = (start_time + end_time) / 2
        start_time = max(0, center - min_duration/2)
        end_time = center + min_duration/2
    
    segment_path = extract_audio_segment(audio_path, start_time, end_time)
    try:
        # Whisper 모델에 전달하기 전에 오디오 길이 확인
        waveform, sample_rate = torchaudio.load(segment_path)
        duration = waveform.shape[1] / sample_rate
        
        if duration < min_duration:
            raise ValueError(f"Audio segment too short: {duration:.2f}s")
            
        text = transcribe(segment_path)
        text = text.strip()
        if not text:  # 빈 문자열인 경우
            raise ValueError("No text extracted from audio segment")
        return text
    finally:
        os.unlink(segment_path)

def transcribe_from_minio(
    key,
    bucket="stream-project-data",
    endpoint_url="http://121.167.129.36:9000",
    access_key="dominic",
    secret_key="gumdong1!530",
    max_workers: int = 4
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
        tmp_path = tmp.name
        
        try:
            # 음성 구간 추출
            voice_segments = extract_voice_segments(tmp_path)
            
            if not voice_segments:  # 음성 구간이 없는 경우
                raise ValueError("No voice segments detected in audio")
            
            # 병렬 처리로 각 구간 처리
            texts = []
            errors = []
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # 각 구간에 대해 처리 작업 제출
                future_to_segment = {
                    executor.submit(process_segment, (tmp_path, start, end)): (start, end)
                    for start, end in voice_segments
                }
                
                # 결과 수집
                for future in as_completed(future_to_segment):
                    start, end = future_to_segment[future]
                    try:
                        text = future.result()
                        texts.append(text)
                    except Exception as e:
                        error_msg = f"Error processing segment {start:.2f}-{end:.2f}: {str(e)}"
                        errors.append(error_msg)
                        print(error_msg)
            
            if not texts:  # 모든 구간에서 텍스트 추출 실패
                raise ValueError("Failed to extract text from any audio segment")
            
            return " ".join(texts)
            
        finally:
            # 원본 임시 파일 삭제
            os.unlink(tmp_path)