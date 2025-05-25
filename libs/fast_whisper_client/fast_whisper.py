# pip install faster-whisper 필요
import boto3
from faster_whisper import WhisperModel
from io import BytesIO
import tempfile
import torch
import torchaudio
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps

# 전역 변수로 모델을 한 번만 로드
WHISPER_MODEL = "base" # 또는 "small", "medium", "large" 중 선택

# GPU 사용 가능 여부 확인 및 안전한 초기화
device = "cpu"  # 기본값을 CPU로 설정
compute_type = "int8"
whisper_model = None

def initialize_whisper_model():
    """Whisper 모델을 안전하게 초기화합니다."""
    global device, compute_type, whisper_model
    
    try:
        # CUDA 사용 가능 여부 확인
        if torch.cuda.is_available():
            try:
                # GPU 테스트: 간단한 CUDA 연산 수행
                test_tensor = torch.tensor([1.0]).cuda()
                _ = test_tensor + 1
                device = "cuda"
                compute_type = "float16"
                print(f"🚀 GPU 감지됨: {torch.cuda.get_device_name()}")
                print(f"🔥 CUDA 버전: {torch.version.cuda}")
            except Exception as cuda_error:
                print(f"⚠️  CUDA 테스트 실패, CPU 모드로 대체: {cuda_error}")
                device = "cpu"
                compute_type = "int8"
        else:
            print("📱 CUDA를 사용할 수 없어 CPU 모드로 실행합니다.")
            device = "cpu"
            compute_type = "int8"
            
        # Whisper 모델 로드 시도 (cuDNN 호환성을 위한 설정)
        try:
            # CTranslate2의 cuDNN 설정 조정
            os.environ["CT2_CUDNN_ALLOW_FALLBACK"] = "1"
            os.environ["CT2_CUDA_ALLOCATOR"] = "cuda_malloc_async"
            
            whisper_model = WhisperModel(
                WHISPER_MODEL, 
                device=device, 
                compute_type=compute_type,
                # CTranslate2 설정 추가
            )
            print(f"✅ Fast-Whisper 모델 로드 성공: {WHISPER_MODEL} on {device}")
        except Exception as model_error:
            print(f"⚠️  {device.upper()} 모델 로드 실패, CPU로 재시도: {model_error}")
            device = "cpu"
            compute_type = "int8"
            whisper_model = WhisperModel(WHISPER_MODEL, device=device, compute_type=compute_type)
            print(f"✅ Fast-Whisper 모델 CPU 로드 성공: {WHISPER_MODEL}")
            
    except Exception as e:
        print(f"💥 모델 초기화 중 치명적 오류 발생: {e}")
        # 최후의 수단으로 기본 설정 사용
        device = "cpu"
        compute_type = "int8"
        whisper_model = WhisperModel(WHISPER_MODEL, device=device, compute_type=compute_type)
        print("🔧 기본 CPU 모드로 모델 로드 완료")

# 모델 초기화 실행
initialize_whisper_model()

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
    try:
        # 오디오 로드 (16kHz로 리샘플링)
        wav = read_audio(audio_path, sampling_rate=16000)
        
        # 오디오 정규화
        wav = wav / (torch.max(torch.abs(wav)) + 1e-8)
        
        # 오디오 길이 확인
        audio_duration = len(wav) / 16000
        print(f"🎵 오디오 분석: 길이={audio_duration:.2f}초")
        
        # 음성 구간 감지 (더 관대한 설정)
        speech_timestamps = get_speech_timestamps(wav, vad_model, 
                                                threshold=0.4,  # 0.6에서 0.4로 낮춤
                                                sampling_rate=16000,
                                                min_speech_duration_ms=500,  # 1000에서 500으로 줄임 (0.5초)
                                                min_silence_duration_ms=200,  # 400에서 200으로 줄임
                                                window_size_samples=1024,
                                                speech_pad_ms=50,  # 20에서 50으로 증가 (더 많은 패딩)
                                                return_seconds=True)
        
        print(f"🔍 VAD 결과: {len(speech_timestamps)}개 음성 구간 감지")
        
        # 구간 병합 로직
        merged_segments = []
        if speech_timestamps:
            current_start, current_end = speech_timestamps[0]['start'], speech_timestamps[0]['end']
            for ts in speech_timestamps[1:]:
                start, end = ts['start'], ts['end']
                if start - current_end < 1.0:  # 500ms에서 1.0초로 증가 (더 적극적인 병합)
                    current_end = end
                else:
                    if current_end - current_start >= min_duration:  # 최소 길이 확인
                        merged_segments.append((current_start, current_end))
                    current_start, current_end = start, end
            
            # 마지막 구간 처리
            if current_end - current_start >= min_duration:
                merged_segments.append((current_start, current_end))
                
        print(f"📊 병합 후: {len(merged_segments)}개 최종 음성 구간")
        for i, (start, end) in enumerate(merged_segments):
            print(f"   구간 {i+1}: {start:.2f}초 ~ {end:.2f}초 ({end-start:.2f}초)")
        
        return merged_segments
        
    except Exception as e:
        print(f"⚠️ VAD 처리 중 오류: {e}")
        # VAD 실패 시 빈 리스트 반환 (fallback 처리는 호출하는 곳에서)
        return []

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
    """
    faster-whisper를 사용하여 오디오를 텍스트로 변환합니다.
    GPU 에러 시 CPU로 자동 fallback됩니다.
    """
    global whisper_model, device, compute_type
    
    try:
        segments, info = whisper_model.transcribe(
            audio_path,   
            language="ko",
            task="transcribe",
            beam_size=2,
            condition_on_previous_text=False,
            # faster-whisper 특유의 설정들
            word_timestamps=False,
            vad_filter=True,  # Voice Activity Detection 필터 사용
            vad_parameters=dict(min_silence_duration_ms=500)
        )
        
        # 세그먼트들을 하나의 텍스트로 결합
        text_segments = []
        for segment in segments:
            text_segments.append(segment.text)
        
        return " ".join(text_segments).strip()
        
    except Exception as e:
        error_msg = str(e)
        if "cudnn" in error_msg.lower() or "cuda" in error_msg.lower():
            print(f"🔄 GPU 에러 감지, CPU 모드로 fallback: {error_msg}")
            
            # CPU 모드로 모델 재로드
            try:
                device = "cpu"
                compute_type = "int8"
                whisper_model = WhisperModel(WHISPER_MODEL, device=device, compute_type=compute_type)
                print(f"✅ CPU 모드로 모델 재로드 완료")
                
                # CPU 모드로 재시도
                segments, info = whisper_model.transcribe(
                    audio_path,   
                    language="ko",
                    task="transcribe",
                    beam_size=2,
                    condition_on_previous_text=False,
                    word_timestamps=False,
                    vad_filter=True,
                    vad_parameters=dict(min_silence_duration_ms=500)
                )
                
                text_segments = []
                for segment in segments:
                    text_segments.append(segment.text)
                
                return " ".join(text_segments).strip()
                
            except Exception as cpu_error:
                print(f"💥 CPU 모드에서도 실패: {cpu_error}")
                raise cpu_error
        else:
            # GPU/CPU와 관련없는 다른 에러는 그대로 전달
            raise e

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
    
    # 최소 길이 확인 (1초)
    min_duration = 1.0  # 0.5초에서 1초로 증가
    if end_time - start_time < min_duration:
        # 세그먼트가 너무 짧은 경우, 앞뒤로 확장
        center = (start_time + end_time) / 2
        start_time = max(0, center - min_duration/2)
        end_time = center + min_duration/2
    
    segment_path = extract_audio_segment(audio_path, start_time, end_time)
    try:
        # 오디오 파일 검증
        waveform, sample_rate = torchaudio.load(segment_path)
        
        # 오디오 길이 확인
        duration = waveform.shape[1] / sample_rate
        if duration < min_duration:
            raise ValueError(f"Audio segment too short: {duration:.2f}s")
            
        # 오디오 형식 검증
        if waveform.shape[0] == 0 or waveform.shape[1] == 0:
            raise ValueError("Invalid audio format: empty waveform")
            
        # 스테레오를 모노로 변환
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        # 볼륨 정규화
        waveform = waveform / (torch.max(torch.abs(waveform)) + 1e-8)
        
        # 정규화된 오디오 저장
        torchaudio.save(segment_path, waveform, sample_rate)
            
        text = transcribe(segment_path)
        text = text.strip()
        if not text:
            raise ValueError("No text extracted from audio segment")
        return text
    finally:
        if os.path.exists(segment_path):
            os.unlink(segment_path)

def transcribe_from_minio(
    key,
    bucket="stream-project-data",
    endpoint_url="http://121.167.129.36:9000",
    access_key="dominic",
    secret_key="gumdong1!530",
    max_workers: int = 4
):
    """
    MinIO에서 오디오 파일을 다운로드하고 faster-whisper로 텍스트 변환을 수행합니다.
    """
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
            
            if not voice_segments:  # 음성 구간이 없는 경우 fallback 처리
                print("⚠️ VAD로 음성 구간을 찾지 못함, 전체 오디오를 직접 transcribe 시도")
                try:
                    # 전체 오디오 파일을 직접 transcribe
                    full_text = transcribe(tmp_path)
                    if full_text and full_text.strip():
                        print(f"✅ 전체 오디오 transcribe 성공: '{full_text[:50]}...'")
                        return full_text.strip()
                    else:
                        print("❌ 전체 오디오에서도 텍스트를 추출하지 못함")
                        raise ValueError("No text extracted from full audio file")
                except Exception as full_transcribe_error:
                    print(f"💥 전체 오디오 transcribe 실패: {full_transcribe_error}")
                    raise ValueError("No voice segments detected and full transcribe failed")
            
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
                print("❌ 모든 음성 구간에서 텍스트 추출 실패, 전체 오디오 fallback 시도")
                try:
                    # 마지막 수단으로 전체 오디오 transcribe
                    full_text = transcribe(tmp_path)
                    if full_text and full_text.strip():
                        print(f"✅ Fallback transcribe 성공: '{full_text[:50]}...'")
                        return full_text.strip()
                    else:
                        raise ValueError("Failed to extract text from any audio segment and full transcribe")
                except Exception as fallback_error:
                    print(f"💥 Fallback transcribe도 실패: {fallback_error}")
                    raise ValueError("Failed to extract text from any audio segment and full transcribe")
            
            return " ".join(texts)
            
        finally:
            # 원본 임시 파일 삭제
            os.unlink(tmp_path) 