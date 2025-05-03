import os
import asyncio
from dotenv import load_dotenv
from apps.hub_server import db
from libs.whisper_client import whisper
from libs.vision_client import vision

load_dotenv()

async def process_jobs():
    await db.connect_db()
    print("[DB] 배치 연결 성공!")

    # 처리 안 된 row만 조회
    jobs = await db.database.fetch_all(
        "SELECT id, audio_path, image_path FROM media_job WHERE processed = FALSE"
    )

    for job in jobs:
        audio_path = job["audio_path"]
        image_path = job["image_path"]
        transcript = None
        image_label = None

        # 오디오 분석
        if audio_path and os.path.exists(audio_path):
            try:
                transcript = whisper.transcribe(audio_path)
            except Exception as e:
                print(f"[오디오 분석 실패] {audio_path}: {e}")

        # 이미지 분석
        if image_path and os.path.exists(image_path):
            try:
                image_label = vision.describe_image(image_path)
            except Exception as e:
                print(f"[이미지 분석 실패] {image_path}: {e}")

        # 결과 DB 업데이트
        try:
            await db.database.execute(
                """
                UPDATE media_job
                SET transcript = :transcript,
                    image_label = :image_label,
                    processed = TRUE
                WHERE id = :id
                """,
                values={
                    "transcript": transcript,
                    "image_label": image_label,
                    "id": job["id"],
                }
            )
            print(f"[업데이트 완료] id={job['id']}")
        except Exception as e:
            print(f"[DB 업데이트 실패] id={job['id']}: {e}")

    await db.disconnect_db()
    print("[DB] 배치 연결 해제!")

if __name__ == "__main__":
    asyncio.run(process_jobs()) 