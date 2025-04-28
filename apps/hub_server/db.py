from dotenv import load_dotenv
import os
import databases

load_dotenv()  # .env 파일 자동 로드

POSTGRES_URL = os.getenv("POSTGRES_URL")
if not POSTGRES_URL:
    user = os.getenv("POSTGRES_USER", "postgres")
    password = os.getenv("POSTGRES_PASSWORD", "password")
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    db = os.getenv("POSTGRES_DB", "postgres")
    POSTGRES_URL = f"postgresql://{user}:{password}@{host}:{port}/{db}"

database = databases.Database(POSTGRES_URL)

async def connect_db():
    await database.connect()

async def disconnect_db():
    await database.disconnect() 