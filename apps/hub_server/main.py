from fastapi import FastAPI
from apps.hub_server.routers import health

app = FastAPI(
    title="Media AI Hub",
    description="멀티미디어 데이터 AI 분석 허브 서버",
    version="0.1.0",
)

# 라우터 등록
app.include_router(health.router)

@app.get("/")
async def root():
    return {"message": "Welcome to Media AI Hub"}
