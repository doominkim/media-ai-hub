import os
import requests
from typing import Dict, Any, Optional
from fastapi import HTTPException

# Nest 서버 설정
NEST_SERVER_URL = os.getenv("NEST_SERVER_URL", "http://121.167.129.36:3000")

class NestClient:
    def __init__(self):
        self.base_url = NEST_SERVER_URL

    def get_next_job(self, queue_key: str) -> Optional[Dict[str, Any]]:
        """큐에서 다음 작업을 가져옴"""
        try:
            response = requests.post(f"{self.base_url}/queue/{queue_key}/next")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            if e.response and e.response.status_code == 404:
                # 작업이 없는 경우 404 에러 발생
                raise HTTPException(status_code=404, detail="No jobs available")
            raise HTTPException(status_code=500, detail=f"Nest 서버 통신 중 에러 발생: {str(e)}")

    def complete_job(self, queue_key: str, job_id: str, result: Dict[str, Any]) -> None:
        """작업 완료 처리"""
        try:
            response = requests.post(
                f"{self.base_url}/queue/{queue_key}/{job_id}/complete",
                json=result
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise HTTPException(status_code=500, detail=f"작업 완료 처리 중 에러 발생: {str(e)}")

    def fail_job(self, queue_key: str, job_id: str, error: str) -> None:
        """작업 실패 처리"""
        try:
            response = requests.post(
                f"{self.base_url}/queue/{queue_key}/{job_id}/fail",
                json={"error": error}
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise HTTPException(status_code=500, detail=f"작업 실패 처리 중 에러 발생: {str(e)}")

# 싱글톤 인스턴스 생성
nest_client = NestClient() 