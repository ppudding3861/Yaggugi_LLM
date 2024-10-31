# cors_config.py

from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv

# STEP 1: 환경 변수 로드
load_dotenv()

# STEP 2: 허용할 오리진 가져오기
allowed_origins = os.getenv("ALLOWED_ORIGINS", "").split(",")

# STEP 3: CORS 미들웨어를 추가하는 함수 정의
def setup_cors(app):
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,  # 환경 변수로부터 CORS 도메인 설정
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
