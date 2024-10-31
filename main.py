'''
install

pip install fastapi[standard]  # FastAPI와 기본 의존성 설치 (CORS, Pydantic 포함)
pip install "fastapi[standard]"
pip install torch              # PyTorch 설치 (딥러닝 모델 사용을 위해)
pip install transformers       # Hugging Face Transformers 설치 (언어 모델 사용을 위해)
pip install sentence-transformers # Sentence Transformers 설치 (문서 임베딩 사용을 위해)
pip install faiss-cpu          # Faiss 설치 (임베딩 검색을 위해)
pip install python-dotenv      # dotenv 설치 (.env 환경 변수 로드를 위해)
pip install requests           # requests 설치 (TTS 서버 호출을 위해)


'''


from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import faiss
import requests  # TTS 서버 호출을 위해 추가

# .env 파일에서 환경 변수 로드
load_dotenv()

# CORS 허용 URL 가져오기
allowed_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")

# 모델과 토크나이저 설정
model_name = "illuni/illuni-llama-2-ko-7b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 문서 임베딩 설정
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
documents = [
    "파이썬은 프로그래밍 언어입니다.",
    "딥러닝 모델은 대량의 데이터를 학습할 수 있습니다.",
    "FastAPI는 Python의 웹 프레임워크입니다."
]
document_embeddings = embedder.encode(documents)
index = faiss.IndexFlatL2(document_embeddings.shape[1])
index.add(document_embeddings)

# FastAPI 객체 생성 및 CORS 설정 추가
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# TTS 서버 URL 설정
TTS_SERVER_URL = "http://127.0.0.1:8080/synthesize"

# 엔드포인트 정의
@app.post("/chat/")
async def inference(text: str = Form(...)):
    query_embedding = embedder.encode([text])
    D, I = index.search(query_embedding, k=3)
    retrieved_docs = [documents[i] for i in I[0]]
    context = " ".join(retrieved_docs)
    input_text = f"{context} {text}"

    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    output = model.generate(inputs["input_ids"], max_length=50, do_sample=True, top_p=0.9, temperature=0.7)
    result = tokenizer.decode(output[0], skip_special_tokens=True)

    # TTS 서버로 텍스트 전송하여 음성 파일 생성 요청
    try:
        tts_response = requests.post(TTS_SERVER_URL, json={"text": result})
        tts_response.raise_for_status()
        audio_stream_url = tts_response.url  # 스트리밍 URL

    except requests.exceptions.RequestException as e:
        print("TTS 서버 오류:", e)
        audio_stream_url = None

    return {"result": result, "audio_url": audio_stream_url}
