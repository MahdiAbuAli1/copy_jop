# src/app/main.py

import os
import logging
import json
import time
from fastapi import FastAPI, Header, HTTPException, Depends
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from dotenv import load_dotenv

# --- تم تعديل هذا السطر ---
from .core_logic import initialize_agent, get_answer_stream

# --- إعداد التسجيل المتقدم بصيغة JSON ---
LOG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../agent_logs"))
os.makedirs(LOG_DIR, exist_ok=True)
log_file_path = os.path.join(LOG_DIR, "interactions.json.log")
json_handler = logging.FileHandler(log_file_path, encoding="utf-8")
json_formatter = logging.Formatter('{"timestamp": "%(asctime)s", "level": "%(levelname)s", "message": %(message)s}')
json_handler.setFormatter(json_formatter)
root_logger = logging.getLogger()
if root_logger.hasHandlers():
    root_logger.handlers.clear()
root_logger.addHandler(json_handler)
root_logger.setLevel(logging.INFO)

# --- قراءة إعدادات البيئة ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
load_dotenv(dotenv_path=os.path.join(PROJECT_ROOT, ".env"), override=True)
EXPECTED_API_KEY = os.getenv("SUPPORT_SERVICE_API_KEY")

# --- تم تعديل هذا الجزء ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    root_logger.info(json.dumps({"event": "startup", "detail": "بدء تشغيل خادم الـ API..."}))
    await initialize_agent()
    yield
    # لا حاجة لـ shutdown_agent بعد الآن
    root_logger.info(json.dumps({"event": "shutdown", "detail": "تم إيقاف تشغيل خادم الـ API."}))

app = FastAPI(title="منصة الدعم الفني المركزي", version="3.0.0-unified", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")

class QueryRequest(BaseModel):
    question: str
    tenant_id: str
    k_results: int = 8 # زيادة عدد النتائج قد يحسن السياق

async def verify_api_key(x_api_key: str = Header(None)):
    if not EXPECTED_API_KEY: return
    if not x_api_key or x_api_key.strip() != EXPECTED_API_KEY.strip():
        raise HTTPException(status_code=401, detail="مفتاح API غير صالح أو مفقود")

@app.post("/ask-stream", dependencies=[Depends(verify_api_key)])
async def ask_question_stream(request: QueryRequest) -> StreamingResponse:
    start_time = time.time()
    
    async def generator_wrapper() -> AsyncGenerator[str, None]:
        final_answer = ""
        log_data = {"tenant_id": request.tenant_id, "question": request.question, "category": "unknown", "error": None}
        
        try:
            async for event in get_answer_stream(request.dict()):
                if event["type"] == "status":
                    log_data["category"] = event.get("category", "unknown")
                elif event["type"] == "chunk":
                    content = event["content"]
                    final_answer += content
                    yield content
                elif event["type"] == "error":
                    content = event["content"]
                    log_data["error"] = content
                    yield content
                    break
        except Exception as e:
            log_data["error"] = str(e)
            root_logger.error(json.dumps({"event": "stream_error", "detail": str(e)}))
            yield "حدث خطأ فادح أثناء معالجة طلبك."
        finally:
            duration = time.time() - start_time
            log_data["duration_ms"] = int(duration * 1000)
            log_data["answer"] = final_answer
            root_logger.info(json.dumps(log_data, ensure_ascii=False))

    return StreamingResponse(generator_wrapper(), media_type="text/plain; charset=utf-8")

@app.get("/", include_in_schema=False)
def read_root():
    return FileResponse(os.path.join(os.path.dirname(__file__), "static", "chat.html"))

@app.get("/tenants")
def list_tenants():
    clients_base = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../4_client_docs"))
    if not os.path.isdir(clients_base): return {"tenants": []}
    return {"tenants": [name for name in os.listdir(clients_base) if os.path.isdir(os.path.join(clients_base, name))]}
