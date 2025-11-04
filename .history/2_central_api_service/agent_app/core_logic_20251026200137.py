import os
import logging
import asyncio
import httpx
from typing import AsyncGenerator, Dict, List
from dotenv import load_dotenv
from async_lru import alru_cache

from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage, AIMessage

# --- 1. الإعدادات ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__ ), "../../"))
load_dotenv(dotenv_path=os.path.join(PROJECT_ROOT, ".env"))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL_NAME")
CHAT_MODEL = os.getenv("CHAT_MODEL_NAME")
OLLAMA_HOST = os.getenv("OLLAMA_HOST")
VECTOR_DBS_DIR = os.path.join(PROJECT_ROOT, "3_shared_resources", "vector_dbs")

if not all([EMBEDDING_MODEL, CHAT_MODEL, OLLAMA_HOST]):
    raise ValueError("متغيرات البيئة الأساسية مفقودة في ملف .env")

# --- متغيرات عالمية ---
llm: Ollama = None
embeddings: OllamaEmbeddings = None
# --- ذاكرة لتخزين قواعد بيانات العملاء المحملة ---
vector_store_cache: Dict[str, FAISS] = {}
chat_history: Dict[str, List[HumanMessage | AIMessage]] = {} 
initialization_lock = asyncio.Lock()

# --- 2. القوالب (لا تغيير) ---
REPHRASE_PROMPT = ChatPromptTemplate.from_messages([...]) # أبقِ القوالب كما هي من النسخة السابقة
ANSWER_PROMPT = ChatPromptTemplate.from_messages([...])   # أبقِ القوالب كما هي من النسخة السابقة

# --- 3. الدوال الأساسية ---
async def initialize_agent():
    global llm, embeddings
    async with initialization_lock:
        if llm is not None: return
        logging.info("بدء تهيئة النماذج الأساسية...")
        try:
            async with httpx.AsyncClient( ) as client:
                await client.get(OLLAMA_HOST, timeout=10.0)
            llm = Ollama(model=CHAT_MODEL, base_url=OLLAMA_HOST, temperature=0.1)
            embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_HOST)
            logging.info("✅ النماذج الأساسية جاهزة.")
        except Exception as e:
            raise RuntimeError(f"فشل فحص الاتصال أو تهيئة النماذج: {e}")

@alru_cache(maxsize=10) # تخزين مؤقت لـ 10 قواعد بيانات عملاء
async def get_tenant_vector_store(tenant_id: str) -> FAISS | None:
    if tenant_id in vector_store_cache:
        logging.info(f"استخدام قاعدة البيانات من الذاكرة المؤقتة للعميل: {tenant_id}")
        return vector_store_cache[tenant_id]

    tenant_db_path = os.path.join(VECTOR_DBS_DIR, tenant_id)
    if not os.path.isdir(tenant_db_path):
        logging.warning(f"قاعدة بيانات العميل '{tenant_id}' غير موجودة في المسار: {tenant_db_path}")
        return None
    
    try:
        logging.info(f"جارٍ تحميل قاعدة بيانات العميل '{tenant_id}' من القرص...")
        store = await asyncio.to_thread(
            FAISS.load_local, tenant_db_path, embeddings, allow_dangerous_deserialization=True
        )
        vector_store_cache[tenant_id] = store
        logging.info(f"تم تحميل وتخزين قاعدة بيانات العميل '{tenant_id}' بنجاح.")
        return store
    except Exception as e:
        logging.error(f"فشل تحميل قاعدة بيانات العميل '{tenant_id}'. الخطأ: {e}")
        return None

# --- 4. دالة get_answer_stream ---
async def get_answer_stream(request_info: dict) -> AsyncGenerator[Dict, None]:
    question = request_info["question"].strip()
    tenant_id = request_info.get("tenant_id")

    if not tenant_id:
        yield {"type": "error", "content": "معرف العميل (tenant_id) مفقود. لا يمكن متابعة الطلب."}
        return

    vector_store = await get_tenant_vector_store(tenant_id)
    if not vector_store:
        yield {"type": "error", "content": f"عذراً، لا توجد قاعدة معرفة مهيأة للعميل '{tenant_id}'."}
        return

    retriever = vector_store.as_retriever(search_kwargs={"k": request_info.get("k_results", 8)})
    user_chat_history = chat_history.get(tenant_id, [])

    history_aware_retriever = create_history_aware_retriever(llm, retriever, REPHRASE_PROMPT)
    document_chain = create_stuff_documents_chain(llm, ANSWER_PROMPT)
    conversational_rag_chain = create_retrieval_chain(history_aware_retriever, document_chain)

    logging.info(f"[{tenant_id}] بدء معالجة السؤال '{question}'...")
    try:
        full_answer = ""
        async for chunk in conversational_rag_chain.astream({"input": question, "chat_history": user_chat_history}):
            if "answer" in chunk and chunk["answer"] is not None:
                answer_chunk = chunk["answer"]
                full_answer += answer_chunk
                yield {"type": "chunk", "content": answer_chunk}
        
        user_chat_history.append(HumanMessage(content=question))
        user_chat_history.append(AIMessage(content=full_answer))
        chat_history[tenant_id] = user_chat_history[-10:]
        logging.info(f"[{tenant_id}] الإجابة الكاملة: '{full_answer}'")
    except Exception as e:
        logging.error(f"[{tenant_id}] فشل في سلسلة RAG. الخطأ: {e}", exc_info=True)
        yield {"type": "error", "content": "عذراً، حدث خطأ فادح أثناء معالجة طلبك."}

# انسخ القوالب من النسخة السابقة وضعها هنا
REPHRASE_PROMPT = ChatPromptTemplate.from_messages(...)
ANSWER_PROMPT = ChatPromptTemplate.from_messages(...)
