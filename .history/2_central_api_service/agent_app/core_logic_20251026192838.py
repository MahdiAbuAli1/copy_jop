# src/app/core_logic.py

import os
import logging
import asyncio
import httpx
import re
from async_lru import alru_cache
from typing import AsyncGenerator, Dict
from dotenv import load_dotenv
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# --- 1. الإعدادات والتحميل الأولي (النسخة المحصّنة ) ---
# تحميل ملف .env بشكل صريح لضمان الوصول إليه
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
load_dotenv(dotenv_path=os.path.join(PROJECT_ROOT, ".env"))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- قراءة الإعدادات مع التحقق من وجودها ---
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL_NAME")
CHAT_MODEL = os.getenv("CHAT_MODEL_NAME")
OLLAMA_HOST = os.getenv("OLLAMA_HOST")

# --- التحقق الحاسم: تأكد من أن الإعدادات الأساسية موجودة ---
if not all([EMBEDDING_MODEL, CHAT_MODEL, OLLAMA_HOST]):
    missing = [
        var for var, val in 
        {"EMBEDDING_MODEL": EMBEDDING_MODEL, "CHAT_MODEL": CHAT_MODEL, "OLLAMA_HOST": OLLAMA_HOST}.items() 
        if not val
    ]
    raise ValueError(f"خطأ فادح: متغيرات البيئة التالية مفقودة في ملف .env: {', '.join(missing)}")

# --- متغيرات عالمية ---
embeddings: OllamaEmbeddings = None
llm_chat: Ollama = None
vector_stores: Dict[str, FAISS] = {}
initialization_lock = asyncio.Lock()

# --- 2. قاموس الردود السريعة ---
FAST_PATH_RESPONSES = {
    "أهلاً": "أهلاً بك! أنا مرشد الدعم. كيف يمكنني مساعدتك اليوم؟",
    "اهلا": "أهلاً بك! أنا مرشد الدعم. كيف يمكنني مساعدتك اليوم؟",
    "مرحباً": "مرحباً بك! أنا مرشد الدعم. كيف يمكنني مساعدتك اليوم؟",
    "مرحبا": "مرحباً بك! أنا مرشد الدعم. كيف يمكنني مساعدتك اليوم؟",
    "السلام عليكم": "وعليكم السلام! أنا مرشد الدعم. كيف يمكنني مساعدتك اليوم؟",
    "شكراً": "على الرحب والسعة! هل هناك أي شيء آخر يمكنني مساعدتك به؟",
    "شكرا": "على الرحب والسعة! هل هناك أي شيء آخر يمكنني مساعدتك به؟",
}

# --- 3. قالب التعليمات الموحد والذكي (QA_PROMPT) ---
QA_PROMPT = PromptTemplate.from_template(
    """### المهمة الأساسية:
أنت "مرشد الدعم"، مساعد ذكي متخصص في الإجابة على الأسئلة المتعلقة بوثائق المشروع المقدمة لك فقط.

### التعليمات الصارمة (يجب اتباعها حرفياً لاتخاذ القرار):
1.  **تحليل السياق أولاً:** انظر إلى قسم "السياق" أدناه.
2.  **القرار الأول (الإجابة من السياق):** إذا كان "السياق" يحتوي على معلومات ذات صلة مباشرة بسؤال المستخدم، قم بصياغة إجابة دقيقة ومفصلة بالاعتماد **حصرياً** على هذا السياق. لا تضف أي معلومات من خارج السياق.
3.  **القرار الثاني (الرفض بأدب):** إذا كان "السياق" فارغاً، أو إذا كانت المعلومات فيه غير مرتبطة بسؤال المستخدم، فهذا يعني أن السؤال خارج نطاق معرفتك. في هذه الحالة، يجب أن تكون إجابتك **فقط**: "أنا مساعد دعم فني متخصص. يمكنني مساعدتك في الأسئلة المتعلقة بالنظام الموثق لدي فقط."
4.  **ممنوع الهلوسة:** لا تحاول أبداً تخمين الإجابة أو استخدام معرفتك العامة. قرارك يعتمد فقط على جودة "السياق".

### السياق (المعلومات المتاحة لك):
{context}

### سؤال المستخدم:
{question}

### الإجابة (بناءً على التعليمات أعلاه):
"""
)

# --- 4. الدوال الأساسية للنظام ---

async def initialize_agent():
    global embeddings, llm_chat
    async with initialization_lock:
        if embeddings is not None: return
        try:
            async with httpx.AsyncClient( ) as client:
                await client.get(OLLAMA_HOST, timeout=5.0)
            logging.info("فحص الاتصال ناجح: خدمة Ollama متاحة.")
        except (httpx.RequestError, httpx.HTTPStatusError ) as e:
            logging.error(f"فشل فحص الاتصال: لا يمكن الوصول إلى خدمة Ollama على {OLLAMA_HOST}. الخطأ: {e}")
            raise RuntimeError("فشل تهيئة الوكيل بسبب عدم توفر خدمة Ollama.")
            
        logging.info("جارٍ تهيئة النماذج الأساسية للوكيل...")
        embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_HOST)
        llm_chat = Ollama(model=CHAT_MODEL, base_url=OLLAMA_HOST, temperature=0.05)
        logging.info("النماذج الأساسية جاهزة للعمل.")

@alru_cache(maxsize=1)
async def get_vector_store() -> FAISS | None:
    db_key = "main_shared_db"
    if db_key in vector_stores: return vector_stores[db_key]
    db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../3_shared_resources/vector_db"))
    if not os.path.isdir(db_path):
        logging.error(f"فشل حاسم: المجلد 'vector_db' غير موجود في المسار المتوقع '{db_path}'.")
        return None
    try:
        logging.info(f"[Lazy Load] جارٍ تحميل قاعدة المعرفة من المسار الصحيح: {db_path}")
        vector_store = await asyncio.to_thread(
            FAISS.load_local, db_path, embeddings, allow_dangerous_deserialization=True
        )
        vector_stores[db_key] = vector_store
        logging.info("تم تحميل قاعدة المعرفة المشتركة بنجاح.")
        return vector_store
    except Exception as e:
        logging.error(f"فشل حاسم أثناء قراءة ملفات FAISS. الخطأ: {e}")
        return None

# --- دالة get_answer_stream - النسخة النهائية الموحدة ---
async def get_answer_stream(request_info: dict) -> AsyncGenerator[Dict, None]:
    question = request_info["question"].strip()
    
    if question in FAST_PATH_RESPONSES:
        logging.info(f"استخدام الرد السريع للسؤال: '{question}'")
        yield {"type": "status", "category": "fast_path"}
        yield {"type": "chunk", "content": FAST_PATH_RESPONSES[question]}
        return

    yield {"type": "status", "category": "retrieval"}
    vector_store = await get_vector_store()
    if not vector_store:
        yield {"type": "error", "content": "عذراً، قاعدة بيانات المعرفة غير متاحة حالياً."}
        return

    retriever = vector_store.as_retriever(search_kwargs={'k': request_info["k_results"]})
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm_chat,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": QA_PROMPT},
        return_source_documents=False
    )
    
    logging.info("بدء توليد الإجابة عبر سلسلة RetrievalQA الموحدة...")
    try:
        result = await qa_chain.ainvoke({"query": question})
        answer = result.get('result', "عذراً، حدث خطأ غير متوقع أثناء توليد الإجابة.")
        yield {"type": "chunk", "content": answer}
    except Exception as e:
        logging.error(f"فشل في سلسلة RetrievalQA. الخطأ: {e}", exc_info=True)
        yield {"type": "error", "content": "عذراً، حدث خطأ فادح أثناء معالجة طلبك."}

