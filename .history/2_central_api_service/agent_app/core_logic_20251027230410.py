# # src/app/core_logic.py
# #كود ممتاز اثبت جدارته ونتائج ممتازه
# import os
# import logging
# import asyncio
# import httpx
# from typing import AsyncGenerator, Dict, List

# from dotenv import load_dotenv
# from langchain_community.embeddings import OllamaEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain_community.llms import Ollama
# from langchain_core.prompts import ChatPromptTemplate
# from langchain.chains import create_history_aware_retriever, create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.messages import HumanMessage, AIMessage

# # --- 1. الإعدادات ---
# PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__ ), "../../"))
# load_dotenv(dotenv_path=os.path.join(PROJECT_ROOT, ".env"))
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# # --- استخدم نفس الإعدادات الموجودة في سكرت البناء الخاص بك ---
# EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL_NAME", "qwen3-embedding:4b")
# CHAT_MODEL = os.getenv("CHAT_MODEL_NAME", "qwen2:7b-instruct-q3_K_M")
# OLLAMA_HOST = os.getenv("OLLAMA_HOST")

# # --- المسار إلى قاعدة البيانات الموحدة التي يبنيها سكرت main_builder.py ---
# UNIFIED_DB_PATH = os.path.join(PROJECT_ROOT, "3_shared_resources", "vector_db")

# # --- متغيرات عالمية ---
# llm: Ollama = None
# vector_store: FAISS = None
# embeddings: OllamaEmbeddings = None
# chat_history: Dict[str, List[HumanMessage | AIMessage]] = {} 
# initialization_lock = asyncio.Lock()

# # --- 2. القوالب ---
# REPHRASE_PROMPT = ChatPromptTemplate.from_template("""
# بالنظر إلى سجل المحادثة والسؤال الأخير، قم بصياغة سؤال مستقل يمكن فهمه بدون سجل المحادثة.
# سجل المحادثة: {chat_history}
# السؤال الأخير: {input}
# السؤال المستقل:""")

# ANSWER_PROMPT = ChatPromptTemplate.from_template("""
# أنت "مرشد الدعم"، مساعد ذكي وخبير. مهمتك هي الإجابة على سؤال المستخدم بالاعتماد **حصرياً** على "السياق" المقدم.
# - كن دائماً متعاوناً ومحترفاً.
# - إذا كان السياق يحتوي على إجابة، قدمها بشكل مباشر ومنظم.
# - إذا كانت المعلومات غير موجودة بشكل واضح في السياق، قل بأسلوب لطيف: "بحثت في قاعدة المعرفة، ولكن لم أجد إجابة واضحة بخصوص هذا السؤال."
# - لا تخترع إجابات أبداً. التزم بالسياق.

# السياق:
# {context}

# السؤال: {input}
# الإجابة:""")

# # --- 3. الدوال الأساسية ---
# async def initialize_agent():
#     global llm, embeddings, vector_store
#     async with initialization_lock:
#         if vector_store is not None: return
#         logging.info("بدء تهيئة النماذج وقاعدة البيانات الموحدة...")
#         try:
#             async with httpx.AsyncClient( ) as client:
#                 await client.get(OLLAMA_HOST, timeout=10.0)
#             llm = Ollama(model=CHAT_MODEL, base_url=OLLAMA_HOST, temperature=0.1)
#             embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_HOST)
            
#             if not os.path.isdir(UNIFIED_DB_PATH):
#                 raise FileNotFoundError(f"قاعدة البيانات الموحدة غير موجودة. يرجى تشغيل سكرت 'main_builder.py' أولاً.")

#             vector_store = await asyncio.to_thread(
#                 FAISS.load_local, UNIFIED_DB_PATH, embeddings, allow_dangerous_deserialization=True
#             )
#             logging.info("✅ الوكيل جاهز للعمل بقاعدة بيانات موحدة.")
#         except Exception as e:
#             logging.error(f"فشل فادح أثناء التهيئة: {e}", exc_info=True)
#             raise

# # --- 4. دالة get_answer_stream ---
# async def get_answer_stream(request_info: dict) -> AsyncGenerator[Dict, None]:
#     question = request_info["question"].strip()
#     tenant_id = request_info.get("tenant_id")
#     session_id = tenant_id or "default_session"

#     if not vector_store:
#         yield {"type": "error", "content": "الوكيل غير جاهز. يرجى إعادة تحميل الصفحة."}
#         return

#     # --- الفلترة حسب العميل تتم هنا، في مرحلة البحث ---
#     retriever = vector_store.as_retriever(
#         search_kwargs={'k': 15, 'filter': {'tenant_id': tenant_id}}
#     )
    
#     user_chat_history = chat_history.get(session_id, [])

#     history_aware_retriever = create_history_aware_retriever(llm, retriever, REPHRASE_PROMPT)
#     document_chain = create_stuff_documents_chain(llm, ANSWER_PROMPT)
#     conversational_rag_chain = create_retrieval_chain(history_aware_retriever, document_chain)

#     logging.info(f"[{session_id}] بدء معالجة السؤال '{question}'...")
#     try:
#         full_answer = ""
#         async for chunk in conversational_rag_chain.astream({"input": question, "chat_history": user_chat_history}):
#             if "answer" in chunk and chunk["answer"] is not None:
#                 answer_chunk = chunk["answer"]
#                 full_answer += answer_chunk
#                 yield {"type": "chunk", "content": answer_chunk}
        
#         user_chat_history.append(HumanMessage(content=question))
#         user_chat_history.append(AIMessage(content=full_answer))
#         chat_history[session_id] = user_chat_history[-10:]
#         logging.info(f"[{session_id}] الإجابة الكاملة: '{full_answer}'")
#     except Exception as e:
#         logging.error(f"[{session_id}] فشل في سلسلة RAG. الخطأ: {e}", exc_info=True)
#         yield {"type": "error", "content": "عذراً، حدث خطأ فادح."}
# src/app/core_logic.py
# src/app/core_logic.py

import os
import logging
import asyncio
import httpx
import yaml
from typing import AsyncGenerator, Dict, List

from dotenv import load_dotenv
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage, AIMessage

from rank_bm25 import BM25Okapi
from rapidfuzz import fuzz
import re

# --- 1. الإعدادات ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
load_dotenv(dotenv_path=os.path.join(PROJECT_ROOT, ".env"))
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL_NAME", "qwen3-embedding:4b")
CHAT_MODEL = os.getenv("CHAT_MODEL_NAME", "qwen2:7b-instruct-q3_K_M")
OLLAMA_HOST = os.getenv("OLLAMA_HOST")

UNIFIED_DB_PATH = os.path.join(PROJECT_ROOT, "3_shared_resources", "vector_db")
FAQ_PATH = os.path.join(PROJECT_ROOT, "kb", "faqs.yaml")

# --- متغيرات عالمية ---
llm: Ollama = None
vector_store: FAISS = None
embeddings: OllamaEmbeddings = None
bm25: BM25Okapi = None
faq_items: List[Dict] = []
chat_history: Dict[str, List[HumanMessage | AIMessage]] = {}
initialization_lock = asyncio.Lock()

# --- 2. دوال مساعدة ---
TASHKEEL = re.compile(r'[\u0617-\u061A\u064B-\u0652]')

def normalize_ar(text: str) -> str:
    t = text.strip()
    t = TASHKEEL.sub("", t)
    t = re.sub("[إأآا]", "ا", t)
    t = re.sub("ى", "ي", t)
    t = re.sub("ؤ", "و", t)
    t = re.sub("ئ", "ي", t)
    t = re.sub("ة", "ه", t)
    return t

# --- 3. إعداد الـ prompts ---
REPHRASE_PROMPT = ChatPromptTemplate.from_template("""
بالنظر إلى سجل المحادثة والسؤال الأخير، قم بصياغة سؤال مستقل يمكن فهمه بدون سجل المحادثة.
سجل المحادثة: {chat_history}
السؤال الأخير: {input}
السؤال المستقل:""")

ANSWER_PROMPT = ChatPromptTemplate.from_template("""
أنت "مرشد الدعم"، مساعد ذكي ودود وخبير.  
- رحب بالمستخدم إذا لزم الأمر.  
- حاول فهم الأسئلة مهما كانت معقدة أو مختصرة.  
- التزم بالسياق فقط، وإذا لم تجد الجواب قل: "بحثت في قاعدة المعرفة، ولكن لم أجد إجابة واضحة."  
- استخدم لغة طبيعية قريبة من الإنسان.  

السياق:
{context}

السؤال: {input}
الإجابة:""")

# --- 4. أسئلة تفاعلية اجتماعية ---
SOCIAL_QS = {
    "سلام": ["السلام عليكم", "هلا", "مرحبا", "اهلا", "السلام"],
    "شكر": ["شكرا", "مشكور", "جزاك الله خير", "متشكر"]
}

SOCIAL_RESPONSES = {
    "سلام": "وعليكم السلام! كيف أستطيع مساعدتك اليوم؟",
    "شكر": "على الرحب والسعة! 😊"
}

def check_social(question: str):
    for key, variants in SOCIAL_QS.items():
        for v in variants:
            if fuzz.partial_ratio(question, v) > 80:
                return SOCIAL_RESPONSES[key]
    return None

# --- 5. دالة FAQ ---
def faq_lookup(question: str, tenant_id: str, threshold: int = 80):
    if not bm25 or not faq_items:
        return False, {}
    # فلترة حسب tenant_id
    relevant_faqs = [f for f in faq_items if f.get("tenant_id") == tenant_id]
    if not relevant_faqs:
        return False, {}
    q_norm = normalize_ar(question)
    corpus_tokens = [normalize_ar(item["title"] + " " + " ".join(item.get("question_variants", []))).split()
                     for item in relevant_faqs]
    local_bm25 = BM25Okapi(corpus_tokens)
    scores = local_bm25.get_scores(q_norm.split())
    top_idx = scores.argmax()
    if scores[top_idx] >= threshold:
        return True, relevant_faqs[top_idx]
    # fallback: fuzzy match
    for item in relevant_faqs:
        text = item["title"] + " " + " ".join(item.get("question_variants", []))
        if fuzz.token_set_ratio(question, text) > threshold:
            return True, item
    return False, {}

# --- 6. تهيئة الوكيل ---
async def initialize_agent():
    global llm, embeddings, vector_store, bm25, faq_items
    async with initialization_lock:
        if vector_store is not None:
            return
        logging.info("بدء تهيئة النماذج وFAQ وFAISS...")
        try:
            # --- التحقق من الاتصال ---
            async with httpx.AsyncClient() as client:
                await client.get(OLLAMA_HOST, timeout=10.0)

            # --- تهيئة LLM وEmbeddings ---
            llm = Ollama(model=CHAT_MODEL, base_url=OLLAMA_HOST, temperature=0.1)
            embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_HOST)

            # --- تحميل FAISS ---
            if not os.path.isdir(UNIFIED_DB_PATH):
                raise FileNotFoundError(f"قاعدة البيانات الموحدة غير موجودة. يرجى تشغيل سكرت 'main_builder.py' أولاً.")

            vector_store = await asyncio.to_thread(
                FAISS.load_local, UNIFIED_DB_PATH, embeddings, allow_dangerous_deserialization=True
            )

            # --- تحميل FAQ ---
            with open(FAQ_PATH, "r", encoding="utf-8") as f:
                kb_yaml = yaml.safe_load(f)
                faq_items = kb_yaml.get("faqs", [])

            # --- إعداد BM25 عام للـ FAQ ---
            corpus_tokens = [normalize_ar(item["title"] + " " + " ".join(item.get("question_variants", []))).split()
                             for item in faq_items]
            bm25 = BM25Okapi(corpus_tokens)

            logging.info("✅ الوكيل جاهز للعمل: FAISS + FAQ + BM25 + تفاعل اجتماعي.")
        except Exception as e:
            logging.error(f"فشل أثناء التهيئة: {e}", exc_info=True)
            raise

# --- 7. دالة get_answer_stream محسّنة ---
async def get_answer_stream(request_info: dict) -> AsyncGenerator[Dict, None]:
    question = request_info["question"].strip()
    tenant_id = request_info.get("tenant_id")
    session_id = tenant_id or "default_session"

    if not vector_store:
        yield {"type": "error", "content": "الوكيل غير جاهز. يرجى إعادة تحميل الصفحة."}
        return

    user_chat_history = chat_history.get(session_id, [])

    # --- 1) تحقق من التحية أو الشكر أولاً ---
    social_resp = check_social(question)
    if social_resp:
        user_chat_history.append(HumanMessage(content=question))
        user_chat_history.append(AIMessage(content=social_resp))
        chat_history[session_id] = user_chat_history[-10:]
        yield {"type": "social", "content": social_resp}
        return

    # --- 2) تحقق FAQ ---
    isfaq, faq_item = faq_lookup(question, tenant_id)
    if isfaq:
        answer_text = faq_item.get("answer", "لا يوجد جواب محدد في FAQ.")
        user_chat_history.append(HumanMessage(content=question))
        user_chat_history.append(AIMessage(content=answer_text))
        chat_history[session_id] = user_chat_history[-10:]
        yield {"type": "faq", "content": answer_text}
        return

    # --- 3) retrieval هجيني: FAISS + LLM ---
    retriever = vector_store.as_retriever(
        search_kwargs={'k': 5, 'filter': {'tenant_id': tenant_id}}  # تقليل k لتسريع
    )
    history_aware_retriever = create_history_aware_retriever(llm, retriever, REPHRASE_PROMPT)
    document_chain = create_stuff_documents_chain(llm, ANSWER_PROMPT)
    conversational_rag_chain = create_retrieval_chain(history_aware_retriever, document_chain)

    logging.info(f"[{session_id}] بدء معالجة السؤال '{question}' عبر pipeline هجيني...")
    try:
        full_answer = ""
        async for chunk in conversational_rag_chain.astream({"input": question, "chat_history": user_chat_history}):
            if "answer" in chunk and chunk["answer"] is not None:
                answer_chunk = chunk["answer"]
                full_answer += answer_chunk
                yield {"type": "chunk", "content": answer_chunk}

        # تحديث سجل المحادثة
        user_chat_history.append(HumanMessage(content=question))
        user_chat_history.append(AIMessage(content=full_answer))
        chat_history[session_id] = user_chat_history[-10:]

        logging.info(f"[{session_id}] الإجابة الكاملة: '{full_answer}'")
    except Exception as e:
        logging.error(f"[{session_id}] فشل في سلسلة RAG. الخطأ: {e}", exc_info=True)
        yield {"type": "error", "content": "عذراً، حدث خطأ فادح."}

