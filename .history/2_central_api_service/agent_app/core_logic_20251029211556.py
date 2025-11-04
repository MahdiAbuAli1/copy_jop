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

import os
import logging
import asyncio
import httpx
from typing import AsyncGenerator, Dict, List, cast

from dotenv import load_dotenv
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document

# --- vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv ---
# --- هذا هو القسم الذي تم تعديله ---
# --- vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv ---

# 1. إضافة استيرادات جديدة للبحث الهجين
from langchain.retrievers import BM25Retriever, EnsembleRetriever

# 2. استخدام المسارات الصحيحة والحديثة لوظائف السلاسل
# هذا يحل مشكلة ImportError
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

# --- ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ ---
# --- نهاية القسم الذي تم تعديله ---
# --- ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ ---

from .performance_tracker import PerformanceLogger

# --- 1. الإعدادات (لا تغيير هنا ) ---
# ... (بقية الكود يبقى كما هو دون أي تغيير) ...
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
load_dotenv(dotenv_path=os.path.join(PROJECT_ROOT, ".env"))
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL_NAME", "qwen3-embedding:0.6b")
CHAT_MODEL = os.getenv("CHAT_MODEL_NAME", "qwen2:7b-instruct-q3_K_M")
OLLAMA_HOST = os.getenv("OLLAMA_HOST")
UNIFIED_DB_PATH = os.path.join(PROJECT_ROOT, "3_shared_resources", "vector_db")

# --- متغيرات عالمية ---
llm: Ollama = None
ensemble_retriever: EnsembleRetriever = None 
chat_history: Dict[str, List[HumanMessage | AIMessage]] = {}
initialization_lock = asyncio.Lock()
perf_logger = PerformanceLogger()

# --- 2. القوالب (لا تغيير هنا) ---
REPHRASE_PROMPT = ChatPromptTemplate.from_template("""
بالنظر إلى سجل المحادثة والسؤال الأخير، قم بصياغة سؤال مستقل يمكن فهمه بدون سجل المحادثة.
سجل المحادثة: {chat_history}
السؤال الأخير: {input}
السؤال المستقل:""")

ANSWER_PROMPT = ChatPromptTemplate.from_template("""
أنت "مرشد الدعم"، مساعد ذكي وخبير. مهمتك هي الإجابة على سؤال المستخدم بالاعتماد **حصرياً** على "السياق" المقدم.
- كن دائماً متعاوناً ومحترفاً.
- إذا كان السياق يحتوي على إجابة، قدمها بشكل مباشر ومنظم.
- إذا كانت المعلومات غير موجودة بشكل واضح في السياق، قل بأسلوب لطيف: "بحثت في قاعدة المعرفة، ولكن لم أجد إجابة واضحة بخصوص هذا السؤال."
- لا تخترع إجابات أبداً. التزم بالسياق.

السياق:
{context}

السؤال: {input}
الإجابة:""")

# --- 3. الدوال الأساسية (لا تغيير هنا) ---
def _load_all_docs_from_faiss(vector_store: FAISS) -> List[Document]:
    return list(cast(dict, vector_store.docstore._dict).values())

async def initialize_agent():
    global llm, ensemble_retriever
    async with initialization_lock:
        if ensemble_retriever is not None: return
        logging.info("بدء تهيئة النماذج والمسترجع الهجين...")
        try:
            async with httpx.AsyncClient( ) as client:
                await client.get(OLLAMA_HOST, timeout=10.0)
            llm = Ollama(model=CHAT_MODEL, base_url=OLLAMA_HOST, temperature=0.1)
            
            logging.info("تحميل قاعدة بيانات FAISS...")
            if not os.path.isdir(UNIFIED_DB_PATH):
                raise FileNotFoundError(f"قاعدة البيانات الموحدة غير موجودة. يرجى تشغيل 'main_builder.py' أولاً.")
            
            embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_HOST)
            faiss_vector_store = await asyncio.to_thread(
                FAISS.load_local, UNIFIED_DB_PATH, embeddings, allow_dangerous_deserialization=True
            )
            faiss_retriever = faiss_vector_store.as_retriever(search_kwargs={'k': 4})
            logging.info("✅ تم تحميل المسترجع الدلالي (FAISS).")

            logging.info("بناء مسترجع الكلمات المفتاحية (BM25)...")
            all_docs = await asyncio.to_thread(_load_all_docs_from_faiss, faiss_vector_store)
            bm25_retriever = BM25Retriever.from_documents(all_docs)
            bm25_retriever.k = 4
            logging.info("✅ تم بناء المسترجع (BM25).")

            ensemble_retriever = EnsembleRetriever(
                retrievers=[bm25_retriever, faiss_retriever],
                weights=[0.5, 0.5]
            )
            logging.info("🚀 الوكيل جاهز للعمل مع المسترجع الهجين.")

        except Exception as e:
            logging.error(f"فشل فادح أثناء التهيئة: {e}", exc_info=True)
            raise

# --- 4. دالة للتحقق من جاهزية الوكيل (لا تغيير هنا) ---
def agent_ready() -> bool:
    return ensemble_retriever is not None and llm is not None

# --- 5. دالة get_answer_stream (لا تغيير هنا) ---
async def get_answer_stream(request_info: Dict) -> AsyncGenerator[Dict, None]:
    question = request_info.get("question", "")
    tenant_id = request_info.get("tenant_id", "default_session")
    
    session_id = tenant_id or "default_session"

    if not ensemble_retriever:
        yield {"type": "error", "content": "الوكيل غير جاهز. يرجى إعادة تحميل الصفحة."}
        return

    perf_logger.start("total_request", tenant_id, question, {"retriever_type": "hybrid"})
    
    user_chat_history = chat_history.get(session_id, [])

    history_aware_retriever = create_history_aware_retriever(llm, ensemble_retriever, REPHRASE_PROMPT)
    document_chain = create_stuff_documents_chain(llm, ANSWER_PROMPT)
    conversational_rag_chain = create_retrieval_chain(history_aware_retriever, document_chain)

    logging.info(f"[{session_id}] بدء معالجة السؤال '{question}'...")
    try:
        full_answer = ""
        perf_logger.start("llm_stream_generation", tenant_id, question)

        async for chunk in conversational_rag_chain.astream({"input": question, "chat_history": user_chat_history}):
            if "answer" in chunk and chunk["answer"] is not None:
                answer_chunk = chunk["answer"]
                full_answer += answer_chunk
                yield {"type": "chunk", "content": answer_chunk}
        
        perf_logger.end("llm_stream_generation", tenant_id, question, {"answer_length": len(full_answer)})

        user_chat_history.append(HumanMessage(content=question))
        user_chat_history.append(AIMessage(content=full_answer))
        chat_history[session_id] = user_chat_history[-10:]
        logging.info(f"[{session_id}] الإجابة الكاملة: '{full_answer}'")
    except Exception as e:
        logging.error(f"[{session_id}] فشل في سلسلة RAG. الخطأ: {e}", exc_info=True)
        yield {"type": "error", "content": "عذراً، حدث خطأ فادح."}
    finally:
        perf_logger.end("total_request", tenant_id, question)
