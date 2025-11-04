# src/app/core_logic.py

import os
import logging
import asyncio
import httpx
from typing import AsyncGenerator, Dict, List, Tuple
from dotenv import load_dotenv

from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader, Docx2txtLoader

# --- 1. الإعدادات والتحميل الأولي ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__ ), "../../"))
load_dotenv(dotenv_path=os.path.join(PROJECT_ROOT, ".env"))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL_NAME")
CHAT_MODEL = os.getenv("CHAT_MODEL_NAME")
OLLAMA_HOST = os.getenv("OLLAMA_HOST")

if not all([EMBEDDING_MODEL, CHAT_MODEL, OLLAMA_HOST]):
    raise ValueError("متغيرات البيئة الأساسية مفقودة في ملف .env")

# --- متغيرات عالمية ---
llm: Ollama = None
retriever: FAISS.as_retriever = None
conversational_rag_chain = None
chat_history: Dict[str, List[Tuple[str, str]]] = {} # ذاكرة المحادثة لكل مستخدم
initialization_lock = asyncio.Lock()

# --- 2. قوالب التعليمات الجديدة المبنية على المحادثة ---

# القالب الأول: لإعادة صياغة السؤال بناءً على سجل المحادثة
REPHRASE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "مهمتك هي تحويل السؤال الأخير للمستخدم وسجل المحادثة إلى سؤال مستقل ومحسن ومثالي للبحث في قاعدة البيانات. لا تجب على السؤال، فقط أعد صياغته."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}")
])

# القالب الثاني: لتوليد الإجابة النهائية من السياق وسجل المحادثة
ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """أنت "مرشد الدعم"، مساعد خبير ودود. أجب على سؤال المستخدم بالاعتماد **حصرياً** على "السياق" المقدم.
    - إذا كان السياق فارغاً أو لا يحتوي على إجابة، أجب بأدب: "بحثت في وثائق المشروع، ولكن لم أجد معلومات كافية للإجابة على هذا السؤال."
    - لا تستخدم ذاكرتك العامة أبداً.
    - كن مختصراً ومباشراً."""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "السؤال: {input}\n\nالسياق:\n{context}")
])

# --- 3. الدوال الأساسية للنظام ---

async def initialize_agent():
    global llm, retriever, conversational_rag_chain
    async with initialization_lock:
        if llm is not None: return

        logging.info("بدء عملية التهيئة الكاملة للوكيل...")
        
        try:
            async with httpx.AsyncClient( ) as client:
                await client.get(OLLAMA_HOST, timeout=10.0)
            logging.info("فحص الاتصال ناجح: خدمة Ollama متاحة.")
        except Exception as e:
            raise RuntimeError(f"فشل فحص الاتصال بـ Ollama: {e}")
        
        llm = Ollama(model=CHAT_MODEL, base_url=OLLAMA_HOST, temperature=0.1)
        embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_HOST)
        logging.info("تم تهيئة نماذج Ollama.")

        docs_path = os.path.abspath(os.path.join(PROJECT_ROOT, "4_client_docs"))
        if not os.path.isdir(docs_path):
            raise FileNotFoundError(f"مجلد المستندات '4_client_docs' غير موجود.")
        
        def get_loader(file_path: str):
            ext = file_path.lower().split('.')[-1]
            if ext == 'pdf': return PyPDFLoader(file_path)
            if ext == 'docx': return Docx2txtLoader(file_path)
            return TextLoader(file_path, encoding='utf-8', autodetect_encoding=True)

        loader = DirectoryLoader(docs_path, glob="**/*.*", loader_cls=get_loader, show_progress=True, use_multithreading=True, silent_errors=True)
        docs = loader.load()
        
        if not docs: raise ValueError("لم يتم العثور على أي مستندات قابلة للقراءة.")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=250)
        splits = text_splitter.split_documents(docs)
        logging.info(f"تم تقسيم المستندات إلى {len(splits)} جزء.")

        vector_store = FAISS.from_documents(splits, embeddings)
        retriever = vector_store.as_retriever(search_kwargs={"k": 10})

        # --- بناء السلسلة الكاملة ذات الذاكرة ---
        history_aware_retriever = create_history_aware_retriever(llm, retriever, REPHRASE_PROMPT)
        document_chain = create_stuff_documents_chain(llm, ANSWER_PROMPT)
        conversational_rag_chain = create_retrieval_chain(history_aware_retriever, document_chain)
        
        logging.info("✅ الوكيل جاهز للعمل مع ذاكرة المحادثة.")

# --- 4. دالة get_answer_stream ---
async def get_answer_stream(request_info: dict) -> AsyncGenerator[Dict, None]:
    question = request_info["question"].strip()
    tenant_id = request_info.get("tenant_id", "default_user")

    if not conversational_rag_chain:
        yield {"type": "error", "content": "الوكيل غير جاهز. يرجى المحاولة مرة أخرى."}
        return

    # استرداد سجل المحادثة لهذا المستخدم
    user_chat_history = chat_history.get(tenant_id, [])

    logging.info(f"بدء معالجة السؤال '{question}' مع سجل محادثة بحجم {len(user_chat_history)}")
    
    try:
        full_answer = ""
        # استدعاء السلسلة مع سجل المحادثة
        async for chunk in conversational_rag_chain.astream({"input": question, "chat_history": user_chat_history}):
            if "answer" in chunk:
                answer_chunk = chunk["answer"]
                full_answer += answer_chunk
                yield {"type": "chunk", "content": answer_chunk}
        
        # تحديث سجل المحادثة
        user_chat_history.append((question, full_answer))
        # الحفاظ على آخر 5 محادثات فقط لمنع الذاكرة من الانفجار
        chat_history[tenant_id] = user_chat_history[-5:]

        logging.info(f"الإجابة الكاملة: '{full_answer}'")

    except Exception as e:
        logging.error(f"فشل في سلسلة RAG للمحادثة. الخطأ: {e}", exc_info=True)
        yield {"type": "error", "content": "عذراً، حدث خطأ فادح أثناء معالجة طلبك."}
