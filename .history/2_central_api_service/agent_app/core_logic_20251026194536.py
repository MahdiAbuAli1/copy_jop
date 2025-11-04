# src/app/core_logic.py

import os
import logging
import asyncio
import httpx
from typing import AsyncGenerator, Dict, List
from dotenv import load_dotenv

# --- استيراد المكتبات الجديدة والمهمة ---
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
# --- تم إضافة Docx2txtLoader ---
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
ensemble_retriever: EnsembleRetriever = None
initialization_lock = asyncio.Lock()

# --- 2. قالب التعليمات النهائي (لا تغيير هنا) ---
ANSWER_PROMPT = PromptTemplate.from_template(
    """### المهمة الأساسية:
أنت "مرشد الدعم"، مساعد خبير في وثائق مشروع "Plant Care". مهمتك هي الإجابة على سؤال المستخدم بدقة وموضوعية بالاعتماد **حصرياً** على "السياق" المقدم.

### التعليمات الصارمة:
1.  **الالتزام المطلق بالسياق:** إذا كانت المعلومات موجودة في "السياق"، قم بصياغة إجابة واضحة منها.
2.  **قاعدة "لا أعرف":** إذا كان "السياق" فارغاً أو لا يحتوي على إجابة، يجب أن تكون إجابتك **فقط**: "بحثت في وثائق المشروع، ولكن لم أجد معلومات كافية للإجابة على هذا السؤال."
3.  **ممنوع الهلوسة:** لا تخترع أي معلومات. لا تستخدم ذاكرتك العامة.
4.  **الأسئلة العامة:** إذا كان السؤال عاماً (مثل "من هو ميسي؟") والسياق فارغ، طبق قاعدة "لا أعرف".

### السياق (المصدر الوحيد للمعلومات):
{context}

### السؤال:
{question}

### الإجابة الدقيقة (من السياق حصراً):
"""
)

# --- 3. الدوال الأساسية للنظام ---

async def initialize_agent():
    global llm, ensemble_retriever
    async with initialization_lock:
        if llm is not None: return

        logging.info("بدء عملية التهيئة الكاملة للوكيل...")
        
        try:
            async with httpx.AsyncClient( ) as client:
                await client.get(OLLAMA_HOST, timeout=5.0)
            logging.info("فحص الاتصال ناجح: خدمة Ollama متاحة.")
        except Exception as e:
            logging.error(f"فشل فحص الاتصال بـ Ollama: {e}")
            raise RuntimeError("فشل تهيئة الوكيل بسبب عدم توفر خدمة Ollama.")
        
        llm = Ollama(model=CHAT_MODEL, base_url=OLLAMA_HOST, temperature=0.0)
        embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_HOST)
        logging.info("تم تهيئة نماذج Ollama بنجاح.")

        docs_path = os.path.abspath(os.path.join(PROJECT_ROOT, "4_client_docs"))
        if not os.path.isdir(docs_path):
            raise FileNotFoundError(f"مجلد المستندات '4_client_docs' غير موجود.")
            
        logging.info(f"جارٍ تحميل المستندات من المسار: {docs_path}")
        
        # --- تم تحديث هذا الجزء ليصبح أكثر ذكاءً ---
        def get_loader(file_path: str):
            """يختار المحمل المناسب بناءً على امتداد الملف."""
            if file_path.lower().endswith('.pdf'):
                return PyPDFLoader(file_path)
            elif file_path.lower().endswith('.docx'):
                return Docx2txtLoader(file_path)
            else: # الافتراضي هو التعامل معه كملف نصي
                return TextLoader(file_path, encoding='utf-8', autodetect_encoding=True)

        loader = DirectoryLoader(
            docs_path, 
            glob="**/*.*", 
            loader_cls=get_loader, 
            show_progress=True, 
            use_multithreading=True,
            silent_errors=True # تجاهل الملفات التي لا يمكن قراءتها
        )
        docs = loader.load()
        
        if not docs:
            raise ValueError("لم يتم العثور على أي مستندات قابلة للقراءة في مجلد '4_client_docs'.")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        logging.info(f"تم تقسيم المستندات إلى {len(splits)} جزء.")

        logging.info("جارٍ بناء الباحث الهجين...")
        bm25_retriever = BM25Retriever.from_documents(splits)
        bm25_retriever.k = 5

        faiss_vectorstore = FAISS.from_documents(splits, embeddings)
        faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": 5})

        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, faiss_retriever],
            weights=[0.5, 0.5]
        )
        logging.info("✅ الوكيل جاهز للعمل مع الباحث الهجين الذكي.")


# --- 4. دالة get_answer_stream (لا تغيير هنا) ---
async def get_answer_stream(request_info: dict) -> AsyncGenerator[Dict, None]:
    question = request_info["question"].strip()

    if not llm or not ensemble_retriever:
        yield {"type": "error", "content": "الوكيل غير جاهز. يرجى المحاولة مرة أخرى بعد قليل."}
        return

    def format_docs(docs: List) -> str:
        unique_contents = {doc.page_content for doc in docs}
        return "\n\n---\n\n".join(unique_contents)

    rag_chain = (
        {"context": ensemble_retriever | format_docs, "question": RunnablePassthrough()}
        | ANSWER_PROMPT
        | llm
        | StrOutputParser()
    )

    logging.info(f"بدء معالجة السؤال '{question}' باستخدام الباحث الهجين...")
    try:
        full_answer = ""
        async for chunk in rag_chain.astream(question):
            full_answer += chunk
            yield {"type": "chunk", "content": chunk}
        logging.info(f"الإجابة الكاملة: '{full_answer}'")

    except Exception as e:
        logging.error(f"فشل في سلسلة RAG. الخطأ: {e}", exc_info=True)
        yield {"type": "error", "content": "عذراً، حدث خطأ فادح أثناء معالجة طلبك."}
