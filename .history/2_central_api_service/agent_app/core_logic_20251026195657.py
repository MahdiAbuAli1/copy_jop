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
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
# --- هذا هو التغيير الحاسم ---
from langchain_core.messages import HumanMessage, AIMessage

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
# --- الذاكرة الآن ستخزن كائنات الرسائل الصحيحة ---
chat_history: Dict[str, List[HumanMessage | AIMessage]] = {} 
initialization_lock = asyncio.Lock()

# --- 2. قوالب التعليمات الجديدة المبنية على المحادثة ---
REPHRASE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "مهمتك هي تحويل السؤال الأخير للمستخدم وسجل المحادثة إلى سؤال مستقل ومحسن ومثالي للبحث في قاعدة البيانات. لا تجب على السؤال، فقط أعد صياغته."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}")
])

ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """أنت "مرشد الدعم"، مساعد خبير ودود لمشروع يسمى "Plant Care". مهمتك هي الإجابة على سؤال المستخدم بالاعتماد **حصرياً** على "السياق" المقدم.
    - إذا سأل المستخدم "من أنت؟"، أجب: "أنا مرشد الدعم، مساعد ذكي متخصص في الإجابة على أسئلتك حول مشروع Plant Care."
    - إذا كان السياق فارغاً أو لا يحتوي على إجابة، أجب بأدب: "بحثت في وثائق المشروع، ولكن لم أجد معلومات كافية للإجابة على هذا السؤال."
    - لا تستخدم ذاكرتك العامة أبداً. كن مختصراً ومباشراً."""),
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

        history_aware_retriever = create_history_aware_retriever(llm, retriever, REPHRASE_PROMPT)
        document_chain = create_stuff_documents_chain(llm, ANSWER_PROMPT)
        conversational_rag_chain = create_retrieval_chain(history_aware_retriever, document_chain)
        
        logging.info("✅ الوكيل جاهز للعمل مع ذاكرة المحادثة الصحيحة.")

# --- 4. دالة get_answer_stream ---
async def get_answer_stream(request_info: dict) -> AsyncGenerator[Dict, None]:
    question = request_info["question"].strip()
    tenant_id = request_info.get("tenant_id", "default_user")

    if not conversational_rag_chain:
        yield {"type": "error", "content": "الوكيل غير جاهز. يرجى المحاولة مرة أخرى."}
        return

    user_chat_history = chat_history.get(tenant_id, [])

    logging.info(f"بدء معالجة السؤال '{question}' مع سجل محادثة بحجم {len(user_chat_history)}")
    
    try:
        full_answer = ""
        async for chunk in conversational_rag_chain.astream({"input": question, "chat_history": user_chat_history}):
            if "answer" in chunk and chunk["answer"] is not None:
                answer_chunk = chunk["answer"]
                full_answer += answer_chunk
                yield {"type": "chunk", "content": answer_chunk}
        
        # --- تحديث سجل المحادثة بالتنسيق الصحيح ---
        user_chat_history.append(HumanMessage(content=question))
        user_chat_history.append(AIMessage(content=full_answer))
        
        chat_history[tenant_id] = user_chat_history[-10:] # الحفاظ على آخر 5 أزواج من المحادثات

        logging.info(f"الإجابة الكاملة: '{full_answer}'")

    except Exception as e:
        logging.error(f"فشل في سلسلة RAG للمحادثة. الخطأ: {e}", exc_info=True)
        yield {"type": "error", "content": "عذراً، حدث خطأ فادح أثناء معالجة طلبك."}

