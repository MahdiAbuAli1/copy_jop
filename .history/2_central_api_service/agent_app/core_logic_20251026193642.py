# src/app/core_logic.py

import os
import logging
import asyncio
import httpx
from async_lru import alru_cache
from typing import AsyncGenerator, Dict
from dotenv import load_dotenv
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- 1. الإعدادات والتحميل الأولي ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__ ), "../../"))
load_dotenv(dotenv_path=os.path.join(PROJECT_ROOT, ".env"))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL_NAME")
CHAT_MODEL = os.getenv("CHAT_MODEL_NAME")
OLLAMA_HOST = os.getenv("OLLAMA_HOST")

if not all([EMBEDDING_MODEL, CHAT_MODEL, OLLAMA_HOST]):
    raise ValueError("متغيرات البيئة الأساسية (EMBEDDING_MODEL_NAME, CHAT_MODEL_NAME, OLLAMA_HOST) مفقودة في ملف .env")

# --- متغيرات عالمية ---
embeddings: OllamaEmbeddings = None
llm: Ollama = None
vector_store: FAISS = None
initialization_lock = asyncio.Lock()

# --- 2. قوالب التعليمات الجديدة (مقسمة ومنطقية) ---

# القالب الأول: لإعادة صياغة السؤال ليكون مثالياً للبحث
REPHRASE_QUESTION_PROMPT = PromptTemplate.from_template(
    """مهمتك هي إعادة صياغة السؤال التالي ليكون أكثر وضوحاً وتحديداً ومثالياً للبحث في قاعدة بيانات تقنية.
    ركز على الكلمات المفتاحية والأسماء المذكورة.
    
    السؤال الأصلي: {question}
    
    السؤال المعاد صياغته للبحث:"""
)

# القالب الثاني: لتوليد الإجابة النهائية من السياق
ANSWER_PROMPT = PromptTemplate.from_template(
    """### المهمة الأساسية:
أنت "مرشد الدعم"، مساعد خبير في وثائق مشروع "Plant Care". مهمتك هي الإجابة على سؤال المستخدم بدقة وموضوعية بالاعتماد **حصرياً** على "السياق" المقدم.

### التعليمات الصارمة:
1.  **الالتزام المطلق بالسياق:** اقرأ "السياق" جيداً. إذا كانت المعلومات موجودة، قم بصياغة إجابة واضحة ومباشرة منها.
2.  **قاعدة "لا أعرف" الإلزامية:** إذا كان "السياق" فارغاً أو لا يحتوي على إجابة للسؤال، يجب أن تكون إجابتك **فقط**: "بحثت في وثائق المشروع، ولكن لم أجد معلومات كافية للإجابة على هذا السؤال."
3.  **ممنوع الهلوسة:** لا تخترع أي معلومات. لا تستخدم ذاكرتك العامة. لا تذكر أي شيء غير موجود في "السياق".
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
    global embeddings, llm, vector_store
    async with initialization_lock:
        if llm is not None: return
        try:
            async with httpx.AsyncClient( ) as client:
                await client.get(OLLAMA_HOST, timeout=5.0)
            logging.info("فحص الاتصال ناجح: خدمة Ollama متاحة.")
        except Exception as e:
            logging.error(f"فشل فحص الاتصال بـ Ollama: {e}")
            raise RuntimeError("فشل تهيئة الوكيل بسبب عدم توفر خدمة Ollama.")
            
        logging.info("جارٍ تهيئة النماذج وقاعدة البيانات...")
        embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_HOST)
        llm = Ollama(model=CHAT_MODEL, base_url=OLLAMA_HOST, temperature=0.0) # درجة حرارة 0 للالتزام بالتعليمات
        
        # تحميل قاعدة البيانات عند بدء التشغيل
        db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../3_shared_resources/vector_db"))
        if not os.path.isdir(db_path):
            raise FileNotFoundError(f"المجلد 'vector_db' غير موجود في المسار المتوقع '{db_path}'.")
        try:
            vector_store = await asyncio.to_thread(
                FAISS.load_local, db_path, embeddings, allow_dangerous_deserialization=True
            )
            logging.info("تم تحميل قاعدة المعرفة بنجاح.")
        except Exception as e:
            logging.error(f"فشل حاسم أثناء تحميل قاعدة بيانات FAISS. الخطأ: {e}")
            raise

# --- 4. دالة get_answer_stream - النسخة النهائية المتقدمة ---
async def get_answer_stream(request_info: dict) -> AsyncGenerator[Dict, None]:
    question = request_info["question"].strip()
    k_results = request_info.get("k_results", 8)

    if not llm or not vector_store:
        yield {"type": "error", "content": "الوكيل غير جاهز. يرجى المحاولة مرة أخرى بعد قليل."}
        return

    retriever = vector_store.as_retriever(search_kwargs={'k': k_results})

    # --- بناء السلسلة المتقدمة (Chain) ---
    # الخطوة 1: إعادة صياغة السؤال
    rephrase_chain = REPHRASE_QUESTION_PROMPT | llm | StrOutputParser()
    
    # الخطوة 2: البحث في قاعدة البيانات باستخدام السؤال المعاد صياغته
    # الخطوة 3: تجميع السياق
    def format_docs(docs):
        return "\n\n---\n\n".join(doc.page_content for doc in docs)

    # الخطوة 4: بناء السلسلة الكاملة
    rag_chain = (
        RunnablePassthrough.assign(
            rephrased_question=rephrase_chain,
        )
        | RunnablePassthrough.assign(
            context=(lambda x: x["rephrased_question"]) | retriever | format_docs
        )
        | ANSWER_PROMPT
        | llm
        | StrOutputParser()
    )

    logging.info(f"بدء معالجة السؤال: '{question}'")
    try:
        # استخدام astream لبث الإجابة
        full_answer = ""
        async for chunk in rag_chain.astream({"question": question}):
            full_answer += chunk
            yield {"type": "chunk", "content": chunk}
        logging.info(f"الإجابة الكاملة للسؤال '{question}': '{full_answer}'")

    except Exception as e:
        logging.error(f"فشل في سلسلة RAG. الخطأ: {e}", exc_info=True)
        yield {"type": "error", "content": "عذراً، حدث خطأ فادح أثناء معالجة طلبك."}

