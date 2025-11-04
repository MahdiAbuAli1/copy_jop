# core_logic.py (النسخة النهائية - مع التوجيه والهوية الديناميكية )

import os
import logging
import time
from typing import List, AsyncGenerator, Dict
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from dotenv import load_dotenv
import langchain
from langchain_core.caches import InMemoryCache
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi

from .performance_tracker import PerformanceLogger

# -----------------------------------------------------------------------------
# 🧩 إعدادات عامة وتسجيل
# -----------------------------------------------------------------------------
perf_logger = PerformanceLogger()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
load_dotenv(dotenv_path=os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.env")))
langchain.llm_cache = InMemoryCache()

# -----------------------------------------------------------------------------
# 📦 متغيرات البيئة والنماذج
# -----------------------------------------------------------------------------
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")
CHAT_MODEL_NAME = os.getenv("CHAT_MODEL_NAME")
VECTOR_DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../3_shared_resources/vector_db"))
RERANK_MODEL_NAME = "BAAI/bge-reranker-base"
# --- التعديل: قراءة عنوان خادم Ollama من متغيرات البيئة ---
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434" )

# -----------------------------------------------------------------------------
# 🧠 قوالب الـ Prompts (مع دعم الشخصية الديناميكية)
# -----------------------------------------------------------------------------

# --- 1. قالب التوجيه (Classifier) ---
ROUTING_PROMPT_TEMPLATE = """
مهمتك هي تصنيف سؤال المستخدم إلى أحد الفئتين التاليتين: "technical" أو "general".
- "technical": إذا كان السؤال يتطلب البحث عن معلومات أو تفاصيل في قاعدة معرفة. (مثل: من هو المشرف، ما هو الرقم الأكاديمي، كيف أحل المشكلة).
- "general": إذا كان السؤال عبارة عن تحية، سؤال عام لا يتطلب بحث (مثل "من أنت؟"، "كيف حالك؟")، حديث صغير، أو إهانة.

أجب بصيغة JSON فقط، مع مفتاح "category".

أمثلة:
- سؤال المستخدم: "اشرح لي خطوات تثبيت البرنامج." -> {{"category": "technical"}}
- سؤال المستخدم: "من هو مهدي أبو علي؟" -> {{"category": "technical"}}
- سؤال المستخدم: "مرحباً يا ساعد" -> {{"category": "general"}}
- سؤال المستخدم: "من تكون؟" -> {{"category": "general"}}

سؤال المستخدم:
{question}
"""

# --- 2. قالب نظام RAG التقني ---
RAG_PROMPT_TEMPLATE = """
**مهمتك:** أنت مساعد دعم فني خبير ومختص لـ **{tenant_name}**. استخدم "السياق" التالي للإجابة على "سؤال المستخدم" بدقة.
- إذا كانت المعلومات غير موجودة في السياق، أجب بـ "أنا آسف، لا أملك معلومات كافية للإجابة على هذا السؤال."
- أجب دائمًا باللغة العربية.

**السياق:**
{context}

**سؤال المستخدم:**
{question}

**الإجابة:**
"""

# --- 3. قالب المحادثة العامة (مع شخصية ديناميكية) ---
GENERAL_PROMPT_TEMPLATE = """
**مهمتك:** أنت "ساعد"، المساعد الآلي لـ **{tenant_name}**. أنت ذكي وودود. تفاعل مع "سؤال المستخدم" بطريقة مناسبة ومهذبة.
- إذا كان السؤال "من أنت؟" أو ما شابه: عرّف بنفسك: "أنا ساعد، مساعد الدعم الآلي لـ {tenant_name}. كيف يمكنني خدمتك؟"
- إذا كان السؤال تحية: رد التحية بلطف. (مثال: "وعليكم السلام! أهلاً بك في خدمة الدعم لـ {tenant_name}.")
- إذا كان السؤال إهانة: حافظ على هدوئك ورد باحترافية: "أنا هنا لمساعدتك في أي استفسارات لديك حول {tenant_name}."
- أجب دائمًا باللغة العربية.

سؤال المستخدم:
{question}
"""

# -----------------------------------------------------------------------------
# 🌍 المتغيرات العالمية وسلاسل العمل
# -----------------------------------------------------------------------------
vector_store: FAISS = None
llm: Ollama = None
embeddings_model: OllamaEmbeddings = None
all_docs_for_bm25: List[Document] = []
cross_encoder: CrossEncoder = None
full_rag_chain = None
general_chain = None
routing_chain = None

# -----------------------------------------------------------------------------
# 🚀 تهيئة الوكيل (مع إعادة التوجيه)
# -----------------------------------------------------------------------------
def initialize_agent():
    global vector_store, llm, embeddings_model, all_docs_for_bm25, cross_encoder, full_rag_chain, general_chain, routing_chain
    if routing_chain:
        logging.info("✅ الوكيل الذكي (مع التوجيه) مُهيأ مسبقًا.")
        return
    
    try:
        logging.info("=" * 80)
        logging.info("🚀 بدء تهيئة الوكيل الذكي (مع التوجيه والشخصية الديناميكية)...")
        logging.info(f"🔗 الاتصال بخادم Ollama على: {OLLAMA_HOST}")
        
        # --- التعديل: إضافة base_url لضمان الاتصال بـ Docker ---
        llm = Ollama(model=CHAT_MODEL_NAME, temperature=0.1, base_url=OLLAMA_HOST)
        embeddings_model = OllamaEmbeddings(model=EMBEDDING_MODEL_NAME, base_url=OLLAMA_HOST)
        
        vector_store = FAISS.load_local(VECTOR_DB_PATH, embeddings=embeddings_model, allow_dangerous_deserialization=True)
        docstore_ids = list(vector_store.docstore._dict.keys())
        all_docs_for_bm25 = [vector_store.docstore._dict[i] for i in docstore_ids]
        # cross_encoder = CrossEncoder(RERANK_MODEL_NAME)
        
        # --- بناء السلاسل ---
        rag_prompt = PromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
        full_rag_chain = (
            RunnablePassthrough.assign(context=lambda x: format_docs_with_source(x["docs"]))
            | rag_prompt
            | llm
            | StrOutputParser()
        )

        general_prompt = PromptTemplate.from_template(GENERAL_PROMPT_TEMPLATE)
        general_chain = general_prompt | llm | StrOutputParser()

        routing_prompt = PromptTemplate.from_template(ROUTING_PROMPT_TEMPLATE)
        routing_chain = routing_prompt | llm | JsonOutputParser()

        logging.info("✨ اكتملت تهيئة الوكيل الذكي بنجاح! ✨")
    except Exception as e:
        logging.critical(f"❌ فشل حاسم أثناء التهيئة: {e}", exc_info=True)
        raise

# -----------------------------------------------------------------------------
# 헬 دوال مساعدة
# -----------------------------------------------------------------------------
def format_docs_with_source(docs: List[Document]) -> str:
    """تنسق المستندات المسترجعة وتضيف المصادر."""
    if not docs:
        return "لا يوجد سياق متوفر."
    sources = {doc.metadata.get("source", "مصدر غير معروف") for doc in docs}
    formatted_docs = "\n\n---\n\n".join(doc.page_content for doc in docs)
    return f"المعلومات التالية تم استرجاعها من المصادر: {', '.join(sources)}\n\n{formatted_docs}"

def perform_hybrid_retrieval_and_rerank(question: str, tenant_id: str, k: int) -> List[Document]:
    """ينفذ البحث الهجين الكامل مع إعادة الترتيب."""
    faiss_retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 15, "filter": {"tenant_id": tenant_id}})
    faiss_docs = faiss_retriever.invoke(question)
    
    tenant_docs_indices = [i for i, doc in enumerate(all_docs_for_bm25) if doc.metadata.get("tenant_id") == tenant_id]
    bm25_docs = []
    if tenant_docs_indices:
        tenant_corpus = [all_docs_for_bm25[i].page_content.split(" ") for i in tenant_docs_indices]
        bm25_for_tenant = BM25Okapi(tenant_corpus)
        tokenized_query = question.split(" ")
        doc_scores = bm25_for_tenant.get_scores(tokenized_query)
        top_n_indices = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)[:15]
        bm25_docs = [all_docs_for_bm25[tenant_docs_indices[i]] for i in top_n_indices]
    
    combined_docs_list = list({doc.page_content: doc for doc in faiss_docs + bm25_docs}.values())
    if not combined_docs_list:
        return []

    # model_input_pairs = [[question, doc.page_content] for doc in combined_docs_list]
    # scores = cross_encoder.predict(model_input_pairs)
    # docs_with_scores = sorted(zip(combined_docs_list, scores), key=lambda x: x[1], reverse=True)
    
    # return [doc for doc, score in docs_with_scores[:k]]

# -----------------------------------------------------------------------------
# 🧠 بث الإجابة (النسخة النهائية مع الهوية الديناميكية المستنبطة)
# -----------------------------------------------------------------------------
async def get_answer_stream(question: str, tenant_id: str, k_results: int = 4) -> AsyncGenerator[str, None]:
    if not routing_chain:
        raise RuntimeError("⚠️ الوكيل الذكي غير مُهيأ. يرجى استدعاء initialize_agent() أولاً.")
    
    logging.info(f"📩 استقبال سؤال من '{tenant_id}': {question}")
    try:
        # 1. مرحلة التوجيه
        perf_logger.start("routing")
        route_decision = await routing_chain.ainvoke({"question": question})
        category = route_decision.get("category", "technical")
        perf_logger.end("routing", tenant_id, question, extra_info={"decision": category})
        logging.info(f"🧠 قرار التوجيه: '{category}'")

        # 2. تنفيذ المسار
        if category == "technical":
            logging.info("🚀 تنفيذ مسار الدعم الفني (RAG)...")
            perf_logger.start("retrieval_rerank")
            final_docs = perform_hybrid_retrieval_and_rerank(question, tenant_id, k_results)
            perf_logger.end("retrieval_rerank", tenant_id, question, extra_info={"final_doc_count": len(final_docs)})
            
            # استنباط الهوية الديناميكية من المستندات المسترجعة
            entity_name = "الخدمة" # اسم افتراضي
            if final_docs and "entity_name" in final_docs[0].metadata:
                entity_name = final_docs[0].metadata["entity_name"]
            logging.info(f"🏢 الهوية الديناميكية المستنبطة: '{entity_name}'")
            
            async for chunk in full_rag_chain.astream({"question": question, "docs": final_docs, "tenant_name": entity_name}):
                yield chunk
        else: # general
            logging.info("💬 تنفيذ مسار المحادثة العامة...")
            
            # استنباط الهوية الديناميكية عبر بحث خفيف جداً
            temp_docs = vector_store.similarity_search("", filter={"tenant_id": tenant_id}, k=1)
            entity_name = "الخدمة" # اسم افتراضي
            if temp_docs and "entity_name" in temp_docs[0].metadata:
                entity_name = temp_docs[0].metadata["entity_name"]
            logging.info(f"🏢 الهوية الديناميكية المستنبطة: '{entity_name}'")

            async for chunk in general_chain.astream({"question": question, "tenant_name": entity_name}):
                yield chunk
    except Exception as e:
        logging.error(f"❌ خطأ أثناء بث الإجابة: {e}", exc_info=True)
        yield "عذرًا، حدث خطأ داخلي أثناء معالجة سؤالك."
        perf_logger.end("error", tenant_id, question, extra_info={"error": str(e)})

