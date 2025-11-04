# هل نسترجع المستندات الصحيحة
import os
import json
import logging
import asyncio

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain_core.documents import Document

# --- 1. الإعدادات ---
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
load_dotenv(dotenv_path=os.path.join(PROJECT_ROOT, ".env"))

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL_NAME", "qwen3-embedding:0.6b")
OLLAMA_HOST = os.getenv("OLLAMA_HOST")
UNIFIED_DB_PATH = os.path.join(PROJECT_ROOT, "3_shared_resources", "vector_db")
SCENARIOS_PATH = os.path.join(PROJECT_ROOT, "5_pro_testing_suite", "scenarios.json")

# --- 2. الدوال المساعدة ---
def load_scenarios():
    """تحميل سيناريوهات الاختبار من ملف JSON."""
    try:
        with open(SCENARIOS_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"فشل في تحميل ملف السيناريوهات: {e}")
        return []

def load_all_docs_from_faiss(vs: FAISS) -> list[Document]:
    """استخراج جميع المستندات من قاعدة بيانات FAISS."""
    return list(vs.docstore._dict.values())

def evaluate_retrieval(docs: list[Document], expected_keywords: list[str]) -> tuple[int, int]:
    """تقييم مدى صلة المستندات المسترجعة بالكلمات المفتاحية المتوقعة."""
    if not expected_keywords:
        return 0, 0 # لا يمكن التقييم إذا لم تكن هناك كلمات متوقعة

    found_keywords = set()
    for doc in docs:
        content = doc.page_content.lower()
        for keyword in expected_keywords:
            if keyword.lower() in content:
                found_keywords.add(keyword)
    
    score = len(found_keywords)
    total = len(expected_keywords)
    return score, total

# --- 3. دالة الاختبار الرئيسية ---
async def test_retrieval_quality():
    """
    تختبر جودة الاسترجاع لكل سيناريو للتأكد من أن المستندات ذات الصلة يتم إحضارها.
    """
    print("\n--- 🔬 اختبار جودة الاسترجاع (المرحلة 2) 🔬 ---")
    
    # --- التهيئة ---
    logging.info("بدء تهيئة قاعدة البيانات والنماذج...")
    try:
        embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_HOST)
        if not os.path.isdir(UNIFIED_DB_PATH):
            raise FileNotFoundError(f"قاعدة البيانات الموحدة غير موجودة في المسار: {UNIFIED_DB_PATH}")
        vector_store = FAISS.load_local(UNIFIED_DB_PATH, embeddings, allow_dangerous_deserialization=True)
        all_docs = load_all_docs_from_faiss(vector_store)
        logging.info("✅ قاعدة البيانات والنماذج جاهزة.")
    except Exception as e:
        logging.error(f"❌ فشل فادح في التهيئة: {e}")
        return

    scenarios = load_scenarios()
    if not scenarios:
        return

    # --- بدء الاختبارات ---
    for scenario_group in scenarios:
        tenant_id = scenario_group['tenant_id']
        system_name = scenario_group['system_name']
        print("\n" + "="*80)
        logging.info(f"📂 بدء اختبارات العميل: {tenant_id} ({system_name})")
        print("="*80)

        # فلترة المستندات الخاصة بالعميل الحالي فقط
        tenant_docs = [doc for doc in all_docs if doc.metadata.get("tenant_id") == tenant_id]
        if not tenant_docs:
            logging.warning(f"⚠️ لا توجد مستندات للعميل '{tenant_id}' في قاعدة البيانات. تم تخطي اختباراته.")
            continue
        
        # إعداد المسترجع الهجين لهذا العميل
        bm25_retriever = BM25Retriever.from_documents(tenant_docs, k=10)
        faiss_retriever = vector_store.as_retriever(search_kwargs={'k': 10, 'filter': {'tenant_id': tenant_id}})
        ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, faiss_retriever], weights=[0.5, 0.5])

        for test in scenario_group['tests']:
            test_id = test['id']
            question = test['question']
            expected_keywords = test['expected_keywords']
            
            print(f"\n--- 🧪 اختبار [{test_id}]: {question} ---")
            
            # تنفيذ الاسترجاع
            retrieved_docs = await ensemble_retriever.ainvoke(question)
            
            # طباعة النتائج
            if not retrieved_docs:
                logging.warning("   -> ⚠️ لم يتم استرجاع أي مستندات لهذا السؤال.")
            else:
                print(f"   -> 📄 تم استرجاع {len(retrieved_docs)} مستند:")
                for i, doc in enumerate(retrieved_docs[:5]): # طباعة أول 5 فقط للاختصار
                    preview = doc.page_content.strip().replace('\n', ' ')[:120]
                    print(f"      {i+1}. \"{preview}...\"")

            # التقييم
            if not expected_keywords:
                 logging.info("   -> 📊 تقييم: لا توجد كلمات مفتاحية متوقعة (سؤال خارج السياق)، التقييم غير مطلوب.")
            else:
                score, total = evaluate_retrieval(retrieved_docs, expected_keywords)
                if score == total:
                    logging.info(f"   -> ✅ تقييم: ممتاز! تم العثور على جميع الكلمات المفتاحية المتوقعة ({score}/{total}).")
                elif score > 0:
                    logging.warning(f"   -> ⚠️ تقييم: جيد جزئيًا. تم العثور على ({score}/{total}) من الكلمات المفتاحية.")
                else:
                    logging.error(f"   -> ❌ تقييم: فشل. لم يتم العثور على أي من الكلمات المفتاحية المتوقعة ({score}/{total}).")

    print("\n--- 🏁 انتهى اختبار جودة الاسترجاع 🏁 ---")


if __name__ == "__main__":
    # ملاحظة: قد تحتاج إلى تثبيت aiohttp إذا لم يكن مثبتًا
    # pip install aiohttp
    asyncio.run(test_retrieval_quality( ))
