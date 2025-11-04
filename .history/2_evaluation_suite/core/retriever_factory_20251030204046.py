# 2_evaluation_suite/core/retriever_factory.py (النسخة النهائية مع عزل البيانات)
import os
from typing import List, Dict, Any
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.documents import Document

# --- المسار الجديد لقواعد البيانات ---
VECTOR_DB_BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../3_shared_resources/vector_db/"))

def get_retriever(
    retriever_type: str,
    embeddings_model: Any,
    tenant_id: str, # وسيط جديد لتحديد العميل
    all_docs: List[Document] # مطلوب لـ BM25
) -> Any:
    """
    يقوم بإنشاء وإرجاع نوع المسترجع المطلوب، مع تحميل قاعدة البيانات الخاصة بالعميل المحدد.
    """
    print(f"\n🔧 جارٍ تهيئة المسترجع من نوع: '{retriever_type}' للعميل '{tenant_id}'...")

    # --- التغيير الجوهري: تحديد مسار قاعدة البيانات الخاصة بالعميل ---
    tenant_db_path = os.path.join(VECTOR_DB_BASE_DIR, tenant_id)
    if not os.path.exists(tenant_db_path):
        raise FileNotFoundError(f"خطأ: لم يتم العثور على قاعدة بيانات للعميل '{tenant_id}' في المسار: {tenant_db_path}")

    # --- تحميل قاعدة FAISS الخاصة بالعميل ---
    try:
        faiss_vectorstore = FAISS.load_local(
            tenant_db_path,
            embeddings=embeddings_model,
            allow_dangerous_deserialization=True
        )
        faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": 20})
        print("  - ✅ تم تهيئة مسترجع FAISS بنجاح.")
    except Exception as e:
        raise RuntimeError(f"فشل تحميل قاعدة بيانات FAISS للعميل '{tenant_id}': {e}")

    if retriever_type == "faiss":
        print("[*] تم إرجاع مسترجع FAISS.")
        return faiss_retriever

    # --- تهيئة BM25 من المستندات الخاصة بالعميل فقط ---
    tenant_docs = [doc for doc in all_docs if doc.metadata.get("tenant_id") == tenant_id]
    if not tenant_docs:
         raise ValueError(f"لم يتم العثور على مستندات للعميل '{tenant_id}' لتهيئة BM25.")
         
    bm25_retriever = BM25Retriever.from_documents(tenant_docs)
    bm25_retriever.k = 20
    print("  - ✅ تم تهيئة مسترجع BM25 بنجاح.")

    if retriever_type == "bm25":
        print("[*] تم إرجاع مسترجع BM25.")
        return bm25_retriever

    if retriever_type == "ensemble" or retriever_type == "hybrid":
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, faiss_retriever],
            weights=[0.5, 0.5]
        )
        print("[*] تم إرجاع مسترجع Ensemble (FAISS + BM25).")
        return ensemble_retriever
    
    raise ValueError(f"نوع المسترجع '{retriever_type}' غير معروف.")
