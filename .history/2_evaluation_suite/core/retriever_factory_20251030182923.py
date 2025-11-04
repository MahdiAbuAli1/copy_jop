# 2_evaluation_suite/core/retriever_factory.py

import os
from typing import List, Literal
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_community.embeddings import OllamaEmbeddings

# --- تعريف المسارات الثابتة ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VECTOR_DB_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../3_shared_resources/vector_db/"))

# --- تعريف أنواع المسترجعات المدعومة ---
# استخدام Literal يوفر فحصًا للأنواع ويجعل الكود أكثر وضوحًا
RetrieverType = Literal["ensemble", "faiss", "bm25"]

def get_retriever(
    retriever_type: RetrieverType,
    docs_for_bm25: List[Document],
    embedding_model_name: str,
    k: int = 5
) -> BaseRetriever:
    """
    مصنع لإنشاء وإرجاع أنواع مختلفة من المسترجعات.

    Args:
        retriever_type (RetrieverType): نوع المسترجع المطلوب ("ensemble", "faiss", "bm25").
        docs_for_bm25 (List[Document]): قائمة المستندات الكاملة، مطلوبة فقط لتهيئة BM25.
        embedding_model_name (str): اسم نموذج التضمين المستخدم (مثل 'qwen2-embedding:0.5b').
        k (int): عدد المستندات التي يجب على كل مسترجع إعادتها.

    Returns:
        BaseRetriever: كائن المسترجع المهيأ والجاهز للاستخدام.
    """
    print(f"\n🔧 جارٍ تهيئة المسترجع من نوع: '{retriever_type}'...")

    # --- تهيئة نموذج التضمين (مطلوب لـ FAISS) ---
    embeddings_model = OllamaEmbeddings(model=embedding_model_name)

    # --- تهيئة مسترجع FAISS (البحث الدلالي) ---
    try:
        if not os.path.exists(os.path.join(VECTOR_DB_DIR, "index.faiss")):
            raise FileNotFoundError("قاعدة بيانات FAISS غير موجودة. يرجى تشغيل خط أنابيب بناء المعرفة أولاً.")
        
        faiss_db = FAISS.load_local(
            VECTOR_DB_DIR, 
            embeddings=embeddings_model, 
            allow_dangerous_deserialization=True
        )
        faiss_retriever = faiss_db.as_retriever(search_kwargs={"k": k})
        print("  - ✅ تم تهيئة مسترجع FAISS بنجاح.")
    except Exception as e:
        print(f"  - ❌ فشل في تهيئة مسترجع FAISS. الخطأ: {e}")
        raise

    # --- تهيئة مسترجع BM25 (البحث بالكلمات المفتاحية) ---
    # BM25 يحتاج إلى المستندات الأصلية لإنشاء الفهرس في الذاكرة
    if retriever_type in ["ensemble", "bm25"]:
        if not docs_for_bm25:
            raise ValueError("قائمة المستندات (docs_for_bm25) مطلوبة لتهيئة مسترجع BM25.")
        bm25_retriever = BM25Retriever.from_documents(docs_for_bm25)
        bm25_retriever.k = k
        print("  - ✅ تم تهيئة مسترجع BM25 بنجاح.")

    # --- اختيار وإرجاع المسترجع المطلوب ---
    if retriever_type == "faiss":
        print(f"[*] تم إرجاع مسترجع FAISS.")
        return faiss_retriever
    
    elif retriever_type == "bm25":
        print(f"[*] تم إرجاع مسترجع BM25.")
        return bm25_retriever
        
    elif retriever_type == "ensemble":
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, faiss_retriever],
            weights=[0.5, 0.5]  # إعطاء وزن متساوٍ لكلا المسترجعين
        )
        print(f"[*] تم إرجاع مسترجع Ensemble (FAISS + BM25).")
        return ensemble_retriever
        
    else:
        raise ValueError(f"نوع المسترجع '{retriever_type}' غير مدعوم. الأنواع المدعومة هي: 'ensemble', 'faiss', 'bm25'.")

