# 2_evaluation_suite/core/reranker.py (النسخة النهائية باستخدام النموذج المتوازن)

from typing import List, Dict, Any
from sentence_transformers import CrossEncoder

# --- تهيئة النموذج الجديد والمتوازن ---
# أصغر حجماً وأسرع من 'large' مع الحفاظ على دقة عالية جداً.
MODEL_NAME = 'BAAI/bge-reranker-base' 
print(f"[*] جارٍ تهيئة نموذج إعادة الترتيب (Cross-Encoder): '{MODEL_NAME}'...")
try:
    # نقوم بتمرير max_length لضمان معالجة النصوص الطويلة بشكل جيد
    cross_encoder = CrossEncoder(MODEL_NAME, max_length=512)
    print("[✅] نموذج إعادة الترتيب جاهز.")
except Exception as e:
    print(f"[❌] فشل في تحميل نموذج Cross-Encoder. تأكد من اتصالك بالإنترنت. الخطأ: {e}")
    cross_encoder = None

def rerank_documents(query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    تأخذ سؤالاً وقائمة من المستندات المسترجعة، وتعيد ترتيبها باستخدام Cross-Encoder.
    """
    if not cross_encoder:
        raise RuntimeError("نموذج Cross-Encoder غير متاح.")
    if not documents:
        return []

    # --- 1. إعداد الأزواج للنموذج ---
    model_input = [(query, doc.get('content', '')) for doc in documents]

    # --- 2. حساب درجات الدقة ---
    scores = cross_encoder.predict(model_input)

    # --- 3. إضافة الدرجات إلى المستندات ---
    for i, doc in enumerate(documents):
        doc['rerank_score'] = float(scores[i])
        doc['original_rank'] = i + 1

    # --- 4. ترتيب المستندات بناءً على الدرجة الجديدة ---
    reranked_docs = sorted(documents, key=lambda x: x['rerank_score'], reverse=True)

    return reranked_docs
