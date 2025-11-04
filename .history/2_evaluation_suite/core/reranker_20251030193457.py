# 2_evaluation_suite/core/reranker.py

from typing import List, Dict, Any
from sentence_transformers import CrossEncoder

# --- تهيئة النموذج ---
# سيتم تحميل النموذج تلقائيًا عند أول استخدام وتخزينه مؤقتًا.
# اخترنا نموذجًا متوازنًا من حيث السرعة والدقة.
print("[*] جارٍ تهيئة نموذج إعادة الترتيب (Cross-Encoder)... قد يستغرق بعض الوقت في المرة الأولى.")
try:
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')
    print("[✅] نموذج إعادة الترتيب جاهز.")
except Exception as e:
    print(f"[❌] فشل في تحميل نموذج Cross-Encoder. تأكد من اتصالك بالإنترنت ومن تثبيت 'sentence-transformers'. الخطأ: {e}")
    cross_encoder = None

def rerank_documents(query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    تأخذ سؤالاً وقائمة من المستندات المسترجعة، وتعيد ترتيبها باستخدام Cross-Encoder.

    Args:
        query (str): سؤال المستخدم الأصلي.
        documents (List[Dict[str, Any]]): قائمة المستندات من المسترجع الأولي.
                                           يجب أن يحتوي كل قاموس على مفتاح 'content'.

    Returns:
        List[Dict[str, Any]]: قائمة المستندات مرتبة حسب درجة الدقة الجديدة.
    """
    if not cross_encoder:
        raise RuntimeError("نموذج Cross-Encoder غير متاح.")
    if not documents:
        return []

    # --- 1. إعداد الأزواج للنموذج ---
    # النموذج يتوقع قائمة من الأزواج: [ (السؤال, محتوى المستند), (السؤال, محتوى المستند), ... ]
    model_input = [(query, doc['content']) for doc in documents]

    # --- 2. حساب درجات الدقة ---
    # يقوم النموذج بحساب درجة لكل زوج
    scores = cross_encoder.predict(model_input)

    # --- 3. إضافة الدرجات إلى المستندات ---
    for i, doc in enumerate(documents):
        doc['rerank_score'] = float(scores[i])
        doc['original_rank'] = i + 1 # نحتفظ بالترتيب الأصلي للمقارنة

    # --- 4. ترتيب المستندات بناءً على الدرجة الجديدة ---
    # نرتب تنازلياً من الأعلى درجة إلى الأقل
    reranked_docs = sorted(documents, key=lambda x: x['rerank_score'], reverse=True)

    return reranked_docs
