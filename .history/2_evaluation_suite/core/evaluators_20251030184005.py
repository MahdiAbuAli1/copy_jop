# 2_evaluation_suite/core/evaluators.py

from typing import List, Dict, Any

def evaluate_retrieval(
    retrieved_docs: List[Dict[str, Any]],
    expected_keywords: List[str],
    expected_source: str
) -> Dict[str, Any]:
    """
    تقوم بتقييم جودة المستندات المسترجعة بناءً على الكلمات المفتاحية والمصدر المتوقع.

    Args:
        retrieved_docs (List[Dict[str, Any]]): قائمة بالمستندات المسترجعة، كل مستند هو قاموس.
        expected_keywords (List[str]): قائمة الكلمات المفتاحية المتوقعة.
        expected_source (str): اسم الملف المصدر المتوقع.

    Returns:
        Dict[str, Any]: قاموس يحتوي على نتائج التقييم.
    """
    # --- التقييم بناءً على الكلمات المفتاحية ---
    found_keywords = set()
    if expected_keywords:
        # دمج محتوى جميع المستندات المسترجعة في نص واحد كبير للبحث
        full_retrieved_text = " ".join([doc['content'] for doc in retrieved_docs])
        
        for keyword in expected_keywords:
            if keyword.lower() in full_retrieved_text.lower():
                found_keywords.add(keyword)
        
        keyword_score = len(found_keywords) / len(expected_keywords)
        keyword_details = {
            "score": f"{len(found_keywords)}/{len(expected_keywords)}",
            "found": sorted(list(found_keywords)),
            "missing": sorted(list(set(expected_keywords) - found_keywords))
        }
    else:
        # إذا لم تكن هناك كلمات مفتاحية متوقعة (أسئلة خارج النطاق)
        keyword_score = 1.0  # يعتبر ناجحاً لأنه لا يجب أن يجد شيئاً
        keyword_details = "لا توجد كلمات مفتاحية متوقعة."

    # --- التقييم بناءً على المصدر الصحيح ---
    # هل المصدر المتوقع موجود في قائمة مصادر المستندات المسترجعة؟
    is_source_found = any(expected_source in doc['source'] for doc in retrieved_docs)
    
    # --- حساب النتيجة الإجمالية (يمكن تطويرها لاحقاً) ---
    # حالياً، نعتبر النجاح هو العثور على المصدر الصحيح و 50% على الأقل من الكلمات المفتاحية
    final_score = (keyword_score + (1 if is_source_found else 0)) / 2
    
    if final_score >= 0.75:
        status = "✅ نجاح"
    elif final_score >= 0.5:
        status = "⚠️ جيد جزئيًا"
    else:
        status = "❌ فشل"

    return {
        "status": status,
        "final_score": round(final_score, 2),
        "source_check": "✅" if is_source_found else "❌",
        "keyword_evaluation": keyword_details
    }

