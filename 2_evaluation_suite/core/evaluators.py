# 2_evaluation_suite/core/evaluators.py (النسخة المحدثة)

from typing import List, Dict, Any

def evaluate_retrieval(
    retrieved_docs: List[Dict[str, Any]],
    expected_keywords: List[str],
    expected_source: str
) -> Dict[str, Any]:
    """
    تقوم بتقييم جودة المستندات المسترجعة بناءً على الكلمات المفتاحية والمصدر المتوقع.
    تم تحديث المعادلة لتعطي تقييماً أكثر دقة.
    """
    # --- التقييم بناءً على الكلمات المفتاحية ---
    found_keywords = set()
    keyword_score = 0.0
    
    if expected_keywords:
        full_retrieved_text = " ".join([doc.get('content', '') for doc in retrieved_docs])
        
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
        keyword_score = 1.0
        keyword_details = "لا توجد كلمات مفتاحية متوقعة."

    # --- التقييم بناءً على المصدر الصحيح ---
    is_source_found = any(expected_source in doc.get('source', '') for doc in retrieved_docs)
    source_score = 1.0 if is_source_found else 0.0
    
    # --- المعادلة الجديدة والمحسنة لحساب النتيجة النهائية ---
    # 60% من الدرجة للعثور على المصدر الصحيح (الأهم)
    # 40% من الدرجة للعثور على الكلمات المفتاحية
    final_score = (0.6 * source_score) + (0.4 * keyword_score)
    
    if final_score >= 0.9:  # نرفع معيار النجاح الكامل
        status = "✅ نجاح"
    elif final_score >= 0.6: # يعتبر جيداً إذا وجد المصدر الصحيح وبعض الكلمات
        status = "⚠️ جيد جزئيًا"
    else:
        status = "❌ فشل"

    return {
        "status": status,
        "final_score": round(final_score, 2),
        "source_check": "✅" if is_source_found else "❌",
        "keyword_evaluation": keyword_details
    }
