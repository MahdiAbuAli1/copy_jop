# 2_evaluation_suite/run_evaluation.py

import os
import json
import argparse
import time
from datetime import datetime
from typing import List, Dict, Any
from dotenv import load_dotenv

# استيراد المكونات الأساسية التي بنيناها
from core.retriever_factory import get_retriever, RetrieverType
from core.evaluators import evaluate_retrieval

# --- إعدادات أساسية ---
load_dotenv()
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_CASES_DIR = os.path.join(BASE_DIR, "test_cases")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
ALL_DOCS_CACHE = {} # ذاكرة تخزين مؤقت لتجنب تحميل المستندات مراراً

def load_all_documents_from_kb() -> List[Dict[str, Any]]:
    """
    يحمل جميع المستندات من قاعدة بيانات FAISS مرة واحدة.
    هذا ضروري لتهيئة BM25 Retriever.
    """
    # هذا جزء متقدم قليلاً، لكنه ضروري. نحن بحاجة إلى كل النصوص لـ BM25.
    # سنقوم بتحميل قاعدة FAISS واستخراج كل المستندات منها.
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import OllamaEmbeddings
    
    db_path = os.path.abspath(os.path.join(BASE_DIR, "../3_shared_resources/vector_db/"))
    if not os.path.exists(os.path.join(db_path, "index.faiss")):
        raise FileNotFoundError("قاعدة بيانات FAISS غير موجودة!")
        
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL_NAME)
    db = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
    
    # استخراج كل المستندات من الفهرس
    # db.docstore._dict هي الطريقة للوصول إلى كل المستندات المخزنة في FAISS
    all_docs = list(db.docstore._dict.values())
    return all_docs


def run_test_for_tenant(
    tenant_id: str,
    retriever_type: RetrieverType,
    all_docs: List[Any]
) -> List[Dict[str, Any]]:
    """
    يشغل جميع حالات الاختبار لعميل واحد باستخدام نوع مسترجع محدد.
    """
    print("\n" + "="*30 + f" 🧪 بدء اختبار العميل: {tenant_id} | النوع: {retriever_type} " + "="*30)
    
    # --- 1. تحميل حالات الاختبار ---
    test_cases_file = os.path.join(TEST_CASES_DIR, f"{tenant_id}_cases.json")
    if not os.path.exists(test_cases_file):
        print(f"⚠️ لم يتم العثور على ملف حالات اختبار للعميل '{tenant_id}'. تم التخطي.")
        return []
    
    with open(test_cases_file, 'r', encoding='utf-8') as f:
        test_cases = json.load(f)
    print(f"  - تم تحميل {len(test_cases)} حالة اختبار.")

    # --- 2. تهيئة المسترجع ---
    # نحتاج إلى تصفية المستندات لتشمل فقط تلك الخاصة بالعميل الحالي لـ BM25
    tenant_specific_docs = [doc for doc in all_docs if doc.metadata.get("tenant_id") == tenant_id]
    if not tenant_specific_docs:
        print(f"⚠️ لم يتم العثور على مستندات في قاعدة المعرفة للعميل '{tenant_id}'. تم التخطي.")
        return []

    retriever = get_retriever(retriever_type, tenant_specific_docs, EMBEDDING_MODEL_NAME, k=5)

    # --- 3. تنفيذ الاختبارات ---
    results = []
    for case in test_cases:
        question = case["question"]
        print(f"\n--- ❓ اختبار [{case['case_id']}]: {question} ---")
        
        start_time = time.time()
        retrieved_docs_langchain = retriever.invoke(question)
        end_time = time.time()
        
        # تحويل كائنات Langchain إلى قواميس بسيطة لسهولة المعالجة والحفظ
        retrieved_docs_simple = [
            {"content": doc.page_content, "source": doc.metadata.get("source", "N/A")}
            for doc in retrieved_docs_langchain
        ]
        
        # --- 4. تقييم النتائج ---
        evaluation = evaluate_retrieval(
            retrieved_docs=retrieved_docs_simple,
            expected_keywords=case["expected_keywords"],
            expected_source=case["expected_source"]
        )
        
        print(f"  - 📊 التقييم: {evaluation['status']} (المصدر: {evaluation['source_check']}, الكلمات: {evaluation['keyword_evaluation']['score']})")
        
        results.append({
            "case_id": case["case_id"],
            "question": question,
            "retrieval_time_seconds": round(end_time - start_time, 2),
            "evaluation": evaluation,
            "retrieved_documents": retrieved_docs_simple
        })
        
    return results


def save_results(tenant_id: str, retriever_type: RetrieverType, results: List[Dict[str, Any]]):
    """
    يحفظ نتائج الاختبار في ملف JSON منظم.
    """
    if not results:
        return
        
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{tenant_id}_{retriever_type}.json"
    filepath = os.path.join(RESULTS_DIR, filename)
    
    report = {
        "report_info": {
            "tenant_id": tenant_id,
            "retriever_type": retriever_type,
            "timestamp": datetime.now().isoformat(),
            "embedding_model": EMBEDDING_MODEL_NAME,
            "total_cases": len(results)
        },
        "evaluation_results": results
    }
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=4)
        
    print("\n" + f"💾 تم حفظ تقرير الاختبار المفصل في: {filepath}")


def main():
    parser = argparse.ArgumentParser(description="إطار عمل تقييم جودة الاسترجاع.")
    parser.add_argument(
        "--tenant", 
        type=str, 
        required=True, 
        help="هوية العميل المطلوب اختباره (مثال: accredit, scholl). اكتب 'all' لاختبار جميع العملاء."
    )
    parser.add_argument(
        "--retriever", 
        type=str, 
        default="all", 
        choices=["all", "ensemble", "faiss", "bm25"],
        help="نوع المسترجع المطلوب اختباره."
    )
    args = parser.parse_args()

    # --- تحميل جميع المستندات مرة واحدة لذاكرة التخزين المؤقت ---
    print("[*] جارٍ تحميل جميع المستندات من قاعدة المعرفة لذاكرة التخزين (مطلوب لـ BM25)...")
    try:
        all_docs_from_kb = load_all_documents_from_kb()
        print(f"[✅] تم تحميل {len(all_docs_from_kb)} قطعة بنجاح.")
    except Exception as e:
        print(f"[❌] فشل حاسم في تحميل قاعدة المعرفة. لا يمكن المتابعة. الخطأ: {e}")
        return

    # --- تحديد العملاء والمسترجعات للاختبار ---
    tenants_to_test = [d for d in os.listdir(TEST_CASES_DIR) if d.endswith('_cases.json')]
    tenants_to_test = [d.replace('_cases.json', '') for d in tenants_to_test]
    
    if args.tenant != "all":
        if args.tenant not in tenants_to_test:
            print(f"خطأ: لا يوجد ملف حالات اختبار للعميل '{args.tenant}'.")
            return
        tenants_to_test = [args.tenant]

    retrievers_to_test = ["ensemble", "faiss", "bm25"] if args.retriever == "all" else [args.retriever]

    # --- بدء حلقة الاختبار ---
    for tenant in tenants_to_test:
        for retriever_name in retrievers_to_test:
            test_results = run_test_for_tenant(tenant, retriever_name, all_docs_from_kb)
            save_results(tenant, retriever_name, test_results)
            
    print("\n" + "="*70 + "\n🎉🎉🎉 اكتملت جميع عمليات التقييم بنجاح! 🎉🎉🎉\n" + "="*70)
    print(f"🔍 يمكنك الآن مراجعة التقارير المفصلة في المجلد: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
