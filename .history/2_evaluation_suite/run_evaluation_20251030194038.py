# 2_evaluation_suite/run_evaluation.py (النسخة المحدثة والشاملة)

import os
import json
import argparse
import time
from datetime import datetime
from typing import List, Dict, Any, Literal
from dotenv import load_dotenv

# استيراد المكونات الأساسية
from core.retriever_factory import get_retriever
from core.evaluators import evaluate_retrieval
from core.reranker import rerank_documents # <-- استيراد وحدة إعادة الترتيب الجديدة

# --- إعدادات أساسية ---
load_dotenv()
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_CASES_DIR = os.path.join(BASE_DIR, "test_cases")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# --- تعريف أنواع المسترجعات المدعومة ---
RetrieverType = Literal["hybrid", "ensemble", "faiss", "bm25"]

def load_all_documents_from_kb() -> List[Any]:
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import OllamaEmbeddings
    
    db_path = os.path.abspath(os.path.join(BASE_DIR, "../3_shared_resources/vector_db/"))
    if not os.path.exists(os.path.join(db_path, "index.faiss")):
        raise FileNotFoundError("قاعدة بيانات FAISS غير موجودة!")
        
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL_NAME)
    db = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
    
    return list(db.docstore._dict.values())

def run_test_for_tenant(
    tenant_id: str,
    retriever_type: RetrieverType,
    all_docs: List[Any]
) -> List[Dict[str, Any]]:
    print("\n" + "="*30 + f" 🧪 بدء اختبار العميل: {tenant_id} | النوع: {retriever_type} " + "="*30)
    
    test_cases_file = os.path.join(TEST_CASES_DIR, f"{tenant_id}_cases.json")
    if not os.path.exists(test_cases_file):
        print(f"⚠️ لم يتم العثور على ملف حالات اختبار للعميل '{tenant_id}'. تم التخطي.")
        return []
    
    with open(test_cases_file, 'r', encoding='utf-8') as f:
        test_cases = json.load(f)
    print(f"  - تم تحميل {len(test_cases)} حالة اختبار.")

    tenant_specific_docs = [doc for doc in all_docs if doc.metadata.get("tenant_id") == tenant_id]
    if not tenant_specific_docs:
        print(f"⚠️ لم يتم العثور على مستندات في قاعدة المعرفة للعميل '{tenant_id}'. تم التخطي.")
        return []

    # --- 1. تهيئة المسترجع ---
    # إذا كان النوع 'hybrid'، فسنستخدم 'ensemble' للاسترجاع الأولي
    base_retriever_type = "ensemble" if retriever_type == "hybrid" else retriever_type
    # نطلب عدداً أكبر من المستندات في حالة الـ hybrid لنعطي فرصة للـ reranker
    k_value = 20 if retriever_type == "hybrid" else 5
    
    retriever = get_retriever(base_retriever_type, tenant_specific_docs, EMBEDDING_MODEL_NAME, k=k_value)

    results = []
    for case in test_cases:
        question = case["question"]
        print(f"\n--- ❓ اختبار [{case['case_id']}]: {question} ---")
        
        # --- 2. مرحلة الاسترجاع ---
        start_time = time.time()
        retrieved_docs_langchain = retriever.invoke(question)
        retrieval_time = time.time() - start_time
        
        retrieved_docs_simple = [
            {
                "content": doc.page_content,
                "source": doc.metadata.get("source", "N/A"),
                "chunk_id": doc.metadata.get("chunk_id", "غير محدد") # مثال لإضافة معرف القطعة
            }
            for doc in retrieved_docs_langchain
        ]

        # --- 3. مرحلة إعادة الترتيب (فقط للنوع الهجين) ---
        final_docs = retrieved_docs_simple
        rerank_time = 0
        if retriever_type == "hybrid":
            print(f"  - 🔃 جارٍ إعادة ترتيب {len(retrieved_docs_simple)} مستند...")
            rerank_start_time = time.time()
            final_docs = rerank_documents(question, retrieved_docs_simple)
            rerank_time = time.time() - rerank_start_time
            print(f"  - ✅ اكتملت إعادة الترتيب في {rerank_time:.2f} ثانية.")
            # نأخذ أفضل 5 نتائج بعد إعادة الترتيب
            final_docs = final_docs[:5]

        # --- 4. التقييم والحفظ ---
        evaluation = evaluate_retrieval(
            retrieved_docs=final_docs, # نستخدم المستندات النهائية للتقييم
            expected_keywords=case["expected_keywords"],
            expected_source=case["expected_source"]
        )
        
        print(f"  - 📊 التقييم: {evaluation['status']} (المصدر: {evaluation['source_check']}, الكلمات: {evaluation['keyword_evaluation']['score']})")
        
        # إضافة تفاصيل شاملة للتقرير
        detailed_docs = []
        for i, doc in enumerate(final_docs):
            detailed_docs.append({
                "final_rank": i + 1,
                "content": doc["content"],
                "source": doc["source"],
                "original_rank": doc.get("original_rank", "N/A"), # من الـ reranker
                "rerank_score": f"{doc.get('rerank_score', 'N/A'):.4f}" if isinstance(doc.get('rerank_score'), float) else "N/A"
            })

        results.append({
            "case_id": case["case_id"],
            "question": question,
            "timing": {
                "retrieval_seconds": round(retrieval_time, 2),
                "rerank_seconds": round(rerank_time, 2),
                "total_seconds": round(retrieval_time + rerank_time, 2)
            },
            "evaluation": evaluation,
            "retrieved_documents": detailed_docs
        })
        
    return results

def save_results(tenant_id: str, retriever_type: RetrieverType, results: List[Dict[str, Any]]):
    if not results: return
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
    parser.add_argument("--tenant", type=str, required=True, help="هوية العميل. اكتب 'all' لاختبار جميع العملاء.")
    parser.add_argument(
        "--retriever", 
        type=str, 
        default="all", 
        choices=["all", "hybrid", "ensemble", "faiss", "bm25"],
        help="نوع المسترجع."
    )
    args = parser.parse_args()

    print("[*] جارٍ تحميل جميع المستندات من قاعدة المعرفة لذاكرة التخزين...")
    try:
        all_docs_from_kb = load_all_documents_from_kb()
        print(f"[✅] تم تحميل {len(all_docs_from_kb)} قطعة بنجاح.")
    except Exception as e:
        print(f"[❌] فشل حاسم في تحميل قاعدة المعرفة: {e}")
        return

    tenants_to_test = [d.replace('_cases.json', '') for d in os.listdir(TEST_CASES_DIR) if d.endswith('_cases.json')]
    if args.tenant != "all":
        if args.tenant not in tenants_to_test:
            print(f"خطأ: لا يوجد ملف حالات اختبار للعميل '{args.tenant}'.")
            return
        tenants_to_test = [args.tenant]

    retrievers_to_test = ["hybrid", "ensemble", "faiss", "bm25"] if args.retriever == "all" else [args.retriever]

    for tenant in tenants_to_test:
        for retriever_name in retrievers_to_test:
            test_results = run_test_for_tenant(tenant, retriever_name, all_docs_from_kb)
            save_results(tenant, retriever_name, test_results)
            
    print("\n" + "="*70 + "\n🎉🎉🎉 اكتملت جميع عمليات التقييم بنجاح! 🎉🎉🎉\n" + "="*70)
    print(f"🔍 يمكنك الآن مراجعة التقارير المفصلة في المجلد: {RESULTS_DIR}")

if __name__ == "__main__":
    main()
