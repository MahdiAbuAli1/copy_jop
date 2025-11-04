# 2_evaluation_suite/run_evaluation.py (النسخة النهائية والمصححة)

import os
import argparse
import json
import time
from datetime import datetime
from dotenv import load_dotenv
from typing import List, Dict, Any

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings

# استيراد الوحدات الأساسية
from core.retriever_factory import get_retriever
from core.evaluators import evaluate_retrieval
from core.reranker import rerank_documents, cross_encoder

# --- الإعدادات الأولية ---
load_dotenv()
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VECTOR_DB_BASE_DIR = os.path.abspath(os.path.join(BASE_DIR, "../3_shared_resources/vector_db/"))
TEST_CASES_DIR = os.path.join(BASE_DIR, "test_cases")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_all_docs_from_faiss() -> List[Document]:
    """
    يقوم بتحميل جميع القطع من جميع قواعد بيانات العملاء المعزولة.
    هذا مطلوب لتهيئة BM25Retriever.
    """
    print("[*] جارٍ تحميل جميع المستندات من قواعد المعرفة لذاكرة التخزين...")
    all_docs = []
    
    # التأكد من أن نموذج التضمين مهيأ
    try:
        embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL_NAME)
    except Exception as e:
        print(f"[❌] فشل في تهيئة نموذج التضمين: {e}")
        return []

    # التحقق من وجود المجلد الأساسي
    if not os.path.exists(VECTOR_DB_BASE_DIR) or not os.listdir(VECTOR_DB_BASE_DIR):
        print("[❌] فشل حاسم في تحميل قاعدة المعرفة: المجلد 'vector_db' فارغ أو غير موجود!")
        return []

    # المرور على كل مجلد عميل وتحميل قاعدة بياناته
    for tenant_id in os.listdir(VECTOR_DB_BASE_DIR):
        tenant_db_path = os.path.join(VECTOR_DB_BASE_DIR, tenant_id)
        if os.path.isdir(tenant_db_path) and os.path.exists(os.path.join(tenant_db_path, "index.faiss")):
            try:
                vector_store = FAISS.load_local(tenant_db_path, embeddings, allow_dangerous_deserialization=True)
                # استرجاع جميع المستندات من قاعدة البيانات الحالية
                # نستخدم retriever للحصول على المستندات مع بياناتها الوصفية الكاملة
                retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={'k': 1000})
                docs = retriever.get_relevant_documents(query=" ") # استعلام فارغ لاسترجاع عينة كبيرة
                all_docs.extend(docs)
            except Exception as e:
                print(f"  - ⚠️ تحذير: فشل تحميل قاعدة بيانات العميل '{tenant_id}'. الخطأ: {e}")
    
    if not all_docs:
        print("[❌] فشل حاسم في تحميل قاعدة المعرفة: لم يتم العثور على أي مستندات في قواعد البيانات الفرعية!")
        return []

    print(f"[✅] تم تحميل {len(all_docs)} قطعة بنجاح من جميع قواعد البيانات.")
    return all_docs


def run_test_for_tenant(tenant_id: str, retriever_types: List[str], all_docs: List[Document]):
    """ينفذ جميع الاختبارات لعميل واحد."""
    test_case_file = os.path.join(TEST_CASES_DIR, f"{tenant_id}_cases.json")
    if not os.path.exists(test_case_file):
        print(f"  - ⚠️ تم تخطي العميل '{tenant_id}': لم يتم العثور على ملف حالات اختبار.")
        return

    with open(test_case_file, 'r', encoding='utf-8') as f:
        test_cases = json.load(f)

    for retriever_type in retriever_types:
        print("\n" + "="*30 + f" 🧪 بدء اختبار العميل: {tenant_id} | النوع: {retriever_type} " + "="*30)
        print(f"  - تم تحميل {len(test_cases)} حالة اختبار.")

        report = {
            "report_info": {
                "tenant_id": tenant_id,
                "retriever_type": retriever_type,
                "timestamp": datetime.now().isoformat(),
                "embedding_model": EMBEDDING_MODEL_NAME,
                "total_cases": len(test_cases)
            },
            "evaluation_results": []
        }
        
        try:
            retriever = get_retriever(retriever_type, OllamaEmbeddings(model=EMBEDDING_MODEL_NAME), tenant_id, all_docs)
        except (FileNotFoundError, ValueError, RuntimeError) as e:
            print(f"  - ❌ فشل تهيئة المسترجع. تخطي هذا الاختبار. الخطأ: {e}")
            continue

        for case in test_cases:
            print(f"\n--- ❓ اختبار [{case['case_id']}]: {case['question']} ---")
            
            start_time = time.time()
            retrieved_docs_langchain = retriever.get_relevant_documents(case['question'])
            retrieval_time = time.time() - start_time
            
            # تحويل المستندات إلى صيغة قابلة للتخزين في JSON
            retrieved_docs_serializable = [{"content": doc.page_content, "source": doc.metadata.get("source", "N/A")} for doc in retrieved_docs_langchain]

            rerank_time = 0
            if retriever_type == "hybrid" and cross_encoder:
                print(f"  - 🔃 جارٍ إعادة ترتيب {len(retrieved_docs_serializable)} مستند...")
                rerank_start_time = time.time()
                reranked_docs = rerank_documents(case['question'], retrieved_docs_serializable)
                rerank_time = time.time() - rerank_start_time
                print(f"  - ✅ اكتملت إعادة الترتيب في {rerank_time:.2f} ثانية.")
                final_docs_to_evaluate = reranked_docs[:5] # نأخذ أفضل 5 بعد إعادة الترتيب
            else:
                final_docs_to_evaluate = retrieved_docs_serializable[:5] # نأخذ أفضل 5 مباشرة

            evaluation = evaluate_retrieval(
                retrieved_docs=final_docs_to_evaluate,
                expected_keywords=case.get('expected_keywords', []),
                expected_source=case.get('expected_source', '')
            )
            print(f"  - 📊 التقييم: {evaluation['status']} (المصدر: {evaluation['source_check']}, الكلمات: {evaluation['keyword_evaluation'].get('score', 'N/A')})")

            # إضافة الترتيب والدرجات إلى المستندات النهائية
            for i, doc in enumerate(final_docs_to_evaluate):
                doc['final_rank'] = i + 1

            report["evaluation_results"].append({
                "case_id": case['case_id'],
                "question": case['question'],
                "timing": {
                    "retrieval_seconds": round(retrieval_time, 2),
                    "rerank_seconds": round(rerank_time, 2),
                    "total_seconds": round(retrieval_time + rerank_time, 2)
                },
                "evaluation": evaluation,
                "retrieved_documents": final_docs_to_evaluate
            })

        # حفظ التقرير
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"{timestamp_str}_{tenant_id}_{retriever_type}.json"
        report_path = os.path.join(RESULTS_DIR, report_filename)
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=4)
        print(f"\n💾 تم حفظ تقرير الاختبار المفصل في: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="إطار تقييم جودة الاسترجاع.")
    parser.add_argument("--tenant", type=str, required=True, help="هوية العميل (أو 'all' للجميع).")
    parser.add_argument("--retriever", type=str, required=True, help="نوع المسترجع (faiss, bm25, ensemble, hybrid, أو 'all' للجميع).")
    args = parser.parse_args()

    all_docs = load_all_docs_from_faiss()
    if not all_docs:
        return

    tenants_to_test = [d for d in os.listdir(VECTOR_DB_BASE_DIR) if os.path.isdir(os.path.join(VECTOR_DB_BASE_DIR, d))] if args.tenant == 'all' else [args.tenant]
    retrievers_to_test = ["hybrid", "ensemble", "faiss", "bm25"] if args.retriever == 'all' else [args.retriever]

    for tenant in tenants_to_test:
        run_test_for_tenant(tenant, retrievers_to_test, all_docs)
    
    print("\n" + "="*70)
    print("🎉🎉🎉 اكتملت جميع عمليات التقييم بنجاح! 🎉🎉🎉")
    print(f"🔍 يمكنك الآن مراجعة التقارير المفصلة في المجلد: {RESULTS_DIR}")
    print("="*70)


if __name__ == "__main__":
    main()
