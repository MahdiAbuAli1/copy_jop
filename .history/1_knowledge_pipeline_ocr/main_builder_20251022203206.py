# 1_knowledge_pipeline/main_builder.py (النسخة الإنتاجية والآلية)

import os
import argparse
from dotenv import load_dotenv
from typing import List
from langchain_core.documents import Document

load_dotenv()
from loaders import load_documents
from cleaners import clean_documents
from splitters import split_documents
from vector_store_manager import add_to_vector_store

EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")
if not EMBEDDING_MODEL_NAME:
    print("[!] خطأ: 'EMBEDDING_MODEL_NAME' غير موجود في .env.")
    exit()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLIENT_DOCS_BASE_DIR = os.path.abspath(os.path.join(BASE_DIR, "../4_client_docs/"))
OUTPUTS_BASE_DIR = os.path.join(BASE_DIR, "_processing_outputs/")

def save_docs_to_file(docs: List[Document], filepath: str, message: str):
    # (هذه الدالة تبقى كما هي دون تغيير)
    print(message)
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"--- إجمالي عدد الأجزاء: {len(docs)} ---\n\n")
            for i, doc in enumerate(docs):
                f.write(f"--- Chunk {i+1} ---\nMetadata: {doc.metadata}\n---\n{doc.page_content}\n\n")
        print(f"[+] تم حفظ المخرجات في: '{filepath}'")
    except IOError as e:
        print(f"[!] خطأ أثناء حفظ الملف '{filepath}': {e}")

def process_tenant(tenant_id: str):
    """
    ينسق عملية المعالجة الآلية الكاملة لمستندات عميل واحد.
    """
    print("-" * 70)
    print(f"[>>] بدء معالجة العميل: {tenant_id}")
    print("-" * 70)

    source_directory = os.path.join(CLIENT_DOCS_BASE_DIR, tenant_id)
    if not os.path.isdir(source_directory):
        print(f"[!] خطأ: لم يتم العثور على مجلد للعميل '{tenant_id}'")
        return

    tenant_output_dir = os.path.join(OUTPUTS_BASE_DIR, tenant_id)

    # --- المرحلة 1: تحميل المستندات واسم الكيان ---
    raw_docs, entity_name = load_documents(source_directory)
    if not raw_docs:
        print(f"[!] لا توجد مستندات للمعالجة للعميل '{tenant_id}'. تم التخطي.")
        return
    save_docs_to_file(raw_docs, os.path.join(tenant_output_dir, "1_raw_content.txt"), 
                      "[*] جارٍ حفظ المحتوى الخام...")

    # --- المرحلة 2: تنظيف النصوص ---
    cleaned_docs = clean_documents(raw_docs)
    save_docs_to_file(cleaned_docs, os.path.join(tenant_output_dir, "2_cleaned_content.txt"), 
                      "[*] جارٍ حفظ المحتوى النظيف...")
    
    # --- المرحلة 3: التقطيع ---
    chunks = split_documents(cleaned_docs)
    
    # --- المرحلة 4: إثراء البيانات الوصفية ---
    print(f"\n[+] المرحلة 4: إثراء البيانات الوصفية لـ {len(chunks)} قطعة...")
    for chunk in chunks:
        chunk.metadata["tenant_id"] = tenant_id
        if entity_name:
            chunk.metadata["entity_name"] = entity_name
    print(f"[*] اكتمل إثراء البيانات الوصفية.")
        
    save_docs_to_file(chunks, os.path.join(tenant_output_dir, "3_final_chunks.txt"), 
                      "[*] جارٍ حفظ القطع النهائية...")

    # --- المرحلة 5: الحفظ في قاعدة المعرفة ---
    print("\n[+] المرحلة 5: إضافة القطع إلى قاعدة المعرفة...")
    add_to_vector_store(chunks, embedding_model_name=EMBEDDING_MODEL_NAME)

    print(f"\n[<<] اكتملت المعالجة بنجاح للعميل: {tenant_id}")

def main():
    # (دالة main تبقى كما هي دون تغيير)
    parser = argparse.ArgumentParser(description="خط أنابيب بناء قاعدة المعرفة الآلي.")
    parser.add_argument("--tenant", type=str, required=False, help="هوية عميل معين لمعالجته.")
    args = parser.parse_args()
    
    if args.tenant:
        process_tenant(args.tenant)
    else:
        print("[*] سيتم معالجة جميع العملاء...")
        try:
            if not os.path.exists(CLIENT_DOCS_BASE_DIR):
                 print(f"[!] خطأ: الدليل '{CLIENT_DOCS_BASE_DIR}' غير موجود.")
                 return
            tenant_ids = [name for name in os.listdir(CLIENT_DOCS_BASE_DIR) if os.path.isdir(os.path.join(CLIENT_DOCS_BASE_DIR, name))]
            if not tenant_ids:
                print("[!] لم يتم العثور على أي عملاء للمعالجة.")
                return
            print(f"[*] تم العثور على {len(tenant_ids)} عميل: {', '.join(tenant_ids)}")
            for tenant_id in tenant_ids:
                process_tenant(tenant_id)
            print("\n" + "="*70 + "\n🎉🎉🎉 اكتملت معالجة جميع العملاء بنجاح! 🎉🎉🎉\n" + "="*70)
        except Exception as e:
            print(f"[!] حدث خطأ غير متوقع: {e}")

if __name__ == "__main__":
    main()
