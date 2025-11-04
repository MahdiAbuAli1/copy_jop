# 1_knowledge_pipeline/main_builder.py (النسخة النهائية والمحسنة)

import os
import argparse
from dotenv import load_dotenv
from typing import List
from langchain_core.documents import Document

# --- الخطوة 0: تحميل الإعدادات ---
load_dotenv()

# --- الخطوة 1: استيراد الوحدات ---
from loaders import load_documents
from cleaners import clean_documents
from splitters import split_documents
from vector_store_manager import add_to_vector_store

# --- الخطوة 2: قراءة الإعدادات الهامة ---
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")
if not EMBEDDING_MODEL_NAME:
    print("[!] خطأ فادح: متغير البيئة 'EMBEDDING_MODEL_NAME' غير موجود في ملف .env. لا يمكن المتابعة.")
    exit()

# --- تعريف الثوابت ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLIENT_DOCS_BASE_DIR = os.path.abspath(os.path.join(BASE_DIR, "../4_client_docs/"))
OUTPUTS_BASE_DIR = os.path.join(BASE_DIR, "_processing_outputs/")

# -----------------------------------------------------------------------------
# 🔴🔴🔴 --- دالة جديدة للحصول على اسم الكيان --- 🔴🔴🔴
# -----------------------------------------------------------------------------
def get_entity_name(tenant_id: str) -> str:
    """
    تطلب من المستخدم إدخال الاسم الرسمي للكيان المرتبط بالـ tenant_id.
    """
    while True:
        prompt = f"\n❓ الرجاء إدخال الاسم الرسمي للكيان المرتبط بالعميل '{tenant_id}' (مثال: 'جامعة العلوم والتكنولوجيا'): "
        entity_name = input(prompt).strip()
        if entity_name:
            return entity_name
        else:
            print("[!] لا يمكن ترك الاسم فارغًا. الرجاء المحاولة مرة أخرى.")

# -----------------------------------------------------------------------------

def save_docs_to_file(docs: List[Document], filepath: str, message: str):
    """
    دالة مساعدة لحفظ محتوى قائمة من كائنات Document في ملف نصي للمراجعة.
    """
    print(message)
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"--- تم إنشاء هذا الملف تلقائيًا للمراجعة ---\n")
            f.write(f"--- إجمالي عدد الأجزاء: {len(docs)} ---\n\n")
            for i, doc in enumerate(docs):
                f.write(f"--- Document/Chunk {i+1} ---\n")
                f.write(f"Metadata: {doc.metadata}\n") # ستظهر البيانات الوصفية المحدثة هنا
                f.write("---\n")
                f.write(doc.page_content)
                f.write("\n\n")
        print(f"[+] تم حفظ المخرجات بنجاح في: '{filepath}'")
    except IOError as e:
        print(f"[!] خطأ أثناء حفظ الملف '{filepath}': {e}")


def process_tenant(tenant_id: str):
    """
    ينسق عملية المعالجة الكاملة لمستندات عميل واحد.
    """
    print("-" * 70)
    print(f"[>>] بدء معالجة مستندات العميل: {tenant_id}")
    print("-" * 70)

    source_directory = os.path.join(CLIENT_DOCS_BASE_DIR, tenant_id)
    if not os.path.isdir(source_directory):
        print(f"[!] خطأ: لم يتم العثور على مجلد للعميل '{tenant_id}' في المسار المتوقع '{source_directory}'")
        return

    # 🔴🔴🔴 --- خطوة جديدة: الحصول على اسم الكيان بشكل تفاعلي --- 🔴🔴🔴
    entity_name = get_entity_name(tenant_id)
    print(f"[+] تم تحديد اسم الكيان: '{entity_name}'")

    tenant_output_dir = os.path.join(OUTPUTS_BASE_DIR, tenant_id)

    # --- المرحلة 1: تحميل المستندات ---
    raw_docs = load_documents(source_directory)
    if not raw_docs:
        print(f"[!] لا توجد مستندات صالحة للمعالجة للعميل '{tenant_id}'. تم التخطي.")
        return
    save_docs_to_file(raw_docs, os.path.join(tenant_output_dir, "1_raw_content.txt"), 
                      "[*] جارٍ حفظ المحتوى الخام بعد التحميل للمراجعة...")

    # --- المرحلة 2: تنظيف النصوص ---
    cleaned_docs = clean_documents(raw_docs)
    save_docs_to_file(cleaned_docs, os.path.join(tenant_output_dir, "2_cleaned_content.txt"), 
                      "[*] جارٍ حفظ المحتوى النظيف بعد التنظيف للمراجعة...")
    
    # --- المرحلة 3: التقطيع ---
    chunks = split_documents(cleaned_docs)
    
    # -----------------------------------------------------------------------------
    # 🔴🔴🔴 --- المرحلة 4: تحديث البيانات الوصفية (Metadata) --- 🔴🔴🔴
    # -----------------------------------------------------------------------------
    print(f"\n[+] المرحلة 4: إثراء البيانات الوصفية لـ {len(chunks)} قطعة...")
    for chunk in chunks:
        # نقوم بتحديث قاموس البيانات الوصفية مباشرة
        chunk.metadata["tenant_id"] = tenant_id
        chunk.metadata["entity_name"] = entity_name # إضافة اسم الكيان
    print(f"[*] اكتمل إثراء البيانات الوصفية.")
        
    # حفظ القطع النهائية مع البيانات الوصفية للمراجعة
    save_docs_to_file(chunks, os.path.join(tenant_output_dir, "3_final_chunks.txt"), 
                      "[*] جارٍ حفظ القطع النهائية مع بياناتها الوصفية المثرية للمراجعة...")

    # --- المرحلة 5: الحفظ في قاعدة المعرفة ---
    print("\n[+] المرحلة 5: إضافة القطع إلى قاعدة المعرفة الموحدة...")
    add_to_vector_store(chunks, embedding_model_name=EMBEDDING_MODEL_NAME)

    print(f"\n[<<] اكتملت المراحل الحالية بنجاح للعميل: {tenant_id}")


def main():
    """
    نقطة الدخول الرئيسية للسكريبت.
    """
    parser = argparse.ArgumentParser(description="خط أنابيب بناء قاعدة المعرفة للعملاء.")
    parser.add_argument("--tenant", type=str, required=False, 
                        help="(اختياري) هوية عميل معين لمعالجته (اسم المجلد).")
    
    args = parser.parse_args()
    
    if args.tenant:
        process_tenant(args.tenant)
    else:
        print("[*] لم يتم تحديد عميل. سيتم محاولة معالجة جميع العملاء في الدليل المصدر...")
        try:
            if not os.path.exists(CLIENT_DOCS_BASE_DIR):
                 print(f"[!] خطأ: الدليل المصدر للعملاء '{CLIENT_DOCS_BASE_DIR}' غير موجود.")
                 return

            tenant_ids = [name for name in os.listdir(CLIENT_DOCS_BASE_DIR) if os.path.isdir(os.path.join(CLIENT_DOCS_BASE_DIR, name))]
            
            if not tenant_ids:
                print("[!] لم يتم العثور على أي مجلدات عملاء للمعالجة.")
                return

            print(f"[*] تم العثور على {len(tenant_ids)} عميل: {', '.join(tenant_ids)}")
            
            for tenant_id in tenant_ids:
                process_tenant(tenant_id)
            
            print("\n" + "="*70)
            print("🎉🎉🎉 اكتملت معالجة جميع العملاء بنجاح! 🎉🎉🎉")
            print("="*70)

        except Exception as e:
            print(f"[!] حدث خطأ غير متوقع أثناء محاولة معالجة جميع العملاء: {e}")

if __name__ == "__main__":
    main()

