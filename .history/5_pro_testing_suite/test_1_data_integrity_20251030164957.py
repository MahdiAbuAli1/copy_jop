# هل بياناتنا نظيفة ومفهومة؟import os
import logging
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader

# إعدادات أساسية
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
CLIENT_DOCS_DIR = os.path.join(PROJECT_ROOT, "4_client_docs")
LOADER_MAPPING = {
    ".pdf": (PyPDFLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf-8"}),
}

def test_all_documents():
    """
    يقرأ كل ملف في مجلدات العملاء للتأكد من سلامة البيانات والترميز.
    """
    print("\n--- 🔬 اختبار سلامة البيانات (المرحلة 1) 🔬 ---")
    has_errors = False
    
    if not os.path.isdir(CLIENT_DOCS_DIR):
        logging.error(f"المجلد الرئيسي للعملاء غير موجود: {CLIENT_DOCS_DIR}")
        return

    for tenant_id in os.listdir(CLIENT_DOCS_DIR):
        tenant_path = os.path.join(CLIENT_DOCS_DIR, tenant_id)
        if not os.path.isdir(tenant_path):
            continue
        
        logging.info(f"\n📂 فحص العميل: {tenant_id}")
        for filename in os.listdir(tenant_path):
            file_path = os.path.join(tenant_path, filename)
            file_ext = os.path.splitext(filename)[1].lower()

            if file_ext in LOADER_MAPPING:
                loader_class, loader_kwargs = LOADER_MAPPING[file_ext]
                try:
                    logging.info(f"  - 📄 جاري قراءة '{filename}'...")
                    loader = loader_class(file_path, **loader_kwargs)
                    docs = loader.load()
                    if not docs or not docs[0].page_content.strip():
                        logging.warning(f"    - ⚠️ تحذير: الملف '{filename}' فارغ أو لا يحتوي على نص.")
                        has_errors = True
                    else:
                        # طباعة أول 50 حرفًا كدليل على النجاح
                        preview = docs[0].page_content.strip()[:50].replace('\n', ' ')
                        logging.info(f"    - ✅ نجح. بداية النص: \"{preview}...\"")

                except Exception as e:
                    logging.error(f"    - ❌ فشل ذريع في قراءة الملف '{filename}'. الخطأ: {e}")
                    has_errors = True
            else:
                if filename != "config.json":
                    logging.warning(f"  - ⏩ تم تخطي ملف غير مدعوم: '{filename}'")

    print("\n--- 🏁 انتهى اختبار سلامة البيانات 🏁 ---")
    if has_errors:
        print("🔴 تم العثور على أخطاء. يرجى مراجعة السجلات وإصلاح الملفات قبل المتابعة.")
    else:
        print("🟢 جميع الملفات قابلة للقراءة بشكل سليم. يمكننا الانتقال إلى المرحلة التالية بثقة.")

if __name__ == "__main__":
    test_all_documents()
