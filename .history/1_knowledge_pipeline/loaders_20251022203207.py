# 1_knowledge_pipeline/loaders.py (النسخة الآلية)

import os
import json
from typing import List, Tuple, Optional
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
    TextLoader,
)

LOADER_MAPPING = {
    ".pdf": PyPDFLoader,
    ".docx": UnstructuredWordDocumentLoader,
    ".txt": TextLoader,
}

def load_documents(source_dir: str) -> Tuple[List[Document], Optional[str]]:
    """
    يقوم بتحميل جميع المستندات المدعومة ويقرأ اسم الكيان من ملف config.json.

    Args:
        source_dir (str): المسار إلى المجلد الذي يحتوي على ملفات العميل.

    Returns:
        Tuple[List[Document], Optional[str]]: 
        - قائمة من كائنات Document.
        - اسم الكيان (entity_name) أو None إذا لم يتم العثور عليه.
    """
    all_documents = []
    entity_name = None
    config_file_path = os.path.join(source_dir, "config.json")

    print(f"📂 جارٍ المسح في المسار: '{source_dir}'")

    if not os.path.isdir(source_dir):
        raise ValueError(f"المسار المحدد ليس مجلدًا صالحًا: {source_dir}")

    # --- الخطوة 1: قراءة ملف الإعدادات أولاً ---
    if os.path.exists(config_file_path):
        try:
            with open(config_file_path, "r", encoding="utf-8") as f:
                config_data = json.load(f)
                entity_name = config_data.get("entity_name")
                if entity_name:
                    print(f"  - ✅ تم العثور على اسم الكيان: '{entity_name}'")
                else:
                    print(f"  - ⚠️ تحذير: ملف 'config.json' موجود ولكنه لا يحتوي على 'entity_name'.")
        except Exception as e:
            print(f"  - ❌ خطأ أثناء قراءة 'config.json': {e}")
    else:
        print(f"  - ⚠️ تحذير: لم يتم العثور على ملف 'config.json'. لن يتم تحديد هوية للعميل.")

    # --- الخطوة 2: تحميل بقية المستندات ---
    for filename in os.listdir(source_dir):
        # تخطي ملف الإعدادات نفسه والملفات غير المدعومة
        if filename == "config.json":
            continue
        
        file_path = os.path.join(source_dir, filename)
        if not os.path.isfile(file_path) or filename.startswith('.'):
            continue

        file_ext = os.path.splitext(filename)[1].lower()
        if file_ext in LOADER_MAPPING:
            loader_class = LOADER_MAPPING[file_ext]
            print(f"  - 📄 جارٍ تحميل الملف: '{filename}'...")
            try:
                loader = loader_class(file_path, encoding="utf-8") if file_ext == ".txt" else loader_class(file_path)
                loaded_docs = loader.load()
                all_documents.extend(loaded_docs)
                print(f"    - ✅ تم تحميل {len(loaded_docs)} جزء/صفحة.")
            except Exception as e:
                print(f"    - ❌ فشل تحميل الملف '{filename}'. الخطأ: {e}")
        else:
            print(f"  -  تم تخطي ملف غير مدعوم: '{filename}'")

    if not all_documents:
        print(" لم يتم العثور على أي مستندات قابلة للمعالجة.")
    
    print(f"\n اكتمل التحميل. إجمالي عدد الأجزاء/الصفحات: {len(all_documents)}")
    return all_documents, entity_name
