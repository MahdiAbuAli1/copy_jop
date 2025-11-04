# 1_knowledge_pipeline_ocr/loaders_ocr.py

import os
import json
from typing import List, Tuple, Optional
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
import io

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    UnstructuredWordDocumentLoader,
    TextLoader,
)

# --- إعدادات OCR ---
# تأكد من أن tesseract-ocr-ara مثبت لديك
TESSERACT_LANG = 'ara+eng' 

# --- محمل PDF المخصص مع دعم OCR ---
class AdvancedPDFLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self) -> List[Document]:
        """
        يقوم بتحميل ملف PDF، ويستخرج النصوص العادية والنصوص من الصور (OCR).
        """
        docs = []
        try:
            pdf_document = fitz.open(self.file_path)
            print(f"    - 📖 جارٍ معالجة {len(pdf_document)} صفحة من '{os.path.basename(self.file_path)}'...")
            
            for page_num, page in enumerate(pdf_document):
                # 1. استخراج النص العادي
                text = page.get_text("text")
                
                # 2. استخراج النصوص من الصور باستخدام OCR
                ocr_text = ""
                image_list = page.get_images(full=True)
                if image_list:
                    print(f"      - 🖼️ تم العثور على {len(image_list)} صورة في الصفحة {page_num + 1}. جارٍ تحليلها...")
                    for img_index, img in enumerate(image_list):
                        xref = img[0]
                        base_image = pdf_document.extract_image(xref)
                        image_bytes = base_image["image"]
                        
                        try:
                            image = Image.open(io.BytesIO(image_bytes))
                            # استخدام tesseract لاستخراج النص من الصورة
                            additional_text = pytesseract.image_to_string(image, lang=TESSERACT_LANG)
                            if additional_text.strip():
                                ocr_text += f"\n--- OCR Text from Image {img_index + 1} ---\n{additional_text.strip()}"
                        except Exception as ocr_e:
                            print(f"        - ⚠️ فشل تحليل صورة في الصفحة {page_num + 1}. الخطأ: {ocr_e}")

                # دمج النصوص
                combined_text = text + ocr_text
                
                # إنشاء كائن Document للصفحة
                metadata = {
                    "source": self.file_path,
                    "page": page_num + 1,
                }
                doc = Document(page_content=combined_text, metadata=metadata)
                docs.append(doc)
                
            pdf_document.close()
        except Exception as e:
            print(f"    - ❌ فشل كبير في معالجة PDF '{self.file_path}'. الخطأ: {e}")
            # في حالة الفشل، نرجع قائمة فارغة لهذا الملف
            return []
            
        return docs

# --- تحديث قاموس التحميل ---
LOADER_MAPPING = {
    ".pdf": AdvancedPDFLoader,  # <-- ✨✨ الترقية هنا ✨✨
    ".docx": UnstructuredWordDocumentLoader,
    ".txt": TextLoader,
}

# --- دالة التحميل الرئيسية (تبقى كما هي تقريباً) ---
def load_documents(source_dir: str) -> Tuple[List[Document], Optional[str]]:
    """
    يقوم بتحميل جميع المستندات باستخدام المحمل المتقدم الذي يدعم OCR.
    """
    # ... (بقية الكود من ملف loaders.py الأصلي تبقى كما هي تماماً) ...
    all_documents = []
    entity_name = None
    config_file_path = os.path.join(source_dir, "config.json")

    print(f"📂 جارٍ المسح في المسار: '{source_dir}'")

    if not os.path.isdir(source_dir):
        raise ValueError(f"المسار المحدد ليس مجلدًا صالحًا: {source_dir}")

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

    for filename in os.listdir(source_dir):
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
                # لا نحتاج لتمرير encoding للمحملات الجديدة
                loader = loader_class(file_path)
                loaded_docs = loader.load()
                all_documents.extend(loaded_docs)
                print(f"    - ✅ تم تحميل ومعالجة {len(loaded_docs)} صفحة.")
            except Exception as e:
                print(f"    - ❌ فشل تحميل الملف '{filename}'. الخطأ: {e}")
        else:
            print(f"  -  تم تخطي ملف غير مدعوم: '{filename}'")

    if not all_documents:
        print(" لم يتم العثور على أي مستندات قابلة للمعالجة.")
    
    print(f"\n اكتمل التحميل. إجمالي عدد الصفحات المعالجة: {len(all_documents)}")
    return all_documents, entity_name
