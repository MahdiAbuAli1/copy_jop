# 1_knowledge_pipeline_ocr/loaders_ocr.py

import os
import json
from typing import List, Tuple, Optional
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
import io
import numpy as np
import cv2  # OpenCV for image pre-processing

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    UnstructuredWordDocumentLoader,
    TextLoader,
)

# --- إعدادات OCR ---
TESSERACT_LANG = 'ara+eng'
# PSM 3: تحليل تلقائي لتخطيط الصفحة، وهو خيار قوي ومتوازن.
TESSERACT_CONFIG = '--psm 3 --dpi 300'

def get_best_ocr_result(image_bytes: bytes) -> str:
    """
    يطبق استراتيجيات معالجة متعددة على الصورة ويختار أفضل نتيجة OCR.
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    original_img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if original_img_cv is None:
        return ""

    gray_img = cv2.cvtColor(original_img_cv, cv2.COLOR_BGR2GRAY)

    # --- قائمة استراتيجيات المعالجة ---
    processing_strategies = {
        "standard": preprocess_standard,
        "inverted": preprocess_inverted,
        "upscaled_denoised": preprocess_upscaled_denoised,
    }

    results = {}

    for name, strategy_func in processing_strategies.items():
        try:
            # تطبيق استراتيجية المعالجة
            processed_img = strategy_func(gray_img.copy())
            pil_image = Image.fromarray(processed_img)
            
            # تشغيل OCR
            text = pytesseract.image_to_string(
                pil_image, 
                lang=TESSERACT_LANG,
                config=TESSERACT_CONFIG
            ).strip()
            
            # تقييم النتيجة
            if is_text_meaningful(text):
                results[name] = text
                print(f"        - (الاستراتيجية: {name}) -> تم العثور على نص: '{text[:40].replace(chr(10), ' ')}...'")
        except Exception as e:
            print(f"        - ⚠️ فشلت استراتيجية '{name}': {e}")
            continue

    # إذا لم تكن هناك نتائج جيدة، أرجع سلسلة فارغة
    if not results:
        return ""

    # اختيار أفضل نتيجة بناءً على طول النص
    best_strategy = max(results, key=lambda k: len(results[k]))
    print(f"        - ✨ تم اختيار أفضل نتيجة من استراتيجية: '{best_strategy}'")
    return results[best_strategy]

# --- دوال المعالجة لكل استراتيجية ---

def preprocess_standard(image: np.ndarray) -> np.ndarray:
    """معالجة قياسية: تباين فقط."""
    return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

def preprocess_inverted(image: np.ndarray) -> np.ndarray:
    """معالجة عكسية: عكس الألوان ثم التباين."""
    inverted = cv2.bitwise_not(image)
    return cv2.adaptiveThreshold(inverted, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

def preprocess_upscaled_denoised(image: np.ndarray) -> np.ndarray:
    """معالجة متقدمة: تكبير، إزالة تشويش، ثم تباين."""
    scale_factor = 2.0
    width = int(image.shape[1] * scale_factor)
    height = int(image.shape[0] * scale_factor)
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_LANCZOS4)
    denoised = cv2.fastNlMeansDenoising(resized, h=30, templateWindowSize=7, searchWindowSize=21)
    return cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 5)

def is_text_meaningful(text: str, min_chars: int = 10, min_words: int = 2) -> bool:
    """
    دالة لتقييم ما إذا كان النص المستخرج ذا معنى.
    """
    text = text.strip()
    if len(text) < min_chars:
        return False
    words = text.split()
    if len(words) < min_words:
        return False
    alpha_chars = sum(1 for char in text if char.isalpha())
    if len(text) > 0 and alpha_chars / len(text) < 0.5:
        return False
    return True

# --- المحمل الجديد الذي يستخدم استراتيجيات متعددة ---
class MultiStrategyOcrPdfLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path
        print(f"🚀 تهيئة المحمل متعدد الاستراتيجيات للملف: {os.path.basename(self.file_path)}")

    def load(self) -> List[Document]:
        docs = []
        try:
            pdf_document = fitz.open(self.file_path)
            print(f"    - 📖 جارٍ معالجة {len(pdf_document)} صفحة...")
            
            for page_num, page in enumerate(pdf_document):
                normal_text = page.get_text("text").strip()
                ocr_texts = []
                image_list = page.get_images(full=True)
                
                if image_list:
                    print(f"      - 🖼️ تم العثور على {len(image_list)} صورة في الصفحة {page_num + 1}. تطبيق استراتيجيات متعددة...")
                    for img_index, img in enumerate(image_list):
                        xref = img[0]
                        base_image = pdf_document.extract_image(xref)
                        image_bytes = base_image["image"]
                        
                        # الحصول على أفضل نتيجة OCR من خلال تجربة عدة طرق
                        best_text = get_best_ocr_result(image_bytes)
                        if best_text:
                            ocr_texts.append(best_text)

                page_content_parts = []
                if normal_text:
                    page_content_parts.append("--- محتوى نصي ---\n" + normal_text)
                if ocr_texts:
                    full_ocr_text = "\n\n".join(ocr_texts)
                    page_content_parts.append("--- محتوى من الصور (OCR) ---\n" + full_ocr_text)
                
                final_page_content = "\n\n".join(page_content_parts)
                
                if final_page_content:
                    metadata = {"source": self.file_path, "page": page_num + 1}
                    docs.append(Document(page_content=final_page_content, metadata=metadata))
                
            pdf_document.close()
        except Exception as e:
            print(f"    - ❌ فشل كبير في معالجة PDF '{self.file_path}'. الخطأ: {e}")
            return []
            
        return docs

# --- تحديث قاموس التحميل ---
LOADER_MAPPING = {
    ".pdf": MultiStrategyOcrPdfLoader,  # <-- ✨✨ الكود الجديد هنا ✨✨
    ".docx": UnstructuredWordDocumentLoader,
    ".txt": TextLoader,
}

# --- دالة التحميل الرئيسية (تبقى كما هي) ---
def load_documents(source_dir: str) -> Tuple[List[Document], Optional[str]]:
    # ... (الكود هنا لم يتغير) ...
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
        except Exception as e:
            print(f"  - ❌ خطأ أثناء قراءة 'config.json': {e}")
    else:
        print(f"  - ⚠️ تحذير: لم يتم العثور على ملف 'config.json'.")

    for filename in os.listdir(source_dir):
        if filename == "config.json" or filename.startswith('.'):
            continue
        
        file_path = os.path.join(source_dir, filename)
        if not os.path.isfile(file_path):
            continue

        file_ext = os.path.splitext(filename)[1].lower()
        if file_ext in LOADER_MAPPING:
            loader_class = LOADER_MAPPING[file_ext]
            print(f"  - 📄 جارٍ تحميل الملف: '{filename}'...")
            try:
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
