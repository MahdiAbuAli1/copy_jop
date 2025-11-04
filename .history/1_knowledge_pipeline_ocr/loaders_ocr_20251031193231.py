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
# DPI 300: تحديد الدقة لتحسين التعرف.
TESSERACT_CONFIG = '--psm 3 --dpi 300'

def is_text_meaningful(text: str, min_chars: int = 15, min_words: int = 3) -> bool:
    """
    دالة لتقييم ما إذا كان النص المستخرج ذا معنى أم مجرد ضوضاء.
    تم جعلها أكثر صرامة.
    """
    text = text.strip()
    if len(text) < min_chars:
        return False
    words = text.split()
    if len(words) < min_words:
        return False
    # حساب نسبة الحروف الأبجدية في النص
    alpha_chars = sum(1 for char in text if char.isalpha())
    if alpha_chars / len(text) < 0.6:  # يجب أن يكون 60% على الأقل من النص حروفًا
        return False
    return True

def deskew_image(image: np.ndarray) -> np.ndarray:
    """
    تصحيح ميلان الصورة لجعل النص أفقيًا تمامًا.
    """
    try:
        # تحويل إلى أبيض وأسود والعكس للعثور على الخطوط
        gray = cv2.bitwise_not(image)
        coords = np.column_stack(np.where(gray > 0))
        angle = cv2.minAreaRect(coords)[-1]

        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        
        if abs(angle) > 20: # لا تقم بالتصحيح إذا كانت الزاوية كبيرة جدًا (قد تكون صورة مائلة عمدًا)
            return image

        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return rotated
    except Exception:
        return image # في حالة حدوث أي خطأ، أرجع الصورة الأصلية

def preprocess_image_for_ocr(image_bytes: bytes) -> Image.Image:
    """
    دالة معالجة صور متقدمة جدًا لتحقيق أقصى دقة من Tesseract.
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # 1. تحويل إلى تدرج الرمادي
    gray_img = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    
    # 2. تصحيح الميلان (Deskewing)
    deskewed_img = deskew_image(gray_img)
    
    # 3. تكبير الصورة (Upscaling) لتحسين التفاصيل
    scale_factor = 2.0
    width = int(deskewed_img.shape[1] * scale_factor)
    height = int(deskewed_img.shape[0] * scale_factor)
    resized_img = cv2.resize(deskewed_img, (width, height), interpolation=cv2.INTER_LANCZOS4)
    
    # 4. تطبيق فلتر لإزالة التشويش
    denoised_img = cv2.fastNlMeansDenoising(resized_img, h=30, templateWindowSize=7, searchWindowSize=21)
    
    # 5. تطبيق Adaptive Thresholding لجعل الحروف بارزة
    processed_img = cv2.adaptiveThreshold(
        denoised_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 5
    )
    
    return Image.fromarray(processed_img)

# --- محمل PDF المحلي فائق الدقة ---
class HighAccuracyLocalPdfLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path
        print(f"🚀 تهيئة المحمل المحلي فائق الدقة للملف: {os.path.basename(self.file_path)}")

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
                    print(f"      - 🖼️ تم العثور على {len(image_list)} صورة في الصفحة {page_num + 1}. جارٍ التحليل المحلي الدقيق...")
                    for img_index, img in enumerate(image_list):
                        xref = img[0]
                        base_image = pdf_document.extract_image(xref)
                        image_bytes = base_image["image"]
                        
                        try:
                            # 1. معالجة متقدمة جدًا للصورة
                            preprocessed_image = preprocess_image_for_ocr(image_bytes)
                            
                            # 2. استخراج النص باستخدام Tesseract
                            ocr_text = pytesseract.image_to_string(
                                preprocessed_image, 
                                lang=TESSERACT_LANG,
                                config=TESSERACT_CONFIG
                            )
                            
                            # 3. فلترة النتائج الضعيفة
                            if is_text_meaningful(ocr_text):
                                print(f"        - ✅ تم استخراج نص مفيد من الصورة {img_index + 1}.")
                                ocr_texts.append(ocr_text.strip())
                            else:
                                print(f"        - 🗑️ تم تجاهل نص غير مفيد من الصورة {img_index + 1}.")

                        except Exception as ocr_e:
                            print(f"        - ⚠️ فشل تحليل صورة في الصفحة {page_num + 1}. الخطأ: {ocr_e}")

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
    ".pdf": HighAccuracyLocalPdfLoader,  # <-- ✨✨ الكود المحلي فائق الدقة هنا ✨✨
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
