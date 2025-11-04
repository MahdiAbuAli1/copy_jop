# 1_knowledge_pipeline_ocr/vector_store_manager.py (النسخة النهائية مع عزل البيانات)
import os
import logging
from typing import List
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from filelock import FileLock, Timeout

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# المسار الأساسي لمجلد قواعد البيانات يبقى كما هو
VECTOR_DB_BASE_DIR = os.path.abspath(os.path.join(BASE_DIR, "../3_shared_resources/vector_db/"))

def get_embeddings_model(model_name: str):
    """ يقوم بتهيئة وإرجاع نموذج التضمين بناءً على الاسم المُمرر. """
    logging.info(f"إعداد نموذج التضمين بالاسم: '{model_name}'...")
    return OllamaEmbeddings(model=model_name)

def add_to_vector_store(chunks: List[Document], embedding_model_name: str):
    if not chunks:
        logging.warning("لا توجد قطع لإضافتها.")
        return

    # --- التغيير الجوهري: استخراج هوية العميل من القطع ---
    tenant_id = chunks[0].metadata.get("tenant_id")
    if not tenant_id:
        logging.error("خطأ فادح: لا يمكن تحديد هوية العميل (tenant_id) من البيانات الوصفية للقطع.")
        return

    logging.info(f"المرحلة 5: سيتم إضافة {len(chunks)} قطعة إلى قاعدة المعرفة الخاصة بالعميل '{tenant_id}'...")

    # --- تحديد المسار الخاص بقاعدة بيانات العميل ---
    tenant_db_dir = os.path.join(VECTOR_DB_BASE_DIR, tenant_id)
    os.makedirs(tenant_db_dir, exist_ok=True)
    
    # --- استخدام ملف قفل خاص بكل قاعدة بيانات لضمان الأمان ---
    lock_file_path = os.path.join(tenant_db_dir, "faiss.lock")
    lock = FileLock(lock_file_path)

    try:
        with lock.acquire(timeout=60):
            logging.info(f"تم الحصول على قفل الكتابة لقاعدة بيانات '{tenant_id}'.")
            
            embeddings_model = get_embeddings_model(embedding_model_name)
            
            db_path = os.path.join(tenant_db_dir, "index.faiss")
            db_exists = os.path.exists(db_path)

            if not db_exists:
                logging.info(f"قاعدة بيانات '{tenant_id}' غير موجودة. سيتم إنشاؤها...")
                vector_store = FAISS.from_documents(documents=chunks, embedding=embeddings_model)
                vector_store.save_local(tenant_db_dir)
                logging.info(f"تم إنشاء وحفظ قاعدة بيانات '{tenant_id}'.")
            else:
                logging.info(f"تحميل قاعدة بيانات '{tenant_id}' الموجودة...")
                vector_store = FAISS.load_local(
                    tenant_db_dir, 
                    embeddings=embeddings_model, 
                    allow_dangerous_deserialization=True
                )
                logging.info("جارٍ دمج القطع الجديدة...")
                vector_store.add_documents(documents=chunks)
                vector_store.save_local(tenant_db_dir)
                logging.info(f"تم دمج وحفظ قاعدة بيانات '{tenant_id}' المحدثة.")
    except Timeout:
        logging.error(f"فشل الحصول على قفل الكتابة لقاعدة بيانات '{tenant_id}'.")
    except Exception as e:
        logging.critical(f"حدث خطأ فادح أثناء معالجة قاعدة بيانات '{tenant_id}': {e}", exc_info=True)
    finally:
        logging.info(f"تم تحرير قفل الكتابة لقاعدة بيانات '{tenant_id}'.")
