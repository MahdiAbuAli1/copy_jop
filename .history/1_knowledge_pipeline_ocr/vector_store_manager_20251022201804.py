# 1_knowledge_pipeline/vector_store_manager.py (النسخة النهائية والمضمونة)
import os
import logging
from typing import List
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from filelock import FileLock, Timeout

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VECTOR_DB_DIR = os.path.abspath(os.path.join(BASE_DIR, "../3_shared_resources/vector_db/"))
LOCK_FILE_PATH = os.path.join(VECTOR_DB_DIR, "faiss.lock")

def get_embeddings_model(model_name: str):
    """ يقوم بتهيئة وإرجاع نموذج التضمين بناءً على الاسم المُمرر. """
    logging.info(f"إعداد نموذج التضمين بالاسم: '{model_name}'...")
    return OllamaEmbeddings(model=model_name)

# لاحظ التغيير هنا: أضفنا `embedding_model_name` كوسيط
def add_to_vector_store(chunks: List[Document], embedding_model_name: str):
    if not chunks:
        logging.warning("لا توجد قطع لإضافتها.")
        return

    logging.info(f"المرحلة 5: سيتم إضافة {len(chunks)} قطعة إلى قاعدة المعرفة...")
    os.makedirs(VECTOR_DB_DIR, exist_ok=True)
    lock = FileLock(LOCK_FILE_PATH)

    try:
        with lock.acquire(timeout=60):
            logging.info("تم الحصول على قفل الكتابة.")
            
            # لم نعد نعتمد على متغير عالمي، بل نستخدم الوسيط مباشرة
            embeddings_model = get_embeddings_model(embedding_model_name)
            
            db_exists = os.path.exists(os.path.join(VECTOR_DB_DIR, "index.faiss"))

            if not db_exists:
                logging.info("قاعدة المعرفة غير موجودة. سيتم إنشاؤها...")
                vector_store = FAISS.from_documents(documents=chunks, embedding=embeddings_model)
                vector_store.save_local(VECTOR_DB_DIR)
                logging.info("تم إنشاء وحفظ قاعدة المعرفة.")
            else:
                logging.info("تحميل قاعدة المعرفة الموجودة...")
                vector_store = FAISS.load_local(
                    VECTOR_DB_DIR, 
                    embeddings=embeddings_model, 
                    allow_dangerous_deserialization=True
                )
                logging.info("جارٍ دمج القطع الجديدة...")
                vector_store.add_documents(documents=chunks)
                vector_store.save_local(VECTOR_DB_DIR)
                logging.info("تم دمج وحفظ قاعدة المعرفة المحدثة.")
    except Timeout:
        logging.error("فشل الحصول على قفل الكتابة.")
    except Exception as e:
        logging.critical(f"حدث خطأ فادح: {e}", exc_info=True)
    finally:
        logging.info("تم تحرير قفل الكتابة.")
