import os
import shutil
from dotenv import load_dotenv
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader, Docx2txtLoader

# --- الإعدادات ---
load_dotenv()
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL_NAME")
OLLAMA_HOST = os.getenv("OLLAMA_HOST")
if not EMBEDDING_MODEL or not OLLAMA_HOST:
    raise ValueError("EMBEDDING_MODEL_NAME و OLLAMA_HOST يجب أن تكون معرفة في ملف .env")

SOURCE_DIR = "4_client_docs"
TARGET_DIR = "3_shared_resources/vector_dbs"

def get_loader(file_path: str):
    ext = file_path.lower().split('.')[-1]
    if ext == 'pdf': return PyPDFLoader(file_path)
    if ext == 'docx': return Docx2txtLoader(file_path)
    return TextLoader(file_path, encoding='utf-8', autodetect_encoding=True)

def build_stores():
    print("🚀 بدء عملية بناء قواعد البيانات المتجهة...")
    
    if os.path.exists(TARGET_DIR):
        print(f"🧹 مسح المجلد القديم: {TARGET_DIR}")
        shutil.rmtree(TARGET_DIR)
    os.makedirs(TARGET_DIR)

    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_HOST)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=250)

    tenants = [d for d in os.listdir(SOURCE_DIR) if os.path.isdir(os.path.join(SOURCE_DIR, d))]
    if not tenants:
        print("⚠️ لم يتم العثور على أي مجلدات عملاء في '4_client_docs'.")
        return

    for tenant in tenants:
        print(f"\n🔄 جاري معالجة العميل: {tenant}...")
        tenant_source_path = os.path.join(SOURCE_DIR, tenant)
        tenant_target_path = os.path.join(TARGET_DIR, tenant)

        try:
            loader = DirectoryLoader(
                tenant_source_path,
                glob="**/*.*",
                loader_cls=get_loader,
                show_progress=True,
                use_multithreading=True,
                silent_errors=True
            )
            docs = loader.load()
            if not docs:
                print(f"🟡 لا توجد مستندات قابلة للقراءة للعميل: {tenant}")
                continue

            splits = text_splitter.split_documents(docs)
            print(f"📄 تم تقسيم مستندات {tenant} إلى {len(splits)} جزء.")

            vector_store = FAISS.from_documents(splits, embeddings)
            vector_store.save_local(tenant_target_path)
            print(f"✅ تم حفظ قاعدة بيانات العميل '{tenant}' بنجاح في: {tenant_target_path}")

        except Exception as e:
            print(f"❌ فشل في معالجة العميل {tenant}. الخطأ: {e}")

    print("\n🎉 اكتملت عملية بناء جميع قواعد البيانات.")

if __name__ == "__main__":
    build_stores()
