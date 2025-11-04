# 1_knowledge_pipeline/auto_watcher.py
# -----------------------------------------------------------------------------
# هذا السكريبت يعمل كخدمة في الخلفية لمراقبة مجلد المستندات.
# عند إضافة مجلد عميل جديد، يقوم تلقائيًا بتشغيل خط أنابيب المعالجة له.
#
# للتشغيل: python 1_knowledge_pipeline/auto_watcher.py
# -----------------------------------------------------------------------------

import time
import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import subprocess # لاستدعاء main_builder.py

# تحديد المسارات
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLIENT_DOCS_BASE_DIR = os.path.abspath(os.path.join(BASE_DIR, "../4_client_docs/"))

class TenantFolderHandler(FileSystemEventHandler):
    """
    معالج الأحداث الذي يتم تفعيله عند حدوث تغييرات في نظام الملفات.
    """
    def on_created(self, event):
        """
        يتم استدعاؤه عند إنشاء ملف أو مجلد جديد.
        """
        # نهتم فقط بإنشاء المجلدات الجديدة (العملاء الجدد)
        if event.is_directory:
            tenant_id = os.path.basename(event.src_path)
            print(f"\n[👀] تم اكتشاف مجلد عميل جديد: {tenant_id}")
            print(f"[*] سيتم بدء عملية المعالجة تلقائيًا...")
            
            # استدعاء main_builder.py كعملية منفصلة
            # هذا يضمن أن كل عملية معالجة معزولة
            try:
                # بناء الأمر لتشغيل السكريبت
                script_path = os.path.join(BASE_DIR, "main_builder.py")
                command = ["python", script_path, "--tenant", tenant_id]
                
                # تشغيل الأمر
                subprocess.run(command, check=True, text=True)
                
                print(f"[✅] اكتملت المعالجة التلقائية للعميل: {tenant_id}")
            except subprocess.CalledProcessError as e:
                print(f"[❌] فشلت المعالجة التلقائية للعميل '{tenant_id}'. الخطأ: {e}")
            except FileNotFoundError:
                print(f"[❌] خطأ: لا يمكن العثور على 'python' أو السكريبت '{script_path}'.")

def start_watcher():
    """
    يبدأ عملية المراقبة.
    """
    if not os.path.exists(CLIENT_DOCS_BASE_DIR):
        os.makedirs(CLIENT_DOCS_BASE_DIR)
        print(f"[+] تم إنشاء مجلد العملاء المصدر: '{CLIENT_DOCS_BASE_DIR}'")

    print("="*70)
    print(f"👁️  بدء مراقبة المجلد: '{CLIENT_DOCS_BASE_DIR}'")
    print("👁️  سيتم معالجة أي مجلد عميل جديد يتم إضافته تلقائيًا.")
    print("👁️  اضغط CTRL+C لإيقاف المراقب.")
    print("="*70)

    event_handler = TenantFolderHandler()
    observer = Observer()
    observer.schedule(event_handler, CLIENT_DOCS_BASE_DIR, recursive=False) # recursive=False لمراقبة المجلد الرئيسي فقط
    
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
    print("\n[🛑] تم إيقاف المراقب.")

if __name__ == "__main__":
    start_watcher()
