import time
import json
import os
import logging

# -----------------------------------------------------------------------------
# ⚙️ إعداد المسار الثابت لملف السجل
# -----------------------------------------------------------------------------
# يحدد المسار بناءً على موقع هذا الملف لضمان الوصول الصحيح دائمًا.
# المسار النسبي: ../../2_central_api_service/agent_app/performance_log.jsonl
# ملاحظة: تم تغيير امتداد الملف إلى .jsonl ليعكس أنه يستخدم صيغة JSON Lines.
try:
    # تحديد المسار الجذري للمشروع (نفترض أن هذا الملف داخل مجلد agent_app)
    # support_service_platform/2_central_api_service/agent_app/performance_tracker.py
    APP_DIR = os.path.dirname(os.path.abspath(__file__))
    LOG_FILE = os.path.join(APP_DIR, "performance_log.jsonl")

    # تأكد من وجود المجلد الذي سيحتوي على ملف السجل
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

except Exception as e:
    logging.error(f"فشل في تحديد مسار ملف السجل: {e}")
    # في حال الفشل، يتم إنشاء الملف في المجلد الحالي كحل بديل
    LOG_FILE = "performance_log.jsonl"


class PerformanceLogger:
    """
    مسجل أداء محسن يقوم بكتابة السجلات بصيغة JSON Lines.
    هذه الطريقة آمنة للعمليات المتزامنة (process-safe) وأكثر كفاءة.
    """
    def __init__(self):
        self.start_times = {}

    def start(self, stage_name: str):
        """تبدأ توقيت مرحلة معينة."""
        self.start_times[stage_name] = time.time()
        logging.info(f"⏱️ بدء المرحلة: {stage_name}")

    def end(self, stage_name: str, tenant_id: str, question: str, extra_info: dict = None):
        """
        تسجل انتهاء مرحلة معينة وتكتب السجل في ملف بصيغة JSON Lines.
        الكتابة في وضع الإلحاق (append) تضمن عدم فقدان البيانات أو تضاربها.
        """
        if stage_name not in self.start_times:
            logging.warning(f"⚠️ محاولة إنهاء مرحلة '{stage_name}' لم يتم بدؤها.")
            return

        end_time = time.time()
        duration = end_time - self.start_times[stage_name]
        
        record = {
            "timestamp": end_time,
            "tenant_id": tenant_id,
            "question": question,
            "stage": stage_name,
            "duration_sec": round(duration, 4),  # تقريب المدة لتسهيل القراءة
        }
        
        if extra_info:
            record.update(extra_info)
        
        try:
            # استخدام وضع الإلحاق "a" للكتابة الآمنة والفعالة
            # ensure_ascii=False لضمان عرض الحروف العربية بشكل صحيح
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            
            logging.info(f"✅ تم تسجيل المرحلة: {stage_name} | المدة: {record['duration_sec']} ثانية")

        except Exception as e:
            logging.error(f"❌ فشل في كتابة سجل الأداء للمرحلة '{stage_name}': {e}", exc_info=True)
        
        # حذف وقت البدء بعد التسجيل لمنع إعادة استخدامه بالخطأ
        del self.start_times[stage_name]

