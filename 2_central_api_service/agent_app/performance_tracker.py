import time
import json
import os
import logging

# -----------------------------------------------------------------------------
# ⚙️ إعداد المسار الثابت لملف السجل
# -----------------------------------------------------------------------------
try:
    APP_DIR = os.path.dirname(os.path.abspath(__file__))
    LOG_FILE = os.path.join(APP_DIR, "performance_log.jsonl")
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
except Exception as e:
    logging.error(f"فشل في تحديد مسار ملف السجل: {e}")
    LOG_FILE = "performance_log.jsonl"


class PerformanceLogger:
    """
    مسجل أداء محسن يقوم بكتابة السجلات بصيغة JSON Lines.
    يستخدم لكل من start() و end() لتتبع مراحل التنفيذ بالتفصيل.
    """
    def __init__(self):
        self.records = {}

    def start(self, stage_name: str, tenant_id: str = None, question: str = None, extra_info: dict = None):
        """بدء مرحلة تسجيل الأداء مع بيانات إضافية اختيارية."""
        self.records[stage_name] = {
            "start_time": time.time(),
            "tenant_id": tenant_id,
            "question": question,
            "extra_info": extra_info or {},
        }
        logging.info(f"⏱️ بدء المرحلة: {stage_name} للمستأجر {tenant_id}")

    def end(self, stage_name: str, tenant_id: str = None, question: str = None, extra_info: dict = None):
        """إنهاء مرحلة وتسجيل بياناتها في ملف JSONL."""
        if stage_name not in self.records:
            logging.warning(f"⚠️ محاولة إنهاء مرحلة '{stage_name}' لم يتم بدؤها.")
            return

        end_time = time.time()
        start_data = self.records.pop(stage_name)
        duration = end_time - start_data["start_time"]

        record = {
            "timestamp": end_time,
            "stage": stage_name,
            "duration_sec": round(duration, 4),
            "tenant_id": tenant_id or start_data.get("tenant_id"),
            "question": question or start_data.get("question"),
            "metadata": {**start_data.get("extra_info", {}), **(extra_info or {})},
        }

        try:
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            logging.info(f"✅ تم تسجيل المرحلة '{stage_name}' | المدة: {record['duration_sec']} ثانية")
        except Exception as e:
            logging.error(f"❌ فشل في كتابة سجل الأداء: {e}", exc_info=True)
