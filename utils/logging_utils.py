import os
import sys
import datetime

# --- Logger Class ---
class Tee(object):
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush() # Ensure immediate write
    def flush(self):
        for f in self.files:
            f.flush()


def _sanitize_log_segment(value: str) -> str:
    value = (value or "").strip()
    for ch in ["/", "\\", ":", " "]:
        value = value.replace(ch, "_")
    return value


def setup_logging(log_dir="./train_log", run_name: str | None = None):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_run_name = _sanitize_log_segment(run_name or "run")
    log_path = os.path.join(log_dir, f"train_{safe_run_name}_{timestamp}.log")
    
    # Open log file
    f = open(log_path, 'w', encoding='utf-8')
    # Backup original stdout
    original_stdout = sys.stdout
    # Redirect stdout to both terminal and file
    sys.stdout = Tee(original_stdout, f)
    
    print(f"Logging initialized. Output saved to: {os.path.abspath(log_path)}")
    return f
