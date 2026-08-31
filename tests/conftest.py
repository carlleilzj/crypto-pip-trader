import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Keep pytest output out of the production reports/logs/ directory. Must be
# set before any module calls get_logger(), which configures file handlers
# at import time.
os.environ.setdefault("PIP_LOG_DIR", tempfile.mkdtemp(prefix="pip-trader-test-logs-"))
