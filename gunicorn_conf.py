import multiprocessing
import os

workers_per_core = int(os.getenv("WORKERS_PER_CORE", "2"))
web_concurrency = workers_per_core * multiprocessing.cpu_count()
host = os.getenv("HOST", "0.0.0.0")
port = os.getenv("PORT", "8000")
bind_env = os.getenv("BIND", None)
use_loglevel = os.getenv("LOG_LEVEL", "info")

if bind_env:
    bind = bind_env
else:
    bind = f"{host}:{port}"

# Worker configuration
workers = web_concurrency
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1024
keepalive = 5

# Logging
accesslog = "-"
errorlog = "-"  # Log to stderr
loglevel = use_loglevel
capture_output = True

# Timeouts
timeout = 120  # seconds

# Security
limit_request_line = 0  # No limit on request line size
limit_request_fields = 32000  # Maximum number of headers
limit_request_field_size = 0  # No limit on header size
