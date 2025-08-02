import multiprocessing
import os

# Gunicorn configuration
# See: https://docs.gunicorn.org/en/stable/configure.html

# Server socket
bind = f"0.0.0.0:{os.getenv('PORT', '10000')}"

# Worker processes
workers_per_core = int(os.getenv("WORKERS_PER_CORE", "2"))
cores = multiprocessing.cpu_count()
workers = max(workers_per_core * cores, 2)
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000

# Timeouts
timeout = 120  # seconds
keepalive = 5

# Logging
loglevel = os.getenv("LOG_LEVEL", "info")
accesslog = "-"  # Log to stdout
errorlog = "-"   # Log to stderr
capture_output = True

# Security
limit_request_line = 0  # No limit on request line size
limit_request_fields = 32000  # Maximum number of headers
limit_request_field_size = 0  # No limit on header size

# Performance settings
max_requests = 500
max_requests_jitter = 50

# Debugging
reload = os.getenv("RELOAD", "false").lower() == "true"

# Set the number of threads per worker
threads = 1

# Set the maximum number of pending connections
backlog = 2048
