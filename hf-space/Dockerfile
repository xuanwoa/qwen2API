FROM ghcr.io/xuanwoa/service-runtime:latest

ENV PORT=7860

CMD ["sh", "-c", "set -eu; if [ -d /data ]; then mkdir -p /data/runtime-data; rm -rf /workspace/data; ln -s /data/runtime-data /workspace/data; else mkdir -p /workspace/data; fi; mkdir -p /workspace/logs; exec python -m uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-7860} --workers ${WORKERS:-1}"]
