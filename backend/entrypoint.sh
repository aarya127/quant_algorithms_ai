#!/bin/sh
exec gunicorn app:app --bind 0.0.0.0:${PORT:-8080} --workers 2 --worker-class gevent --timeout 120
