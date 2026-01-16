#!/usr/bin/env bash
set -e
PORT="${LOCAL_HELPER_PORT:-57123}"
echo "Starting Centaur Local Helper on http://127.0.0.1:${PORT}"
python -m uvicorn local_helper.main:app --host 127.0.0.1 --port "${PORT}"
