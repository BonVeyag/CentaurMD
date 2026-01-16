# Centaur Local Helper (dev mode)

Minimal FastAPI server that exposes a local transcription endpoint using the existing `app.transcription` faster‑whisper pipeline. Binds to `127.0.0.1:57123` by default (override with `LOCAL_HELPER_PORT`).

## Run in a fresh venv
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r local_helper/requirements.txt
uvicorn local_helper.main:app --host 127.0.0.1 --port 57123
```

## Quick test
```bash
curl -s http://127.0.0.1:57123/health
curl -s -F "file=@/path/to/audio.wav" http://127.0.0.1:57123/transcribe_chunk
```

## Notes
- No auth in dev mode.
- Reuses repo `app/transcription.py` (faster‑whisper). Ensure the repo root is on `PYTHONPATH` or run from repo root.
- If ffmpeg or required codecs are missing, transcription will return an error.
