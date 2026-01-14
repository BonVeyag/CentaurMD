# Centaur Stress Test

This folder contains a concurrent-session stress runner for the live Centaur API.

## Audio fixtures
Place real WAV audio files here:

```
stress_test/audio/
  doctor_1_10min.wav
  doctor_2_10min.wav
  doctor_3_10min.wav
  doctor_4_10min.wav
  doctor_5_15min.wav
  doctor_6_15min.wav
```

Guidelines:
- Use real dictation-style audio (not synthetic TTS).
- Include interruptions and mild background noise.
- Do not reuse the same file for all sessions.

## Usage

Set environment variables (or pass CLI flags):

```
export CENTAUR_BASE_URL="http://127.0.0.1:8000"
export CENTAUR_USERNAME="<username>"
export CENTAUR_PASSWORD="<password>"

python3 stress_test/run_concurrent_sessions.py --concurrency 4
python3 stress_test/run_failure_tests.py --concurrency 4
```

Optional:
- `--server-pid` to capture CPU/mem from the server process.

Output:
- `stress_test/results/*.jsonl` raw event logs
- `stress_test/report.md` summary

## Notes
- The runner uses the same API endpoints as the live web app.
- If you run against production, ensure rate limits and privacy policies are respected.
