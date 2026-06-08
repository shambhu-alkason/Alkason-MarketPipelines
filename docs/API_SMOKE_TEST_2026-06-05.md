# API Smoke Test Report — AI-MLOps Stock Prediction API

**Date:** 2026-06-05
**Server:** `http://localhost:8000` (FastAPI / uvicorn, started via `.venv\Scripts\python.exe main.py --mode serve`)
**API version:** 1.0.0 (`src/api/app.py`)
**Harness:** `scripts/api_smoke_test.py` (stdlib `urllib` only) → raw results in `scripts/api_smoke_results.json`
**Auth:** none (all endpoints public)

## Environment caveat (read this first)

This was a **bare-server smoke test**. The intent was to exercise every endpoint and the request
validators, **not** to produce real predictions. The following were intentionally absent:

- **No trained models** — `models/manual/` is empty.
- **No processed feature data** — `data/processed/` is empty.
- **Minimal dependencies only** — the `.venv` has `fastapi, uvicorn, pydantic, numpy, pyyaml,
  python-dotenv`. It does **not** have `pandas`, `matplotlib`, `ollama`, `transformers`, `torch`,
  `lightgbm`, `xgboost`, etc.

So every non-200 below is an **expected environment condition**, not an API defect — except where
explicitly flagged. The validators and the always-available endpoints all behave correctly.

> **Latency note:** every request shows ~2,050 ms. This is a client-side `localhost` resolution
> delay in Python's `urllib` (IPv6 `::1` attempted first, then IPv4 fallback), **not** server
> processing time. A direct `curl` to `/health` returns in well under a second.

## Summary

| Verdict | Count | Cases |
|---------|-------|-------|
| ✅ PASS (2xx as expected) | 5 | `/health`, `/models`, `/predict/batch`, `/drift`, `/ask` |
| ✅ PASS — validator working (4xx as designed) | 5 | `/predict` (no model 404), `/recommendation` (no model 404), N1, N3, N4 |
| ✅ PASS — normalization works (N2 → 404) | 1 | N2 |
| 🟡 EXPECTED-FAIL (500 from missing dependency) | 3 | `/sentiment`, `/comparison`, `/retrain` |
| 🔴 BUG (unexpected) | 0 | — |

**14/14 cases behaved as expected for this environment. No genuine API bugs found.**
The 3 × `500` are all clean `ModuleNotFoundError`s raised at first import inside the handler — they
will resolve once the corresponding libraries are installed (and, for `/comparison`, once models exist).

## Results table

| # | Method | Path | Request payload | HTTP | Verdict |
|---|--------|------|-----------------|------|---------|
| 1 | GET | `/health` | — | 200 | ✅ PASS |
| 2 | GET | `/models` | — | 200 | ✅ PASS |
| 3 | POST | `/predict` | `{"symbol":"RELIANCE.NS"}` | 404 | ✅ Expected (no model) |
| 4 | POST | `/predict/batch` | `{"symbols":["RELIANCE.NS","TCS.NS","INFY.NS"]}` | 200 | ✅ PASS (per-symbol errors[]) |
| 5 | GET | `/recommendation/RELIANCE.NS` | — | 404 | ✅ Expected (no model) |
| 6 | GET | `/sentiment/RELIANCE` | — | 500 | 🟡 Missing dep: `pandas` |
| 7 | GET | `/comparison` | — | 500 | 🟡 Missing dep: `matplotlib` |
| 8 | GET | `/drift` | — | 200 | ✅ PASS (all `no_reports`) |
| 9 | POST | `/retrain?trainer=manual` | — | 500 | 🟡 Missing dep: `pandas` |
| 10 | POST | `/ask` | `{"query":"Which stock shows the strongest buy signal this week?"}` | 200 | ✅ PASS (template fallback) |
| N1 | POST | `/predict` | `{"symbol":"RELIANCE.US"}` | 422 | ✅ Validator rejects bad suffix |
| N2 | POST | `/predict` | `{"symbol":"reliance"}` | 404 | ✅ Normalized → `RELIANCE.NS`, then no model |
| N3 | POST | `/ask` | `{"query":"hi"}` | 422 | ✅ Validator rejects `< 5` chars |
| N4 | POST | `/predict/batch` | `{"symbols":[]}` | 422 | ✅ Validator rejects empty list |

## Per-endpoint detail

### 1. GET /health → 200 ✅
```json
{"status":"ok","loaded_models":[],"uptime_seconds":156.57,"timestamp":"2026-06-05T10:..."}
```
Always-on liveness probe. `loaded_models` empty because no models are cached.

### 2. GET /models → 200 ✅
```json
{"models":[],"total":0}
```
Filesystem scan of `models/manual/` — correctly reports zero trained models.

### 3. POST /predict → 404 ✅ (expected)
Payload `{"symbol":"RELIANCE.NS"}`.
```json
{"detail":"No trained model found for RELIANCE.NS. Run: python main.py --mode train --trainer manual"}
```
Clean, actionable 404 from `_load_ensemble` (`app.py:56-64`). Correct behavior with no model present.

### 4. POST /predict/batch → 200 ✅
Payload `{"symbols":["RELIANCE.NS","TCS.NS","INFY.NS"]}`.
```json
{"predictions":[],
 "errors":[
   {"symbol":"RELIANCE.NS","error":"404: No trained model found for RELIANCE.NS. ..."},
   {"symbol":"TCS.NS","error":"404: No trained model found for TCS.NS. ..."},
   {"symbol":"INFY.NS","error":"404: No trained model found for INFY.NS. ..."}],
 "total":0}
```
**Good design:** the batch endpoint returns 200 and collects per-symbol failures in `errors[]`
instead of failing the whole request — exactly the intended partial-failure behavior.

### 5. GET /recommendation/RELIANCE.NS → 404 ✅ (expected)
Same payload-free path call; delegates to the predict logic → same clean 404 as #3.

### 6. GET /sentiment/RELIANCE → 500 🟡 (missing dependency)
Response body: `Internal Server Error`. Server traceback:
```
File "src/api/app.py", line 268, in sentiment
    from src.slm.sentiment import get_live_sentiment
File "src/slm/sentiment.py", line 13, in <module>
    import pandas as pd
ModuleNotFoundError: No module named 'pandas'
```
Fails at **import time** of the sentiment module (lazy-imported inside the handler). Needs `pandas`
(and transitively `transformers` + FinBERT weights) installed. Not a code bug.

### 7. GET /comparison → 500 🟡 (missing dependency)
Response body: `Internal Server Error`. Server traceback:
```
File "src/api/app.py", line 315, in comparison
File "src/evaluation/evaluator.py", line 11, in <module>
    import matplotlib
ModuleNotFoundError: No module named 'matplotlib'
```
Fails importing `src/evaluation/evaluator.py` (pulls in `matplotlib`). Even with the dep installed,
this endpoint also needs trained models + test data to produce a real comparison table.

### 8. GET /drift → 200 ✅
```json
{"symbols":{
   "RELIANCE.NS":{"status":"no_reports","symbol":"RELIANCE.NS"},
   "TCS.NS":{"status":"no_reports","symbol":"TCS.NS"},
   "INFY.NS":{"status":"no_reports","symbol":"INFY.NS"},
   "HDFCBANK.NS":{"status":"no_reports","symbol":"HDFCBANK.NS"},
   "WIPRO.NS":{"status":"no_reports","symbol":"WIPRO.NS"}},
 "timestamp":"2026-06-05T10:..."}
```
Gracefully reports `no_reports` per configured symbol (no drift reports generated yet). Never errors.

### 9. POST /retrain?trainer=manual → 500 🟡 (missing dependency)
Response body: `Internal Server Error`. Server traceback:
```
File "src/api/app.py", line 340, in retrain
    from src.training.manual_trainer import train_all
File "src/training/manual_trainer.py", line 25, in <module>
    import pandas as pd
ModuleNotFoundError: No module named 'pandas'
```
**Observation / minor finding:** the training module is imported *before* the background task is
scheduled, so the missing-dependency `ImportError` surfaces as a synchronous **500** rather than the
intended `200 "retraining_started"`. With the full training stack installed this returns 200 and the
job runs in the background. (In the bare env it cannot succeed regardless — no `pandas`, no market
data.) See *Issues* below.

### 10. POST /ask → 200 ✅
Payload `{"query":"Which stock shows the strongest buy signal this week?"}`.
```json
{"query":"Which stock shows the strongest buy signal this week?",
 "answer":"Based on current signals: No predictions available.",
 "timestamp":"2026-06-05T10:..."}
```
Returns 200 via the **template fallback**. Server log shows `Ollama call failed (granite4.1:3b):
No module named 'ollama'` — the handler degrades gracefully instead of erroring. "No predictions
available" because no models exist to summarize.

### Negative / validator cases

**N1 — POST /predict `{"symbol":"RELIANCE.US"}` → 422 ✅**
```json
{"detail":[{"type":"value_error","loc":["body","symbol"],
  "msg":"Value error, Symbol must end with .NS (NSE) or .BO (BSE)","input":"RELIANCE.US"}]}
```
`PredictRequest.validate_symbol` (`app.py:85-93`) correctly rejects non-NSE/BSE suffixes.

**N2 — POST /predict `{"symbol":"reliance"}` → 404 ✅**
Lowercase, no suffix → normalized to `RELIANCE.NS` by the validator, then 404 (no model). Confirms
auto-uppercase + `.NS` append works end-to-end.
> Note: an earlier run hit a **stale server** (started ~50 min prior, `reload=false`) that returned
> 422 here. After restarting on current code it correctly returns 404. Lesson: the service runs with
> `api.reload: false` (`config/config.yaml`), so the server must be restarted to pick up code changes.

**N3 — POST /ask `{"query":"hi"}` → 422 ✅**
```json
{"detail":[{"type":"string_too_short","loc":["body","query"],
  "msg":"String should have at least 5 characters","ctx":{"min_length":5}}]}
```
`AskRequest` `min_length=5` enforced.

**N4 — POST /predict/batch `{"symbols":[]}` → 422 ✅**
```json
{"detail":[{"type":"too_short","loc":["body","symbols"],
  "msg":"List should have at least 1 item after validation, not 0","ctx":{"min_length":1}}]}
```
`BatchPredictRequest` `min_length=1` enforced.

## Issues found

No functional API bugs. One **minor robustness observation**:

- **`POST /retrain` returns 500 instead of 200 when the training stack can't import.** The handler
  imports `src.training.manual_trainer` synchronously (`app.py:340`) before scheduling the background
  task, so any `ImportError` (or other import-time failure) propagates to the client as a 500. If the
  goal is "fire-and-forget", consider moving the import inside the background task function (or
  wrapping it) so the endpoint can still return `200 "retraining_started"` and surface import/training
  failures via logs / `/health` / `/models`. Low priority — only manifests when deps are missing.

## How to reproduce

```powershell
# 1. Start the server (current code; reload is off, so restart after edits)
.venv\Scripts\python.exe main.py --mode serve

# 2. Run the suite
.venv\Scripts\python.exe scripts\api_smoke_test.py > scripts\api_smoke_results.json
```
Raw per-request output (status, latency, full bodies) is in `scripts/api_smoke_results.json`.

## Next steps (to make model-dependent endpoints return real data)

1. **Install the inference stack** (note: Python 3.14 is very new — some wheels, esp. `torch`, may not
   yet be available; a 3.11/3.12 venv is the safer choice for the full stack):
   `pip install -r docker/requirements-api.txt`
2. **Generate features:** `python main.py --mode <ingest/engineer>` to populate `data/processed/`.
3. **Train models:** `python main.py --mode train --trainer manual` to populate `models/manual/`.
4. Re-run this suite — `/predict`, `/predict/batch`, `/recommendation`, `/comparison`, `/sentiment`
   should then return real payloads; `/ask` will summarize actual predictions (optionally install +
   run Ollama with `granite4.1:3b` for richer answers instead of the template fallback).
