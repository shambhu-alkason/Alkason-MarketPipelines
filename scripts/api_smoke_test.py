"""
Throwaway API smoke-test harness (stdlib only — urllib).
Runs all 10 endpoints plus negative validator cases against the local server,
captures status / latency / body, and writes a JSON result blob to stdout.
"""
import json
import time
import urllib.request
import urllib.error

BASE = "http://localhost:8000"
TIMEOUT = 60  # seconds per request

# (id, label, method, path, json_body_or_None, kind)
# kind: "positive" | "negative"
CASES = [
    ("1",  "Health",            "GET",  "/health",                       None,                                     "positive"),
    ("2",  "Models",            "GET",  "/models",                       None,                                     "positive"),
    ("3",  "Predict",           "POST", "/predict",                      {"symbol": "RELIANCE.NS"},                "positive"),
    ("4",  "Predict batch",     "POST", "/predict/batch",                {"symbols": ["RELIANCE.NS", "TCS.NS", "INFY.NS"]}, "positive"),
    ("5",  "Recommendation",    "GET",  "/recommendation/RELIANCE.NS",   None,                                     "positive"),
    ("6",  "Sentiment",         "GET",  "/sentiment/RELIANCE",           None,                                     "positive"),
    ("7",  "Comparison",        "GET",  "/comparison",                   None,                                     "positive"),
    ("8",  "Drift",             "GET",  "/drift",                        None,                                     "positive"),
    ("9",  "Retrain",           "POST", "/retrain?trainer=manual",       None,                                     "positive"),
    ("10", "Ask",               "POST", "/ask",                          {"query": "Which stock shows the strongest buy signal this week?"}, "positive"),
    # Negative / validator cases
    ("N1", "Predict bad suffix","POST", "/predict",                      {"symbol": "RELIANCE.US"},                "negative"),
    ("N2", "Predict normalize", "POST", "/predict",                      {"symbol": "reliance"},                   "negative"),
    ("N3", "Ask too short",     "POST", "/ask",                          {"query": "hi"},                          "negative"),
    ("N4", "Batch empty",       "POST", "/predict/batch",                {"symbols": []},                          "negative"),
]


def run_case(case):
    cid, label, method, path, body, kind = case
    url = BASE + path
    data = None
    headers = {"Accept": "application/json"}
    if body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    t0 = time.perf_counter()
    status = None
    resp_body = None
    err = None
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            status = r.status
            resp_body = r.read().decode(errors="replace")
    except urllib.error.HTTPError as e:
        status = e.code
        resp_body = e.read().decode(errors="replace")
    except Exception as e:  # noqa: BLE001
        err = f"{type(e).__name__}: {e}"
    latency_ms = round((time.perf_counter() - t0) * 1000, 1)
    # try to pretty-parse JSON
    parsed = None
    if resp_body is not None:
        try:
            parsed = json.loads(resp_body)
        except Exception:  # noqa: BLE001
            parsed = None
    return {
        "id": cid, "label": label, "method": method, "path": path,
        "request_body": body, "kind": kind,
        "status": status, "latency_ms": latency_ms,
        "transport_error": err,
        "response_text": resp_body,
        "response_json": parsed,
    }


def main():
    results = [run_case(c) for c in CASES]
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
