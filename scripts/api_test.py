from pathlib import Path
from typing import Any, Dict, Optional
import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient
from src.api.app import app

client = TestClient(app)

ENDPOINTS = [
    {
        "name": "health",
        "method": "GET",
        "path": "/health",
        "payload": None,
        "description": "API health check",
    },
    {
        "name": "models",
        "method": "GET",
        "path": "/models",
        "payload": None,
        "description": "List available trained models",
    },
    {
        "name": "predict_infy",
        "method": "POST",
        "path": "/predict",
        "payload": {"symbol": "INFY"},
        "description": "Predict signal for INFY",
    },
    {
        "name": "predict_tcs",
        "method": "POST",
        "path": "/predict",
        "payload": {"symbol": "TCS"},
        "description": "Predict signal for TCS",
    },
    {
        "name": "predict_batch",
        "method": "POST",
        "path": "/predict/batch",
        "payload": {"symbols": ["INFY", "TCS"]},
        "description": "Batch prediction for INFY and TCS",
    },
    {
        "name": "recommendation_infy",
        "method": "GET",
        "path": "/recommendation/INFY",
        "payload": None,
        "description": "Recommendation endpoint for INFY",
    },
    {
        "name": "recommendation_tcs",
        "method": "GET",
        "path": "/recommendation/TCS",
        "payload": None,
        "description": "Recommendation endpoint for TCS",
    },
    {
        "name": "sentiment_infy",
        "method": "GET",
        "path": "/sentiment/INFY",
        "payload": None,
        "description": "Latest sentiment for INFY",
    },
    {
        "name": "sentiment_tcs",
        "method": "GET",
        "path": "/sentiment/TCS",
        "payload": None,
        "description": "Latest sentiment for TCS",
    },
    {
        "name": "ask",
        "method": "POST",
        "path": "/ask",
        "payload": {"query": "What is the recommendation for INFY and TCS today?"},
        "description": "Natural language query against current predictions",
    },
    {
        "name": "comparison",
        "method": "GET",
        "path": "/comparison",
        "payload": None,
        "description": "Comparison of trained models",
    },
    {
        "name": "drift",
        "method": "GET",
        "path": "/drift",
        "payload": None,
        "description": "Latest drift report status",
    },
    {
        "name": "retrain",
        "method": "POST",
        "path": "/retrain?trainer=manual",
        "payload": None,
        "description": "Trigger retraining in background",
    },
]


def run_request(method: str, path: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if method == "GET":
        response = client.get(path)
    elif method == "POST":
        response = client.post(path, json=payload)
    else:
        raise ValueError(f"Unsupported method: {method}")

    result = {
        "status_code": response.status_code,
        "headers": {k: v for k, v in response.headers.items() if k.lower() in ["content-type"]},
    }
    try:
        result["body"] = response.json()
    except ValueError:
        result["body"] = response.text
    return result


def format_yaml_block(value: Any) -> str:
    import json
    return json.dumps(value, indent=2, ensure_ascii=False)


def main() -> None:
    out_lines = [
        "# API Payloads and Executed Responses",
        "",
        "This document lists the API payloads for all routes plus actual internal responses from the current codebase.",
        "",
    ]

    for endpoint in ENDPOINTS:
        out_lines.append(f"## {endpoint['name']}")
        out_lines.append("")
        out_lines.append(f"**Route:** `{endpoint['method']} {endpoint['path']}`")
        out_lines.append("")
        out_lines.append(f"**Description:** {endpoint['description']}")
        out_lines.append("")
        if endpoint["payload"] is not None:
            out_lines.append("**Request body:**")
            out_lines.append("```json")
            out_lines.append(format_yaml_block(endpoint["payload"]))
            out_lines.append("```")
            out_lines.append("")
        response = run_request(endpoint["method"], endpoint["path"], endpoint["payload"])
        out_lines.append("**Response status:** ``" + str(response["status_code"]) + "``")
        out_lines.append("")
        out_lines.append("**Response body:**")
        out_lines.append("```json")
        import json
        body = response["body"]
        if isinstance(body, str):
            out_lines.append(body)
        else:
            out_lines.append(json.dumps(body, indent=2, ensure_ascii=False))
        out_lines.append("```")
        out_lines.append("")
        out_lines.append("---")
        out_lines.append("")

    out_path = Path(__file__).resolve().parents[1] / "API_PAYLOADS.md"
    out_path.write_text("\n".join(out_lines), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
