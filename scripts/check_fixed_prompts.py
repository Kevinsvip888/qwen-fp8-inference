import argparse
import hashlib
import json
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, ProxyHandler, build_opener

parser = argparse.ArgumentParser()
parser.add_argument("--precision", choices=["bf16", "fp8"], required=True)
args = parser.parse_args()

root = Path(__file__).resolve().parents[1]
model_name = f"qwen-{args.precision}"

cases = [
    (
        "arithmetic",
        "Calculate 29 multiplied by 14. Answer with only the integer."
    ),
    (
        "reading",
        "A sensor recorded 18 degrees at 09:00, 21 degrees at 10:00, "
        "and 19 degrees at 11:00. What was the reading at 10:00? "
        "Answer with only the integer."
    ),
    (
        "json",
        "Extract the city and sensor count. Return only a JSON object "
        "with keys city and count. Text: There are three sensors in Stockholm."
    ),
    (
        "chinese",
        "一周有多少天？只输出一个阿拉伯数字。"
    ),
    (
        "summary",
        "Summarize this in one sentence: A building installed wireless "
        "sensors to monitor temperature. The sensors transmit readings "
        "every minute. Engineers use the readings to detect heating faults."
    ),
    (
        "code",
        "Write a Python function named clamp(x, low, high) that limits x "
        "to the inclusive interval [low, high]. Assume low <= high. "
        "Return only the function, without examples or explanation."
    ),
]

settings = {
    "temperature": 0,
    "top_p": 1,
    "max_tokens": 256,
    "seed": 42,
}

protocol = {"cases": cases, "settings": settings}
protocol_hash = hashlib.sha256(
    json.dumps(protocol, sort_keys=True, ensure_ascii=False).encode()
).hexdigest()

# Send requests directly to the local server.
opener = build_opener(ProxyHandler({}))
records = []

for case_id, prompt in cases:
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        **settings,
    }
    request = Request(
        "http://127.0.0.1:8000/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )

    try:
        with opener.open(request, timeout=120) as response:
            result = json.load(response)
    except HTTPError as error:
        raise RuntimeError(error.read().decode(errors="replace")) from error

    records.append({
        "id": case_id,
        "request": payload,
        "response": result,
    })

    choice = result["choices"][0]
    print(f"\n[{case_id}]")
    print(choice["message"]["content"])
    print("Finish reason:", choice["finish_reason"], flush=True)

report = {
    "precision": args.precision,
    "protocol_sha256": protocol_hash,
    "source_revision": json.loads(
        (root / "logs/model_revision.json").read_text()
    )["revision"],
    "results": records,
}

destination = root / "outputs" / f"{args.precision}_fixed_prompts.json"
destination.write_text(
    json.dumps(report, ensure_ascii=False, indent=2) + "\n"
)
print("\nSaved:", destination)
