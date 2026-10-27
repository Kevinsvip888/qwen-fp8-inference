import argparse
import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, ProxyHandler, build_opener

ROOT = Path("/root/autodl-tmp/qwen-inference-lab")
BASE_URL = "http://127.0.0.1:8000"

parser = argparse.ArgumentParser()
parser.add_argument(
    "--method", required=True,
    choices=["Random32", "Selected32", "Random128"],
)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

alias = f"qwen-{args.method.lower()}"
checkpoint = ROOT / (
    f"models/Qwen3-4B-Instruct-2507-FP8-{args.method}-seed{args.seed}"
)
manifest = json.loads(
    (checkpoint / "calibration_manifest.json").read_text()
)
assert manifest["identity"]["method"] == args.method
assert manifest["identity"]["pool_seed"] == args.seed

baseline = json.loads(
    (ROOT / "outputs/bf16_fixed_prompts.json").read_text()
)
assert baseline["source_revision"] == (
    manifest["protocol"]["source"]["revision"]
)
assert len(baseline["results"]) == 6

cases = []
for original in baseline["results"]:
    request = copy.deepcopy(original["request"])
    request["model"] = alias
    cases.append({
        "id": original["id"],
        "request": request,
        "expected_prompt_tokens": (
            original["response"]["usage"]["prompt_tokens"]
        ),
    })

# Exclude the served-model alias from the common request-protocol hash.
canonical = [
    {
        "id": case["id"],
        "request": {
            key: value
            for key, value in case["request"].items()
            if key != "model"
        },
    }
    for case in cases
]
protocol_hash = hashlib.sha256(
    json.dumps(canonical, sort_keys=True).encode()
).hexdigest()

# Bypass proxy settings for all localhost requests.
opener = build_opener(ProxyHandler({}))

def api_json(path, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    request = Request(
        BASE_URL + path,
        data=data,
        headers={"Content-Type": "application/json"},
    )
    try:
        with opener.open(request, timeout=180) as response:
            return json.load(response)
    except HTTPError as error:
        detail = error.read().decode(errors="replace")
        raise RuntimeError(
            f"HTTP {error.code}: {detail[:2000]}"
        ) from error

with opener.open(BASE_URL + "/health", timeout=10) as response:
    assert response.status == 200

models = api_json("/v1/models")
available = {card["id"] for card in models["data"]}
assert alias in available, (
    f"Expected {alias}; server exposes {sorted(available)}"
)

stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
output_dir = ROOT / "outputs/serving_smoke" / f"seed{args.seed}"
output_dir.mkdir(parents=True, exist_ok=True)
output = output_dir / f"{args.method}_{stamp}.json"

report = {
    "method": args.method,
    "pool_seed": args.seed,
    "checkpoint": str(checkpoint),
    "export_identity": manifest["identity"],
    "baseline_protocol_sha256": baseline["protocol_sha256"],
    "request_protocol_sha256": protocol_hash,
    "model_listing": models,
    "completed": False,
    "results": [],
}

def save():
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(output)

save()

for case in cases:
    response = api_json("/v1/chat/completions", case["request"])
    report["results"].append({
        "id": case["id"],
        "request": case["request"],
        "response": response,
    })
    save()

    assert response["model"] == alias
    assert len(response["choices"]) == 1
    choice = response["choices"][0]
    content = choice["message"]["content"]
    usage = response["usage"]

    assert isinstance(content, str) and content.strip()
    assert usage["prompt_tokens"] == case["expected_prompt_tokens"]
    assert choice["finish_reason"] == "stop", (
        case["id"], choice["finish_reason"]
    )

    print(
        f"\n[{case['id']}] "
        f"input={usage['prompt_tokens']}; "
        f"output={usage['completion_tokens']}; "
        f"finish={choice['finish_reason']}"
    )
    print(content)

log_path = ROOT / f"logs/serve_{args.method}_seed{args.seed}.log"
lines = log_path.read_text(errors="replace").splitlines()
kernel_lines = [
    line for line in lines
    if "Selected" in line and "LinearKernel" in line
]
report["linear_kernel_log_lines"] = kernel_lines
report["completed"] = True
save()

print("\nLinear kernel evidence:")
for line in kernel_lines:
    print(line)
if not kernel_lines:
    print("CHECK: no selected linear-kernel lines found in the log.")

print("\nRequest protocol SHA256:", protocol_hash)
print("Saved:", output)
print(
    f"PASS: {args.method}: six responses completed; "
    "input token counts match BF16."
)
