#!/usr/bin/env python3
"""Run the complete, frozen final tests for one selected checkpoint.

The forwarding recorder accepts only the frozen final-test prompts. It records
raw responses, checks prompt token counts, and cross-checks ARC likelihoods
against the harness samples. It never stops the user's vLLM server.
"""

import argparse
import ast
import csv
import re
import copy
from collections import defaultdict, deque
from datetime import datetime, timezone
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.metadata import distribution, version
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener

from transformers import AutoTokenizer

TASKS = ("day7_gsm8k_test", "day7_arc_challenge_test")
PLAN_SHA256 = "1bd1f014f7409f32e93347cbd25e39eca79a782a614acd0234a5ed7c003e4558"
PREPARATION_SHA256 = "1ddf3a16aad32e1273d90b2e0ba7dd0d84f9e0edd281051c396fc9ca405faa95"
BASE = "http://127.0.0.1:8000"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def http_json(path):
    with build_opener(ProxyHandler({})).open(BASE + path, timeout=10) as response:
        return response.status, json.load(response)


def one_prompt(payload):
    prompt = payload["prompt"]
    if isinstance(prompt, list) and len(prompt) == 1 and isinstance(prompt[0], list):
        prompt = prompt[0]
    require(isinstance(prompt, list) and prompt and
            all(type(token) is int for token in prompt),
            "Expected one tokenized prompt per API request")
    return prompt


class Recorder(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, directory, expected, alias):
        super().__init__(("127.0.0.1", 0), Handler)
        self.directory, self.expected, self.alias = directory, expected, alias
        self.lock = threading.Lock()
        self.sequence = 0


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def do_POST(self):
        server = self.server
        with server.lock:
            server.sequence += 1
            number = server.sequence
        receipt = {"sequence": number, "path": self.path}
        start = time.perf_counter()
        status, result = 502, {"error": "Forwarding did not complete"}
        try:
            require(self.path == "/v1/completions", "Unexpected endpoint")
            size = int(self.headers.get("Content-Length", "0"))
            require(0 < size <= 10 * 1024 * 1024, "Invalid request body size")
            body = self.rfile.read(size)
            payload = json.loads(body)
            receipt["request"] = payload
            require(payload["model"] == server.alias, "Unexpected model alias")
            require(not payload.get("stream", False), "Recorder expects non-streaming requests")
            require(payload.get("temperature") == 0, "Expected greedy requests")
            require(payload.get("seed") == 42, "Request seed differs from 42")
            prompt = one_prompt(payload)
            kind = "arc" if payload.get("echo", False) else "math"
            if kind == "math":
                require(payload["max_tokens"] == 2048 and payload.get("top_p", 1) == 1,
                        "Math generation settings differ")
                require(set(("Question:", "</s>", "<|im_end|>")) <= set(payload.get("stop", [])),
                        "Expected math stop strings are missing")
            else:
                require(payload["max_tokens"] == 1 and payload.get("logprobs") == 1,
                        "Expected echoed ARC token likelihoods")
            with server.lock:
                matches = server.expected.get((kind, tuple(prompt)))
                require(matches is not None and len(matches) > 0,
                        "Prompt differs from the frozen tests, or was sent more than once")
                receipt["expected"] = matches.popleft()
            request = Request(BASE + self.path, data=body,
                              headers={"Content-Type": "application/json"})
            opener = build_opener(ProxyHandler({}))
            try:
                with opener.open(request, timeout=300) as response:
                    status, raw = response.status, response.read()
            except HTTPError as error:
                status, raw = error.code, error.read()
            result = json.loads(raw)
            receipt.update(http_status=status, response=result)
        except Exception as error:
            receipt["error"] = f"{type(error).__name__}: {error}"
            status, result = 502, {"error": receipt["error"]}
        finally:
            receipt["wall_seconds"] = time.perf_counter() - start
            save_json(server.directory / f"record_{number:06d}.json", receipt)
        output = json.dumps(result, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(output)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(output)
        self.close_connection = True


def loglikelihood(value):
    """Read a saved likelihood pair, including string-serialized numbers."""
    while isinstance(value, list) and len(value) == 1:
        value = value[0]
    require(isinstance(value, list) and len(value) == 2, "Unexpected likelihood pair")
    score, flag = value
    require(not isinstance(score, bool), "Invalid likelihood score")
    try:
        score = float(score)
    except (TypeError, ValueError):
        raise RuntimeError("Likelihood score is not numeric")
    require(math.isfinite(score), "Likelihood score is not finite")
    require(type(flag) is bool or
            (isinstance(flag, str) and flag in ("True", "False")),
            "Invalid greedy flag")
    return score


def audit_saved(out, alias, counts, expected, total, original_wall_seconds=None):
    """Audit saved requests/results without contacting any model server."""
    raw_dir = out / "http_records"
    models = read_json(out / "models.json")
    cards = [card for card in models["data"] if card["id"] == alias]
    require(len(cards) == 1 and cards[0].get("max_model_len") == 8192,
            "Saved server identity/context differs")
    receipts = [read_json(path) for path in sorted(raw_dir.glob("record_*.json"))]
    require(len(receipts) == total, "Unexpected API request count")
    math_receipts, arc_receipts = {}, {}
    for item in receipts:
        require("error" not in item and item.get("http_status") == 200,
                f"Failed recorded request {item['sequence']}")
        expected_item, response = item["expected"], item["response"]
        payload = item["request"]
        kind = "arc" if payload.get("echo", False) else "math"
        matches = expected.get((kind, tuple(one_prompt(payload))))
        require(matches is not None and len(matches) > 0,
                "Saved prompt differs from the frozen final requests or is duplicated")
        require(matches.popleft() == expected_item,
                "Saved prompt/document identity differs from the prepared request")
        require(payload["model"] == alias and payload.get("seed") == 42
                and payload.get("temperature") == 0,
                "Saved generation settings differ")
        require(payload["max_tokens"] == (2048 if kind == "math" else 1),
                "Saved token limit differs")
        require(response["model"] == alias and len(response["choices"]) == 1,
                "Unexpected response model or choice count")
        require(response["usage"]["prompt_tokens"] == expected_item["prompt_tokens"],
                "Server prompt token count differs from the frozen request")
        choice = response["choices"][0]
        if expected_item["kind"] == "math":
            require(choice["finish_reason"] in ("stop", "length"), "Unexpected math finish reason")
            require(0 < response["usage"]["completion_tokens"] <= 2048,
                    "Unexpected math output token count")
            math_receipts[expected_item["doc_id"]] = item
        else:
            require(response["usage"]["completion_tokens"] == 1,
                    "Expected one generated token after the echoed ARC prompt")
            values = choice["logprobs"]["token_logprobs"]
            require(len(values) == expected_item["prompt_tokens"] + 1,
                    "Expected all prompt logprobs followed by one generated token")
            terms = values[expected_item["context_tokens"]:expected_item["prompt_tokens"]]
            require(terms and all(isinstance(x, (int, float)) and math.isfinite(x) for x in terms),
                    "ARC continuation logprobs are missing or invalid")
            item["recomputed_loglikelihood"] = sum(terms)
            arc_receipts[(expected_item["doc_id"], expected_item["choice_index"])] = item

    require(sum(len(items) for items in expected.values()) == 0,
            "Saved run is missing frozen requests")
    require(len(math_receipts) == counts[TASKS[0]], "Saved math request count differs")

    results = []
    for path in (out / "harness").rglob("*.json"):
        data = read_json(path)
        if isinstance(data, dict) and set(TASKS) <= set(data.get("results", {})):
            results.append((path, data))
    require(len(results) == 1, "Could not identify a unique harness results file")
    samples_by_task = {}
    sample_filter_counts = {}
    for task in TASKS:
        paths = [p for p in (out / "harness").rglob("*.jsonl") if task in p.name]
        require(len(paths) == 1, f"Expected one samples file for {task}")
        samples = [json.loads(line) for line in paths[0].read_text().splitlines() if line.strip()]
        expected_filters = ({"strict-match", "flexible-extract"}
                            if task == TASKS[0] else {"none"})
        pairs = [(sample["doc_id"], sample.get("filter")) for sample in samples]
        require(all(type(doc_id) is int for doc_id, _ in pairs),
                "Expected integer harness document IDs")
        require(len(pairs) == counts[task] * len(expected_filters)
                and len(set(pairs)) == len(pairs)
                and set(pairs) == {(doc_id, name) for doc_id in range(counts[task])
                                  for name in expected_filters},
                f"Missing or duplicate document/filter sample pairs for {task}")
        sample_filter_counts[task] = {
            name: sum(sample["filter"] == name for sample in samples)
            for name in sorted(expected_filters)
        }
        samples_by_task[task] = samples
        for sample in samples:
            doc_id = sample["doc_id"]
            if task == TASKS[0]:
                raw = math_receipts[doc_id]["response"]["choices"][0]["text"]
                generated = sample["resps"][0]
                while isinstance(generated, list) and len(generated) == 1:
                    generated = generated[0]
                require(isinstance(generated, str) and generated.strip() == raw.strip(),
                        "Harness math text differs from the recorded response")
            else:
                for index, value in enumerate(sample["resps"]):
                    raw_ll = arc_receipts[(doc_id, index)]["recomputed_loglikelihood"]
                    require(math.isclose(loglikelihood(value), raw_ll, rel_tol=1e-8, abs_tol=1e-5),
                            "ARC harness loglikelihood differs from the raw token probabilities")

    metrics = results[0][1]["results"]
    length_finishes = sum(x["response"]["choices"][0]["finish_reason"] == "length"
                          for x in math_receipts.values())
    for task, samples in samples_by_task.items():
        for name in sample_filter_counts[task]:
            group = [sample for sample in samples if sample["filter"] == name]
            names = ("exact_match",) if task == TASKS[0] else ("acc", "acc_norm")
            for metric in names:
                values = [sample[metric] for sample in group]
                require(all(isinstance(x, (int, float)) and math.isfinite(x)
                            and 0 <= x <= 1 for x in values), "Invalid per-question metric")
                require(math.isclose(sum(values) / len(values),
                                     metrics[task][f"{metric},{name}"], abs_tol=1e-10),
                        "Per-filter sample scores differ from the aggregate result")
    hashes = {str(p.relative_to(out)): sha256(p) for p in out.rglob("*")
              if p.is_file() and p.name != "final_summary.json"}
    strict_samples = [x for x in samples_by_task[TASKS[0]] if x["filter"] == "strict-match"]
    strict_extraction_failures = sum(re.search(r"#### (\-?[0-9\.\,]+)",
        math_receipts[x["doc_id"]]["response"]["choices"][0]["text"]) is None for x in strict_samples)
    summary = {"strict_extraction_failures": strict_extraction_failures,
               "correct_math": int(sum(x["exact_match"] for x in strict_samples)),
               "correct_science_acc_norm": int(sum(x["acc_norm"] for x in samples_by_task[TASKS[1]])),
               "scope": "full official tests", "questions_per_task": counts,
               "math_length_finishes": length_finishes, "http_requests_verified": total,
               "wall_seconds_including_client_startup": original_wall_seconds,
               "recorded_api_wall_seconds": sum(item["wall_seconds"] for item in receipts),
               "sample_records_per_filter": sample_filter_counts,
               "auditor_sha256": sha256(Path(__file__).resolve()),
               "metrics": metrics, "raw_response_and_result_sha256": hashes}
    save_json(out / "final_summary.json", summary)
    print(f"\nMath strict-match: {100 * metrics[TASKS[0]]['exact_match,strict-match']:.2f}%")
    print(f"Science acc_norm: {100 * metrics[TASKS[1]]['acc_norm,none']:.2f}%")
    print(f"Math length finishes: {length_finishes}/{counts[TASKS[0]]}")
    print(f"Strict answer-extraction failures: {strict_extraction_failures}")
    print(f"Saved: {out}")
    print("PASS: raw responses, exact prompt tokens and ARC likelihoods match the harness.")
    print("Full frozen test evaluation complete for this checkpoint.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/root/autodl-tmp/qwen-inference-lab"))
    parser.add_argument("--model", required=True, help="Name in final_plan.json, e.g. Random32-seed42")
    parser.add_argument("--server-log", type=Path, default=Path("logs/evaluation/day7/fp8_server.log"))
    parser.add_argument("--audit-only", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    folder = root / "data/evaluation/day7_quality_v1"
    plan_path, prepared = folder / "final_plan.json", folder / "final"
    preparation_path = prepared / "preparation_manifest.json"
    require(sha256(plan_path) == PLAN_SHA256, "Frozen final plan changed")
    require(sha256(preparation_path) == PREPARATION_SHA256, "Final preparation changed")
    plan, preparation = read_json(plan_path), read_json(preparation_path)
    selected = [m for m in plan["models"] if m["name"] == args.model]
    require(len(selected) == 1, "Select one model from the frozen ten-model plan")
    selected = selected[0]
    development_path = folder / "development_protocol.json"
    require(sha256(development_path) == plan["development_protocol_sha256"], "Development settings changed")
    development = read_json(development_path)
    for mapping in (development["file_sha256"], preparation["file_sha256"]):
        for relative, digest in mapping.items():
            require(sha256(root / relative) == digest, f"Frozen input changed: {relative}")
    for name, expected_version in development["packages"].items():
        require(version(name) == expected_version, f"Evaluation package changed: {name}")
    harness = Path(distribution("lm_eval").locate_file("lm_eval"))
    require({str(p.relative_to(harness)): sha256(p) for p in harness.rglob("*.py")}
            == read_json(folder / "harness_source_hashes.json"), "Harness source changed")
    model_path = root / selected["path"]
    if args.model != "BF16":
        checkpoint_path = model_path / "calibration_manifest.json"
        require(sha256(checkpoint_path) == selected["calibration_manifest_sha256"], "FP8 checkpoint identity changed")
    alias = "qwen-quality-bf16" if args.model == "BF16" else "qwen-quality-fp8"
    counts = {name: preparation["tasks"][name]["questions"] for name in TASKS}
    require(counts == {TASKS[0]: 1319, TASKS[1]: 1172}, "Full test counts differ")
    tokenizer = AutoTokenizer.from_pretrained(root / "models/Qwen3-4B-Instruct-2507",
                                             local_files_only=True, trust_remote_code=False)
    expected, total = defaultdict(deque), 0
    for task in TASKS:
        for line in (prepared / (task + "_requests.jsonl")).read_text().splitlines():
            entry = json.loads(line)
            context_ids = tokenizer.encode(entry["context"], add_special_tokens=False)
            kind = "math" if task == TASKS[0] else "arc"
            text = entry["context"] + entry.get("continuation", "")
            tokens = tokenizer.encode(text, add_special_tokens=False)
            require(tokens[:len(context_ids)] == context_ids, "Token boundary changed")
            metadata = {"task": task, "doc_id": entry["doc_id"], "source_id": entry["source_id"],
                        "kind": kind, "context_tokens": len(context_ids), "prompt_tokens": len(tokens)}
            if kind == "arc":
                metadata["choice_index"] = entry["choice_index"]
            expected[(kind, tuple(tokens))].append(metadata)
            total += 1
    require(total == sum(preparation["tasks"][name]["api_requests"] for name in TASKS),
            "Final API request count differs")
    if args.audit_only is not None:
        out = (args.audit_only if args.audit_only.is_absolute() else root / args.audit_only).resolve()
        saved = read_json(out / "run_manifest.json")
        require(saved["model"] == args.model and saved["final_plan_sha256"] == PLAN_SHA256
                and saved["preparation_sha256"] == PREPARATION_SHA256, "Saved run identity differs")
        print("Auditing saved FULL TEST responses; no inference.", flush=True)
        prior = out / "final_summary.json"
        elapsed = read_json(prior).get("wall_seconds_including_client_startup") if prior.exists() else None
        audit_saved(out, alias, counts, expected, total, elapsed)
        return

    parent = root / "outputs/evaluation/day7_quality_v1/final" / args.model
    require(not list(parent.glob("*/final_summary.json")),
            "A completed final run already exists; use --audit-only to verify it")
    status, models = http_json("/v1/models")
    cards = [c for c in models["data"] if c["id"] == alias]
    require(status == 200 and len(cards) == 1 and cards[0].get("max_model_len") == 8192,
            "Expected the quality server with an 8192-token context")
    require((root / cards[0]["root"]).resolve() == model_path.resolve(),
            "The running server is using a different checkpoint")
    server_log = args.server_log if args.server_log.is_absolute() else root / args.server_log
    log_text = server_log.read_text(errors="replace")
    log_text = re.sub(r"\x1b\[[0-9;]*m", "", log_text)
    configs = [ast.literal_eval(line.split("non-default args:", 1)[1].strip())
               for line in log_text.splitlines() if "non-default args:" in line]
    require(configs, "Cannot find startup configuration in server log")
    server_config = configs[-1]
    for name, value in {"max_model_len": 8192, "dtype": "bfloat16", "kv_cache_dtype": "bfloat16",
                        "kv_cache_memory_bytes": 4294967296, "attention_backend": "FLASH_ATTN",
                        "max_num_seqs": 4, "max_num_batched_tokens": 4096, "seed": 42,
                        "generation_config": "vllm", "enable_prefix_caching": False}.items():
        require(server_config.get(name) == value, f"Server setting differs: {name}")
    require(server_config.get("quantization") == (None if args.model == "BF16" else "modelopt"),
            "Server quantization differs")
    require((root / server_config["model"]).resolve() == model_path.resolve(), "Server log checkpoint differs")
    if args.model != "BF16":
        require("Selected CutlassFP8ScaledMMLinearKernel" in log_text, "Missing FP8 kernel evidence")
        checkpoint = read_json(model_path / "calibration_manifest.json")
        print("Verifying exported checkpoint files...", flush=True)
        for item in checkpoint["export_files"]:
            path = model_path / item["path"]
            require(path.stat().st_size == item["bytes"] and sha256(path) == item["sha256"],
                    f"Exported checkpoint file changed: {path}")
    weight_hashes = {path.name: sha256(path) for path in sorted(model_path.glob("*.safetensors"))}
    require(weight_hashes, "Checkpoint weight files are missing")
    gpu_lines = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid,name,driver_version",
                                         "--format=csv,noheader"], text=True).strip().splitlines()
    require(len(gpu_lines) == 1, "Expected a single GPU")
    gpu = [part.strip() for part in next(csv.reader(gpu_lines))]
    serve_versions = json.loads(subprocess.check_output([
        str(root / ".venv-serve/bin/python"), "-c",
        "import json;from importlib.metadata import version;print(json.dumps({n:version(n) for n in ('torch','vllm','transformers')}))"], text=True))
    require(serve_versions == {"torch": "2.13.0+cu132", "vllm": "0.32.0", "transformers": "5.19.0"},
            "Serving environment differs from the validated environment")
    runtime = {"final_plan_sha256": PLAN_SHA256, "preparation_sha256": PREPARATION_SHA256,
               "gpu_uuid_name_driver": gpu, "serve_versions": serve_versions,
               "evaluation_versions": development["packages"], "api_concurrency": 1,
               "tokenized_requests": True, "add_bos_token": False, "generation_seed": 42}
    runtime_path = root / "outputs/evaluation/day7_quality_v1/final/runtime_protocol.json"
    runtime_path.parent.mkdir(parents=True, exist_ok=True)
    if runtime_path.exists():
        require(read_json(runtime_path) == runtime, "Final execution environment/GPU differs")
    else:
        save_json(runtime_path, runtime)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out = parent / stamp
    raw_dir = out / "http_records"
    raw_dir.mkdir(parents=True)
    shutil.copy2(server_log, out / "server_before.log")
    save_json(out / "models.json", models)
    expected_for_audit = copy.deepcopy(expected)
    recorder = Recorder(raw_dir, expected, alias)
    thread = threading.Thread(target=recorder.serve_forever, daemon=True)
    thread.start()
    model_args = ",".join((f"model={alias}",
        f"base_url=http://127.0.0.1:{recorder.server_port}/v1/completions",
        f"tokenizer={root / 'models/Qwen3-4B-Instruct-2507'}", "tokenizer_backend=huggingface",
        "tokenized_requests=True", "add_bos_token=False", "max_length=8192", "max_gen_toks=2048",
        "num_concurrent=1", "max_retries=1", "timeout=300", "seed=42"))
    command = [sys.executable, "-m", "lm_eval", "run", "--model", "local-completions",
               "--model_args", model_args, "--tasks", ",".join(TASKS),
               "--include_path", str(prepared / "tasks"), "--batch_size", "1",
               "--seed", "42,42,42,42", "--log_samples", "--output_path", str(out / "harness")]
    environment = os.environ.copy()
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        environment.pop(name, None)
    environment.update(HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                       NO_PROXY="127.0.0.1,localhost", no_proxy="127.0.0.1,localhost",
                       OPENAI_API_KEY="local-evaluation", PYTHONUNBUFFERED="1")
    save_json(out / "run_manifest.json", {"scope": "full official tests", "model": args.model,
        "checkpoint": selected, "weight_sha256": weight_hashes, "final_plan_sha256": PLAN_SHA256,
        "preparation_sha256": PREPARATION_SHA256, "runtime_protocol_sha256": sha256(runtime_path),
        "runner_sha256": sha256(Path(__file__).resolve()), "command": command,
        "questions_per_task": counts, "expected_http_requests": total,
        "native_api_attempts_per_request": 1, "server_config": server_config,
        "note": "Recorded timings are not serving benchmark measurements."})
    print(f"Running FULL TESTS: {args.model}; math=1319, science=1172; API requests={total}", flush=True)
    started, process = time.perf_counter(), None
    try:
        with (out / "client.log").open("wb") as log:
            process = subprocess.Popen(command, cwd=root, env=environment,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            while chunk := process.stdout.read1(4096):
                log.write(chunk); log.flush()
                sys.stdout.buffer.write(chunk); sys.stdout.buffer.flush()
            returncode = process.wait()
        require(returncode == 0, f"Harness failed; saved responses remain in {out}")
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait()
        recorder.shutdown(); recorder.server_close(); thread.join(timeout=5)
        shutil.copy2(server_log, out / "server_after.log")
    require(sum(len(items) for items in expected.values()) == 0, "Some final requests were not sent")
    audit_saved(out, alias, counts, expected_for_audit, total, time.perf_counter() - started)


if __name__ == "__main__":
    main()


