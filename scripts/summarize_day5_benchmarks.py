#!/usr/bin/env python3
"""Validate the saved Day 5 study and write full-precision JSON/Markdown reports.

Uses only Python's standard library. Reads completed results; performs no GPU
work, networking, package installation, or benchmark reruns. The two derived
report files are atomically replaced only after all 36 runs pass validation.
"""

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
import sys
from pathlib import Path


METHODS = ("BF16", "Random32")
INPUT_LENGTHS = (256, 1024, 2048)
CONCURRENCIES = (1, 4)
METRICS = (
    "output_throughput", "mean_ttft_ms", "mean_tpot_ms", "mean_e2el_ms",
)
TAIL_METRICS = ("p95_ttft_ms", "p95_tpot_ms", "p95_e2el_ms")
REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read {path}: {error}") from error


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def positive(value, label):
    value = float(value)
    require(math.isfinite(value) and value > 0, f"Invalid {label}: {value}")
    return value


def describe(values):
    return {
        "mean": statistics.mean(values),
        "sample_sd": statistics.stdev(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def read_protocol(project, base):
    protocol = read_json(base / "protocol.json")
    expected = {
        "study": "day5_formal_v1", "requests": 200, "warmup_requests": 20,
        "output_tokens": 128, "workload_seed": 42, "generation_seed": 42,
        "temperature": 0, "top_p": 1, "ignore_eos": True,
        "kv_cache_bytes": 4294967296, "kv_cache_dtype": "bfloat16",
        "attention_backend": "FLASH_ATTN", "prefix_caching": False,
        "source_revision": REVISION,
    }
    for key, value in expected.items():
        require(protocol.get(key) == value, f"Unexpected protocol {key}")
    require(protocol.get("gpu_uuid"), "Protocol is missing GPU UUID")
    require(protocol.get("packages") == {
        "torch": "2.13.0+cu132", "vllm": "0.32.0", "transformers": "5.19.0",
    }, "Unexpected recorded serving package versions")
    bindings = {
        "server_script_sha256": project / "scripts/serve_day5_benchmark.sh",
        "runner_script_sha256": project / "scripts/run_day5_benchmark.sh",
        "fp8_manifest_sha256": project / "models" /
        "Qwen3-4B-Instruct-2507-FP8-Random32-seed42/calibration_manifest.json",
    }
    for key, path in bindings.items():
        require(sha256(path) == protocol[key], f"File identity changed: {path}")
    source = read_json(project / "logs/model_revision.json")
    require(source.get("repo_id") == "Qwen/Qwen3-4B-Instruct-2507", "Wrong source repo")
    require(source.get("revision") == REVISION, "Wrong source revision")
    exported = read_json(bindings["fp8_manifest_sha256"])
    identity = exported["identity"]
    require(identity.get("method") == "Random32", "Wrong FP8 calibration method")
    require(identity.get("pool_seed") == 42, "Wrong FP8 calibration pool")
    require(identity.get("protocol_sha256") ==
            "aaca5144109cec26410a142939d5d65e69551bbb8990dd5d1267ec8b25528e9c",
            "Wrong FP8 calibration protocol")
    return protocol


def validate_run(project, base, protocol, method, length, concurrency, repetition):
    directory = base / method / f"isl{length}_osl128_c{concurrency}" / f"rep{repetition}"
    report = read_json(directory / "benchmark.json")
    memory = read_json(directory / "memory_summary.json")
    require(report.get("completed") == 200, f"Incomplete run: {directory}")
    require(report.get("input_lens") == [length] * 200, f"Wrong input lengths: {directory}")
    require(report.get("output_lens") == [128] * 200, f"Wrong output lengths: {directory}")
    if "failed" in report:
        require(report["failed"] == 0, f"Failed requests: {directory}")
    if "errors" in report:
        require(all(not error for error in report["errors"]), f"Request errors: {directory}")
    if "max_concurrency" in report:
        require(report["max_concurrency"] == concurrency, f"Wrong concurrency: {directory}")
    require(sha256(directory / "environment.txt") == protocol["environment_sha256"],
            f"Different environment: {directory}")

    client_log = (directory / "client.log").read_text(errors="replace")
    for field in (
        f"max_concurrency={concurrency}", f"random_input_len={length}",
        "num_prompts=200", "num_warmups=20", "ignore_eos=True",
    ):
        require(field in client_log, f"Client configuration evidence missing: {field}: {directory}")
    server_log = (directory / "server.log").read_text(errors="replace")
    alias = "qwen-bench-bf16" if method == "BF16" else "qwen-bench-fp8"
    require(alias in server_log, f"Wrong/missing server alias: {directory}")
    capacities = re.findall(r"GPU KV cache size:\s*([\d,]+)\s*tokens", server_log)
    require(capacities and all(int(x.replace(",", "")) == 29120 for x in capacities),
            f"Unverified KV-cache token capacity: {directory}")
    require("reserved 4.0 GiB memory for KV Cache" in server_log,
            f"Unverified KV-cache byte budget: {directory}")
    require("flash_attn" in server_log.lower(), f"Unverified attention backend: {directory}")
    if method == "Random32":
        require("CutlassFP8ScaledMMLinearKernel" in server_log,
                f"Unverified FP8 linear kernels: {directory}")

    samples = []
    with (directory / "gpu_samples.csv").open(newline="") as handle:
        for row in csv.reader(handle):
            if len(row) != 6:
                continue
            try:
                used = float(row[2])
            except ValueError:
                continue
            require(row[1].strip() == protocol["gpu_uuid"], f"GPU UUID changed: {directory}")
            require(math.isfinite(used) and used >= 0, f"Invalid GPU sample: {directory}")
            samples.append(used)
    require(samples, f"Missing GPU samples: {directory}")
    require(memory.get("sample_interval_seconds") == 1, f"Wrong sample interval: {directory}")
    require(memory.get("scope") == "whole GPU during benchmark client, including warmup",
            f"Wrong memory scope: {directory}")
    require(memory.get("samples") == len(samples), f"GPU sample count mismatch: {directory}")
    require(memory.get("sampled_max_used_mib") == max(samples), f"Memory maximum mismatch: {directory}")
    require(memory.get("sampled_min_used_mib") == min(samples), f"Memory minimum mismatch: {directory}")

    measured = {key: positive(report[key], f"{key}: {directory}") for key in METRICS}
    for key in TAIL_METRICS:
        if key in report:
            measured[key] = positive(report[key], f"{key}: {directory}")
    measured["sampled_max_used_mib"] = max(samples)
    files = ("benchmark.json", "memory_summary.json", "environment.txt",
             "client.log", "server.log", "gpu_samples.csv")
    return {
        "method": method, "input_tokens": length, "output_tokens": 128,
        "concurrency": concurrency, "repetition": repetition, "requests": 200,
        "directory": str(directory.relative_to(project)), "metrics": measured,
        "artifact_sha256": {name: sha256(directory / name) for name in files},
    }


def aggregate(runs):
    cases = []
    for length in INPUT_LENGTHS:
        for concurrency in CONCURRENCIES:
            methods = {}
            for method in METHODS:
                entries = [r for r in runs if r["method"] == method and
                           r["input_tokens"] == length and r["concurrency"] == concurrency]
                require(len(entries) == 3, "Expected three repetitions per method/case")
                common = set.intersection(*(set(r["metrics"]) for r in entries))
                methods[method] = {
                    metric: describe([r["metrics"][metric] for r in entries])
                    for metric in sorted(common)
                }
            bf16, fp8 = methods["BF16"], methods["Random32"]
            cases.append({
                "input_tokens": length, "output_tokens": 128,
                "concurrency": concurrency, "methods": methods,
                "throughput_speedup": fp8["output_throughput"]["mean"] /
                bf16["output_throughput"]["mean"],
                "reduction_percent": {
                    key: 100 * (1 - fp8[key]["mean"] / bf16[key]["mean"])
                    for key in (*METRICS[1:], "sampled_max_used_mib")
                },
                "sampled_memory_saved_gib": (
                    bf16["sampled_max_used_mib"]["mean"] -
                    fp8["sampled_max_used_mib"]["mean"]
                ) / 1024,
            })
    return cases


def render_markdown(summary):
    primary = summary["primary"]
    p = primary["methods"]
    lines = [
        "# Day 5 — Validated serving benchmark report", "",
        "All 36 saved runs passed validation: 7,200 measured requests and 921,600 output tokens. "
        "Initial sanity requests, warm-up requests, and pilots are excluded from these totals.", "",
        "## Primary result", "",
        "Preselected workload: 1,024 input tokens, 128 output tokens, concurrency 4; "
        "three repetitions of 200 measured requests per model.", "",
        f"FP8 throughput speedup: **{primary['throughput_speedup']:.4f}×**. "
        f"Sampled memory maximum decreased by **{primary['sampled_memory_saved_gib']:.2f} GiB**.", "",
        "| Metric | BF16 mean ± sample SD | FP8 mean ± sample SD |",
        "| --- | ---: | ---: |",
    ]
    labels = {"output_throughput": "Output throughput (tok/s)",
              "mean_ttft_ms": "Mean TTFT (ms)", "mean_tpot_ms": "Mean TPOT (ms)",
              "mean_e2el_ms": "Mean request latency (ms)",
              "sampled_max_used_mib": "Sampled GPU maximum (MiB)"}
    for key in (*METRICS, "sampled_max_used_mib"):
        values = []
        for method in METHODS:
            item = p[method][key]
            values.append(f"{item['mean']:.2f} ± {item['sample_sd']:.2f}")
        lines.append(f"| {labels[key]} | {values[0]} | {values[1]} |")
    lines += ["", "## Throughput matrix", "",
              "Values are arithmetic means of three run-level rates; SD is the sample SD across those runs.", "",
              "| Input tokens | Concurrency | BF16 tok/s ± SD | FP8 tok/s ± SD | Speedup |",
              "| ---: | ---: | ---: | ---: | ---: |"]
    for case in summary["cases"]:
        b = case["methods"]["BF16"]["output_throughput"]
        f = case["methods"]["Random32"]["output_throughput"]
        lines.append(f"| {case['input_tokens']} | {case['concurrency']} | "
                     f"{b['mean']:.2f} ± {b['sample_sd']:.2f} | "
                     f"{f['mean']:.2f} ± {f['sample_sd']:.2f} | {case['throughput_speedup']:.4f}× |")
    lines += ["", "## Mean latency matrix", "",
              "Each entry is BF16 / FP8, in milliseconds, averaged across the three run means.", "",
              "| Input tokens | Concurrency | TTFT | TPOT | Request latency |",
              "| ---: | ---: | ---: | ---: | ---: |"]
    for case in summary["cases"]:
        pairs = []
        for key in METRICS[1:]:
            pairs.append(" / ".join(f"{case['methods'][m][key]['mean']:.2f}" for m in METHODS))
        lines.append(f"| {case['input_tokens']} | {case['concurrency']} | " + " | ".join(pairs) + " |")
    lines += ["", "## Sampled GPU memory", "",
              "Mean of each run's whole-GPU sampled maximum, in MiB. Sampling interval: one second. "
              "The window includes client startup and warm-up, and excludes server startup. "
              "Brief spikes between samples can be missed. Values include persistent allocator/workspace state "
              "from earlier workloads. The primary workload supplies the headline memory comparison.", "",
              "| Input tokens | Concurrency | BF16 MiB | FP8 MiB | Difference GiB |",
              "| ---: | ---: | ---: | ---: | ---: |"]
    for case in summary["cases"]:
        b = case["methods"]["BF16"]["sampled_max_used_mib"]["mean"]
        f = case["methods"]["Random32"]["sampled_max_used_mib"]["mean"]
        lines.append(f"| {case['input_tokens']} | {case['concurrency']} | {b:.2f} | {f:.2f} | "
                     f"{case['sampled_memory_saved_gib']:.3f} |")
    if all(all(key in case["methods"][method] for key in TAIL_METRICS)
           for case in summary["cases"] for method in METHODS):
        lines += ["", "## Tail latency", "",
                  "Each entry is BF16 / FP8. These are means of the three native run-level P95 values, "
                  "not P95 values calculated by pooling all requests.", "",
                  "| Input tokens | Concurrency | Mean run P95 TTFT (ms) | Mean run P95 TPOT (ms) | Mean run P95 request latency (ms) |",
                  "| ---: | ---: | ---: | ---: | ---: |"]
        for case in summary["cases"]:
            pairs = [" / ".join(f"{case['methods'][m][key]['mean']:.2f}" for m in METHODS)
                     for key in TAIL_METRICS]
            lines.append(f"| {case['input_tokens']} | {case['concurrency']} | " + " | ".join(pairs) + " |")
    lines += ["", "## Protocol and interpretation", "",
              f"- Physical GPU UUID: `{summary['protocol']['gpu_uuid']}`.",
              "- Recorded serving packages: " + ", ".join(
                  f"`{key}=={value}`" for key, value in sorted(summary["protocol"]["packages"].items())) + ".",
              "- Original Qwen3-4B-Instruct-2507 BF16 versus ModelOpt static FP8 Random32, calibration pool seed 42.",
              "- Both servers: BF16 KV cache, 4 GiB budget, 29,120-token capacity, FLASH_ATTN, "
              "context limit 4,096, maximum 4 sequences, batching budget 4,096 tokens, prefix caching disabled.",
              "- Synthetic completions, output length 128, temperature 0, top-p 1, workload and generation seed 42, "
              "EOS ignored, 20 warm-up requests per run, client and server on the same machine.",
              "- Primary execution order: FP8-1, BF16-1, BF16-2, FP8-2, FP8-3, BF16-3. "
              "Secondary cases were collected in server blocks; run variation is descriptive.",
              "- Native peak concurrent-request statistics count requests touching one-second buckets. "
              "Use the configured concurrency limit when interpreting these experiments.",
              "- This is a warm serving comparison on one GPU. Startup/compilation time, "
              "production-network latency, other GPUs, and answer quality are separate measurements.",
              "- Three repetitions describe observed variability; no population confidence intervals or significance tests are claimed.",
              "- Keep the Day 4 quality results separate: Selected32 showed no consistent advantage over Random32. "
              "This benchmark uses the fixed Random32 seed-42 checkpoint.", "",
              "## Evidence", "",
              "The companion JSON retains all 36 run summaries, the frozen protocol, "
              "and SHA256 hashes of the result, environment, client log, server log, and GPU samples for every run. "
              "The raw evidence remains in the original run directories.", "",
              f"Protocol SHA256: `{summary['protocol_sha256']}`.", ""]
    return "\n".join(lines)


def atomic_write(path, content):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path,
                        default=Path("/root/autodl-tmp/qwen-inference-lab"))
    args = parser.parse_args()
    project = args.root.resolve()
    base = project / "outputs/benchmark/day5/formal_v1"
    protocol = read_protocol(project, base)
    runs = []
    for length in INPUT_LENGTHS:
        for concurrency in CONCURRENCIES:
            for method in METHODS:
                for repetition in (1, 2, 3):
                    runs.append(validate_run(project, base, protocol, method,
                                             length, concurrency, repetition))
            print(f"Verified: input={length}, concurrency={concurrency}, both models × 3 runs")
    cases = aggregate(runs)
    primary = next(c for c in cases if c["input_tokens"] == 1024 and c["concurrency"] == 4)
    summary = {
        "study": "day5_formal_v1", "protocol": protocol,
        "protocol_sha256": sha256(base / "protocol.json"),
        "summarizer_sha256": sha256(Path(__file__)), "validated_runs": len(runs),
        "measured_requests": sum(r["requests"] for r in runs),
        "measured_output_tokens": sum(r["requests"] * r["output_tokens"] for r in runs),
        "measured_input_tokens": sum(r["requests"] * r["input_tokens"] for r in runs),
        "aggregation": "arithmetic mean and sample SD of three run-level metrics per cell",
        "memory_scope": "1-second whole-GPU samples during client execution, including warmup; excludes server startup",
        "runs": runs, "cases": cases, "primary": primary,
    }
    json_path = base / "day5_summary.json"
    markdown_path = base / "day5_report.md"
    rendered = render_markdown(summary)
    atomic_write(json_path, json.dumps(summary, indent=2, sort_keys=True) + "\n")
    atomic_write(markdown_path, rendered)
    print("\nInput  C  BF16 tok/s  FP8 tok/s  Speedup")
    for case in cases:
        b = case["methods"]["BF16"]["output_throughput"]["mean"]
        f = case["methods"]["Random32"]["output_throughput"]["mean"]
        print(f"{case['input_tokens']:>5} {case['concurrency']:>2} {b:>11.2f} {f:>10.2f} "
              f"{case['throughput_speedup']:>7.4f}x")
    print(f"\nPrimary throughput speedup: {primary['throughput_speedup']:.4f}x")
    print(f"Primary sampled memory difference: {primary['sampled_memory_saved_gib']:.2f} GiB")
    print("Saved:", json_path)
    print("Saved:", markdown_path)
    print("PASS: 36 runs, 7,200 measured requests, exact token counts, common protocol/GPU/environment, and serving evidence verified.")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, OSError) as error:
        print(f"VALIDATION FAILED: {error}", file=sys.stderr)
        sys.exit(1)
