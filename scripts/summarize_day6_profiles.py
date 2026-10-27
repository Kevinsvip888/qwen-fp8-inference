#!/usr/bin/env python3
"""Validate selected Day 6 repeated captures and write JSON/Markdown findings.

Standard-library-only analysis: no model loads, GPU queries, API requests,
or inference. Selects the unique complete three-repetition session per model
and workload, excluding one-repetition pilots. Explicit session paths can be
supplied if more than one complete session exists. Original evidence is read
only; derived reports are replaced after all checks pass.
"""

import argparse
import gzip
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path


METHODS = ("BF16", "Random32")
CASES = ("prefill_c1", "decode_c1", "decode_c4")
LABELS = {
    "prefill_c1": "execute_context_1(1024)_generation_0(0)",
    "decode_c1": "execute_context_0(0)_generation_1(1)",
    "decode_c4": "execute_context_0(0)_generation_4(4)",
}
WINDOWS = {"prefill_c1": (0, 1), "decode_c1": (8, 8), "decode_c4": (16, 8)}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load(path):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read {path}: {error}") from error


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def describe(values):
    return {"mean": statistics.mean(values), "sample_sd": statistics.stdev(values),
            "minimum": min(values), "maximum": max(values)}


def category(name):
    # Deliberately the same coarse rules used in the earlier C1 analysis.
    text = name.lower()
    if "flash::" in text or "flash_fwd" in text:
        return "Attention"
    if text.startswith("triton_") and any(
        word in text for word in ("clamp", "to_copy", "quant", "scaled_mm")
    ):
        return "Fused arithmetic / conversion"
    if "gemm" in text or "gemvx::" in text:
        return "Matrix calculations"
    if "quant" in text or "convert" in text:
        return "Explicit quantization / conversion"
    if "rms_norm" in text or "layer_norm" in text:
        return "Normalization / residual"
    if "rotary" in text:
        return "Position encoding"
    if "silu" in text or "act_and_mul" in text or text.startswith("triton_poi_"):
        return "Activation / pointwise"
    return "Other"


def validate_protocols(project, base):
    c1_path = base / "phase_protocol_v1.json"
    c4_path = base / "decode_c4_protocol_v1.json"
    c1, c4 = load(c1_path), load(c4_path)
    request_path = base / "phase_request_v1.json"
    request_sha = sha(request_path)
    body = load(request_path)
    require(len(body["prompt"]) == 1024 and all(isinstance(x, int) for x in body["prompt"]),
            "Unexpected fixed prompt")
    for key, expected in {"max_tokens": 128, "min_tokens": 128, "temperature": 0,
                          "top_p": 1, "seed": 42, "ignore_eos": True,
                          "add_special_tokens": False, "stream": False}.items():
        require(body.get(key) == expected, f"Unexpected fixed-request {key}")
    require(c1["request_sha256"] == c4["request_sha256"] == request_sha, "Request identity differs")
    require(c4["parent_protocol_sha256"] == sha(c1_path), "C4 parent protocol differs")
    require(c1["gpu_uuid"] == c4["gpu_uuid"], "GPU identities differ")
    require(c1["packages"] == c4["packages"], "Serving environments differ")
    require(c1["input_tokens"] == 1024 and c1["output_tokens"] == 128 and c1["concurrency"] == 1,
            "Unexpected C1 workload")
    require(c1["windows"] == {"prefill": [0, 1], "decode": [8, 8]}, "Unexpected C1 windows")
    for key, expected in {"concurrency": 4, "requests_per_batch": 4,
                          "warmup_batches": 4, "delay_iterations": 16,
                          "max_iterations": 8, "input_tokens_per_request": 1024,
                          "output_tokens_per_request": 128}.items():
        require(c4.get(key) == expected, f"Unexpected C4 {key}")
    require(c1["warmup_requests"] == 4, "Unexpected C1 warmup count")
    bindings = {
        "server_script_sha256": project / "scripts/serve_day6_profile.sh",
        "client_script_sha256": project / "scripts/profile_day6_phases.py",
        "fp8_manifest_sha256": project / "models" /
        "Qwen3-4B-Instruct-2507-FP8-Random32-seed42/calibration_manifest.json",
    }
    for key, path in bindings.items():
        require(sha(path) == c1[key], f"Changed source/manifest: {path}")
    require(sha(project / "scripts/profile_day6_decode_c4.py") == c4["client_script_sha256"],
            "C4 capture script changed")
    source = load(project / "logs/model_revision.json")
    require(c1["source_revision"] == source["revision"] ==
            "cdbee75f17c01a7cc42f958dc650907174af0554", "Unexpected source revision")
    return c1, c4, request_sha


def expected_pairs(kind):
    phases = ("prefill", "decode") if kind == "c1" else ("steady_decode_c4",)
    return {(phase, repetition) for phase in phases for repetition in (1, 2, 3)}


def complete_session(path, method, kind, protocol_sha, request_sha):
    data = load(path / "capture_summary.json")
    pairs = [(r["requested_phase"], r["repetition"]) for r in data["captures"]]
    return (data["method"] == method and data["protocol_sha256"] == protocol_sha
            and data["request_sha256"] == request_sha
            and len(pairs) == len(expected_pairs(kind)) and set(pairs) == expected_pairs(kind))


def select_session(base, method, kind, explicit, protocol_sha, request_sha):
    if explicit is not None:
        path = explicit.resolve()
        require(path.is_relative_to(base.resolve()), "Session must be under the Day 6 output directory")
        require(complete_session(path, method, kind, protocol_sha, request_sha),
                f"Not a matching complete session: {path}")
        return path
    subdirectory = "captures" if kind == "c1" else "captures_c4"
    matches = []
    for summary in sorted(base.glob(f"{method}_*/{subdirectory}/*/capture_summary.json")):
        if complete_session(summary.parent, method, kind, protocol_sha, request_sha):
            matches.append(summary.parent)
    require(len(matches) == 1,
            f"Expected one complete {method}/{kind} session; found {len(matches)}. "
            f"Use --{method.lower()}-{kind} with the intended session path.")
    return matches[0]


def response_ok(response, alias):
    require(response["usage"]["prompt_tokens"] == 1024 and
            response["usage"]["completion_tokens"] == 128, "Unexpected response token counts")
    if "model" in response:
        require(response["model"] == alias, "Unexpected response model alias")


def validate_session(project, session, method, kind, protocol_sha, request_sha):
    data = load(session / "capture_summary.json")
    alias = "qwen-profile-bf16" if method == "BF16" else "qwen-profile-fp8"
    evidence = {str((session / "capture_summary.json").relative_to(project)):
                sha(session / "capture_summary.json")}
    for index in range(1, 5):
        filename = f"warmup_{index}.json" if kind == "c1" else f"warmup_batch_{index}.json"
        path = session / filename
        responses = [load(path)] if kind == "c1" else load(path)
        require(len(responses) == (1 if kind == "c1" else 4), "Unexpected warmup batch size")
        for response in responses:
            response_ok(response, alias)
        evidence[str(path.relative_to(project))] = sha(path)
    output = []
    for record in data["captures"]:
        phase, repetition = record["requested_phase"], record["repetition"]
        case = phase + "_c1" if kind == "c1" else "decode_c4"
        delay, limit = WINDOWS[case]
        require(record["method"] == method and record["protocol_sha256"] == protocol_sha
                and record["request_sha256"] == request_sha, "Capture identity differs")
        settings = record["profile_settings"]
        require(settings["delay_iterations"] == delay and settings["max_iterations"] == limit,
                "Unexpected capture window")
        if kind == "c1":
            response_path = session / f"{phase}_r{repetition}_response.json"
            responses = [load(response_path)]
        else:
            response_path = session / f"r{repetition}_responses.json"
            responses = load(response_path)
        require(len(responses) == (1 if kind == "c1" else 4), "Unexpected capture request count")
        for response in responses:
            response_ok(response, alias)
        evidence[str(response_path.relative_to(project))] = sha(response_path)
        totals, calls, labels = defaultdict(float), Counter(), Counter()
        for trace_info in record["traces"]:
            path = session.parent.parent / trace_info["file"]
            require(path.name.startswith(settings["profile_prefix"]), "Trace prefix differs")
            require(sha(path) == trace_info["sha256"], f"Trace bytes changed: {path}")
            evidence[str(path.relative_to(project))] = sha(path)
            opener = gzip.open if path.suffix == ".gz" else open
            with opener(path, "rt") as handle:
                events = json.load(handle).get("traceEvents", [])
            trace_count = 0
            for event in events:
                if event.get("ph") != "X":
                    continue
                name = event.get("name", "")
                cats = {x.strip().lower() for x in event.get("cat", "").split(",")}
                if "kernel" in cats and event.get("dur", 0) > 0:
                    duration = float(event["dur"]) / 1000
                    require(math.isfinite(duration), "Invalid kernel duration")
                    totals[name] += duration
                    calls[name] += 1
                    trace_count += 1
                elif name.startswith("execute_context_"):
                    labels[name] += 1
            require(trace_count == trace_info["gpu_kernel_events"], "Trace kernel count differs")
        total = sum(totals.values())
        require(labels and set(labels) == {LABELS[case]}, f"Wrong phase labels: {session}/{case}")
        require(sum(calls.values()) == record["gpu_kernel_events"] > 0, "Kernel count differs")
        require(abs(total - record["summed_kernel_ms"]) < 1e-6, "Kernel duration total differs")
        groups = defaultdict(float)
        for name, duration in totals.items():
            groups[category(name)] += duration
        output.append({
            "method": method, "case": case, "repetition": repetition,
            "window_engine_steps": limit, "concurrency": 1 if kind == "c1" else 4,
            "summed_kernel_ms": total, "gpu_kernel_events": sum(calls.values()),
            "phase_annotations": dict(labels), "groups_ms": dict(groups),
            "kernels": [{"name": name, "summed_ms": duration, "calls": calls[name],
                         "category": category(name)} for name, duration in
                        sorted(totals.items(), key=lambda item: item[1], reverse=True)],
        })
    return output, evidence


def aggregate(runs):
    results = {}
    for case in CASES:
        methods = {}
        for method in METHODS:
            rows = [row for row in runs if row["case"] == case and row["method"] == method]
            require(sorted(row["repetition"] for row in rows) == [1, 2, 3], "Missing repetition")
            labels = set().union(*(set(row["groups_ms"]) for row in rows))
            methods[method] = {
                "kernel_work_ms": describe([row["summed_kernel_ms"] for row in rows]),
                "kernel_events": describe([row["gpu_kernel_events"] for row in rows]),
                "groups_ms": {label: describe([row["groups_ms"].get(label, 0) for row in rows])
                              for label in sorted(labels)},
            }
        b, f = (methods[method]["kernel_work_ms"]["mean"] for method in METHODS)
        results[case] = {
            "methods": methods, "engine_steps": WINDOWS[case][1],
            "gpu_work_reduction_percent": 100 * (1 - f / b),
            "gpu_work_saved_ms": b - f, "gpu_work_ratio_bf16_over_fp8": b / f,
        }
    return results


def markdown(summary):
    cases = summary["cases"]
    lines = ["# Day 6 — GPU profiling findings", "",
             "Validated 18 repeated capture windows: BF16 and FP8, three repetitions each for "
             "single-request prefill, single-request decode, and four-request steady decode. "
             "One-repetition pilots are excluded.", "",
             "## GPU work comparison", "",
             "Values are summed GPU kernel durations, averaged across three captures. "
             "They are not end-to-end request latency or throughput.", "",
             "| Window | Engine steps | BF16 ms ± sample SD | FP8 ms ± sample SD | Less GPU work |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for case in CASES:
        row = cases[case]
        b, f = (row["methods"][method]["kernel_work_ms"] for method in METHODS)
        lines.append(f"| {case} | {row['engine_steps']} | {b['mean']:.3f} ± {b['sample_sd']:.3f} | "
                     f"{f['mean']:.3f} ± {f['sample_sd']:.3f} | {row['gpu_work_reduction_percent']:.2f}% |")
    lines += ["", "## Kernel groups", "",
              "Coarse categories use kernel names. Full names, assignments, per-capture durations, "
              "and trace hashes remain in the companion JSON. Fused kernels stay combined: "
              "their full time cannot be attributed to quantization alone. Fusion can move work "
              "between categories, so category changes need this context."]
    for case in CASES:
        row = cases[case]
        b, f = (row["methods"][method]["groups_ms"] for method in METHODS)
        lines += ["", f"### {case}", "",
                  "| Category | BF16 ms | FP8 ms | BF16 minus FP8 ms |",
                  "| --- | ---: | ---: | ---: |"]
        for label in sorted(set(b) | set(f)):
            bv = b.get(label, {"mean": 0})["mean"]
            fv = f.get(label, {"mean": 0})["mean"]
            lines.append(f"| {label} | {bv:.3f} | {fv:.3f} | {bv-fv:.3f} |")
        differences = {label: b.get(label, {"mean": 0})["mean"] -
                       f.get(label, {"mean": 0})["mean"] for label in set(b) | set(f)}
        largest = max(differences, key=differences.get)
        lines += ["", f"Largest positive category difference: **{largest}**, "
                  f"{differences[largest]:.3f} ms. Net GPU kernel-work saving: "
                  f"{row['gpu_work_saved_ms']:.3f} ms."]
    lines += ["", "## What the evidence supports", "",
              "The comparisons identify where recorded GPU work changed. Use the tables above "
              "to determine whether matrix kernels account for the largest savings at both "
              "concurrency settings. Attention, conversion, activation, normalization, and other "
              "kernels can offset part of a matrix-kernel improvement.", "",
              "Day 5 remains the source for unprofiled serving speed and memory measurements. "
              "Day 6 uses a fixed synthetic token-ID prompt, not the exact Day 5 prompt set. "
              "Both models receive the same Day 6 prompt and output-token budget.", "",
              "GPU kernel-time sums do not measure the whole critical path: kernels can overlap, "
              "and CPU scheduling, frontend/network work, and idle gaps are outside these sums. "
              "No hardware-counter evidence of memory bandwidth saturation or Tensor Core "
              "utilization was collected. Profiling overhead and GPU clock state can influence durations. "
              "Three repetitions summarize observed variation; no statistical significance claim is made.", "",
              "## Protocol", "",
              "- Original BF16 Qwen3-4B-Instruct-2507 versus fixed ModelOpt FP8 Random32 seed 42.",
              "- Input/output: 1,024 / 128 tokens per request, temperature 0, seed 42, EOS ignored.",
              "- Prefill C1: delay 0, maximum 1 engine step.",
              "- Decode C1: delay 8, maximum 8 engine steps.",
              "- Decode C4: delay 16, maximum 8 engine steps; annotations must show four generating "
              "requests and zero context requests. Eight steps serve four requests together; "
              "the GPU-work total is not a per-request latency.",
              "- PyTorch Profiler: worker CPU/CUDA activity, stacks/shapes/memory tracking disabled, "
              "frontend capture disabled; Day 5 compilation/CUDA-graph behavior retained.",
              "- BF16 KV cache, 4 GiB cache budget, FLASH_ATTN, prefix caching disabled, "
              "context limit 4,096, maximum four sequences and 4,096 batched tokens.",
              f"- GPU UUID: `{summary['protocol_c1']['gpu_uuid']}`.",
              "- Recorded serving packages: " + ", ".join(
                  f"`{name}=={value}`" for name, value in sorted(summary["protocol_c1"]["packages"].items())) + ".",
              "", "## Selected sessions", ""]
    for key, path in summary["sessions"].items():
        lines.append(f"- {key}: `{path}`")
    lines += ["", "## Reproduce the analysis", "", "```bash",
              "cd /root/autodl-tmp/qwen-inference-lab",
              "python scripts/summarize_day6_profiles.py", "```", "",
              "The script checks frozen protocol/request/source identities, warm-up and capture "
              "response counts, raw trace hashes, matching phase labels, and kernel totals before "
              "writing this report. It performs CPU-only analysis and preserves original evidence.", ""]
    return "\n".join(lines)


def atomic_write(path, text):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path,
                        default=Path("/root/autodl-tmp/qwen-inference-lab"))
    for method in METHODS:
        for kind in ("c1", "c4"):
            parser.add_argument(f"--{method.lower()}-{kind}", type=Path)
    args = parser.parse_args()
    project = args.root.resolve()
    base = project / "outputs/profiling/day6"
    c1, c4, request_sha = validate_protocols(project, base)
    sessions, runs, evidence = {}, [], {}
    for method in METHODS:
        for kind in ("c1", "c4"):
            protocol_path = base / ("phase_protocol_v1.json" if kind == "c1" else "decode_c4_protocol_v1.json")
            session = select_session(base, method, kind,
                                     getattr(args, f"{method.lower()}_{kind}"), sha(protocol_path), request_sha)
            rows, files = validate_session(project, session, method, kind, sha(protocol_path), request_sha)
            sessions[f"{method}_{kind}"] = str(session.relative_to(project))
            runs.extend(rows)
            evidence.update(files)
            print(f"Verified {method}/{kind}: {len(rows)} repeated capture windows")
    require(len(runs) == 18, "Expected 18 selected capture windows")
    cases = aggregate(runs)
    summary = {
        "study": "day6_gpu_profiling", "validated_capture_windows": len(runs),
        "protocol_c1": c1, "protocol_c4": c4, "request_sha256": request_sha,
        "summarizer_sha256": sha(Path(__file__)), "sessions": sessions,
        "cases": cases, "captures": runs, "evidence_sha256": evidence,
        "grouping": "Conservative kernel-name rules; fused operations remain combined",
        "timing_scope": "Summed GPU kernel durations, not request latency; one or eight worker engine steps per window",
    }
    rendered = markdown(summary)
    atomic_write(base / "day6_summary.json", json.dumps(summary, indent=2, sort_keys=True) + "\n")
    atomic_write(base / "day6_report.md", rendered)
    print("\nWindow       BF16 GPU ms  FP8 GPU ms  Reduction")
    for case in CASES:
        row = cases[case]
        b, f = (row["methods"][method]["kernel_work_ms"]["mean"] for method in METHODS)
        print(f"{case:<12} {b:>11.3f} {f:>11.3f} {row['gpu_work_reduction_percent']:>9.2f}%")
    print("\nC4 decode groups — eight engine steps")
    b, f = (cases["decode_c4"]["methods"][method]["groups_ms"] for method in METHODS)
    for label in sorted(set(b) | set(f)):
        bv, fv = b.get(label, {"mean": 0})["mean"], f.get(label, {"mean": 0})["mean"]
        print(f"{label:<36} BF16={bv:8.3f}  FP8={fv:8.3f}  saved={bv-fv:8.3f} ms")
    print("\nSaved:", base / "day6_summary.json")
    print("Saved:", base / "day6_report.md")
    print("PASS: 18 repeated windows, matching protocols/prompts, raw responses, trace hashes, phase labels, and GPU kernel totals verified.")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, OSError) as error:
        raise SystemExit(f"VALIDATION FAILED: {error}")
