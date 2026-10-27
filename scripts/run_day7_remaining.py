#!/usr/bin/env python3
"""Finish the frozen ten-model study, skipping audited completed checkpoints.

Stop any manually started server before running this controller. It terminates
only process groups that it starts itself. Resume is at checkpoint boundaries:
an interrupted partial checkpoint is not automatically replayed from a cache.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from urllib.request import ProxyHandler, build_opener

PLAN_SHA256 = "1bd1f014f7409f32e93347cbd25e39eca79a782a614acd0234a5ed7c003e4558"
PREPARATION_SHA256 = "1ddf3a16aad32e1273d90b2e0ba7dd0d84f9e0edd281051c396fc9ca405faa95"
RUNNER_SHA256 = "59e6c38182085dc6da8e0808e0bf2624135193674590f89b1dd062308875466b"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def read_json(path):
    return json.loads(path.read_text())


def port_busy():
    try:
        with socket.create_connection(("127.0.0.1", 8000), timeout=1):
            return True
    except OSError:
        return False


def stop_owned(process):
    """The PID is a new session/process-group leader created by this script."""
    if process is None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=10)


def run_client(command, log_path, root, environment):
    process = None
    try:
        with log_path.open("wb") as log:
            process = subprocess.Popen(command, cwd=root, env=environment,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            while chunk := process.stdout.read1(4096):
                log.write(chunk); log.flush()
                sys.stdout.buffer.write(chunk); sys.stdout.buffer.flush()
            require(process.wait() == 0, f"Client failed; inspect {log_path}")
    finally:
        if process is not None and process.poll() is None:
            stop_owned(process)


def completed_run(output_root, name):
    summaries = sorted((output_root / name).glob("*/final_summary.json"))
    require(len(summaries) <= 1, f"Multiple completed runs for {name}; resolve explicitly")
    if not summaries:
        return None
    directory = summaries[0].parent
    manifest = read_json(directory / "run_manifest.json")
    summary = read_json(summaries[0])
    require(manifest["model"] == name and manifest["final_plan_sha256"] == PLAN_SHA256
            and manifest["preparation_sha256"] == PREPARATION_SHA256,
            f"Completed run identity differs: {name}")
    require(summary["questions_per_task"] == {
        "day7_gsm8k_test": 1319, "day7_arc_challenge_test": 1172},
        f"Incomplete test coverage for {name}")
    return directory


def wait_for_cleanup(timeout=60):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        free = subprocess.check_output([
            "nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            text=True).strip().splitlines()
        require(len(free) == 1, "Expected one GPU")
        if not port_busy() and float(free[0]) >= 18000:
            return
        time.sleep(1)
    raise RuntimeError("Port 8000 or GPU memory is still occupied. Inspect running processes before retrying.")


def wait_ready(process, alias, model_path, root, log_path):
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        require(process.poll() is None, f"Server exited during startup; inspect {log_path}")
        try:
            opener = build_opener(ProxyHandler({}))
            with opener.open("http://127.0.0.1:8000/health", timeout=3) as response:
                healthy = response.status == 200
            with opener.open("http://127.0.0.1:8000/v1/models", timeout=3) as response:
                models = json.load(response)
            cards = [c for c in models["data"] if c["id"] == alias]
            if healthy and len(cards) == 1 and cards[0].get("max_model_len") == 8192:
                require((root / cards[0]["root"]).resolve() == model_path.resolve(),
                        "Ready server checkpoint differs")
                return
        except (OSError, ValueError):
            pass
        time.sleep(1)
    raise RuntimeError(f"Server startup timed out; inspect {log_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/root/autodl-tmp/qwen-inference-lab"))
    root = parser.parse_args().root.resolve()
    plan_path = root / "data/evaluation/day7_quality_v1/final_plan.json"
    preparation = root / "data/evaluation/day7_quality_v1/final/preparation_manifest.json"
    runner = root / "scripts/run_day7_final.py"
    require(sha256(plan_path) == PLAN_SHA256 and sha256(preparation) == PREPARATION_SHA256,
            "Frozen plan or preparation changed")
    require(sha256(runner) == RUNNER_SHA256, "Final runner differs from the version validated on BF16/Random32")
    plan = read_json(plan_path)
    require(len(plan["models"]) == 10, "Expected the frozen ten-model scope")
    require(not port_busy(), "Stop the manually started server in Terminal A with Ctrl+C, then run this controller.")
    serve_binary = root / ".venv-serve/bin/vllm"
    eval_python = root / ".venv-eval/bin/python"
    require(serve_binary.is_file() and eval_python.is_file(), "Required environments are missing")
    output_root = root / "outputs/evaluation/day7_quality_v1/final"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    logs = root / "logs/evaluation/day7/final_batch" / stamp
    logs.mkdir(parents=True)
    environment = os.environ.copy()
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        environment.pop(name, None)
    environment.update(HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                       PYTHONUNBUFFERED="1", NO_PROXY="127.0.0.1,localhost",
                       no_proxy="127.0.0.1,localhost")
    state = {"final_plan_sha256": PLAN_SHA256, "status": "running", "completed": [],
             "active_model": None, "logs": str(logs), "controller_sha256": sha256(Path(__file__))}
    progress_path = output_root / "batch_progress.json"
    try:
        for selected in plan["models"]:
            name = selected["name"]
            state["active_model"] = name
            save_json(progress_path, state)
            command = [str(eval_python), "-u", str(runner), "--model", name]
            complete = completed_run(output_root, name)
            if complete is not None:
                print(f"\nVERIFY COMPLETED: {name}; saved responses only", flush=True)
                run_client(command + ["--audit-only", str(complete)],
                           logs / f"{name}_verify.log", root, environment)
                state["completed"].append(name)
                save_json(progress_path, state)
                continue
            print(f"\nSTART FULL TESTS: {name} ({len(state['completed'])}/10 complete)", flush=True)
            wait_for_cleanup()
            model_path = root / selected["path"]
            alias = "qwen-quality-bf16" if name == "BF16" else "qwen-quality-fp8"
            server_log = logs / f"{name}_server.log"
            server_command = [str(serve_binary), "serve", str(model_path),
                "--served-model-name", alias, "--host", "127.0.0.1", "--port", "8000",
                "--dtype", "bfloat16", "--kv-cache-dtype", "bfloat16",
                "--kv-cache-memory-bytes", "4294967296", "--attention-backend", "FLASH_ATTN",
                "--max-model-len", "8192", "--max-num-seqs", "4",
                "--max-num-batched-tokens", "4096", "--generation-config", "vllm",
                "--no-enable-prefix-caching", "--seed", "42"]
            if name != "BF16":
                server_command += ["--quantization", "modelopt"]
            server_environment = environment.copy()
            server_environment["VIRTUAL_ENV"] = str(root / ".venv-serve")
            server_environment["PATH"] = str(root / ".venv-serve/bin") + os.pathsep + environment.get("PATH", "")
            save_json(logs / f"{name}_launch.json", {"model": selected, "command": server_command})
            server = None
            try:
                with server_log.open("wb") as log:
                    server = subprocess.Popen(server_command, cwd=root, env=server_environment,
                                              stdout=log, stderr=subprocess.STDOUT,
                                              start_new_session=True)
                    print(f"Waiting for server startup; log: {server_log}", flush=True)
                    wait_ready(server, alias, model_path, root, server_log)
                    run_client(command + ["--server-log", str(server_log)],
                               logs / f"{name}_client.log", root, environment)
            finally:
                stop_owned(server)
            require(completed_run(output_root, name) is not None,
                    f"Client exited without a completed audit for {name}")
            state["completed"].append(name)
            save_json(progress_path, state)
        state.update(status="complete", active_model=None)
        save_json(progress_path, state)
        print("\nPASS: all 10 checkpoints have completed audited full test runs.", flush=True)
        print(f"Progress: {progress_path}")
        print(f"Logs: {logs}")
    except BaseException as error:
        state.update(status="stopped", error=f"{type(error).__name__}: {error}")
        save_json(progress_path, state)
        raise


def interrupted(signum, frame):
    raise KeyboardInterrupt(f"Received signal {signum}")


if __name__ == "__main__":
    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, interrupted)
    main()
