import json
import os
import signal
import socket
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import ProxyHandler, build_opener

ROOT = Path("/root/autodl-tmp/qwen-inference-lab")
PYTHON = ROOT / ".venv-serve/bin/python"
METHODS = ("Random32", "Selected32", "Random128")
SEEDS = (43, 44)
HOST = "127.0.0.1"
PORT = 8000

opener = build_opener(ProxyHandler({}))
environment = os.environ.copy()
environment["HF_HUB_OFFLINE"] = "1"

requests_path = ROOT / "data/evaluation/gsm8k_dev_v1/requests.jsonl"
question_ids = [
    json.loads(line)["id"]
    for line in requests_path.read_text().splitlines()
]
assert len(question_ids) == len(set(question_ids)) == 256
assert PYTHON.is_file()

# Check that every required checkpoint exists before starting work.
for seed in SEEDS:
    for method in METHODS:
        checkpoint = ROOT / (
            f"models/Qwen3-4B-Instruct-2507-FP8-{method}-seed{seed}"
        )
        assert (checkpoint / "calibration_manifest.json").is_file(), checkpoint

def require_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind((HOST, PORT))
        except OSError as error:
            raise RuntimeError(
                "Port 8000 is occupied. Stop the existing server first."
            ) from error

def wait_for_gpu_memory():
    deadline = time.monotonic() + 90
    next_notice = 0.0

    while True:
        output = subprocess.check_output(
            [
                "nvidia-smi", "--id=0",
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=10,
        )
        free_mib = int(output.strip())

        if free_mib >= 20 * 1024:
            print(f"Free GPU memory: {free_mib / 1024:.2f} GiB", flush=True)
            return

        now = time.monotonic()
        if now >= deadline:
            raise RuntimeError(
                f"GPU still has only {free_mib / 1024:.2f} GiB free."
            )
        if now >= next_notice:
            print("Waiting for GPU memory to be released...", flush=True)
            next_notice = now + 10
        time.sleep(2)

def wait_until_ready(process, log_path):
    start = time.monotonic()
    next_notice = 0.0

    while time.monotonic() - start < 600:
        if process.poll() is not None:
            raise RuntimeError(
                f"Server exited with code {process.returncode}. "
                f"See {log_path}"
            )

        try:
            with opener.open(
                f"http://{HOST}:{PORT}/health", timeout=2
            ) as response:
                if response.status == 200:
                    print("Server ready.", flush=True)
                    return
        except OSError:
            pass

        elapsed = time.monotonic() - start
        if elapsed >= next_notice:
            print(f"Waiting for startup: {elapsed:.0f}s", flush=True)
            next_notice = elapsed + 15
        time.sleep(2)

    raise TimeoutError(f"Server startup exceeded 600s. See {log_path}")

def group_exists(group_id):
    try:
        os.killpg(group_id, 0)
        return True
    except ProcessLookupError:
        return False

def stop_owned_server(process):
    # start_new_session=True gives this server its own process group.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass

    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        process.poll()
        if not group_exists(process.pid):
            return
        time.sleep(0.5)

    if group_exists(process.pid):
        print("Stopping remaining processes in the owned server group.",
              flush=True)
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    process.wait(timeout=10)

def run_collector(method, seed):
    command = [
        str(PYTHON), "-u",
        str(ROOT / "scripts/evaluate_gsm8k_development.py"),
        "--method", method,
        "--seed", str(seed),
        "--limit", "256",
    ]
    client = subprocess.Popen(command, cwd=ROOT, env=environment)
    try:
        while True:
            try:
                code = client.wait(timeout=30)
                break
            except subprocess.TimeoutExpired:
                print(
                    f"Evaluation active: seed={seed}, method={method}",
                    flush=True,
                )
        if code:
            raise subprocess.CalledProcessError(code, command)
    finally:
        if client.poll() is None:
            client.terminate()
            try:
                client.wait(timeout=10)
            except subprocess.TimeoutExpired:
                client.kill()
                client.wait(timeout=10)

for seed in SEEDS:
    for method in METHODS:
        print(f"\nEvaluating seed={seed}, method={method}", flush=True)
        folder = ROOT / (
            f"outputs/evaluation/gsm8k_dev_v1/seed{seed}/{method}"
        )
        record_dir = folder / "records"
        all_cached = all(
            (record_dir / f"{qid}.json").is_file()
            for qid in question_ids
        )

        if all_cached:
            print("All responses present; validating cached evaluation.",
                  flush=True)
            run_collector(method, seed)
            continue

        require_free_port()
        wait_for_gpu_memory()

        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        log_path = ROOT / (
            f"logs/replication_server_{method}_seed{seed}_{stamp}.log"
        )
        command = [
            "bash",
            str(ROOT / "scripts/serve_fp8_experiment.sh"),
            method, str(seed),
        ]
        print("Server log:", log_path, flush=True)

        with log_path.open("w") as log:
            server = subprocess.Popen(
                command,
                cwd=ROOT,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                wait_until_ready(server, log_path)
                run_collector(method, seed)
            except BaseException:
                print("Server log tail:", flush=True)
                lines = log_path.read_text(errors="replace").splitlines()
                print("\n".join(lines[-40:]), flush=True)
                raise
            finally:
                print("Stopping this run's server...", flush=True)
                stop_owned_server(server)

        kernel_lines = [
            line
            for line in log_path.read_text(errors="replace").splitlines()
            if "Selected" in line and "LinearKernel" in line
        ]
        print(
            "FP8 linear kernel logged:",
            any("FP8" in line for line in kernel_lines),
            flush=True,
        )
        print(f"PASS: seed={seed}, method={method}", flush=True)

print("\nCompleted development evaluations")
print(f"{'Seed':<6} {'Method':<12} {'Correct':>8} "
      f"{'EM %':>8} {'Parse':>7} {'Length':>7}")

for seed in SEEDS:
    for method in METHODS:
        path = ROOT / (
            f"outputs/evaluation/gsm8k_dev_v1/seed{seed}/"
            f"{method}/summary_full.json"
        )
        summary = json.loads(path.read_text())
        assert summary["complete_256"]
        assert summary["questions_evaluated"] == 256
        print(
            f"{seed:<6} {method:<12} "
            f"{summary['correct']:>4}/256 "
            f"{summary['numeric_exact_match_percent']:>8.2f} "
            f"{summary['parse_failures']:>7} "
            f"{summary['length_finishes']:>7}"
        )

print("\nPASS: all six replication evaluations complete.")
