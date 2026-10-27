# Commands and reproduction notes

This guide reconstructs the workflow from the published scripts and experiment records. It is not a complete shell-history export. Commands are grouped by the actual project days. Run from the project root unless a command says otherwise.

## Choose a reproduction scope

| Goal | What is needed |
|---|---|
| Inspect published results and configuration consistency | Checkout plus Python 3.10+; no GPU or external packages |
| Run a new calibration/serving experiment | Suitable NVIDIA GPU, recorded dependencies, downloaded source model and datasets, and a fresh working directory |
| Replay the original full raw-evidence audits | Original checkpoint directories plus the local experiment backup containing data, logs, responses, and traces |

The original scripts are coupled to `/root/autodl-tmp/qwen-inference-lab`. Use that path on a separate machine or preserve the current experiment machine. Do not run setup over a different study's artifacts. The guide uses the recorded Linux path because changing paths changes some serialized metadata and hashes.

The frozen Day 7 runner is an archived-study runner. A new experiment needs its own plan and artifact identities; it must not bypass the old checks or claim freshly generated checkpoints are byte-identical to the archived ones. This snapshot does not include a general new-study plan generator.

## Day 1 — environment and BF16 serving

### Inspect hardware

```bash
cd /root/autodl-tmp/qwen-inference-lab
nvidia-smi
cat /etc/os-release
uname -m
python3 --version
free -h
df -h .
```

These report the GPU/driver, Linux distribution, CPU architecture, Python, shared host memory, and available disk. The shared host RAM figure is not a container memory guarantee.

### Create isolated environments on a fresh machine

The following assumes `uv` is already installed. It is a reconstruction using the captured package snapshots, not a newly validated cold installation. Keep the working experiment's environments unchanged.

```bash
uv venv --python 3.12.15 .venv-serve
uv venv --python 3.12.15 .venv-quant
uv venv --python 3.12.15 .venv-eval

uv pip install --python .venv-serve/bin/python \
    --torch-backend=auto -r environment/serve-requirements.txt
uv pip install --python .venv-quant/bin/python \
    --torch-backend=auto -r environment/quant-requirements.txt
uv pip install --python .venv-eval/bin/python \
    -r environment/eval-requirements.txt

uv pip check --python .venv-serve/bin/python
uv pip check --python .venv-quant/bin/python
uv pip check --python .venv-eval/bin/python
```

The three lists intentionally differ. Quantization used Transformers 5.14.1; serving and the harness client used 5.19.0. The installed CUDA-enabled PyTorch build was `2.13.0+cu132`; a compatible GPU wheel source is needed. `--torch-backend=auto` is not a guarantee of that exact wheel on another host. Compare installed versions with the snapshots before proceeding. Do not substitute CPU PyTorch for quantization or serving.

### Restore source pins, create working folders, and download the model

```bash
mkdir -p logs outputs models data scripts
mkdir -p logs/benchmark/day5 logs/profiling/day6 logs/evaluation/day7

python3 tools/restore_source_pins.py
source .venv-serve/bin/activate
python -u scripts/download_model.py
```

The restoration tool copies only archived source-revision records and the fixed training partition into their live locations. It refuses to overwrite a differing file. It does not restore final-run metadata or pretend that weights exist. Restoring the pins before downloading avoids resolving a newer model/dataset revision by accident.

The original model downloader uses `logs/model_revision.json` when present; otherwise it resolves the current upstream revision. The recorded model revision is `cdbee75f17c01a7cc42f958dc650907174af0554`.

```bash
python - <<'PY'
import torch
print('PyTorch:', torch.__version__)
print('CUDA build:', torch.version.cuda)
assert torch.cuda.is_available()
print('GPU:', torch.cuda.get_device_name(0))
print('BF16 supported:', torch.cuda.is_bf16_supported())
x = torch.randn(512, 512, device='cuda', dtype=torch.bfloat16)
y = x @ x
torch.cuda.synchronize()
print('Output:', tuple(y.shape), y.dtype)
PY
```

This checks actual BF16 CUDA execution. It is more informative than comparing the CUDA label in `nvidia-smi` with the PyTorch build string alone.

### Serve BF16 and run smoke requests

Terminal A:

```bash
bash scripts/serve_bf16.sh 2>&1 | tee logs/bf16_server.log
```

Terminal B:

```bash
source .venv-serve/bin/activate
curl --noproxy '*' --fail --silent --show-error http://127.0.0.1:8000/health
python scripts/check_fixed_prompts.py --precision bf16
```

Wait for the health check to succeed. The six prompts cover arithmetic, reading, JSON, Chinese, summarization, and code. Their responses are a serving sanity check, not a benchmark accuracy estimate. Stop Terminal A with Ctrl+C before loading another model or quantizing.

## Day 2 — FP8 smoke checkpoint

With the server stopped and enough free GPU memory:

```bash
source .venv-quant/bin/activate
python -u scripts/quantize_fp8_smoke.py
du -sh models/Qwen3-4B-Instruct-2507-FP8-smoke
cat models/Qwen3-4B-Instruct-2507-FP8-smoke/hf_quant_config.json
```

This uses eight synthetic calibration dialogues, inserts ModelOpt quantizers, calibrates static FP8 ranges, and exports a smoke checkpoint. The script deliberately refuses an existing export directory.

Terminal A:

```bash
bash scripts/serve_fp8_smoke.sh 2>&1 | tee logs/fp8_server.log
```

Terminal B, after the server becomes healthy:

```bash
source .venv-serve/bin/activate
python scripts/check_fixed_prompts.py --precision fp8
```

Compare semantics and formatting with BF16. Inspect the server log for the selected FP8 linear kernels. The smoke checkpoint is separate from the nine later calibration-study checkpoints.

## Day 3 — data, calibration pools, features, and nine exports

Stop the GPU server first. In `.venv-quant`:

```bash
source .venv-quant/bin/activate
set -o pipefail

env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
    -u http_proxy -u https_proxy -u all_proxy \
    HF_HUB_DISABLE_XET=1 HF_HUB_DOWNLOAD_TIMEOUT=120 \
    python -u scripts/prepare_data_sources.py \
    2>&1 | tee logs/prepare_data_sources.log

python scripts/split_math_training.py
python scripts/build_calibration_pool.py --seed 42
python scripts/audit_and_sample_pool.py --seed 42
python scripts/extract_activation_features.py --seed 42
python scripts/select_calibration_sequences.py --seed 42

for method in Random32 Selected32 Random128; do
    python -u scripts/quantize_fp8_experiment.py --method "$method" --seed 42 || break
done

python scripts/compare_calibration_ranges.py --seed 42
```

Check each command's success before advancing. The proxy-unsetting command was the workaround for this machine's broken proxy configuration; it is not needed on every network. Disabling Xet avoided a CAS authorization failure on that host.

The source script downloads only training shards and saves revisions and hashes. The splitter reconstructs/checks the 256/16/7,201 partition. The pool builder packs 512-token sequences; the auditor reconstructs them from source records and produces balanced nested random subsets. Feature extraction runs the original BF16 model; selection chooses a matched-size activation-coverage subset. Each quantization run loads fresh original weights. The final range comparison verifies the shared recipe and the nested-max property of Random32 versus Random128.

The convenience script performs the corresponding pool/feature/export steps for seeds 43 and 44:

```bash
bash scripts/prepare_remaining_seeds.sh
```

The seed-42 pool SHA256 is `2e7617c9373418c024716f768436cb79ad75dbfc48bdf10b25abad340a889241`. Each pool's metadata contains the expected identity and source-row counts. Range calibration and checkpoint serialization can produce environment-dependent artifacts; match scientific settings without assuming every regenerated file has the archived hash.

For a seed-42 checkpoint smoke check, Terminal A:

```bash
bash scripts/serve_fp8_experiment.sh Random32 42
```

Terminal B:

```bash
source .venv-serve/bin/activate
python scripts/smoke_fp8_experiment.py --method Random32 --seed 42
```

Repeat with Selected32 and Random128 after stopping/replacing the server. The earlier BF16 fixed-prompt response file must exist for token-count comparison.

## Day 4 — separate development evaluation

```bash
source .venv-quant/bin/activate
python scripts/gsm8k_eval_protocol.py
HF_HUB_OFFLINE=1 python scripts/prepare_gsm8k_development.py
```

This freezes the original 256-question zero-shot development requests and custom final-answer parser. Its output limit is 1,024 tokens. It is not the later five-shot official test protocol.

Start BF16 in Terminal A using `serve_bf16.sh`. In Terminal B:

```bash
source .venv-serve/bin/activate
python scripts/evaluate_gsm8k_development.py --method BF16 --seed 42 --limit 8
python scripts/evaluate_gsm8k_development.py --method BF16 --seed 42 --limit 256
```

Then, one model at a time, start `serve_fp8_experiment.sh METHOD 42` in Terminal A and run the matching method in Terminal B:

```bash
python scripts/evaluate_gsm8k_development.py --method Random32 --seed 42 --limit 256
# Use Selected32 or Random128 only when that matching checkpoint is serving.
```

After the four seed-42 evaluations:

```bash
python scripts/analyze_gsm8k_development.py --seed 42
```

With port 8000 free and seeds 43/44 exported, the controller starts their servers sequentially:

```bash
python -u scripts/evaluate_remaining_seeds.py
python scripts/summarize_calibration_study.py
```

The development study found no consistent selection benefit. Parser diagnostics were reported separately without retroactively changing primary scores.

## Day 5 — serving performance

Activate `.venv-serve` and set `pipefail` in both terminals. Terminal A:

```bash
source .venv-serve/bin/activate
set -o pipefail
bash scripts/serve_day5_benchmark.sh BF16 \
    2>&1 | tee logs/benchmark/day5/bf16_server.log
```

Terminal B, after health succeeds:

```bash
source .venv-serve/bin/activate
set -o pipefail
bash scripts/run_day5_benchmark.sh BF16 1 1024 4
```

Arguments are method, repetition, input-token length, and concurrency. Output length is fixed at 128. Each run makes 20 warm-up requests and 200 measured requests. The native vLLM client uses synthetic prompts, streaming completions, ignored EOS, and seed 42. The runner records environment, GPU samples, native results, and exact token-length checks.

The full matrix is methods `{BF16, Random32}` × input lengths `{256, 1024, 2048}` × concurrency `{1, 4}` × repetitions `{1, 2, 3}`. Change Terminal A to the matching method and log filename before running a case for that method. For FP8:

```bash
# Terminal A, after stopping BF16:
bash scripts/serve_day5_benchmark.sh Random32 \
    2>&1 | tee logs/benchmark/day5/fp8_server.log

# Terminal B:
bash scripts/run_day5_benchmark.sh Random32 1 1024 4
```

The primary case's original collection order was FP8-1, BF16-1, BF16-2, FP8-2, FP8-3, BF16-3. Secondary workloads were collected in server blocks. New run directories are never overwritten. All cases within a study must share the recorded GPU and environment.

After all 36 runs:

```bash
python scripts/summarize_day5_benchmarks.py
```

This audits raw benchmark files and writes the Day 5 report. A fresh clone has copied summaries only, so this command requires newly generated runs or restored raw evidence.

## Day 6 — prefill and decode profiling

In Terminal A, start BF16 with profiling enabled:

```bash
bash scripts/serve_day6_profile.sh BF16 \
    2>&1 | tee logs/profiling/day6/bf16_server.log
```

In Terminal B, after health succeeds:

```bash
source .venv-serve/bin/activate
python scripts/profile_day6_phases.py --method BF16 --repetitions 3
python scripts/profile_day6_decode_c4.py --method BF16 --repetitions 3
```

Replace the server with Random32 using `serve_day6_profile.sh Random32` and an FP8 log, then run:

```bash
python scripts/profile_day6_phases.py --method Random32 --repetitions 3
python scripts/profile_day6_decode_c4.py --method Random32 --repetitions 3
python scripts/summarize_day6_profiles.py
```

The phase client captures one prefill step and eight steady decode steps at concurrency 1. The second client records eight engine steps serving four requests together. Both use the fixed 1,024/128-token request. Verify phase annotations before attributing a trace to prefill or decode. Profiling results explain kernel work; Day 5 supplies unprofiled serving performance.

## Day 7 — harness quality evaluation

### Prepare data and inspect the pipeline

These preparation steps can reconstruct local data from pinned upstream sources, but later archived final-plan checks still require the original identities.

```bash
source .venv-eval/bin/activate
set -o pipefail
mkdir -p logs/evaluation/day7

# Preserve the exact archived client environment record used by the protocol.
cp environment/eval-requirements.txt logs/evaluation/day7/eval-requirements.txt

env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
    -u http_proxy -u https_proxy -u all_proxy \
    HF_HUB_DISABLE_XET=1 HF_HUB_DOWNLOAD_TIMEOUT=120 \
    python scripts/prepare_day7_sources.py

HF_HUB_OFFLINE=1 python scripts/prepare_day7_quality.py
```

The pinned Day 7 source manifest is restored by `tools/restore_source_pins.py`; without it, the original source preparer would resolve the then-current ARC revision. Task defaults must match the installed harness. The quality preparer constructs actual harness contexts, fixes five reserved examples, and verifies split separation and token budgets without model requests.

The quality server uses context 8,192, BF16 KV cache, exactly 4 GiB KV memory, FLASH_ATTN, four maximum sequences, a 4,096-token batching budget, generation-config `vllm`, prefix caching disabled, and seed 42. It differs from the 4,096-context performance server. A BF16 server command for Terminal A is:

```bash
source .venv-serve/bin/activate
vllm serve ./models/Qwen3-4B-Instruct-2507 \
    --served-model-name qwen-quality-bf16 \
    --dtype bfloat16 --kv-cache-dtype bfloat16 \
    --kv-cache-memory-bytes 4294967296 --attention-backend FLASH_ATTN \
    --max-model-len 8192 --max-num-seqs 4 --max-num-batched-tokens 4096 \
    --generation-config vllm --no-enable-prefix-caching --seed 42 \
    --host 127.0.0.1 --port 8000 \
    2>&1 | tee logs/evaluation/day7/bf16_server.log
```

In Terminal B:

```bash
source .venv-eval/bin/activate
python scripts/run_day7_development.py --method BF16 --limit 8
python scripts/run_day7_development.py --method BF16 --limit 100
```

For the FP8 development check, use the Random32 seed-42 model path, add `--quantization modelopt`, change the server alias to `qwen-quality-fp8`, and write `logs/evaluation/day7/fp8_server.log`. Then:

```bash
python scripts/run_day7_development.py --method Random32 --limit 8 \
    --server-log logs/evaluation/day7/fp8_server.log
python scripts/run_day7_development.py --method Random32 --limit 100 \
    --server-log logs/evaluation/day7/fp8_server.log
```

These limited runs are development pipeline checks, not final scores. A PyTorch-not-found message in `.venv-eval` is expected. The no-chat-template choice is part of this fixed completion-format benchmark; do not silently switch formats after seeing results.

### Replay the archived final evaluation

**The commands below require the original live study state.** The public snapshot alone is insufficient. Restore the original data/log/output archive to the original path and retain the original model directories. The earlier 443 MiB backup did not include model weights. Checkpoint manifests in `provenance/original/models/` are sidecars only and cannot replace missing weights.

The final plan was frozen before official test inference and included all ten models. Its exact copy is published in `provenance/original/data/evaluation/day7_quality_v1/final_plan.json`. The archived runner requires this identity, the original calibration-manifest identities, prepared requests, and runtime settings.

Once the corresponding live files are restored:

```bash
source .venv-eval/bin/activate
HF_HUB_OFFLINE=1 python scripts/prepare_day7_final.py

# With the matching manually launched quality server:
python scripts/run_day7_final.py --model BF16 \
    --server-log logs/evaluation/day7/bf16_server.log

# For one FP8 checkpoint, use its matching quality server and exact plan name:
python scripts/run_day7_final.py --model Random32-seed42 \
    --server-log logs/evaluation/day7/fp8_server.log
```

Alternatively, stop any manual server and run the controller. It uses the fixed ten-model plan, verifies/skips completed runs, and sequentially starts/stops only its own server processes:

```bash
python -u scripts/run_day7_remaining.py \
    2>&1 | tee logs/evaluation/day7/final_remaining.log
```

It resumes at completed checkpoint boundaries, not halfway through an unfinished checkpoint. The final runner refuses a duplicate completed inference run; use its `--audit-only RUN_DIRECTORY` option with the correct `--model` to recheck saved evidence instead. Do not delete good runs just to force a rerun.

After all ten runs:

```bash
python scripts/summarize_day7_quality.py \
    2>&1 | tee logs/evaluation/day7/summarize_day7.log
```

This rechecks the saved audits and paired question identities, then computes full-precision scores, paired gains/losses, exact McNemar diagnostics, and question-bootstrap intervals. It sends no inference requests. It needs the local raw run directories, not only the published aggregate JSON.

## Network and audit troubleshooting

- A proxy tunnel reset is a network issue; use the per-command proxy-unsetting workaround only when appropriate to the host.
- HF CAS 401 was avoided with `HF_HUB_DISABLE_XET=1` before Python starts.
- A missing hash-pinned file is missing evidence. Restore or regenerate the appropriate artifact; do not disable the check.
- A changed runtime GPU UUID is expected on another instance, but it is a different experiment identity. Never reuse an old same-GPU protocol claim for it.
- `fatal: not a git repository` appeared during the original harness metadata probe and did not prevent results being saved. The publication repository now has Git metadata; this was not an inference failure.
- Keep the original scripts' bytes unchanged when auditing this snapshot. `.gitattributes` prevents Git line-ending normalization from changing their frozen hashes.

The full final runs and GPU benchmark results were collected on the experiment machine. Publication preparation checked source syntax, metadata consistency, and public-snapshot verification; it did not reinstall all GPU dependencies or repeat the GPU experiment.
