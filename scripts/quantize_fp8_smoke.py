import copy
import json
from importlib.metadata import version
from pathlib import Path

import torch
import modelopt.torch.quantization as mtq
from modelopt.torch.export import export_hf_checkpoint
from modelopt.torch.quantization.nn import TensorQuantizer
from transformers import AutoModelForCausalLM, AutoTokenizer

root = Path(__file__).resolve().parents[1]
source_dir = root / "models" / "Qwen3-4B-Instruct-2507"
export_dir = root / "models" / "Qwen3-4B-Instruct-2507-FP8-smoke"

if export_dir.exists():
    raise FileExistsError(
        f"Export directory already exists: {export_dir}. "
        "Use a fresh destination for this run."
    )

torch.manual_seed(42)
assert torch.cuda.is_available(), "CUDA is unavailable."

free_bytes, _ = torch.cuda.mem_get_info()
print(f"Free GPU memory: {free_bytes / 2**30:.2f} GiB", flush=True)
if free_bytes < 16 * 2**30:
    raise RuntimeError("Please free at least 16 GiB before this test.")

# Synthetic calibration dialogues, separate from our arithmetic test.
samples = [
    (
        "Explain photosynthesis briefly.",
        "Plants use light energy to convert water and carbon dioxide "
        "into sugars, releasing oxygen."
    ),
    (
        "What causes multipath propagation?",
        "A transmitted signal reaches the receiver along multiple paths "
        "after reflecting from objects and surfaces."
    ),
    (
        "What is a matrix?",
        "A matrix is a rectangular arrangement of numbers. Matrix "
        "multiplication represents compositions of linear transformations."
    ),
    (
        "Describe a Python dictionary.",
        "A Python dictionary stores key-value pairs. For example, "
        "{'temperature': 21.5, 'count': 12} associates names with values."
    ),
    (
        "Explain the mean of a dataset.",
        "The arithmetic mean is the sum of the observations divided by "
        "their number. It can be sensitive to extreme values."
    ),
    (
        "What does a validation set do?",
        "A validation set helps compare model configurations during "
        "development. Final evaluation uses a separate test set."
    ),
    (
        "请简要解释无线信道。",
        "无线信道描述信号从发射机传播到接收机的过程，"
        "其特性受到距离、反射、遮挡和移动物体的影响。"
    ),
    (
        "Explain scientific notation.",
        "Scientific notation expresses a number as a coefficient times "
        "a power of ten. For example, 0.00025 equals 2.5 times 10 to the minus four."
    ),
]

print("Loading the local tokenizer...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(
    source_dir, local_files_only=True
)

batches = []
for question, answer in samples:
    text = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ],
        tokenize=False,
        add_generation_prompt=False,
    )
    batch = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=False,
        truncation=True,
        max_length=256,
    )
    batches.append(batch)

token_counts = [int(batch["input_ids"].numel()) for batch in batches]
print("Calibration token counts:", token_counts, flush=True)

print("Loading the original BF16 model...", flush=True)
model = AutoModelForCausalLM.from_pretrained(
    source_dir,
    dtype=torch.bfloat16,
    device_map={"": 0},
    attn_implementation="eager",
    local_files_only=True,
)
model.eval()
model.requires_grad_(False)

# Modify the existing entries so later layer exclusions retain precedence.
recipe = copy.deepcopy(mtq.FP8_DEFAULT_CFG)
for entry in recipe["quant_cfg"]:
    if entry["quantizer_name"] in (
        "*weight_quantizer", "*input_quantizer"
    ):
        entry["cfg"]["type"] = "static"

def forward_loop(calibration_model):
    with torch.no_grad():
        for i, batch in enumerate(batches, start=1):
            print(f"Calibration sample {i}/{len(batches)}", flush=True)
            calibration_model(**batch.to("cuda"), use_cache=False)

print("Starting FP8 calibration...", flush=True)
with torch.no_grad():
    model = mtq.quantize(model, recipe, forward_loop)

# Verify that active input quantizers have calibrated scalar ranges.
activation_ranges = {}
for name, quantizer in model.named_modules():
    if not isinstance(quantizer, TensorQuantizer):
        continue
    if not quantizer.is_enabled or not name.endswith("input_quantizer"):
        continue

    amax = quantizer.amax
    if (
        not quantizer.is_fp8
        or amax is None
        or amax.numel() != 1
        or not torch.isfinite(amax).all().item()
        or not (amax > 0).all().item()
    ):
        raise RuntimeError(f"Invalid FP8 calibration state: {name}")

    activation_ranges[name] = float(amax.item())

if not activation_ranges:
    raise RuntimeError("No calibrated FP8 input quantizers were found.")

print(
    "Calibrated FP8 input quantizers:",
    len(activation_ranges),
    flush=True,
)

print("Exporting the checkpoint...", flush=True)
with torch.inference_mode():
    export_hf_checkpoint(
        model,
        dtype=torch.bfloat16,
        export_dir=str(export_dir),
    )

tokenizer.save_pretrained(export_dir)

quant_config = json.loads(
    (export_dir / "hf_quant_config.json").read_text()
)
algorithm = quant_config["quantization"]["quant_algo"]
if algorithm.upper() != "FP8":
    raise RuntimeError(f"Unexpected exported format: {algorithm}")

if not list(export_dir.glob("*.safetensors")):
    raise RuntimeError("No exported safetensors weights were found.")

manifest = {
    "purpose": "FP8 pipeline compatibility test",
    "source": json.loads((root / "logs/model_revision.json").read_text()),
    "torch": torch.__version__,
    "modelopt": version("nvidia-modelopt"),
    "transformers": version("transformers"),
    "recipe": recipe,
    "calibration_dialogues": samples,
    "calibration_token_counts": token_counts,
    "activation_amax": activation_ranges,
    "exported_quant_algo": algorithm,
}
(export_dir / "calibration_manifest.json").write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
)

print("Exported format:", algorithm)
print("Checkpoint:", export_dir)
print("FP8 EXPORT COMPLETE")
