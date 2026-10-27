#!/usr/bin/env python3
"""Prepare and audit DEVELOPMENT tasks; never load or evaluate official tests.

Requires prepare_day7_sources.py in the same scripts directory and .venv-eval.
The task/score settings are immutable on rerun. No server or GPU is required.
"""

import argparse
import copy
import hashlib
import json
from importlib.metadata import distribution, version
from pathlib import Path

import pyarrow.parquet as pq
import yaml
from lm_eval.api.task import ConfigurableTask
from transformers import AutoTokenizer

from prepare_day7_sources import GSM_REVISION, require, sha256

CONTEXT_LIMIT = 8192
OUTPUT_LIMIT = 2048


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()


def jsonl_bytes(records):
    return b"".join((json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n").encode()
                    for item in records)


def put(path, payload):
    """Create once, or require identical bytes; do not replace a prior protocol."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(path.read_bytes() == payload, f"Existing prepared file differs: {path}")
    else:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_bytes(payload)
        temporary.replace(path)


def normalized_question_hash(question):
    text = " ".join(question.split())
    return hashlib.sha256(text.encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(
        "/root/autodl-tmp/qwen-inference-lab"))
    root = parser.parse_args().root.resolve()
    require(version("lm_eval") == "0.4.13", "Expected frozen lm_eval 0.4.13")
    source_dir = root / "data/evaluation/day7_sources"
    source_path = source_dir / "source_revisions.json"
    source = json.loads(source_path.read_text())
    out = root / "data/evaluation/day7_quality_v1"
    receipts = {}

    def remember(path):
        receipts[str(path.relative_to(root))] = sha256(path)

    remember(source_path)
    split_path = root / "data/splits/gsm8k_train_split.json"
    splits = json.loads(split_path.read_text())["partitions"]
    remember(split_path)
    expected_counts = {"development": 256, "fewshot_reserve": 16, "calibration": 7201}
    row_sets = {}
    for partition, count in expected_counts.items():
        records = splits[partition]
        require(len(records) == count, f"Unexpected {partition} size")
        row_sets[partition] = {record["row"] for record in records}
        require(len(row_sets[partition]) == count, f"Duplicate rows in {partition}")
    require(sum(map(len, row_sets.values())) == len(set.union(*row_sets.values())) == 7473,
            "Training partitions overlap or do not cover all training rows")

    def read_split(name, split):
        record = source["datasets"][name]
        require(record["complete"], f"Source preparation incomplete: {name}")
        if name == "gsm8k":
            require(record["revision"] == GSM_REVISION, "GSM8K source revision changed")
        files = sorted((x for x in record["files"] if x["split"] == split),
                       key=lambda item: item["repository_path"])
        require(files, f"Missing source files for {name}/{split}")
        rows = []
        for item in files:
            path = root / item["path"]
            require(path.stat().st_size == item["bytes"] and sha256(path) == item["sha256"],
                    f"Source file changed: {path}")
            remember(path)
            chunk = pq.read_table(path).to_pylist()
            require(len(chunk) == item["rows"], "Source row count differs")
            rows.extend(chunk)
        return rows

    # Deliberately load only GSM8K TRAIN and ARC VALIDATION here.
    training = read_split("gsm8k", "train")
    require(len(training) == 7473, "Unexpected GSM8K training row count")
    for partition in expected_counts:
        for record in splits[partition]:
            row = record["row"]
            require(isinstance(row, int) and 0 <= row < len(training), "Invalid training row")
            require(normalized_question_hash(training[row]["question"]) == record["question_sha256"],
                    f"Saved split/question mismatch at row {row}")

    def math_record(split_record):
        row = split_record["row"]
        return {"source_id": f"gsm8k-train-{row:05d}",
                "question": training[row]["question"], "answer": training[row]["answer"]}

    development = [math_record(item) for item in splits["development"]]
    # Preserve the saved reserve ordering. No selection based on model outputs.
    fewshot = [math_record(item) for item in splits["fewshot_reserve"][:5]]
    require(len({normalized_question_hash(x["question"]) for x in fewshot}) == 5,
            "Few-shot examples are duplicated")
    for example in fewshot:
        require("#### " in example["answer"], "Few-shot answer lacks GSM8K final-answer marker")
    arc_validation = read_split("arc_challenge", "validation")
    require(len(arc_validation) == 299, "Unexpected ARC validation size")
    for row in arc_validation:
        labels, texts = row["choices"]["label"], row["choices"]["text"]
        require(len(labels) == len(texts) and len(labels) >= 2, "Invalid ARC choices")
        require(row["answerKey"] in labels, "ARC reference is missing from choices")

    gsm_data = out / "gsm8k_development.jsonl"
    arc_data = out / "arc_validation.jsonl"
    fewshot_data = out / "gsm8k_fewshot5.jsonl"
    for path, rows in ((gsm_data, development), (arc_data, arc_validation),
                       (fewshot_data, fewshot)):
        put(path, jsonl_bytes(rows))
        remember(path)

    defaults_path = source_dir / "task_defaults_manifest.json"
    defaults = json.loads(defaults_path.read_text())
    require(defaults["lm_eval_version"] == version("lm_eval"), "Task default version differs")
    remember(defaults_path)
    for item in defaults["files"].values():
        path = root / item["path"]
        require(sha256(path) == item["sha256"], "Archived task template changed")
        remember(path)
    gsm = yaml.safe_load((source_dir / "task_defaults/gsm8k/gsm8k.yaml").read_text())
    arc = yaml.safe_load((source_dir / "task_defaults/arc/arc_easy.yaml").read_text())
    for config, name, data in ((gsm, "day7_gsm8k_dev", gsm_data),
                               (arc, "day7_arc_challenge_dev", arc_data)):
        config.pop("tag", None)
        config.update(task=name, dataset_path="json", dataset_name=None,
                      dataset_kwargs={"data_files": {"validation": str(data)}},
                      training_split=None, validation_split="validation", test_split=None,
                      fewshot_split=None, target_delimiter=" ", fewshot_delimiter="\n\n",
                      should_decontaminate=False,
                      metadata={"version": 1.0, "study": "day7_quality_v1",
                                "scope": "development_only"})
    gsm.update(num_fewshot=5, fewshot_config={"sampler": "first_n", "samples": fewshot})
    gsm["generation_kwargs"].update(max_gen_toks=OUTPUT_LIMIT, temperature=0.0, top_p=1.0)
    arc.update(num_fewshot=0, fewshot_config={"sampler": "first_n"})
    configs = [(gsm, development, 5), (arc, arc_validation, 0)]
    for config, _, _ in configs:
        path = out / "tasks/development" / (config["task"] + ".yaml")
        put(path, yaml.safe_dump(config, allow_unicode=True, sort_keys=False).encode())
        remember(path)

    tokenizer_dir = root / "models/Qwen3-4B-Instruct-2507"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True,
                                             trust_remote_code=False)
    for filename in ("tokenizer.json", "tokenizer_config.json", "config.json"):
        remember(tokenizer_dir / filename)
    budgets, request_counts = {}, {}
    for config, rows, shots in configs:
        print(f"Building actual harness prompts: {config['task']}", flush=True)
        task = ConfigurableTask(config=copy.deepcopy(config))
        task.set_fewshot_seed(42)
        if shots:
            actual = task.sampler.sample(n=5)
            require([x["source_id"] for x in actual] == [x["source_id"] for x in fewshot],
                    "Harness did not use the five reserved examples in their fixed order")
        task.build_all_requests()
        requests, lengths, by_doc = [], [], {}
        for instance in task.instances:
            doc_id = int(instance.doc_id)
            args = instance.args
            context = args[0]
            require(isinstance(context, str), "Expected plain completion prompts")
            # Compare the actual harness context to the intended fixed plain-text format.
            expected = "".join(
                f"Question: {x['question']}\nAnswer: {x['answer']}\n\n"
                for x in fewshot) if shots else ""
            expected += f"Question: {rows[doc_id]['question']}\nAnswer:"
            require(context == expected, f"Unexpected rendered prompt for {config['task']}/{doc_id}")
            entry = {"doc_id": doc_id, "request_type": instance.request_type,
                     "context": context, "input_tokens": len(tokenizer.encode(
                         context, add_special_tokens=False))}
            if shots:
                require(instance.request_type == "generate_until", "Expected math generation")
                require(args[1]["max_gen_toks"] == OUTPUT_LIMIT, "Output limit differs")
                entry.update(source_id=rows[doc_id]["source_id"], generation_kwargs=args[1])
                total = entry["input_tokens"] + OUTPUT_LIMIT
            else:
                require(instance.request_type == "loglikelihood", "Expected ARC choice scoring")
                index = int(instance.idx)
                require(args[1] == " " + rows[doc_id]["choices"]["text"][index],
                        "ARC continuation differs from the answer-choice text")
                entry.update(source_id=rows[doc_id]["id"], choice_index=index,
                             continuation=args[1])
                total = len(tokenizer.encode(context + args[1], add_special_tokens=False)) + 1
            require(total <= CONTEXT_LIMIT, f"Context budget exceeded: {total}")
            lengths.append(entry["input_tokens"])
            by_doc[doc_id] = by_doc.get(doc_id, 0) + 1
            requests.append(entry)
        expected_counts_by_doc = {i: (1 if shots else len(row["choices"]["text"]))
                                  for i, row in enumerate(rows)}
        require(by_doc == expected_counts_by_doc, "Missing or duplicate development requests")
        path = out / (config["task"] + "_requests.jsonl")
        put(path, jsonl_bytes(requests)); remember(path)
        budgets[config["task"]] = {"prompt_tokens_min": min(lengths),
                                    "prompt_tokens_max": max(lengths)}
        request_counts[config["task"]] = len(requests)
        preview = out / (config["task"] + "_prompt_preview.txt")
        put(preview, requests[0]["context"].encode()); remember(preview)
        print(f"  questions={len(rows)}; requests={len(requests)}; "
              f"prompt tokens={min(lengths)}..{max(lengths)}")

    harness = Path(distribution("lm_eval").locate_file("lm_eval"))
    harness_hashes = {str(path.relative_to(harness)): sha256(path)
                      for path in sorted(harness.rglob("*.py"))}
    source_hashes = out / "harness_source_hashes.json"
    put(source_hashes, json_bytes(harness_hashes)); remember(source_hashes)
    remember(Path(__file__).resolve())
    remember(Path(__file__).with_name("prepare_day7_sources.py").resolve())
    freeze = root / "logs/evaluation/day7/eval-requirements.txt"
    remember(freeze)
    settings = {
        "study": "day7_quality_v1", "stage": "development_checks_pending",
        "final_model_scope": None, "official_test_inference_run": False,
        "context_limit": CONTEXT_LIMIT, "gsm8k_output_limit": OUTPUT_LIMIT,
        "generation_seed": 42, "fewshot_seed": 42, "apply_chat_template": False,
        "tokenizer_add_special_tokens": False,
        "gsm8k_primary": "exact_match,strict-match",
        "gsm8k_secondary": "exact_match,flexible-extract",
        "arc_primary": "acc_norm,none", "arc_secondary": "acc,none",
        "fewshot_policy": "first five entries in the saved fewshot_reserve partition",
        "fewshot_ids": [x["source_id"] for x in fewshot],
        "development_questions": {"gsm8k": 256, "arc_challenge": 299},
        "request_counts": request_counts, "token_budgets": budgets,
        "packages": {name: version(name) for name in (
            "lm_eval", "datasets", "transformers", "huggingface_hub")},
        "file_sha256": receipts,
        "day4_results": "Unchanged; its zero-shot prompt/parser differs from this protocol.",
    }
    manifest_path = out / "development_protocol.json"
    put(manifest_path, json_bytes(settings))
    print("\nFive-shot IDs:", ", ".join(settings["fewshot_ids"]))
    print(f"Math output allowance: {OUTPUT_LIMIT}; context limit: {CONTEXT_LIMIT}")
    print("Primary scores: GSM8K strict-match; ARC-Challenge acc_norm")
    print(f"Protocol SHA256: {sha256(manifest_path)}")
    print(f"Saved: {out}")
    print("PASS: reserved examples, split separation, harness prompts and token budgets verified.")
    print("No model requests sent. Official test question/answer files were not loaded.")


if __name__ == "__main__":
    main()
