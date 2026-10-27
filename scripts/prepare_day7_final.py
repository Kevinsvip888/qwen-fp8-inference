#!/usr/bin/env python3
"""Prepare all frozen final-test requests, without calling a model or scoring it."""

import argparse
import copy
import json
from importlib.metadata import version
from pathlib import Path

import pyarrow.parquet as pq
import yaml
from lm_eval.api.task import ConfigurableTask
from transformers import AutoTokenizer

from prepare_day7_sources import require, sha256

PLAN_SHA256 = "1bd1f014f7409f32e93347cbd25e39eca79a782a614acd0234a5ed7c003e4558"


def put(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(path.read_bytes() == payload, f"Existing prepared file differs: {path}")
    else:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_bytes(payload)
        temporary.replace(path)


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()


def jsonl_bytes(rows):
    return b"".join((json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode()
                    for row in rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(
        "/root/autodl-tmp/qwen-inference-lab"))
    root = parser.parse_args().root.resolve()
    folder = root / "data/evaluation/day7_quality_v1"
    plan_path = folder / "final_plan.json"
    require(sha256(plan_path) == PLAN_SHA256, "Final plan differs from the recorded pin")
    plan = json.loads(plan_path.read_text())
    development_path = folder / "development_protocol.json"
    require(sha256(development_path) == plan["development_protocol_sha256"],
            "Development protocol changed")
    development = json.loads(development_path.read_text())
    for relative, expected in development["file_sha256"].items():
        require(sha256(root / relative) == expected, f"Prepared input changed: {relative}")
    for package, expected in development["packages"].items():
        require(version(package) == expected, f"Package version changed: {package}")
    require(len(plan["models"]) == 10, "Expected the frozen ten-model comparison")
    require(plan["context_limit"] == 8192 and plan["math_max_output_tokens"] == 2048,
            "Frozen token budgets differ")
    require(plan["apply_chat_template"] is False and plan["generation_seed"] == 42,
            "Frozen prompt/generation settings differ")
    for model in plan["models"]:
        if model["name"] != "BF16":
            path = root / model["path"] / "calibration_manifest.json"
            require(sha256(path) == model["calibration_manifest_sha256"],
                    f"Checkpoint manifest changed: {model['name']}")

    source_path = root / "data/evaluation/day7_sources/source_revisions.json"
    require(sha256(source_path) == plan["source_manifest_sha256"], "Source manifest changed")
    sources = json.loads(source_path.read_text())["datasets"]
    tokenizer = AutoTokenizer.from_pretrained(
        root / "models/Qwen3-4B-Instruct-2507", local_files_only=True,
        trust_remote_code=False)
    out = folder / "final"
    files, prepared_tasks = {}, {}

    def remember(path):
        files[str(path.relative_to(root))] = sha256(path)

    for dataset, old_name, task_name in (
        ("gsm8k", "day7_gsm8k_dev", "day7_gsm8k_test"),
        ("arc_challenge", "day7_arc_challenge_dev", "day7_arc_challenge_test"),
    ):
        math_task = dataset == "gsm8k"
        source = sources[dataset]
        require(source["complete"], f"Incomplete source: {dataset}")
        rows = []
        for item in sorted(source["files"], key=lambda x: x["repository_path"]):
            if item["split"] != "test":
                continue
            path = root / item["path"]
            require(path.stat().st_size == item["bytes"] and sha256(path) == item["sha256"],
                    f"Test source file changed: {path}")
            chunk = pq.read_table(path).to_pylist()
            require(len(chunk) == item["rows"], "Test shard row count differs")
            rows.extend(chunk)
            remember(path)
        require(len(rows) == plan["test_questions"][dataset], "Final test size differs")
        data_path = out / f"{dataset}_test.jsonl"
        put(data_path, jsonl_bytes(rows)); remember(data_path)
        config = yaml.safe_load((folder / "tasks/development" / f"{old_name}.yaml").read_text())
        config.update(task=task_name,
                      dataset_kwargs={"data_files": {"test": str(data_path)}},
                      validation_split=None, test_split="test")
        config["metadata"] = {"version": 1.0, "study": plan["study"],
                              "scope": "official_test", "final_plan_sha256": PLAN_SHA256}
        require(config["num_fewshot"] == (5 if math_task else 0), "Few-shot count differs")
        shots = config["fewshot_config"].get("samples", [])
        if math_task:
            require([x["source_id"] for x in shots] == plan["fewshot_ids"],
                    "Reserved few-shot examples changed")
            require(config["generation_kwargs"]["max_gen_toks"] == 2048,
                    "Math generation limit changed")
        config_path = out / "tasks" / f"{task_name}.yaml"
        put(config_path, yaml.safe_dump(config, allow_unicode=True, sort_keys=False).encode())
        remember(config_path)

        print(f"Building all final prompts: {task_name} ({len(rows)} questions)", flush=True)
        task = ConfigurableTask(config=copy.deepcopy(config))
        task.set_fewshot_seed(42)
        task.build_all_requests()
        requests, budgets, by_doc = [], [], {}
        for instance in task.instances:
            doc_id = int(instance.doc_id)
            context = instance.args[0]
            require(isinstance(context, str), "Expected plain completion prompts")
            prefix = ("".join(f"Question: {x['question']}\nAnswer: {x['answer']}\n\n"
                              for x in shots) if math_task else "")
            require(context == prefix + f"Question: {rows[doc_id]['question']}\nAnswer:",
                    "Final prompt format differs from the verified development format")
            entry = {"doc_id": doc_id, "request_type": instance.request_type,
                     "context": context,
                     "source_id": (f"gsm8k-test-{doc_id:05d}" if math_task else rows[doc_id]["id"]),
                     "input_tokens": len(tokenizer.encode(context, add_special_tokens=False))}
            if math_task:
                require(instance.request_type == "generate_until", "Unexpected math request type")
                entry["generation_kwargs"] = instance.args[1]
                budget = entry["input_tokens"] + 2048
            else:
                require(instance.request_type == "loglikelihood", "Expected science answer scoring")
                index = int(instance.idx)
                continuation = instance.args[1]
                require(continuation == " " + rows[doc_id]["choices"]["text"][index],
                        "Science continuation differs from its answer choice")
                entry.update(choice_index=index, continuation=continuation)
                budget = len(tokenizer.encode(context + continuation, add_special_tokens=False)) + 1
            require(budget <= 8192, f"Final request exceeds frozen context: {task_name}/{doc_id}")
            budgets.append(budget)
            by_doc[doc_id] = by_doc.get(doc_id, 0) + 1
            requests.append(entry)
        expected_counts = {i: (1 if math_task else len(row["choices"]["text"]))
                           for i, row in enumerate(rows)}
        require(by_doc == expected_counts, "Missing or duplicate final requests")
        request_path = out / f"{task_name}_requests.jsonl"
        put(request_path, jsonl_bytes(requests)); remember(request_path)
        prepared_tasks[task_name] = {"dataset": dataset, "questions": len(rows),
                                     "api_requests": len(requests),
                                     "max_input_plus_output_budget": max(budgets),
                                     "config_path": str(config_path.relative_to(root)),
                                     "requests_path": str(request_path.relative_to(root))}
        print(f"  Verified {len(requests)} requests; maximum token budget: {max(budgets)}/8192")

    remember(Path(__file__).resolve())
    manifest = {"study": plan["study"], "final_plan_sha256": PLAN_SHA256,
                "models": len(plan["models"]), "tasks": prepared_tasks,
                "file_sha256": files, "model_inference_performed": False}
    manifest_path = out / "preparation_manifest.json"
    put(manifest_path, json_bytes(manifest))
    print(f"\nPreparation SHA256: {sha256(manifest_path)}")
    print(f"Saved: {out}")
    print("PASS: full test tasks, fixed examples, prompts and token budgets verified.")
    print("No model requests sent; no test scores computed.")


if __name__ == "__main__":
    main()
