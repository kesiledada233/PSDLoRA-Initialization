"""Matrix expansion shared by the runner and completeness checker."""

from __future__ import annotations

import itertools
from pathlib import Path

import yaml


def load_matrix(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict) or "matrix_name" not in config:
        raise ValueError(f"Invalid matrix config: {path}")
    return config


def _products(models, tasks, methods, seeds, max_steps, target="qv"):
    for model, task, method, seed in itertools.product(models, tasks, methods, seeds):
        yield {
            "model": model, "task": task, "method": method, "seed": int(seed),
            "max_steps": int(max_steps), "target": target,
        }


def checkpoint_steps_for_run(config: dict, run: dict) -> set[int]:
    checkpoints = {int(step) for step in config.get("checkpoints", [])}
    checkpoints.update(int(step) for step in run.get("checkpoints", []))
    section = config.get(run.get("section"), {})
    if isinstance(section, dict):
        checkpoints.update(int(step) for step in section.get("checkpoints", []))
    return checkpoints


def expand_matrix(config: dict) -> list[dict]:
    name = config["matrix_name"]
    runs: list[dict] = []
    if name == "integration_smoke":
        if config.get("purpose") != "integration_smoke":
            raise ValueError("integration_smoke matrix must declare purpose=integration_smoke")
        defaults = config["training_defaults"]
        for item in config.get("runs", []):
            training = dict(defaults)
            training.update(item.get("training_overrides", {}))
            run = {
                "model": item["model"], "task": item["task"], "method": item["method"],
                "seed": int(item["seed"]), "max_steps": int(item["max_steps"]),
                "target": "qv", "case": item["case"], "variant": f"smoke_{item['case']}",
                "training": training, "target_modules": training["target_modules"],
                "checkpoints": [int(step) for step in item.get("checkpoints", [])],
                "logging": dict(item.get("logging", {})),
            }
            for key in ("gradient_batches", "gradient_batch_size", "gradient_max_length", "stable_gamma"):
                if key in item:
                    run[key] = item[key]
            runs.append(run)
    elif name == "scope":
        defaults = config["training_defaults"]
        for section_name in ("all_linear", "long_run"):
            section = config[section_name]
            target = "all_linear" if section_name == "all_linear" else "qv"
            for model, task in section["model_task_pairs"]:
                entries = _products([model], [task], section["methods"], section["seeds"], section["max_steps"], target)
                for entry in entries:
                    entry.update({"section": section_name, "training": defaults, "target_modules": section["target_modules"]})
                    runs.append(entry)
    elif name == "baseline_fairness_qwen_table3":
        defaults = config["training_defaults"]
        for task in config["tasks"]:
            for search_method, search in config["screening"]["methods"].items():
                central_lr = float(search["central_learning_rate"])
                method_training = dict(defaults)
                method_training.update(search.get("lora_config_overrides", {}))
                initialization = {
                    key: search[key]
                    for key in ("gradient_batches", "gradient_batch_size", "gradient_max_length", "stable_gamma")
                    if key in search
                }
                if "learning_rate_multipliers" in search:
                    for multiplier in search["learning_rate_multipliers"]:
                        runs.append({
                            "model": config["model"], "task": task, "method": search["method"],
                            "seed": int(config["screening"]["seed"]), "max_steps": int(config["screening"]["max_steps"]),
                            "target": "qv", "section": "screening", "search_method": search_method,
                            "learning_rate": central_lr * float(multiplier), "central_learning_rate": central_lr,
                            "learning_rate_multiplier": float(multiplier), "variant": f"lr_x{multiplier}",
                            "training": method_training, "target_modules": defaults["target_modules"], **initialization,
                        })
                else:
                    for alpha in search["alpha_values"]:
                        runs.append({
                            "model": config["model"], "task": task, "method": f"powerlaw_global_a{int(alpha*10):02d}",
                            "seed": int(config["screening"]["seed"]), "max_steps": int(config["screening"]["max_steps"]),
                            "target": "qv", "section": "screening", "search_method": search_method,
                            "learning_rate": central_lr, "central_learning_rate": central_lr,
                            "alpha": float(alpha), "variant": f"alpha_{alpha}",
                            "training": method_training, "target_modules": defaults["target_modules"],
                        })
        final_tasks = config["final"].get("tasks", config["tasks"])
        for run in _products([config["model"]], final_tasks, config["final"]["methods"], config["final"]["seeds"], config["final"]["max_steps"]):
            method_training = dict(defaults)
            initialization = {}
            if run["method"] == "lora_one":
                search = config["screening"]["methods"]["lora_one"]
                method_training.update(search.get("lora_config_overrides", {}))
                initialization = {
                    key: search[key]
                    for key in ("gradient_batches", "gradient_batch_size", "gradient_max_length", "stable_gamma")
                }
            run.update({
                "section": "final", "training": method_training,
                "target_modules": defaults["target_modules"], **initialization,
            })
            if run["method"] != "peft_default":
                run["selection_method"] = "proposed" if run["method"] == "validation_selected_proposed" else run["method"]
            runs.append(run)
    else:
        training = config["training"]
        pairs = config.get("model_task_pairs")
        if pairs:
            for model, task in pairs:
                for run in _products([model], [task], config["methods"], config["seeds"], config["max_steps"]):
                    run.update({"training": training, "target_modules": training["target_modules"]})
                    runs.append(run)
        else:
            runs.extend(_products(config["models"], config["tasks"], config["methods"], config["seeds"], config["max_steps"]))
            for run in runs:
                run.update({"training": training, "target_modules": training["target_modules"]})
            for block in config.get("additional_runs", []):
                for run in _products(block["models"], block["tasks"], block["methods"], config["seeds"], config["max_steps"]):
                    run.update({"training": training, "target_modules": training["target_modules"]})
                    runs.append(run)
    for run in runs:
        suffix_parts = []
        if run.get("variant"):
            suffix_parts.append(run["variant"])
        if config.get("run_suffix"):
            suffix_parts.append(str(config["run_suffix"]))
        suffix = "".join(f"__{part}" for part in suffix_parts)
        run["run_id"] = (
            f"{run['model']}__{run['task']}__{run['target']}__{run['method']}"
            f"__s{run['seed']}__n{run['max_steps']}{suffix}"
        )
    seen = set()
    duplicate = [run["run_id"] for run in runs if run["run_id"] in seen or seen.add(run["run_id"])]
    if duplicate:
        raise ValueError(f"Duplicate run IDs: {duplicate[:5]}")
    return runs
