"""Pinned public HyPoradise N-best loader; never partitions the official test set."""
import random
from pathlib import Path
from .common import read_json, save_json, save_jsonl, ensure_disjoint, fingerprint, normalized

def prepare_hyporadise(config, output):
    from huggingface_hub import HfApi, hf_hub_download
    repo = "PeacefulData/HyPoradise-v0"
    revision = config.get("dataset_revision") or HfApi().dataset_info(repo).sha
    domain = config.get("domain", "atis")
    if domain not in ("atis", "chime4", "coraal", "cv", "lrs2", "swbd", "td3", "wsj_score"):
        raise ValueError("Unsupported HyPoradise domain with matching public train/test files")
    source = {}
    for split in ("train", "test"):
        path = hf_hub_download(repo, f"{split}/{split}_{domain}.json", repo_type="dataset", revision=revision)
        raw = read_json(path)
        source[split] = [{"id": row["id"], "reference": row["output"],
                          "hypotheses": row["input"], "domain": domain} for row in raw]
    # Some published domains repeat IDs between files. Exclude those training
    # records before creating a selection split; never move test rows to train.
    test_ids = {r["id"] for r in source["test"]}
    test_inputs = {tuple(normalized(h) for h in r["hypotheses"]) for r in source["test"]}
    shuffled = [r for r in source["train"] if r["id"] not in test_ids
                and tuple(normalized(h) for h in r["hypotheses"]) not in test_inputs]
    removed = len(source["train"]) - len(shuffled)
    random.Random(int(config.get("seed", 42))).shuffle(shuffled)
    n = int(config.get("train_limit", 3))
    m = int(config.get("selection_limit", 3))
    if min(n, m) < 1 or n + m > len(shuffled):
        raise ValueError("Invalid train/selection subset sizes")
    splits = {"train": shuffled[:n], "selection": shuffled[n:n+m],
              "test": source["test"][:int(config.get("limit", 3))]}
    for split, rows in splits.items():
        save_jsonl(output / f"{split}.jsonl", rows)
    metadata = {"dataset": repo, "revision": revision, "domain": domain,
                "original_counts": {k:len(v) for k,v in source.items()},
                "excluded_training_rows_overlapping_test": removed,
                "counts": {k:len(v) for k,v in splits.items()},
                "selected_ids": {k:[r["id"] for r in v] for k,v in splits.items()},
                "selection_rule": "seeded shuffle of official train only; official test untouched",
                "data_sha256": fingerprint(splits), "license": "MIT dataset card; verify original corpus terms"}
    save_json(output / "source.json", metadata)
    return metadata
