"""Stable entry point: prepare, demo, evaluate, train, generate, and download."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys
import time
import uuid
from .common import read_json, save_json, fingerprint, package_versions
from . import pipeline

def main():
    parser = argparse.ArgumentParser(description=pipeline.TITLE)
    parser.add_argument("command", choices=["prepare", "demo", "evaluate", "train", "generate", "download", "transcribe"])
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--mode", choices=["fixture", "live"], default="fixture")
    parser.add_argument("--output-dir", default="runs")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--backend", default=None)
    parser.add_argument("--results", default=None, help="Existing run directory to rescore without model calls")
    parser.add_argument("--audio", action="append", default=None, help="Audio file for the optional NeMo Speech input path")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    args = parser.parse_args()
    config = read_json(args.config)
    if args.audio: config['audio_paths']=args.audio
    for name in ("data_dir", "backend", "limit", "steps"):
        value = getattr(args, name)
        if value is not None:
            config[name] = value
    if int(config.get("limit", 3)) < 1:
        parser.error("limit must be positive")
    source = None
    if args.results:
        import hashlib
        source_path=Path(args.results)/"execution.json"
        source=read_json(source_path)
        if source.get("status") != "passed":
            parser.error("Cannot rescore a source run that did not pass")
        config=source["config"]
        args.mode=source["mode"]
        source={"path":str(source_path),"sha256":hashlib.sha256(source_path.read_bytes()).hexdigest()}
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    out = Path(args.output_dir) / f"{stamp}-{args.command}-{uuid.uuid4().hex[:8]}"
    out.mkdir(parents=True, exist_ok=False)
    record = {"project": pipeline.TITLE, "command": args.command, "mode": args.mode,
              "started_utc": stamp, "status": "running", "config_sha256": fingerprint(config),
              "config": config, "python": platform.python_version(), "packages": package_versions()}
    if source: record["rescore_source"]=source
    save_json(out / "execution.json", record)
    start = time.monotonic()
    try:
        if args.results:
            if args.command != "evaluate": raise ValueError("--results is only valid with evaluate")
            result = pipeline.rescore(Path(args.results))
        else:
            result = pipeline.run(args.command, config, args.mode, out)
        save_json(out / "results.json", result)
        record["status"] = "passed"
    except Exception as error:
        import traceback
        (out/"traceback.txt").write_text(traceback.format_exc(),encoding="utf-8")
        record["status"] = "blocked" if isinstance(error, (ImportError, FileNotFoundError)) else "failed"
        record["error_type"] = type(error).__name__
        record["error"] = str(error)
        record["elapsed_seconds"] = time.monotonic() - start
        save_json(out / "execution.json", record)
        print(json.dumps({"status": record["status"], "error": str(error), "run_dir": str(out)}, ensure_ascii=False))
        return 1
    record["elapsed_seconds"] = time.monotonic() - start
    save_json(out / "execution.json", record)
    print(json.dumps({"status": "passed", "mode": args.mode, "run_dir": str(out), "result": result}, ensure_ascii=False))
    return 0

if __name__ == "__main__":
    sys.exit(main())
