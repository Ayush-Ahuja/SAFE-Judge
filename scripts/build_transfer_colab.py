"""Build an uploadable notebook and a compact, checksummed offline input ZIP."""

import hashlib
import json
from pathlib import Path
import sys
import textwrap
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def build_notebook():
    cells = []
    def md(value):
        cells.append({"cell_type": "markdown", "metadata": {}, "source": textwrap.dedent(value).strip() + "\n"})
    def code(value):
        cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                      "outputs": [], "source": textwrap.dedent(value).strip() + "\n"})

    md('''
    # Qwen inference: DL2020, TREC-COVID, ChatReport

    **Start here:** Select **Runtime → Change runtime type → GPU** (T4 or better),
    then **Runtime → Run all**. Authorize Google Drive and upload the companion
    **`qwen_transfer_colab.zip`** when requested. No GitHub setup is needed.

    | Input | Queries | Pairs | Positive label |
    |---|---:|---:|---|
    | TREC DL2020 passages | 54 | 11,386 | Grades 2 and 3 |
    | TREC-COVID frozen judged-pool sample | 50 | 5,000 (100/query) | Grade 2 only |
    | ChatReport expert test | 11 | 660 | Partial and fully relevant |

    COVID uses title + abstract, with the project's existing strict label mapping;
    it is **not the full COVID judged set**. ChatReport uses question + passage,
    without expert background. All inputs and provenance are in the ZIP.

    This runs the project's pinned **Qwen3.5-4B, 4-bit, float16, 512-token** scorer.
    It saves uncalibrated No/Yes probabilities and logits, not just hard labels.
    The final cell downloads one results ZIP to return for policy comparisons.
    No policy is trained and no risk guarantee is claimed here.
    ''')
    code('''
    # Keep these settings unchanged to resume. Use a new RUN_NAME for a new experiment.
    RUN_NAME = "qwen35_transfer_v1"  # @param {type:"string"}
    DATASETS = ["chatreport", "trec_covid", "dl2020"]  # Edit to run a subset.
    RUBRIC_MODE = "dataset"  # @param ["dataset", "shared"]
    PAIR_LIMIT = 0  # @param {type:"integer"}
    # 0 = all pairs. Use 12 for a quick smoke test, saved separately.
    MAX_RUN_MINUTES = 0  # @param {type:"number"}
    # 0 = no voluntary limit; Colab's own session limits still apply.
    MAX_BATCH_SIZE = 8  # @param {type:"integer"}
    BACKUP_EVERY_SECONDS = 120
    DOWNLOAD_RESULTS = True  # @param {type:"boolean"}
    ''')
    md('''
    ## 1. Mount Drive and load the inputs

    Backups go to `MyDrive/SAFE-Judge/<RUN_NAME>/<RUBRIC_MODE>/`.
    On a later session the saved input ZIP is reused automatically. A sudden VM
    loss may lose predictions since the last Drive snapshot (up to roughly two
    minutes). Leave the settings unchanged and run all cells to resume.
    ''')
    code('''
    import hashlib, json, os, re, shutil, subprocess, sys, time, zipfile
    from pathlib import Path, PurePosixPath
    from google.colab import drive, files

    assert re.fullmatch(r"[A-Za-z0-9_-]+", RUN_NAME), "Use letters, numbers, underscores or hyphens in RUN_NAME."
    assert DATASETS and len(DATASETS) == len(set(DATASETS))
    assert set(DATASETS) <= {"chatreport", "trec_covid", "dl2020"}
    assert RUBRIC_MODE in {"dataset", "shared"}
    assert isinstance(PAIR_LIMIT, int) and PAIR_LIMIT >= 0
    assert MAX_RUN_MINUTES >= 0 and MAX_BATCH_SIZE >= 1
    drive.mount("/content/drive")
    DRIVE_BASE = Path("/content/drive/MyDrive/SAFE-Judge") / RUN_NAME / RUBRIC_MODE
    DRIVE_BASE.mkdir(parents=True, exist_ok=True)
    bundle = DRIVE_BASE / "input_bundle.zip"
    if not bundle.exists():
        print("Upload the companion qwen_transfer_colab.zip file now.")
        uploaded = files.upload()
        matches = [name for name in uploaded if Path(name).name == "qwen_transfer_colab.zip"]
        if len(matches) != 1:
            raise ValueError("Please upload exactly the companion qwen_transfer_colab.zip (keep its filename).")
        temporary = bundle.with_suffix(".tmp")
        temporary.write_bytes(uploaded[matches[0]])
        temporary.replace(bundle)
        del uploaded
    digest = hashlib.sha256(bundle.read_bytes()).hexdigest()
    ROOT = Path("/content/safe_judge_transfer") / digest[:16]
    ROOT.mkdir(parents=True, exist_ok=True)
    # Validate member paths before extracting any code from the upload.
    with zipfile.ZipFile(bundle) as archive:
        for item in archive.infolist():
            name = item.filename
            if ("\\\\" in name or ":" in name or PurePosixPath(name).is_absolute()
                or ".." in PurePosixPath(name).parts
                or ((item.external_attr >> 16) & 0o170000) == 0o120000
                or not (ROOT / name).resolve().is_relative_to(ROOT.resolve())):
                raise ValueError(f"Unsafe archive member: {name}")
        archive.extractall(ROOT)
    # Clear previously imported project modules if this runtime loaded another bundle.
    for name in list(sys.modules):
        if name in ("scripts", "src") or name.startswith(("scripts.", "src.")):
            del sys.modules[name]
    sys.path.insert(0, str(ROOT))
    from scripts.transfer_colab_support import verify_bundle, restore_run, archive_results
    verified = verify_bundle(ROOT)
    SUITE = json.loads((ROOT / "data/transfer/suite.json").read_text())
    SUBRUN = "full" if PAIR_LIMIT == 0 else f"smoke_{PAIR_LIMIT}"
    DURABLE = DRIVE_BASE / SUBRUN
    DURABLE.mkdir(parents=True, exist_ok=True)
    LOCAL = Path("/content/safe_judge_transfer_runs") / digest[:16] / RUN_NAME / RUBRIC_MODE / SUBRUN
    LOCAL.mkdir(parents=True, exist_ok=True)
    CONFIG = ROOT / "configs/transfer_colab.yaml"
    # JSON is valid YAML; no PyYAML dependency is needed for this setup cell.
    CONFIG.write_text(json.dumps({"assessor": {
        "model_name": "Qwen/Qwen3.5-4B",
        "revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        "quantization": "4bit", "compute_dtype": "float16",
        "max_length": 512, "batch_size": 1, "rubric_mode": RUBRIC_MODE,
    }, "seed": 42}, indent=2))
    settings = {"run_name": RUN_NAME, "rubric_mode": RUBRIC_MODE, "pair_limit": PAIR_LIMIT,
                "input_bundle_sha256": digest, "datasets": DATASETS}
    (DURABLE / "notebook_settings.json").write_text(json.dumps(settings, indent=2))
    shutil.copy2(CONFIG, DURABLE / "transfer_colab.yaml")
    print(f"Verified {verified} bundled files. Inputs: {digest[:16]}")
    print("Drive results:", DURABLE)
    ''')
    md('''
    ## 2. Check the frozen manifests

    Default `dataset` rubrics implement each dataset's declared positive class.
    This changes task instructions across domains. Optional `shared` uses one
    generic rubric for all three, with separate outputs. Strict fixed-prompt
    transfer from DL2019 would also require a source run using that same shared
    rubric. Neither mode changes the frozen gold labels.
    ''')
    code('''
    from src.utils.io import read_manifest, file_hash
    from src.assessor.transfer_prompts import validate_task, RUBRICS, SHARED_RUBRIC
    for dataset in DATASETS:
        info = SUITE["datasets"][dataset]
        manifest = ROOT / info["manifest"]
        assert file_hash(manifest) == info["manifest_sha256"], f"Changed input: {dataset}"
        rows = read_manifest(manifest)
        for row in rows:
            validate_task(row)
        assert len(rows) == info["pairs"]
        assert len({r["query_id"] for r in rows}) == info["queries"]
        selected = min(PAIR_LIMIT, len(rows)) if PAIR_LIMIT else len(rows)
        print(f"{dataset}: {info['queries']} queries, {info['pairs']:,} pairs; selected {selected:,}")
        print("  Positive grade >=", info["relevance_threshold"])
        print("  Rubric:", SHARED_RUBRIC if RUBRIC_MODE == "shared" else RUBRICS[dataset])
    if PAIR_LIMIT:
        print("SMOKE TEST ONLY: the selected prefixes are not a benchmark evaluation.")
    ''')
    md('''
    ## 3. Install inference dependencies and check the GPU

    Keep this install cell before importing Torch or Transformers. The notebook
    retains Colab's CUDA PyTorch build and records its exact version. If you
    already imported older libraries in this runtime, restart the session and
    run all cells again. A GPU is required for inference.
    ''')
    code('''
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet",
        "torch>=2.6,<3", "transformers==5.3.0", "accelerate==1.15.0",
        "bitsandbytes==0.50.2", "PyYAML==6.0.2", "matplotlib==3.10.1"])
    import torch, transformers, accelerate, bitsandbytes
    assert torch.cuda.is_available(), "Select Runtime > Change runtime type > GPU, then reconnect."
    assert torch.cuda.get_device_capability(0)[0] >= 6, "This GPU does not support NF4."
    assert transformers.__version__ == "5.3.0", "Restart this session to load the installed Transformers version."
    print("GPU:", torch.cuda.get_device_name(0))
    print("VRAM (GiB):", round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1))
    print("PyTorch:", torch.__version__, "CUDA:", torch.version.cuda)
    print("Transformers:", transformers.__version__)
    ''')
    md('''
    ## 4. Run or resume inference

    ChatReport runs first as the smallest dataset. Model weights download on the
    first run. The worker tests batch sizes, prints an estimated remaining time,
    and commits each completed batch locally. Consistent backups reach Drive
    every 120 seconds and when the worker exits normally or handles an interrupt.

    Do not run two copies with the same `RUN_NAME` simultaneously. To pause,
    interrupt this cell, allow the worker to save, then run the final export cell.
    Changed code, package versions, prompts, or model settings are rejected on
    resume; use matching versions or a new `RUN_NAME`.
    ''')
    code('''
    import signal
    started = time.monotonic()
    RUN_RESULTS = []
    for dataset in DATASETS:
        remaining = MAX_RUN_MINUTES * 60 - (time.monotonic() - started) if MAX_RUN_MINUTES else 0
        if MAX_RUN_MINUTES and remaining <= 0:
            print("Voluntary time budget reached. Remaining datasets can resume next session.")
            break
        working, saved = LOCAL / dataset, DURABLE / dataset
        restore_run(working, saved)
        manifest = ROOT / SUITE["datasets"][dataset]["manifest"]
        command = [sys.executable, "-u", str(ROOT / "scripts/run_transfer_inference.py"),
            "--manifest", str(manifest), "--config", str(CONFIG),
            "--output-dir", str(working), "--backup-dir", str(saved),
            "--backup-every-seconds", str(BACKUP_EVERY_SECONDS),
            "--auto-batch", "--max-batch-size", str(MAX_BATCH_SIZE)]
        if PAIR_LIMIT:
            command += ["--limit", str(PAIR_LIMIT)]
        if MAX_RUN_MINUTES:
            command += ["--max-seconds", str(remaining)]
        print("\\n" + "=" * 20 + f" {dataset} " + "=" * 20, flush=True)
        process = None
        interrupted = False
        log_path = working / "console.log"
        try:
            with log_path.open("a", encoding="utf-8") as log:
                process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, text=True, bufsize=1)
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                returncode = process.wait()
        except KeyboardInterrupt:
            interrupted = True
            print("Stopping worker and saving its checkpoint...", flush=True)
            if process is not None and process.poll() is None:
                process.send_signal(signal.SIGINT)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            returncode = process.returncode if process is not None else -1
        finally:
            saved.mkdir(parents=True, exist_ok=True)
            if log_path.exists():
                shutil.copy2(log_path, saved / "console.log")
        RUN_RESULTS.append({"dataset": dataset, "returncode": returncode, "interrupted": interrupted})
        if returncode != 0 or interrupted:
            print("Paused (resumable)." if returncode == 3 or interrupted else
                  "Worker failed; inspect the traceback above. Available backups will still be exported.")
            break
    (DURABLE / "notebook_run_results.json").write_text(json.dumps(RUN_RESULTS, indent=2))
    print("\\nInference cell ended. Continue to the final cell to check completion and export.")
    ''')
    md('''
    ## 5. Check completion, compute diagnostics, and download results

    This cell can be run manually after stopping inference. It exports durable
    Drive snapshots, including partial runs. It computes raw-score diagnostics
    only for complete full runs; it never fits calibration on the test labels.

    **Return the downloaded results ZIP** for the policy comparison. The ZIP
    includes predictions, logits, checkpoints, metadata, inputs and diagnostics;
    it excludes model weights. Inference alone does not show that our policy
    beats LARA or Oosterhuis.
    ''')
    code('''
    status_rows = []
    for dataset in DATASETS:
        saved = DURABLE / dataset
        progress_path = saved / "progress.json"
        progress = json.loads(progress_path.read_text()) if progress_path.exists() else {}
        full_count = SUITE["datasets"][dataset]["pairs"]
        target = min(PAIR_LIMIT, full_count) if PAIR_LIMIT else full_count
        done = progress.get("completed_pairs", 0)
        complete = bool(progress.get("complete")) and done == target
        status = "SMOKE COMPLETE" if complete and PAIR_LIMIT else ("COMPLETE" if complete else "INCOMPLETE")
        status_rows.append({"dataset": dataset, "completed_pairs": done,
                            "target_pairs": target, "full_pairs": full_count, "status": status})
        print(f"{dataset:12} {done:>6,}/{target:,}  {status}")
        if complete and not PAIR_LIMIT and (saved / "predictions.csv").exists():
            command = [sys.executable, str(ROOT / "scripts/evaluate_pilot.py"),
                "--manifest", str(ROOT / SUITE["datasets"][dataset]["manifest"]),
                "--predictions", str(saved / "predictions.csv"),
                "--output-dir", str(saved / "evaluation"), "--plots"]
            result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
            print(result.stdout)
            if result.returncode:
                print("Diagnostics failed; raw predictions are still exported:", result.stderr)
    (DURABLE / "completion_summary.json").write_text(json.dumps(status_rows, indent=2))
    RESULT_ZIP = DRIVE_BASE / f"{RUN_NAME}_{RUBRIC_MODE}_{SUBRUN}_results.zip"
    archive_results(ROOT, DURABLE, RESULT_ZIP)
    print("Saved:", RESULT_ZIP)
    print("ZIP size (MiB):", round(RESULT_ZIP.stat().st_size / 2**20, 2))
    if DOWNLOAD_RESULTS:
        files.download(str(RESULT_ZIP))
    ''')
    notebook = {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
                "metadata": {"accelerator": "GPU", "colab": {"name": "qwen_transfer_colab.ipynb", "provenance": []},
                             "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                             "language_info": {"name": "python"}}}
    for index, cell in enumerate(cells):
        cell["id"] = f"transfer-{index:02d}"
        if cell["cell_type"] == "code":
            compile(cell["source"], f"cell-{index}", "exec")
    path = ROOT / "notebooks/qwen_transfer_colab.ipynb"
    path.write_text(json.dumps(notebook, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def main():
    notebook = build_notebook()
    suite_path = ROOT / "data/transfer/suite.json"
    if not suite_path.exists():
        raise SystemExit("Run scripts/prepare_transfer_inputs.py first.")
    names = ["configs/default.yaml", "docs/qwen_transfer_colab.md",
             "scripts/run_transfer_inference.py", "scripts/transfer_colab_support.py",
             "scripts/evaluate_pilot.py", "scripts/compare_predictions.py",
             "scripts/prepare_transfer_inputs.py", "scripts/build_transfer_colab.py",
             "notebooks/qwen_transfer_colab.ipynb"]
    names += [p.relative_to(ROOT).as_posix() for p in (ROOT / "data/transfer").glob("*.json*")]
    for directory in ("src", "src/assessor", "src/evaluation", "src/utils"):
        names.append(f"{directory}/__init__.py")
    names += ["src/assessor/model.py", "src/assessor/prompts.py", "src/assessor/transfer_model.py",
              "src/assessor/transfer_prompts.py", "src/utils/io.py", "src/evaluation/metrics.py"]
    names = sorted(set(names))
    receipt = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in names}
    target = ROOT / "dist/qwen_transfer_colab.zip"
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in names:
            archive.write(ROOT / name, name)
        archive.writestr("bundle_hashes.json", json.dumps(receipt, indent=2) + "\n")
    print(notebook)
    print(target, f"({target.stat().st_size / 2**20:.2f} MiB)")


if __name__ == "__main__":
    main()
