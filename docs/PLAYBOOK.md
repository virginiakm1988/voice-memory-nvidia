# Demo playbook

Audience: solution architects, researchers and workshop participants.

## 1. Explain the research question

Optimize a short, editable text memory while keeping the speech-correction model frozen. A candidate edit is accepted only when selection-set word error rate strictly improves. The test set is evaluated after optimization.

Use the overview diagram and identify the data input, NVIDIA SDK operation, evaluation gate and final artifact. Explain which behavior comes from the original research and which is a new SDK adaptation.

## 2. Run the offline contract

```bash
python -m unittest discover -s tests -v
python -m project_demo demo --mode fixture
```

The fixture verifies deterministic control flow and metric calculations. Its numbers are not model benchmark results.

## 3. Prepare actual data and record provenance

```bash
python -m project_demo prepare --mode live
```

For Voice Memory, supply separate train/selection/test JSONL files first, or use an authored-fixture preparation directory explicitly labeled as a teaching corpus. Preserve official benchmark splits. Record the source revision and selected IDs; do not tune on test outputs.

## 4. Run actual inference

```bash
# Replace DATA with the preparation directory containing train/selection/test.jsonl
python -m project_demo evaluate --mode live --data-dir DATA
```

Inspect per-item outputs before averaging scores. Demonstrate a failure or uncertainty case. A model refusing access, malformed output, missing data or unavailable CUDA must produce an explicit failed/blocked record.

## 5. Adapt and compare

```bash
python -m project_demo train --mode live --data-dir DATA --steps 20
```

Confirm nonzero gradients, changed adapter weights, exact checkpoint reload and before/after held-out outputs. Report the measured change even if it is negative. Keep optimization/selection data separate from final test data.

## 6. Handoff

Share the pinned repo commit, environment record, config, source links and public-safe evidence. Keep credentials and restricted model/data artifacts out of Git. See README for scope limitations and third-party terms.
