# Independent code review

Reviewer: separate agent `build_agents_finance_repos` (not the implementation owner).
Date: 2026-11-16. Review covers local source and executed CPU checks; GPU completion is governed by the repository execution evidence.

## Round 1 — method and correctness

Reviewed frozen NIM corrector, bounded memory edits, strict selection-WER acceptance, train/selection/test separation, reference exclusion from inference, and the independent LoRA adaptation baseline.

Requested corrections were implemented: exact prompt-token prefix validation before assistant loss masking; saved-result rescoring with no new model calls; source-mode/configuration provenance preservation and rejection of failed source runs. For audio and images, content-level leakage checks now cover renamed files. Data Designer reserves its worst-case HTTP attempt allowance persistently before generation, rejects repeated use of the same generation directory, and retains pending human-review status.

## Round 2 — readability and reproducibility

Verified the corrections in source and independently executed `python -m unittest discover -s tests -v`: **5/5 tests passed**. Checked English documentation, explicit fixture/live labels, public paper/project links, source attribution, the separately labelled training baseline, and failure preservation. The documented AutoModel0.7/Triton compatibility patch backports an identified upstream one-line correction and is included as a runnable setup script rather than an undisclosed machine-only patch.

## Disposition

No unresolved blocking issue in the reviewed CPU method, scoring, and data separation code. No paper-scale or semantic-gate reproduction is claimed. Public upstream HF content is identified as method/code/memory artifacts rather than pretrained weights.

Real HF/AutoModel GPU execution and final notebook output checks remain separately recorded requirements. This review does not turn a pending or failed GPU route into a successful result, and does not certify accuracy or improved benchmark performance from a tiny sample.
