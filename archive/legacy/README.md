# Legacy Archive

This directory keeps older experiments and scripts that are no longer part of the
teacher-facing reproducible workflow. They are retained for provenance only.

Active entrypoints live in the top-level `scripts/`, `comparison/scripts/`, and
`configs/` directories.

| Legacy item | Current replacement |
|---|---|
| `comparison/scripts/compare.py` | `scripts/plot_canonical_synthetic_comparison.py` |
| `comparison/scripts/run_pytorch.py` | `scripts/train.py` and `scripts/evaluate.py` |
| `comparison/scripts/run_autocpd.py` | `comparison/scripts/train_autocpd_paper_faithful.py` |
| `comparison/scripts/train_autocpd.py` | `comparison/scripts/train_autocpd_paper_faithful.py` |
| `comparison/scripts/gen_test_set.py` | `scripts/generate_reproducible_data.py` |

Do not use files here for README or report results unless they are explicitly
restored and re-validated against the current codebase.
