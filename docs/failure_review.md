# Manual failure review

`scripts/export_failures.py` selects fifty unique disagreements by default and
refuses to overwrite an existing review CSV. Sampling is stratified across
query, error type and confidence; it is diagnostic, not a prevalence estimate.

For each case, read the query, full document and model-visible prefix. Keep
official gold unchanged. Fill `failure_category`, `notes`, `reviewer` and
`reviewed`; record uncertain causes explicitly rather than guessing model reasoning.

Categories: `keyword_distraction`, `incomplete_answer`, `entity_mismatch`,
`temporal_mismatch`, `subtle_relevance`, `domain_vocabulary`, `truncation`,
`judgment_disagreement`, or `other`. Choose one primary category and explain
secondary causes in notes. High-confidence error is a separate flag.

The submitted analysis contains fifty personally reviewed DL 2019 cases across
31 queries: 27 false positives and 23 false negatives. Notes and categories are
in `artifacts/final/manual_failure_review.csv`. Judgment disagreement records
the reviewer's interpretation, not an independently established gold-label error.
