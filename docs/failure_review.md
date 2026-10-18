# Manual failure analysis protocol

Use scripts/export_failures.py after complete inference. It refuses to overwrite an
existing review CSV and selects 50 unique disagreements by default. Its round-robin
query/error-type/confidence sampling is diagnostic, not a prevalence estimate.

Read the query, original passage and model_document_text. The last field records the
actual document prefix for truncated inputs. Keep official human labels unchanged;
record debatable gold labels in notes rather than silently relabelling them.

Choose one primary failure_category; record secondary causes in notes:

- keyword_distraction: lexical overlap without the requested information;
- incomplete_answer: partial or related information below the chosen relevance cutoff;
- entity_mismatch: wrong person, object, disease, location or relation;
- temporal_mismatch: wrong date, period or outdated information;
- subtle_relevance: a relevant answer requires an inference the assessor missed;
- domain_vocabulary: domain terminology appears to explain the disagreement;
- truncation: answer-bearing text was outside the model's retained prefix;
- judgment_disagreement: the official grade is debatable on reading the passage;
- other: explain the cause explicitly.

High-confidence wrong is already a separate flag, not a mutually exclusive cause.
If the available evidence cannot establish the cause, say so in notes rather than
asserting the model's internal reasoning. For each example fill failure_category,
notes, reviewer, and reviewed (for example, yes). Do not use an LLM-generated category
as a substitute for the required personal reading.

The report should give the number personally reviewed, false-positive/false-negative
counts, queries represented, category counts and several illustrative examples.
Category percentages describe this selected sample only. Original sampling metadata
records zero reviews at export time. The report builder validates a completed external
sheet and writes its current review counts into metadata beside that sheet.
