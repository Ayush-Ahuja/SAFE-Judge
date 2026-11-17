"""Explicit task rubrics for the three frozen transfer datasets."""

from .prompts import SYSTEM, RUBRICS as ORIGINAL_RUBRICS, answer_token_ids

PROMPT_VERSION = "transfer-binary-rubric-v1"
TASKS = {"dl2020": 2, "trec_covid": 2, "chatreport": 1}
RUBRICS = {
    "dl2020": ORIGINAL_RUBRICS[2],
    "trec_covid": (
        "Use the TREC-COVID rubric: grade 0 (not relevant) and grade 1 (partially "
        "relevant) are No; grade 2 (fully relevant) is Yes. The document must "
        "provide information that fully addresses the query to receive Yes."
    ),
    "chatreport": (
        "Assess whether this corporate-report passage provides information relevant "
        "to the climate disclosure question. No means not relevant; Yes means "
        "partially or fully relevant. Judge the information supplied in the passage."
    ),
}
SHARED_RUBRIC = (
    "Assess whether the document provides substantive information that directly "
    "addresses the query. Answer Yes if it does, and No otherwise. "
    "Topical overlap alone is not sufficient for Yes."
)


def validate_task(row):
    name = row.get("rubric_name")
    if name not in TASKS or row["relevance_threshold"] != TASKS[name]:
        raise ValueError("Manifest requires a supported rubric_name and matching relevance_threshold")
    if row.get("rubric_mode", "dataset") not in ("dataset", "shared"):
        raise ValueError("rubric_mode must be dataset or shared")


def render_prompt(tokenizer, row, document):
    validate_task(row)
    rubric = SHARED_RUBRIC if row.get("rubric_mode", "dataset") == "shared" else RUBRICS[row["rubric_name"]]
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM},
         {"role": "user", "content": (
             f"{rubric}\n\n<query>\n{row['query_text']}\n</query>\n\n"
             f"<document>\n{document}\n</document>\n\nIs this document relevant? Answer Yes or No.")}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )


def encode_pair(tokenizer, row, max_length):
    """Same document-prefix truncation as the frozen DL2019 scorer."""
    if max_length < 1:
        raise ValueError("max_length must be positive")
    def encoded(document):
        return tokenizer.encode(render_prompt(tokenizer, row, document), add_special_tokens=False)
    full = encoded(row["doc_text"])
    if len(full) <= max_length:
        return full, {"input_tokens": len(full), "original_input_tokens": len(full),
                      "truncated": False, "truncated_document_text": ""}
    if len(encoded("")) >= max_length:
        raise ValueError(f"Query and rubric exceed token budget for {row['query_id']}; increase max_length")
    document_ids = tokenizer.encode(row["doc_text"], add_special_tokens=False)
    count = min(len(document_ids), max_length)
    while count > 0:
        document = tokenizer.decode(document_ids[:count], skip_special_tokens=True)
        result = encoded(document)
        if len(result) <= max_length:
            return result, {"input_tokens": len(result), "original_input_tokens": len(full),
                            "truncated": True, "truncated_document_text": document}
        count -= max(1, len(result) - max_length)
    raise ValueError("Token budget leaves no document content")
