"""Explicit text fields; no stringified dataset objects or silent dropped pairs."""


def select_text(record, fields):
    values = []
    for field in fields:
        if not hasattr(record, field):
            raise ValueError(f"Dataset record has no requested field {field!r}")
        value = getattr(record, field)
        if value is not None and str(value).strip():
            values.append(str(value).strip())
    if not values:
        raise ValueError(f"Empty text for fields {fields}")
    return "\n".join(values)


def default_fields(name):
    if name == "trec_covid":
        return ["title", "description"], ["title", "abstract"]
    return ["text"], ["text"]


DATASETS = {
    "trec_dl_2019": "msmarco-passage/trec-dl-2019/judged",
    "trec_dl_2020": "msmarco-passage/trec-dl-2020/judged",
    "trec_covid": "cord19/trec-covid",
    "msmarco_train": "msmarco-passage/train",
}
