"""GPU-only, direct-answer Qwen scoring. Training is outside the Stage 2 pilot."""

from .prompts import answer_token_ids, encode_pair


def resolve_revision(model_name, revision):
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(model_name, revision=revision)
    commit = getattr(config, "_commit_hash", None)
    if not commit:
        raise ValueError("Use a Hugging Face model ID with a resolvable commit revision")
    if config.model_type != "qwen3_5":
        raise ValueError("This scorer currently supports Qwen3.5 checkpoints only")
    return commit


class QwenAssessor:
    def __init__(self, config, revision, seed):
        import torch
        from transformers import AutoModelForImageTextToText, AutoTokenizer, BitsAndBytesConfig
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU required. Use --validate-only or evaluate_pilot.py on CPU.")
        if torch.cuda.get_device_capability(0)[0] < 6:
            raise RuntimeError("NF4 requires CUDA compute capability >=6.0; Tesla K40 GPUs are unsupported.")
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        self.torch = torch
        self.max_length = config["max_length"]
        self.tokenizer = AutoTokenizer.from_pretrained(config["model_name"], revision=revision)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        requested_dtype = config.get("compute_dtype", "auto")
        dtype_name = ("bfloat16" if torch.cuda.is_bf16_supported() else "float16") if requested_dtype == "auto" else requested_dtype
        if dtype_name not in ("bfloat16", "float16"):
            raise ValueError("compute_dtype must be auto, bfloat16 or float16")
        if dtype_name == "bfloat16" and not torch.cuda.is_bf16_supported():
            raise ValueError("Requested bfloat16 is unsupported on this GPU")
        dtype = getattr(torch, dtype_name)
        self.compute_dtype = dtype_name
        if config.get("quantization") != "4bit":
            raise ValueError("The Stage 2 scorer expects quantization: 4bit")
        quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=dtype,
                                         bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
        self.model = AutoModelForImageTextToText.from_pretrained(
            config["model_name"], revision=revision, device_map={"": 0}, dtype=dtype,
            quantization_config=quantization, attn_implementation="sdpa",
        ).eval()

    def score_batch(self, rows):
        torch = self.torch
        encodings = [encode_pair(self.tokenizer, row, self.max_length) for row in rows]
        choices = [answer_token_ids(self.tokenizer, ids) for ids, _ in encodings]
        if any(ids != choices[0] for ids in choices):
            raise ValueError("Answer token IDs changed between prompts")
        inputs = self.tokenizer.pad({"input_ids": [ids for ids, _ in encodings]},
                                    padding=True, return_tensors="pt").to("cuda:0")
        with torch.inference_mode():
            # Only project the final hidden state; full sequence x vocabulary logits waste VRAM.
            outputs = self.model(**inputs, use_cache=False, logits_to_keep=1)
            logits = outputs.logits[:, -1, :].float()
            selected = logits[:, choices[0]]  # No, Yes
            if not torch.isfinite(logits).all():
                raise ValueError("Non-finite model logits; no predictions were checkpointed for this batch")
            probabilities = torch.softmax(selected, dim=-1)[:, 1].cpu().tolist()
            mass = torch.softmax(logits, dim=-1)[:, choices[0]].sum(dim=-1).cpu().tolist()
            top_tokens = logits.argmax(dim=-1).cpu().tolist()
            answer_logits = selected.cpu().tolist()
        output = []
        for row, (_, token_info), p, m, token, scores in zip(
                rows, encodings, probabilities, mass, top_tokens, answer_logits):
            output.append({"query_id": row["query_id"], "doc_id": row["doc_id"],
                           "true_relevance": row["true_relevance"], "true_binary": row["true_binary"],
                           "prob_relevant": p, "predicted_label": int(p >= 0.5),
                           "confidence": max(p, 1 - p), "no_logit": scores[0], "yes_logit": scores[1],
                           "compute_dtype": self.compute_dtype,
                           "answer_probability_mass": m, "top_token_id": token,
                           "top_token": self.tokenizer.decode([token]), **token_info})
        return output
