"""Transfer prompts with the frozen Qwen loading and answer-token scoring recipe."""

from .model import QwenAssessor
from .transfer_prompts import answer_token_ids, encode_pair


class TransferAssessor(QwenAssessor):
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
