"""Whole-prompt tokenization preserves the original text baseline exactly."""
import json

import torch


def prompt_and_span(legacy, tokenizer, row):
    messages = legacy.messages(row, "B", False)
    user = messages[-1]["content"]
    evidence = json.dumps(legacy.evidence(row), ensure_ascii=False)
    encoded_top1 = json.dumps(row["input"]["asr_top1"], ensure_ascii=False)
    key = '{"asr_top1": '
    if not row["input"]["asr_top1"] or not evidence.startswith(key + encoded_top1):
        raise ValueError("Missing top1 or unexpected evidence serialization")
    suffix = "识别证据：" + evidence
    if not user.endswith(suffix):
        raise ValueError("Evidence is not the expected user suffix")
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False)
    if prompt.count(user) != 1:
        raise ValueError("Cannot uniquely locate user message in chat template")
    start = prompt.index(user) + len(user) - len(evidence) + len(key) + 1
    end = start + len(encoded_top1) - 2
    if prompt[start:end] != encoded_top1[1:-1]:
        raise AssertionError("Top1 span mismatch")
    return prompt, (start, end)


def encode_prompt(legacy, tokenizer, row):
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("Fast tokenizer with offset mapping is required")
    prompt, (start, end) = prompt_and_span(legacy, tokenizer, row)
    encoded = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
    offsets = encoded["offset_mapping"]
    selected = [i for i, (a, b) in enumerate(offsets) if b > a and a < end and b > start]
    if not selected or selected != list(range(selected[0], selected[-1] + 1)):
        raise ValueError("Empty or noncontiguous top1 token span")
    covered = set()
    for i in selected:
        a, b = offsets[i]
        covered.update(range(max(start, a), min(end, b)))
    if covered != set(range(start, end)):
        raise ValueError("Tokenizer offsets do not cover the serialized top1")
    return encoded["input_ids"], selected


def residual_batch(legacy, model, tokenizer, connector, row, frames=None, mode="true", training=False):
    if mode not in {"true", "zero", "mismatch", "off", "text"}:
        raise ValueError(mode)
    ids, selected = encode_prompt(legacy, tokenizer, row)
    embed = model.get_input_embeddings()
    device = embed.weight.device
    token_ids = torch.tensor(ids, device=device, dtype=torch.long)
    prefix = embed(token_ids)
    if mode not in {"off", "text"}:
        if frames is None:
            raise ValueError("Audio is required for an enabled branch")
        if mode == "zero":
            frames = torch.zeros_like(frames)
        index = torch.tensor(selected, device=device, dtype=torch.long)
        delta = connector(prefix.index_select(0, index), frames)
        prefix = prefix.index_add(0, index, delta)
    # Reference is only accessed below, after all connector inputs have been built.
    labels = None
    if training:
        target = json.dumps({"text": row["reference"]}, ensure_ascii=False) + tokenizer.eos_token
        target_ids = tokenizer(target, add_special_tokens=False)["input_ids"]
        target_ids = torch.tensor(target_ids, device=device, dtype=torch.long)
        labels = torch.cat([torch.full((len(ids),), -100, device=device, dtype=torch.long), target_ids])[None]
        prefix = torch.cat([prefix, embed(target_ids)])
    if len(prefix) > 4096:
        raise ValueError(f"Sequence exceeds 4096: {row['id']}")
    result = {"inputs_embeds": prefix[None],
              "attention_mask": torch.ones((1, len(prefix)), device=device, dtype=torch.long)}
    if labels is not None:
        result["labels"] = labels
    return result
