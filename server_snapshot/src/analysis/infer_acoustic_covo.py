#!/usr/bin/env python
"""Generation with the acoustic-conditioned COVO backend (AC-COVO).

Mirrors covo/scripts/infer_lora_text.py but adds the acoustic channel.  The span
of the ASR top-1 hypothesis is located by the SAME shared function the trainer
uses (`build_acoustic_inputs`), so training and inference cannot disagree about
which character gets which acoustic vector.

Rows are paired to acoustics explicitly: the input file must carry `ac_row` (the
linked files do).  When it does not, the row's own `input.asr_top1` is looked up
in the acoustic index by exact string match, and any row that cannot be paired is
generated WITHOUT acoustics and counted -- never silently paired with another
utterance's audio.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
for p in (str(REPO / "src"), str(REPO / "covo" / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from acoustic_conditioning import (  # noqa: E402
    AcousticConditionedLM, AcousticProjector, AcousticStore,
    build_acoustic_inputs, collate_acoustic,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--model-name-or-path", required=True)
    p.add_argument("--adapter-path", required=True)
    p.add_argument("--projector-path", required=True,
                   help="acoustic_projector.pt saved next to the adapter")
    p.add_argument("--acoustic-bin", required=True)
    p.add_argument("--acoustic-index", required=True)
    p.add_argument("--acoustic-dim", type=int, default=512)
    p.add_argument("--acoustic-layers", type=int, default=2)
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--max-new-tokens", type=int, default=96)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--progress-every", type=int, default=200)
    p.add_argument("--disable-thinking", action="store_true")
    p.add_argument("--constrained-sets", default="",
                   help="optional *_allowed_sets.jsonl; enables SEAL-style constrained "
                        "decoding, restricting each output position to the hypothesis "
                        "character plus its per-frame acoustic alternatives")
    p.add_argument("--constrained-topk", type=int, default=5)
    return p.parse_args()


def extract_json_object(text: str) -> str:
    """Extract the first balanced {...}, ignoring braces INSIDE string literals.

    A naive depth counter treats a '}' inside the corrected sentence as the end of
    the object, which makes json.loads fail and silently falls back to the raw
    model output -- costing CER on exactly the rows whose text contains a brace.
    """
    start = text.find("{")
    if start < 0:
        return text.strip()
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        c = text[i]
        if esc:
            esc = False
            continue
        if c == "\\":
            esc = True
            continue
        if c == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return text[start:].strip()


def parse_prediction(model_output: str, fallback: str = "") -> str:
    """Pull the corrected sentence out of the model's {"text": ...} reply.

    This matters more than it looks: `restore_eval.py` compares `prediction`
    against the reference character by character.  Writing the raw JSON here
    instead of the sentence inside it would inflate CER by the length of the JSON
    wrapper on every row and make the whole evaluation meaningless.
    """
    obj = extract_json_object(model_output)
    try:
        d = json.loads(obj)
        if isinstance(d, dict) and isinstance(d.get("text"), str) and d["text"].strip():
            return d["text"].strip()
    except Exception:
        pass
    return (fallback or model_output).strip()


def main() -> int:
    args = parse_args()
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    store = AcousticStore(args.acoustic_bin, args.acoustic_index)
    rows = [json.loads(l) for l in Path(args.input).read_text(encoding="utf-8").splitlines()
            if l.strip()]
    if args.limit:
        rows = rows[: args.limit]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, trust_remote_code=True, torch_dtype=torch.bfloat16,
        device_map="auto")
    base = PeftModel.from_pretrained(base, args.adapter_path)

    projector = AcousticProjector(args.acoustic_dim, int(base.config.hidden_size),
                                  args.acoustic_layers)
    sd = torch.load(args.projector_path, map_location="cpu")
    projector.load_state_dict(sd)
    projector.to(base.device).eval()
    print(json.dumps({"projector_loaded": str(args.projector_path),
                      "projector_params": sum(p.numel() for p in projector.parameters()),
                      "rows": len(rows)}), flush=True)

    wrapper = AcousticConditionedLM(base, projector).eval()

    # Pairing by hypothesis string is AMBIGUOUS -- many utterances share the same
    # top-1 text -- so it is only accepted when that string is unique in the store.
    # An ambiguous row runs WITHOUT acoustics rather than silently receiving some
    # other utterance's, which would look like the acoustic channel misbehaving.
    top1_count, top1_first = {}, {}
    for i, ar in enumerate(store.rows):
        t = ar.get("top1") or ""
        if t:
            top1_count[t] = top1_count.get(t, 0) + 1
            top1_first.setdefault(t, i)

    unresolved = 0
    by_unique_string = 0
    prepared = []
    for r in rows:
        ac_row = r.get("ac_row")
        if ac_row is None:
            top1 = str(((r.get("input") or {}).get("asr_top1")) or "")
            if top1 and top1_count.get(top1) == 1:
                ac_row = top1_first[top1]
                by_unique_string += 1
        if ac_row is None or int(ac_row) < 0 or int(ac_row) >= len(store):
            unresolved += 1
            ac_row = 0
            has = False
        else:
            has = True
        prepared.append((r, int(ac_row), has))
    print(json.dumps({"rows_without_acoustics": unresolved,
                      "rows_paired_by_unique_hypothesis": by_unique_string,
                      "rows_with_acoustics": len(prepared) - unresolved}), flush=True)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_empty = 0
    n_done = 0
    n_jsonlike = 0

    sets_by_id = None
    char_to_ids = None
    extra_ids = []
    if args.constrained_sets:
        from constrained_decoding import build_char_to_ids, load_allowed_sets
        sets_by_id = load_allowed_sets(args.constrained_sets, args.constrained_topk)
        all_chars = [c for s in sets_by_id.values() for st in s for c in st]
        char_to_ids = build_char_to_ids(tokenizer, all_chars)
        for ch in ('"', "}", "\n", " "):
            extra_ids.extend(tokenizer.encode(ch, add_special_tokens=False))
        # Only characters that encode to EXACTLY one token may appear in a masked
        # position; see the note in constrained_decoding.py.
        safe_chars = [c for c in set(all_chars)
                      if len(tokenizer.encode(c, add_special_tokens=False)) == 1]
        n_pos = sum(len(st) for s in sets_by_id.values() for st in s)
        n_safe = sum(1 for s in sets_by_id.values() for st in s
                     if all(c in set(safe_chars) for c in st))
        print(json.dumps({"constrained_sets": args.constrained_sets,
                          "rows_with_sets": len(sets_by_id),
                          "distinct_chars": len(char_to_ids),
                          "single_token_chars": len(safe_chars),
                          "positions": n_pos, "positions_constrainable": n_safe,
                          "positions_released_for_multitoken": n_pos - n_safe,
                          "extra_allowed_ids": sorted(set(extra_ids))}), flush=True)

    with out_path.open("w", encoding="utf-8") as w, torch.no_grad():
        for b0 in range(0, len(prepared), args.batch_size):
            chunk = prepared[b0:b0 + args.batch_size]
            items = []
            for r, ac_row, has in chunk:
                user = [m for m in (r.get("messages") or []) if m.get("role") != "assistant"]
                d = build_acoustic_inputs(
                    tokenizer, user, ac_row, store, max_length=args.max_length,
                    disable_thinking=args.disable_thinking, train=False)
                if not has:
                    d["acoustic"] = torch.zeros(0, store.dim)
                    d["acoustic_start"], d["acoustic_len"] = 0, 0
                items.append(d)
            batch = collate_acoustic(items, tokenizer.pad_token_id, pad_side="left")
            batch = {k: v.to(base.device) for k, v in batch.items()}
            gen_kwargs = {}
            if sets_by_id is not None:
                from constrained_decoding import AcousticConstrainedLogitsProcessor
                row_sets = [sets_by_id.get(str(r.get("uttid") or r.get("id")), [])
                            for r, _, _ in chunk]
                if any(row_sets):
                    gen_kwargs["logits_processor"] = [AcousticConstrainedLogitsProcessor(
                        row_sets, char_to_ids, tokenizer,
                        eos_token_id=tokenizer.eos_token_id,
                        extra_allowed_ids=extra_ids, safe_chars=safe_chars)]
            gen = wrapper.generate(
                input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                acoustic=batch["acoustic"], acoustic_start=batch["acoustic_start"],
                acoustic_len=batch["acoustic_len"], max_new_tokens=args.max_new_tokens,
                do_sample=False, pad_token_id=tokenizer.pad_token_id, **gen_kwargs)
            # With `inputs_embeds`, HF's generate returns ONLY the newly generated
            # tokens (the prompt has no token ids to prepend).  Slicing off
            # prompt_len unconditionally would therefore discard the entire output.
            # Discriminate on length instead of assuming.
            plen = batch["input_ids"].size(1)
            if gen.size(1) > args.max_new_tokens + 4:
                new_tokens = gen[:, plen:]
            else:
                new_tokens = gen
            texts = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
            for (r, ac_row, has), text in zip(chunk, texts):
                fallback = str(((r.get("input") or {}).get("asr_top1")) or "")
                pred = parse_prediction(text, fallback)
                if not pred:
                    n_empty += 1
                # A parser failure that leaves the JSON wrapper as the prediction is
                # NOT empty, so it would sail past the empty-rate guard below and show
                # up only as a mysteriously bad CER. Count it explicitly.
                if pred.lstrip().startswith("{"):
                    n_jsonlike += 1
                rec = dict(r)
                rec["model_output"] = text
                rec["raw_prediction"] = text
                rec["prediction"] = pred
                rec["acoustic_used"] = bool(has)
                w.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_done += 1
            if args.progress_every and (b0 + len(chunk)) % args.progress_every < args.batch_size:
                print(json.dumps({"done": n_done, "empty_predictions": n_empty,
                                  "json_like_predictions": n_jsonlike}), flush=True)
    empty_rate = n_empty / max(n_done, 1)
    json_rate = n_jsonlike / max(n_done, 1)
    print(json.dumps({"output": str(out_path), "rows": n_done,
                      "empty_predictions": n_empty, "empty_rate": round(empty_rate, 4),
                      "json_like_predictions": n_jsonlike,
                      "json_like_rate": round(json_rate, 4)}), flush=True)
    if empty_rate > 0.5:
        print("WARNING: more than half of the rows produced an EMPTY prediction; "
              "the generation path is probably broken (check the inputs_embeds "
              "return-length assumption).", flush=True)
        return 2
    if json_rate > 0.1:
        print("WARNING: more than 10%% of predictions still look like the raw JSON "
              "wrapper, so parse_prediction is failing on them; CER will be badly "
              "inflated. Inspect --output rows whose prediction starts with '{'.",
              flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
