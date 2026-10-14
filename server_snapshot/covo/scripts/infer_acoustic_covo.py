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
    return p.parse_args()


def extract_json_object(text: str) -> str:
    start = text.find("{")
    if start < 0:
        return text.strip()
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return text[start:].strip()


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

    # resolve ac_row for every row, explicitly
    unresolved = 0
    prepared = []
    for r in rows:
        ac_row = r.get("ac_row")
        if ac_row is None:
            top1 = str(((r.get("input") or {}).get("asr_top1")) or "")
            ac_row = -1
            for i, ar in enumerate(store.rows):
                if ar.get("top1") == top1 and top1:
                    ac_row = i
                    break
        if ac_row is None or int(ac_row) < 0 or int(ac_row) >= len(store):
            unresolved += 1
            ac_row = 0
            has = False
        else:
            has = True
        prepared.append((r, int(ac_row), has))
    print(json.dumps({"rows_without_acoustics": unresolved,
                      "rows_with_acoustics": len(prepared) - unresolved}), flush=True)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
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
            batch = collate_acoustic(items, tokenizer.pad_token_id)
            batch = {k: v.to(base.device) for k, v in batch.items()}
            gen = wrapper.generate(
                input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                acoustic=batch["acoustic"], acoustic_start=batch["acoustic_start"],
                acoustic_len=batch["acoustic_len"], max_new_tokens=args.max_new_tokens,
                do_sample=False, pad_token_id=tokenizer.pad_token_id)
            texts = tokenizer.batch_decode(gen[:, batch["input_ids"].size(1):],
                                           skip_special_tokens=True)
            for (r, ac_row, has), text in zip(chunk, texts):
                pred = extract_json_object(text)
                rec = dict(r)
                rec["prediction_raw"] = text
                rec["prediction"] = pred
                rec["acoustic_used"] = bool(has)
                w.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if args.progress_every and (b0 + len(chunk)) % args.progress_every < args.batch_size:
                print(json.dumps({"done": b0 + len(chunk)}), flush=True)
    print(json.dumps({"output": str(out_path)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
