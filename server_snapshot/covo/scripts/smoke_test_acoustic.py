#!/usr/bin/env python
"""Smoke test for train_lora_sft_acoustic.py -- proves the four required properties.

  1. projector at zero init  =>  loss == text-only loss on the same batch (<=1e-6)
  2. after one backward the projector's final Linear receives non-zero gradients
  3. span-location failure rate, and acoustic_vecs are non-zero at the right characters
  4. a run with --acoustic-bin omitted is unchanged (bit-comparable loss against the
     ORIGINAL train_lora_sft.py on the same file/flags)

Run:  python smoke_test_acoustic.py --n-rows 50
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for _p in (str(HERE),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

R = Path("/root/autodl-tmp/.dsh_checks/rerank")
DEFAULT_MODEL = "/root/autodl-tmp/cbwhisper_covo_migration_20260609_tar_extracted/models/Qwen3.5-9B"
DEFAULT_SFT = R / "train_aishell_sft.jsonl"
DEFAULT_BIN = R / "ac_aishell_train.bin"
DEFAULT_INDEX = R / "ac_aishell_train.jsonl"

LOSS_RE = re.compile(r'\{"trainer_log": \{"loss": ([0-9.eE+-]+)')


def write_subset(src: Path, dst: Path, n: int) -> None:
    lines = []
    with src.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                lines.append(line)
            if len(lines) >= n:
                break
    dst.write_text("".join(lines), encoding="utf-8")


def cli_loss(stdout: str, log_path: Path) -> float:
    log_path.write_text(stdout, encoding="utf-8")
    hits = LOSS_RE.findall(stdout)
    if not hits:
        head = "\n".join(stdout.strip().splitlines()[-25:])
        raise SystemExit("no trainer_log loss in %s; tail:\n%s" % (log_path, head))
    return float(hits[-1])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--sft", default=str(DEFAULT_SFT))
    ap.add_argument("--ac-bin", default=str(DEFAULT_BIN))
    ap.add_argument("--ac-index", default=str(DEFAULT_INDEX))
    ap.add_argument("--n-rows", type=int, default=50)
    ap.add_argument("--work", default="/tmp/acoustic_smoke")
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--skip-cli", action="store_true", help="skip the three end-to-end CLI runs")
    a = ap.parse_args()

    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    subset = work / ("train_%d.jsonl" % a.n_rows)
    write_subset(Path(a.sft), subset, a.n_rows)
    print("== subset: %s (%d rows) ==" % (subset, a.n_rows), flush=True)

    import numpy as np
    import torch

    import train_lora_sft_acoustic as mod

    base_flags = [
        "--train-file", str(subset),
        "--model-name-or-path", a.model,
        "--max-length", str(a.max_length),
        "--max-steps", "1",
        "--per-device-train-batch-size", "2",
        "--gradient-accumulation-steps", "1",
        "--logging-steps", "1",
        "--save-steps", "1000",
        "--lora-r", "8",
        "--lora-dropout", "0.0",
        "--learning-rate", "1e-5",
        "--warmup-ratio", "0.0",
        "--bf16",
        "--gradient-checkpointing",
        "--disable-thinking",
        "--preprocessing-num-workers", "1",
    ]

    # ---------------- Part A: in-process, single GPU ------------------------
    print("\n== A. in-process zero-init / gradient / span checks ==", flush=True)
    argv = base_flags + [
        "--output-dir", str(work / "a"),
        "--acoustic-bin", a.ac_bin,
        "--acoustic-index", a.ac_index,
        "--acoustic-dim", "512",
        "--acoustic-dropout", "0.0",
    ]
    saved_argv, sys.argv = sys.argv, ["train_lora_sft_acoustic.py"] + argv
    try:
        args = mod.parse_args()
    finally:
        sys.argv = saved_argv
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    store = mod.AcousticVectorStore(args.acoustic_bin, args.acoustic_index, expect_dim=args.acoustic_dim)
    print("store:", json.dumps(store.summary(), ensure_ascii=False), flush=True)
    train_ds, _eval_ds, collator, _texts, _ = mod.build_data(args, tokenizer, store)

    span_audit = mod.audit_span_location(train_ds, store, anchor=args.acoustic_anchor)
    print("span audit:", json.dumps(span_audit, ensure_ascii=False), flush=True)

    model = mod.build_model(args, tokenizer, acoustic_enabled=True)
    assert model.proj.is_zero_initialised(), "final Linear is not zero-initialised"

    features = [train_ds[i] for i in range(min(2, len(train_ds)))]
    batch = collator(features)
    device = model.get_input_embeddings().weight.device
    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    text_batch = {k: v for k, v in batch.items() if not k.startswith("acoustic")}

    # (3) acoustic vectors really are non-zero at the hypothesis characters
    raw0 = store.vectors(0)
    raw_norms = np.linalg.norm(raw0, axis=1)
    av = batch["acoustic_vecs"]
    av_norms = av.float().norm(dim=-1)
    print(json.dumps({
        "acoustic_check": {
            "store_row0_top1": store.top1(0),
            "store_row0_char_norms_first8": [round(float(x), 3) for x in raw_norms[:8]],
            "store_row0_nonzero_chars": int((raw_norms > 0).sum()),
            "batch_acoustic_starts": batch["acoustic_starts"].tolist(),
            "batch_acoustic_lens": batch["acoustic_lens"].tolist(),
            "batch_acoustic_vecs_nonzero_rows": int((av_norms > 0).sum()),
            "batch_acoustic_vecs_norm_row0_first8": [round(float(x), 3) for x in av_norms[0][:8]],
            "projector_bias_norm_at_init": None,  # filled below
        }
    }, ensure_ascii=False), flush=True)

    model.train()
    with torch.no_grad():
        # embedding id at the first span token must decode back to a hypothesis char
        emb_ids = batch["input_ids"][0][int(batch["acoustic_starts"][0]): int(batch["acoustic_starts"][0]) + 4]
        print("tokens under the acoustic span:", tokenizer.decode(emb_ids), flush=True)
        bias0 = model.proj(batch["acoustic_vecs"].to(device=device,
                                                    dtype=model.get_input_embeddings().weight.dtype))
        print("projector output norm at init:", float(bias0.float().norm()), flush=True)

    out_text = model.base(**text_batch)
    out_ac = model(**batch)
    loss_text = float(out_text.loss.detach().float())
    loss_ac = float(out_ac.loss.detach().float())
    print(json.dumps({"zero_init_loss": {
        "text_only": loss_text, "with_acoustic": loss_ac, "abs_diff": abs(loss_text - loss_ac)}},
        ensure_ascii=False), flush=True)

    model.zero_grad(set_to_none=True)
    out_ac.loss.backward()
    grads = {n: (None if p.grad is None else float(p.grad.detach().float().norm()))
             for n, p in model.proj.named_parameters()}
    print(json.dumps({"projector_grad_norms_after_one_backward": grads}, ensure_ascii=False), flush=True)

    # ---------------- Part B: end-to-end CLI --------------------------------
    results = {
        "text_only_loss_A": loss_text,
        "acoustic_zero_init_loss_A": loss_ac,
        "zero_init_abs_diff_A": abs(loss_text - loss_ac),
        "fc2_weight_grad_norm_A": grads.get("fc2.weight"),
        "fc2_bias_grad_norm_A": grads.get("fc2.bias"),
        "span_audit_A": span_audit,
    }

    if not a.skip_cli:
        print("\n== B. end-to-end CLI: original / mine-without-acoustics / mine-with-acoustics ==", flush=True)
        common = base_flags
        runs = [
            ("orig_text_only", [sys.executable, str(HERE / "train_lora_sft.py")] + common
             + ["--output-dir", str(work / "orig")]),
            ("mine_text_only", [sys.executable, str(HERE / "train_lora_sft_acoustic.py")] + common
             + ["--output-dir", str(work / "mine_text")]),
            ("mine_acoustic", [sys.executable, str(HERE / "train_lora_sft_acoustic.py")] + common
             + ["--acoustic-bin", a.ac_bin, "--acoustic-index", a.ac_index, "--acoustic-dim", "512",
                "--acoustic-dropout", "0.0", "--output-dir", str(work / "mine_ac")]),
        ]
        losses = {}
        for name, cmd in runs:
            print("  running %s ..." % name, flush=True)
            proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(HERE))
            log = work / ("%s.log" % name)
            log.write_text(proc.stdout + "\n--- stderr ---\n" + proc.stderr, encoding="utf-8")
            if proc.returncode != 0:
                tail = "\n".join((proc.stdout + proc.stderr).strip().splitlines()[-30:])
                raise SystemExit("run %s failed (exit %d); tail:\n%s" % (name, proc.returncode, tail))
            losses[name] = cli_loss(proc.stdout, log)
            print("    %-16s loss=%.10f  (log %s)" % (name, losses[name], log), flush=True)
        results["cli_losses"] = losses
        results["cli_orig_vs_mine_text_only_abs_diff"] = abs(losses["orig_text_only"] - losses["mine_text_only"])
        results["cli_text_only_vs_acoustic_abs_diff"] = abs(losses["mine_text_only"] - losses["mine_acoustic"])
        ac_out = work / "mine_ac" / "adapter_model.safetensors"
        proj_out = work / "mine_ac" / "acoustic_projector.pt"
        results["saved_adapter"] = ac_out.exists()
        results["saved_projector"] = proj_out.exists()
        if proj_out.exists():
            blob = torch.load(proj_out, map_location="cpu", weights_only=False)
            results["projector_keys_saved"] = sorted(blob["projector"].keys())

    # ---------------- verdict ----------------------------------------------
    checks = {
        "1 zero-init loss == text-only (<=1e-6)": results["zero_init_abs_diff_A"] <= 1e-6,
        "2 projector final Linear got non-zero gradients":
            bool(results["fc2_weight_grad_norm_A"] and results["fc2_weight_grad_norm_A"] > 0
                 and results["fc2_bias_grad_norm_A"] and results["fc2_bias_grad_norm_A"] > 0),
        "3 span hard-fail rate <= 2%": results["span_audit_A"]["hard_fail_rate"] <= 0.02,
        "3b acoustic_vecs non-zero on the span": results["span_audit_A"]["chars_dropped_no_token"] == 0,
    }
    if not a.skip_cli:
        checks["4 no-acoustic CLI run == original CLI run"] = results["cli_orig_vs_mine_text_only_abs_diff"] == 0.0
        checks["4b acoustic CLI run == text-only CLI run"] = results["cli_text_only_vs_acoustic_abs_diff"] == 0.0
        checks["5 checkpoint saved adapter + projector"] = bool(results["saved_adapter"] and results["saved_projector"])

    print("\n===== SMOKE TEST SUMMARY =====")
    print(json.dumps({"results": results, "checks": checks}, ensure_ascii=False, indent=2))
    print("\n===== VERDICT: %s =====" % ("PASS" if all(checks.values()) else "FAIL"))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
