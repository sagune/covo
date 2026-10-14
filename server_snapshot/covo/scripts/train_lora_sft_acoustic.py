#!/usr/bin/env python
"""LoRA/QLoRA SFT with an additive per-character ACOUSTIC conditioning channel.

This is `train_lora_sft.py` plus exactly ONE new thing: a projector that turns the
frozen ASR encoder state of each character of the ASR top-1 hypothesis into an
additive bias on that character's input embedding.

    text prompt (same template as the text-only trainer)
        +  AcousticProjector: LayerNorm(512) -> Linear(512, d) -> GELU -> Linear(d, d)
        -> bias added to emb[token(char j)] for every character j of the top-1 span

The projector's last Linear is ZERO-INITIALISED, so at step 0 the model computes
exactly the text-only model; training therefore starts at the baseline and the
acoustic channel can only add value (or gradients).

With no `--acoustic-bin` this script is a drop-in replacement for the text-only
trainer: same data loading, same tokenizer, same collator, same Trainer wiring
(all reused from `train_lora_sft`).

Acoustic data format (see src/analysis/export_acoustic_vectors.py):
  <--acoustic-bin>    concatenated little-endian float16 arrays, each (n, 512)
  <--acoustic-index>  one json object per row:
                      {"id", "offset", "n", "dim", "top1", "reference", "n_no_frame"}
  Row i of the index corresponds POSITIONALLY to row i of the training file, and
  `index[i]["top1"]` is the exact string rendered into that row's user message after
  the "ASR top-1: " anchor.  Entry j of the (n, 512) array is the frozen encoder
  state at the frame where the CTC head emitted character j of `top1` (zero when the
  character has no acoustic frame, e.g. punctuation).

The script refuses to train when it cannot locate the hypothesis span in more than
`--acoustic-max-span-fail` (default 2%) of rows, because that means the template
assumption is wrong and the acoustics would be attached to the wrong tokens.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
_SCRIPT_DIR = Path(__file__).resolve().parent
for _p in (str(SRC_DIR), str(_SCRIPT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Reuse, do not rewrite: data loading, chat templating, arg parsing, metadata.
import train_lora_sft as base_trainer  # noqa: E402

ACOUSTIC_ANCHOR = "ASR top-1: "

# Rendering differences the bridge is known to introduce inside the hypothesis line:
# full-width punctuation replaced by its ASCII twin (，->, ？->?).  These are 1:1
# substitutions, so character j of the prompt line is still character j of the
# acoustic row and the alignment stays valid.  Anything else is a hard failure.
_PUNCT_LIKE = set(
    "，。,、；;：:？！?!…．.~～·-—–_'\"'“”‘’()（）《》〈〉「」『』【】[]{}<>/\\|"
)


def _punct_like(ch: str) -> bool:
    return ch in _PUNCT_LIKE


# --------------------------------------------------------------------------- #
# 1. Acoustic projector
# --------------------------------------------------------------------------- #
class AcousticProjector(nn.Module):
    """LayerNorm(d) -> Linear(d, d_model) -> GELU -> Linear(d_model, d_model).

    The final Linear is zero-initialised (weight AND bias), so `forward` returns an
    exact zero tensor at initialisation and the conditioned model is bit-identical
    to the text-only model before the first update.
    """

    def __init__(self, in_dim: int, d_model: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.d_model = int(d_model)
        self.norm = nn.LayerNorm(int(in_dim))
        self.fc1 = nn.Linear(int(in_dim), int(d_model))
        self.act = nn.GELU()
        self.fc2 = nn.Linear(int(d_model), int(d_model))
        self.dropout = nn.Dropout(float(dropout))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, acoustic_vecs: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(self.act(self.fc1(self.norm(acoustic_vecs)))))

    def is_zero_initialised(self) -> bool:
        return bool(
            torch.count_nonzero(self.fc2.weight) == 0 and torch.count_nonzero(self.fc2.bias) == 0
        )


# --------------------------------------------------------------------------- #
# 2. Injection wrapper
# --------------------------------------------------------------------------- #
class AcousticConditionedLM(nn.Module):
    """Base causal LM + an additive acoustic bias on the hypothesis token span.

    `forward(input_ids, acoustic_vecs, acoustic_starts, acoustic_lens, **kw)` builds
    the input embeddings itself, adds `proj(acoustic_vecs)[b, :n_b]` into
    `emb[b, start_b : start_b + n_b]`, and calls the base model with `inputs_embeds`
    and every remaining kwarg (labels, attention_mask, ...) forwarded unchanged.

    `labels` are NOT shifted here.  Qwen3_5ForCausalLM.forward passes inputs_embeds
    straight to the text model and computes
    `loss_function(logits, labels, ...)`, which does the -1/+1 shift itself, so labels
    aligned with `input_ids` remain correct when `inputs_embeds` has the same length.
    (Verified against .../transformers/models/qwen3_5/modeling_qwen3_5.py:1827-1844.)
    """

    _BASE_PREFIX = "base."
    _PROJ_PREFIX = "proj."

    def __init__(self, base: nn.Module, projector: AcousticProjector, anchor: str = ACOUSTIC_ANCHOR) -> None:
        super().__init__()
        self.base = base
        self.proj = projector
        self.anchor = anchor
        self.is_acoustic_conditioned = True
        self.last_acoustic_stats: Dict[str, float] = {}

    # -- keep the wrapped model transparent for Trainer / peft / accelerate -----
    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            modules = object.__getattribute__(self, "__dict__").get("_modules", {})
            base = modules.get("base")
            if base is None:
                raise
            return getattr(base, name)

    def get_input_embeddings(self) -> nn.Module:
        return self.base.get_input_embeddings()

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.base.set_input_embeddings(value)

    # -- the actual conditioning ----------------------------------------------
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        acoustic_vecs: Optional[torch.Tensor] = None,
        acoustic_starts: Optional[torch.Tensor] = None,
        acoustic_lens: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ):
        emb = self.get_input_embeddings()(input_ids)
        if acoustic_vecs is not None and acoustic_vecs.numel() > 0:
            vecs = acoustic_vecs.to(device=emb.device)
            b, lmax, _ = vecs.shape
            if lmax > 0:
                bias = self.proj(vecs.to(dtype=emb.dtype))  # (B, L, d)
                starts = acoustic_starts.to(device=emb.device, dtype=torch.long).reshape(b, 1)
                lens = acoustic_lens.to(device=emb.device, dtype=torch.long).reshape(b, 1)
                t = emb.size(1)
                pos = torch.arange(t, device=emb.device, dtype=torch.long).reshape(1, t) - starts
                valid = (pos >= 0) & (pos < lens)
                idx = pos.clamp_(0, lmax - 1).unsqueeze(-1).expand(b, t, bias.size(-1))
                gathered = torch.gather(bias, 1, idx)
                emb = emb + gathered * valid.unsqueeze(-1).to(emb.dtype)
                with torch.no_grad():
                    self.last_acoustic_stats = {
                        "rows_with_acoustics": int((lens > 0).sum().item()),
                        "mean_span_len": float(lens.float().mean().item()),
                        "bias_norm": float(bias.detach().float().norm().item()),
                    }
        return self.base(inputs_embeds=emb, **kwargs)

    # -- saving ---------------------------------------------------------------
    def projector_state_dict(self) -> Dict[str, torch.Tensor]:
        return {k: v.detach().cpu() for k, v in self.proj.state_dict().items()}

    def save_pretrained(self, output_dir: str, state_dict: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        base_sd = None
        proj_sd = self.projector_state_dict()
        if state_dict is not None:
            base_sd = {
                k[len(self._BASE_PREFIX):]: v
                for k, v in state_dict.items()
                if k.startswith(self._BASE_PREFIX)
            }
            proj_sd = {
                k[len(self._PROJ_PREFIX):]: v
                for k, v in state_dict.items()
                if k.startswith(self._PROJ_PREFIX)
            }
        self.base.save_pretrained(str(out), state_dict=base_sd, **kwargs)
        torch.save(
            {
                "projector": proj_sd,
                "config": {
                    "in_dim": self.proj.in_dim,
                    "d_model": self.proj.d_model,
                    "architecture": "LayerNorm->Linear->GELU->Linear(zero-init)",
                    "anchor": self.anchor,
                },
            },
            out / "acoustic_projector.pt",
        )


# --------------------------------------------------------------------------- #
# 3. Acoustic vector store
# --------------------------------------------------------------------------- #
class AcousticVectorStore:
    """Memmap reader for the concatenated fp16 (n, dim) arrays plus their jsonl index."""

    def __init__(self, bin_path: str, index_path: str, expect_dim: int = 512) -> None:
        self.bin_path = Path(bin_path)
        self.index_path = Path(index_path)
        for p in (self.bin_path, self.index_path):
            if not p.is_file():
                raise SystemExit("acoustic file missing: %s" % p)
        self.rows: List[Dict[str, Any]] = []
        with self.index_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    self.rows.append(json.loads(line))
        if not self.rows:
            raise SystemExit("acoustic index is empty: %s" % self.index_path)

        self.size_bytes = self.bin_path.stat().st_size
        bad_n = bad_range = bad_dim = 0
        dims = set()
        for i, r in enumerate(self.rows):
            n = int(r.get("n", -1))
            d = int(r.get("dim", -1))
            off = int(r.get("offset", -1))
            dims.add(d)
            if n != len(str(r.get("top1") or "")):
                bad_n += 1
            if off < 0 or off % 2 != 0 or off + n * d * 2 > self.size_bytes:
                bad_range += 1
        if len(dims) != 1:
            raise SystemExit("acoustic index has mixed dims: %s" % sorted(dims))
        self.dim = int(next(iter(dims)))
        if int(expect_dim) != self.dim:
            raise SystemExit(
                "--acoustic-dim %d does not match the index dim %d" % (int(expect_dim), self.dim)
            )
        if bad_n or bad_range:
            raise SystemExit(
                "acoustic index is inconsistent: n!=len(top1) on %d rows, offset out of range on %d rows"
                % (bad_n, bad_range)
            )
        self.bad_dim = bad_dim
        self.arr = np.memmap(self.bin_path, dtype=np.float16, mode="r")
        self._n_no_frame = sum(int(r.get("n_no_frame", 0)) for r in self.rows)

    def __len__(self) -> int:
        return len(self.rows)

    def top1(self, i: int) -> str:
        return str(self.rows[i].get("top1") or "")

    def vectors(self, i: int) -> np.ndarray:
        r = self.rows[i]
        n = int(r["n"])
        off = int(r["offset"]) // 2
        flat = self.arr[off: off + n * self.dim]
        return np.asarray(flat, dtype=np.float32).reshape(n, self.dim)

    def summary(self) -> Dict[str, Any]:
        first = self.rows[0]
        return {
            "bin": str(self.bin_path),
            "index": str(self.index_path),
            "rows": len(self.rows),
            "dim": self.dim,
            "bin_bytes": self.size_bytes,
            "index_rows_with_missing_frames": int(sum(1 for r in self.rows if int(r.get("n_no_frame", 0)) > 0)),
            "chars_without_frame": self._n_no_frame,
            "first_id": first.get("id"),
        }


# --------------------------------------------------------------------------- #
# 4. Span location: character offsets -> token indices
# --------------------------------------------------------------------------- #
def locate_hypothesis_span(
    text: str, top1: str, anchor: str = ACOUSTIC_ANCHOR
) -> Tuple[Optional[int], str]:
    """Return (start_char_index_of_top1_char_0, status).

    The anchor ("ASR top-1: ") is what makes this unambiguous: the hypothesis string
    also appears 1-3 more times inside the N-best list of the rendered prompt, so a
    bare `str.find` is only accidentally right.  We require the anchor to be unique,
    then require the hypothesis to start right after it.
    """
    if not top1:
        return None, "empty_top1"
    n_anchor = text.count(anchor)
    if n_anchor == 0:
        pos = text.find(top1)
        return (pos, "find_fallback") if pos >= 0 else (None, "not_found")
    if n_anchor > 1:
        return None, "ambiguous_anchor"
    p = text.find(anchor) + len(anchor)
    if text.startswith(top1, p):
        return p, "exact"
    cand = text[p: p + len(top1)]
    if len(cand) == len(top1) and cand:
        diffs = [j for j in range(len(top1)) if cand[j] != top1[j]]
        if diffs and all(_punct_like(top1[j]) and _punct_like(cand[j]) for j in diffs):
            # 1:1 punctuation re-rendering: character j still lines up with row j.
            return p, "punct_substituted"
    pos = text.find(top1)
    return (pos, "find_fallback") if pos >= 0 else (None, "not_found")


def chars_to_tokens(offsets: Sequence[Sequence[int]], start: int, n_chars: int) -> List[int]:
    """Map each character index j in [0, n_chars) to the token covering text[start+j].

    Characters whose text position is covered by no token get -1 (dropped).
    `offsets` must be monotonically increasing, as tokenizer offset mappings are.
    """
    out = [-1] * n_chars
    n_tok = len(offsets)
    t = 0
    for j in range(n_chars):
        c = start + j
        while t < n_tok and offsets[t][1] <= c:
            t += 1
        if t < n_tok and offsets[t][0] <= c < offsets[t][1]:
            out[j] = t
    return out


# --------------------------------------------------------------------------- #
# 5. Collator
# --------------------------------------------------------------------------- #
class AcousticCollator:
    """DataCollatorForLanguageModeling + padded acoustic vectors.

    Delegates all padding / label construction to the same
    `DataCollatorForLanguageModeling(mlm=False)` the text-only trainer uses, so the
    text half of the batch is bit-identical; it then adds

        acoustic_vecs   (B, Lmax, dim) float32   per-token bias, position 0 = start_b
        acoustic_starts (B,)          int64     first token index of the span
        acoustic_lens   (B,)          int64     number of tokens in the span

    Batches whose rows carry no `ac_row` (e.g. an eval file the acoustic index does
    not cover) fall through unchanged.
    """

    _BASE_KEYS = ("input_ids", "attention_mask", "token_type_ids", "special_tokens_mask")

    def __init__(
        self,
        tokenizer: Any,
        base_collator: Any,
        store: AcousticVectorStore,
        anchor: str = ACOUSTIC_ANCHOR,
        max_length: int = 2048,
    ) -> None:
        self.tokenizer = tokenizer
        self.base_collator = base_collator
        self.store = store
        self.anchor = anchor
        self.max_length = int(max_length)
        self.stats: Counter = Counter()

    @property
    def dim(self) -> int:
        return self.store.dim

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        clean = [{k: f[k] for k in self._BASE_KEYS if k in f} for f in features]
        batch = self.base_collator(clean)
        if not features or "ac_row" not in features[0]:
            return batch

        per_row: List[Tuple[int, np.ndarray]] = []
        for f in features:
            i = int(f["ac_row"])
            top1 = self.store.top1(i)
            start, status = locate_hypothesis_span(f.get("text", ""), top1, self.anchor)
            self.stats["row_" + status] += 1
            self.stats["rows"] += 1
            m = np.zeros((0, self.dim), dtype=np.float32)
            s = 0
            if start is not None and top1:
                c2t = chars_to_tokens(f.get("offset_mapping") or [], start, len(top1))
                hits = [t for t in c2t if t >= 0]
                dropped = len(top1) - len(hits)
                self.stats["chars"] += len(top1)
                self.stats["chars_dropped"] += dropped
                if dropped:
                    self.stats["rows_with_dropped_chars"] += 1
                if self.stats["rows"] > 0 and dropped == len(top1):
                    self.stats["rows_all_chars_dropped"] += 1
                if hits:
                    s, e = min(hits), max(hits)
                    vecs = self.store.vectors(i)  # (n, dim)
                    acc = np.zeros((e - s + 1, self.dim), dtype=np.float32)
                    cnt = np.zeros((e - s + 1,), dtype=np.float32)
                    for j, t in enumerate(c2t):
                        if t < 0:
                            continue
                        acc[t - s] += vecs[j]
                        cnt[t - s] += 1.0
                    cnt[cnt == 0.0] = 1.0
                    m = acc / cnt[:, None]
                    self.stats["tokens_spanned"] += int(e - s + 1)
            per_row.append((s, m))

        b = len(per_row)
        lmax = max((m.shape[0] for _, m in per_row), default=0)
        vecs = torch.zeros(b, max(lmax, 1), self.dim, dtype=torch.float32)
        starts = torch.zeros(b, dtype=torch.long)
        lens = torch.zeros(b, dtype=torch.long)
        for k, (s, m) in enumerate(per_row):
            if m.shape[0] > 0:
                vecs[k, : m.shape[0]] = torch.from_numpy(m)
                starts[k] = int(s)
                lens[k] = int(m.shape[0])
        batch["acoustic_vecs"] = vecs
        batch["acoustic_starts"] = starts
        batch["acoustic_lens"] = lens
        return batch

    def report(self) -> Dict[str, Any]:
        return dict(self.stats)


# --------------------------------------------------------------------------- #
# 6. Startup audit
# --------------------------------------------------------------------------- #
def audit_span_location(
    dataset: Any,
    store: AcousticVectorStore,
    anchor: str = ACOUSTIC_ANCHOR,
    limit: int = 0,
) -> Dict[str, Any]:
    """Locate every row's hypothesis span and aggregate the failure counts."""
    n = len(dataset) if not limit else min(int(limit), len(dataset))
    stats: Counter = Counter()
    examples: List[Dict[str, Any]] = []
    for i in range(n):
        row = dataset[i]
        top1 = store.top1(int(row["ac_row"]))
        text = row.get("text", "")
        start, status = locate_hypothesis_span(text, top1, anchor)
        stats["rows"] += 1
        stats[status] += 1
        if top1 and text.count(top1) > 1:
            stats["multiply_present_in_prompt"] += 1
        if start is not None and top1:
            c2t = chars_to_tokens(row.get("offset_mapping") or [], start, len(top1))
            hits = [t for t in c2t if t >= 0]
            dropped = len(top1) - len(hits)
            stats["chars"] += len(top1)
            stats["chars_dropped"] += dropped
            if dropped:
                stats["rows_with_dropped_chars"] += 1
            if hits:
                stats["tokens_spanned"] += max(hits) - min(hits) + 1
        elif len(examples) < 5:
            examples.append({"row": i, "ac_row": int(row["ac_row"]), "status": status, "top1": top1[:60]})

    hard = sum(stats[k] for k in ("not_found", "no_anchor", "ambiguous_anchor", "empty_top1"))
    soft = stats["find_fallback"] + stats["punct_substituted"]
    rows = max(stats["rows"], 1)
    out = {
        "rows_audited": stats["rows"],
        "located_exact": stats["exact"],
        "located_punct_substituted": stats["punct_substituted"],
        "located_find_fallback": stats["find_fallback"],
        "hard_fail": hard,
        "hard_fail_rate": hard / rows,
        "not_found": stats["not_found"],
        "ambiguous_anchor": stats["ambiguous_anchor"],
        "empty_top1": stats["empty_top1"],
        "degraded_or_failed_rate": (hard + soft) / rows,
        "top1_present_more_than_once_in_prompt": stats["multiply_present_in_prompt"],
        "hypothesis_chars": stats["chars"],
        "chars_dropped_no_token": stats["chars_dropped"],
        "rows_with_dropped_chars": stats["rows_with_dropped_chars"],
        "tokens_spanned": stats["tokens_spanned"],
        "examples": examples,
    }
    return out


# --------------------------------------------------------------------------- #
# 7. Args
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    """Base flags parsed by train_lora_sft (reused); the acoustic flags are ours."""
    extra = argparse.ArgumentParser(add_help=False, description="acoustic conditioning flags")
    extra.add_argument("--acoustic-bin", default="", help="concatenated fp16 (n, 512) arrays")
    extra.add_argument("--acoustic-index", default="", help="jsonl index for --acoustic-bin")
    extra.add_argument("--acoustic-dim", type=int, default=512)
    extra.add_argument("--acoustic-dropout", type=float, default=0.0)
    extra.add_argument("--acoustic-anchor", default=ACOUSTIC_ANCHOR,
                       help="literal prompt anchor that the top-1 hypothesis follows")
    extra.add_argument("--acoustic-max-span-fail", type=float, default=0.02,
                       help="abort if more than this fraction of rows cannot be located")
    extra.add_argument("--acoustic-audit-limit", type=int, default=0,
                       help="audit only the first N rows (0 = all)")
    extra.add_argument("--acoustic-audit-only", action="store_true",
                       help="run the span audit and exit without loading the model")
    known, passthrough = extra.parse_known_args(sys.argv[1:])
    if any(t in ("-h", "--help") for t in passthrough):
        print("Acoustic conditioning flags (this script):", file=sys.stderr)
        extra.print_help(sys.stderr)
        print("", file=sys.stderr)
    saved = sys.argv
    sys.argv = [saved[0]] + passthrough
    try:
        args = base_trainer.parse_args()
    finally:
        sys.argv = saved
    for key, value in vars(known).items():
        setattr(args, key, value)
    return args


# --------------------------------------------------------------------------- #
# 8. Data
# --------------------------------------------------------------------------- #
def build_data(
    args: argparse.Namespace,
    tokenizer: Any,
    store: Optional[AcousticVectorStore] = None,
) -> Tuple[Any, Optional[Any], Any, List[Dict[str, str]], List[Dict[str, str]]]:
    """Texts -> tokenized datasets + collator.  Reuses the text-only loader verbatim."""
    from datasets import Dataset
    from transformers import DataCollatorForLanguageModeling

    acoustic = bool(args.acoustic_bin)
    load = lambda path: base_trainer._load_texts(  # noqa: E731
        path,
        tokenizer,
        args.input_format,
        args.max_length,
        args.disable_thinking,
        args.strict_json_system,
        args.edits_only_output,
        args.position_aware_system,
    )
    train_texts = load(args.train_file)
    eval_texts = load(args.eval_file) if args.eval_file else []

    if acoustic:
        if store is None:
            raise SystemExit("internal error: acoustic store not built")
        if len(store) < len(train_texts):
            raise SystemExit(
                "STOP: the acoustic index covers %d rows but --train-file has %d rows; the pairing "
                "is positional, so a shorter index would silently mis-align acoustics and text. "
                "Wait for the export to finish, or pass a --train-file truncated to the first %d rows."
                % (len(store), len(train_texts), len(store))
            )
        train_rows = [{"text": r["text"], "ac_row": i} for i, r in enumerate(train_texts)]
        eval_rows = [{"text": r["text"]} for r in eval_texts]

        def tokenize(batch):
            return tokenizer(
                batch["text"],
                max_length=int(args.max_length),
                truncation=True,
                padding=False,
                return_offsets_mapping=True,
            )

        remove_columns: List[str] = []
    else:
        train_rows = train_texts
        eval_rows = eval_texts

        def tokenize(batch):  # identical to train_lora_sft's tokenize
            return tokenizer(
                batch["text"],
                max_length=int(args.max_length),
                truncation=True,
                padding=False,
            )

        remove_columns = ["text"]

    map_num_proc = max(1, int(args.preprocessing_num_workers))
    map_kwargs = {
        "batched": True,
        "remove_columns": remove_columns,
        "num_proc": map_num_proc if map_num_proc > 1 else None,
    }
    train_dataset = Dataset.from_list(train_rows).map(tokenize, **map_kwargs)
    eval_dataset = Dataset.from_list(eval_rows).map(tokenize, **map_kwargs) if eval_rows else None

    base_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    if acoustic:
        collator = AcousticCollator(
            tokenizer,
            base_collator,
            store,
            anchor=args.acoustic_anchor,
            max_length=int(args.max_length),
        )
    else:
        collator = base_collator
    return train_dataset, eval_dataset, collator, train_texts, eval_texts


# --------------------------------------------------------------------------- #
# 9. Model
# --------------------------------------------------------------------------- #
def build_model(
    args: argparse.Namespace,
    tokenizer: Any,
    acoustic_enabled: bool = False,
) -> nn.Module:
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    quantization_config = None
    if args.qlora:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 or args.qlora else None),
        device_map="auto",
        quantization_config=quantization_config,
    )
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    if args.qlora:
        model = prepare_model_for_kbit_training(model)

    if args.adapter_path:
        model = PeftModel.from_pretrained(model, args.adapter_path, is_trainable=True)
    else:
        peft_config = LoraConfig(
            r=int(args.lora_r),
            lora_alpha=int(args.lora_alpha),
            lora_dropout=float(args.lora_dropout),
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[item.strip() for item in args.target_modules.split(",") if item.strip()],
        )
        model = get_peft_model(model, peft_config)

    if not acoustic_enabled:
        return model

    emb_weight = model.get_input_embeddings().weight
    d_model = int(emb_weight.shape[1])
    device = emb_weight.device
    dtype = emb_weight.dtype
    if not dtype.is_floating_point:  # e.g. a 4-bit quantised embedding table
        dtype = torch.bfloat16 if args.bf16 else torch.float16
    projector = AcousticProjector(
        in_dim=int(args.acoustic_dim),
        d_model=d_model,
        dropout=float(args.acoustic_dropout),
    ).to(device=device, dtype=dtype)
    wrapper = AcousticConditionedLM(model, projector, anchor=args.acoustic_anchor)
    print(
        json.dumps(
            {
                "acoustic_projector": {
                    "in_dim": projector.in_dim,
                    "d_model": projector.d_model,
                    "dtype": str(dtype),
                    "device": str(device),
                    "params": int(sum(p.numel() for p in projector.parameters())),
                    "zero_initialised_final_linear": projector.is_zero_initialised(),
                    "dropout": float(args.acoustic_dropout),
                }
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return wrapper


# --------------------------------------------------------------------------- #
# 10. Trainer that can persist wrapper + projector
# --------------------------------------------------------------------------- #
try:  # pragma: no cover - location differs across transformers versions
    from transformers.trainer import TRAINING_ARGS_NAME
except Exception:  # pragma: no cover
    TRAINING_ARGS_NAME = "training_args.bin"


def make_trainer_class():
    """Trainer whose _save persists the LoRA adapter AND the acoustic projector.

    transformers' Trainer._save only understands PreTrainedModel/PeftModel; for any
    other nn.Module it dumps a raw state_dict (which for a 9B base is ~18 GB and is
    not loadable as an adapter), so the wrapper needs this hook.
    """
    from transformers import Trainer

    class AcousticTrainer(Trainer):
        def _save(self, output_dir: Optional[str] = None, state_dict: Optional[Dict[str, Any]] = None) -> None:
            out = Path(output_dir if output_dir is not None else self.args.output_dir)
            out.mkdir(parents=True, exist_ok=True)
            model = self.accelerator.unwrap_model(self.model, keep_torch_compile=False)
            if not getattr(model, "is_acoustic_conditioned", False):
                return super()._save(output_dir, state_dict=state_dict)
            model.save_pretrained(str(out), state_dict=state_dict)
            if self.processing_class is not None:
                self.processing_class.save_pretrained(str(out))
            elif (
                self.data_collator is not None
                and hasattr(self.data_collator, "tokenizer")
                and self.data_collator.tokenizer is not None
            ):
                self.data_collator.tokenizer.save_pretrained(str(out))
            torch.save(self.args, os.path.join(str(out), TRAINING_ARGS_NAME))

    return AcousticTrainer


# --------------------------------------------------------------------------- #
# 11. main
# --------------------------------------------------------------------------- #
def main() -> int:
    args = parse_args()
    started = time.monotonic()
    acoustic_enabled = bool(args.acoustic_bin)
    if acoustic_enabled and not args.acoustic_index:
        raise SystemExit("--acoustic-bin requires --acoustic-index")

    from transformers import AutoTokenizer, Trainer, TrainerCallback, TrainingArguments

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    store = None
    if acoustic_enabled:
        store = AcousticVectorStore(args.acoustic_bin, args.acoustic_index, expect_dim=int(args.acoustic_dim))
        print(json.dumps({"acoustic_store": store.summary()}, ensure_ascii=False), flush=True)

    train_dataset, eval_dataset, data_collator, train_texts, eval_texts = build_data(args, tokenizer, store)
    if args.dry_run:
        print(json.dumps({"train_rows": len(train_texts), "eval_rows": len(eval_texts), "sample": train_texts[:1]},
                         ensure_ascii=False, indent=2))
        return 0

    span_audit = None
    if acoustic_enabled:
        span_audit = audit_span_location(
            train_dataset, store, anchor=args.acoustic_anchor, limit=int(args.acoustic_audit_limit)
        )
        print(json.dumps({"acoustic_span_audit": span_audit}, ensure_ascii=False), flush=True)
        if span_audit["hard_fail_rate"] > float(args.acoustic_max_span_fail):
            raise SystemExit(
                "STOP: could not locate the ASR top-1 span in %.4f of rows (limit %.4f). The prompt "
                "template assumption is wrong, so the acoustic bias would be added to the wrong tokens."
                % (span_audit["hard_fail_rate"], float(args.acoustic_max_span_fail))
            )
        if span_audit["degraded_or_failed_rate"] > float(args.acoustic_max_span_fail):
            print(json.dumps({"acoustic_span_audit_warning":
                              "located only via fallback/punctuation substitution"}), flush=True)

    if args.acoustic_audit_only:
        print(json.dumps({"audit_only": True, "elapsed_s": round(time.monotonic() - started, 1)}), flush=True)
        return 0

    model = build_model(args, tokenizer, acoustic_enabled=acoustic_enabled)
    inner = getattr(model, "base", model)
    inner.print_trainable_parameters()
    if acoustic_enabled:
        print(json.dumps({"acoustic_trainable_params": int(
            sum(p.numel() for p in model.proj.parameters() if p.requires_grad))}), flush=True)

    class JsonLogCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs:
                return
            keep = {
                key: value
                for key, value in logs.items()
                if key in {"loss", "eval_loss", "learning_rate", "epoch", "grad_norm"}
            }
            if keep:
                keep["step"] = int(state.global_step)
                print(json.dumps({"trainer_log": keep}, ensure_ascii=False), flush=True)

    training_kwargs = dict(
        output_dir=args.output_dir,
        num_train_epochs=float(args.epochs),
        max_steps=int(args.max_steps),
        learning_rate=float(args.learning_rate),
        lr_scheduler_type=args.lr_scheduler_type,
        per_device_train_batch_size=int(args.per_device_train_batch_size),
        per_device_eval_batch_size=int(args.per_device_eval_batch_size),
        gradient_accumulation_steps=int(args.gradient_accumulation_steps),
        warmup_ratio=float(args.warmup_ratio),
        logging_steps=int(args.logging_steps),
        save_steps=int(args.save_steps),
        eval_steps=int(args.eval_steps),
        save_strategy="steps",
        dataloader_num_workers=max(0, int(args.dataloader_num_workers)),
        dataloader_prefetch_factor=max(1, int(args.dataloader_prefetch_factor)) if int(args.dataloader_num_workers) > 0 else None,
        bf16=bool(args.bf16),
        fp16=bool(args.fp16),
        report_to="none",
        remove_unused_columns=not acoustic_enabled,
    )
    try:
        training_args = TrainingArguments(
            **training_kwargs,
            eval_strategy="steps" if eval_dataset is not None else "no",
        )
    except TypeError:
        training_args = TrainingArguments(
            **training_kwargs,
            evaluation_strategy="steps" if eval_dataset is not None else "no",
        )

    trainer_cls = make_trainer_class() if acoustic_enabled else Trainer
    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=[JsonLogCallback()],
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    base_trainer._write_training_metadata(args, len(train_texts), len(eval_texts))
    meta_path = Path(args.output_dir) / "training_metadata.json"
    if acoustic_enabled:
        try:
            payload = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            payload = {}
        payload["acoustic"] = {
            "enabled": True,
            "bin": args.acoustic_bin,
            "index": args.acoustic_index,
            "dim": int(args.acoustic_dim),
            "dropout": float(args.acoustic_dropout),
            "anchor": args.acoustic_anchor,
            "injection": "emb[start_b : start_b+n_b] += proj(vecs)[b, :n_b]",
            "projector": "LayerNorm->Linear->GELU->Linear(zero-init)",
            "span_audit": span_audit,
        }
        meta_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": args.output_dir, "saved": True, "metadata": str(meta_path),
                      "elapsed_s": round(time.monotonic() - started, 1)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
