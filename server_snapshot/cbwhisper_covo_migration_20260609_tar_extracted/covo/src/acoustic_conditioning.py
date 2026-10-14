"""Acoustic conditioning for COVO: the coupling component (AC-COVO).

RESULTS section 17 measured the gap this module exists to close: at the frames
where CB-SenseVoice's top-1 is wrong, the correct character is inside the
model's own top-5 for 87.5% of errors and is the immediate runner-up for 58.8%
(above this project's 54% break-even) -- yet no gate built on single-character
acoustic confidence recovers more than 48 characters of the 438 needed.  The
acoustics carry the answer, the criterion is favourable, and the missing piece
is a decision function that reads the acoustics AND the surrounding text at the
same time.  That function is this module.

Design, and why each choice is forced by a measurement:

  * The projector consumes the FROZEN encoder state at the exact frame where the
    CTC head emitted each character (exported by
    src/analysis/export_acoustic_vectors.py).  A sentence-level summary would
    destroy the per-position information that the 87.5% figure is about.
  * The acoustic term is ADDED to that character's input embedding rather than
    concatenated as a prefix, so the sequence length is unchanged and every
    position stays tied to its own character.
  * The final projection layer is ZERO-INITIALISED, so at step 0 the model is
    numerically identical to the text-only model.  This removes the usual reason
    for a separate alignment stage (a randomly initialised projector injecting
    noise) while making the acoustic channel something the model must actively
    earn.  It also gives a free end-to-end check: at init, this model must
    reproduce the baseline adapter's output exactly.

Train and inference share `build_acoustic_inputs` on purpose: a span-location
difference between the two would silently attach acoustics to the wrong
characters and would look like a modelling failure rather than a bug.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
# projector / wrapper
# --------------------------------------------------------------------------- #
class AcousticProjector(nn.Module):
    """LayerNorm -> Linear -> GELU -> Linear, with the last Linear zeroed."""

    def __init__(self, dim_in: int = 512, dim_out: int = 4096, n_layers: int = 2,
                 dropout: float = 0.0):
        super().__init__()
        layers: List[nn.Module] = [nn.LayerNorm(dim_in)]
        d = dim_in
        for i in range(n_layers):
            last = i == n_layers - 1
            layers.append(nn.Linear(d, dim_out))
            if not last:
                layers.append(nn.GELU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
            d = dim_out
        self.net = nn.Sequential(*layers)
        final = self.net[-1]
        if isinstance(final, nn.Linear):
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)
        self.dim_in = dim_in
        self.dim_out = dim_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _splice(emb: torch.Tensor, bias: torch.Tensor,
            start: torch.Tensor, length: torch.Tensor) -> torch.Tensor:
    """Add `bias[b, :n]` into `emb[b, s:s+n]` WITHOUT any in-place operation.

    In-place assignment fails under gradient checkpointing, because the embedding
    output is then a leaf that requires grad ("a view of a leaf Variable that
    requires grad is being used in an in-place operation").  Building each row with
    cat/stack is equivalent and stays differentiable.
    """
    rows = []
    for b in range(emb.size(0)):
        s = int(start[b])
        n = int(length[b])
        if n <= 0:
            rows.append(emb[b])
            continue
        s = max(0, min(s, emb.size(1)))
        n = min(n, emb.size(1) - s, bias.size(1))
        if n <= 0:
            rows.append(emb[b])
            continue
        row = emb[b]
        rows.append(torch.cat([row[:s], row[s:s + n] + bias[b, :n], row[s + n:]], dim=0))
    return torch.stack(rows, dim=0)


class AcousticConditionedLM(nn.Module):
    """Wraps a causal LM and adds projected acoustics into character embeddings."""

    def __init__(self, base: nn.Module, projector: AcousticProjector):
        super().__init__()
        self.base = base
        self.projector = projector

    def forward(self, input_ids=None, attention_mask=None,
                acoustic: Optional[torch.Tensor] = None,
                acoustic_start: Optional[torch.Tensor] = None,
                acoustic_len: Optional[torch.Tensor] = None,
                inputs_embeds: Optional[torch.Tensor] = None, **kw):
        if inputs_embeds is None:
            inputs_embeds = self.base.get_input_embeddings()(input_ids)
        if acoustic is not None and acoustic.numel() > 0:
            # The projector stays in fp32 (it is trained from scratch, so its own
            # precision matters) while the base model runs in bf16; cast at the
            # boundary in both directions.
            pdtype = next(self.projector.parameters()).dtype
            bias = self.projector(acoustic.to(pdtype)).to(inputs_embeds.dtype)
            inputs_embeds = _splice(inputs_embeds, bias, acoustic_start, acoustic_len)
        return self.base(inputs_embeds=inputs_embeds,
                         attention_mask=attention_mask, **kw)

    @torch.no_grad()
    def generate(self, input_ids=None, attention_mask=None,
                 acoustic=None, acoustic_start=None, acoustic_len=None, **kw):
        emb = self.base.get_input_embeddings()(input_ids)
        if acoustic is not None and acoustic.numel() > 0:
            pdtype = next(self.projector.parameters()).dtype
            bias = self.projector(acoustic.to(pdtype)).to(emb.dtype)
            emb = _splice(emb, bias, acoustic_start, acoustic_len)
        return self.base.generate(inputs_embeds=emb, attention_mask=attention_mask, **kw)

    def gradient_checkpointing_enable(self, **kw):
        return self.base.gradient_checkpointing_enable(**kw)

    def gradient_checkpointing_disable(self, **kw):
        return self.base.gradient_checkpointing_disable(**kw)

    def enable_input_require_grads(self):
        return self.base.enable_input_require_grads()

    def save_pretrained(self, *a, **k):
        return self.base.save_pretrained(*a, **k)

    def __getattr__(self, name):
        # Behave like the wrapped model for everything Trainer touches (config,
        # device, is_loaded_in_8bit, ...).  nn.Module resolves submodules through
        # its own __getattr__, so try that FIRST -- an earlier version raised for
        # "base" before delegating and broke ordinary `wrapper.base` access.
        # Fall back to the wrapped model via _modules directly, never via
        # `self.base`, which would re-enter this method and recurse.
        try:
            return super().__getattr__(name)
        except AttributeError:
            pass
        base = self.__dict__.get("_modules", {}).get("base")
        if base is None:
            raise AttributeError(name)
        return getattr(base, name)


# --------------------------------------------------------------------------- #
# span location (shared by training and inference -- no train/serve skew)
# --------------------------------------------------------------------------- #
def _strip_punct(s: str) -> str:
    keep = []
    for ch in s or "":
        if "\u4e00" <= ch <= "\u9fff" or "\u3400" <= ch <= "\u4dbf" or ch.isalnum():
            keep.append(ch)
    return "".join(keep)


def locate_span(prompt: str, offsets: Sequence[Tuple[int, int]], top1: str):
    """Character span of `top1` in `prompt`, tolerant of punctuation rendering.

    The prompt renders the hypothesis WITHOUT Chinese punctuation while the raw
    top-1 (which the acoustics are aligned to) still carries it, so a byte-exact
    search fails on ~1.3% of rows.  We therefore try the exact string first and
    fall back to a punctuation-stripped search, mapping back to the raw index.

    Returns (char_start, char_len) into `prompt`, or None.
    """
    if not top1:
        return None
    pos = prompt.find(top1)
    if pos >= 0:
        return pos, len(top1)
    stripped = _strip_punct(top1)
    if not stripped:
        return None
    # walk the prompt with the same stripping so lengths map back exactly
    keep_idx = [i for i, ch in enumerate(prompt) if _strip_punct(ch)]
    flat = "".join(prompt[i] for i in keep_idx)
    p = flat.find(stripped)
    if p < 0:
        return None
    return keep_idx[p], keep_idx[p + len(stripped) - 1] - keep_idx[p] + 1


def chars_to_tokens(offsets: Sequence[Tuple[int, int]], start: int, n_chars: int):
    """token index -> list of character indices (relative to `start`)."""
    end = start + n_chars
    mapping: Dict[int, List[int]] = {}
    for i, (a, b) in enumerate(offsets):
        if b <= start or a >= end:
            continue
        js = [c - start for c in range(max(a, start), min(b, end))]
        if js:
            mapping[i] = js
    return mapping


# --------------------------------------------------------------------------- #
# acoustic store
# --------------------------------------------------------------------------- #
class AcousticStore:
    """Random access to the exported fp16 per-character encoder states."""

    def __init__(self, bin_path: str, index_path: str):
        self.bin_path = bin_path
        self.rows: List[Dict[str, Any]] = [
            json.loads(l) for l in Path(index_path).read_text(encoding="utf-8").splitlines() if l.strip()
        ]
        self.by_id = {str(r["id"]): i for i, r in enumerate(self.rows)}
        self._arr = None
        self.dim = int(self.rows[0]["dim"]) if self.rows else 512

    @property
    def arr(self) -> np.memmap:
        if self._arr is None:
            self._arr = np.memmap(self.bin_path, dtype=np.float16, mode="r")
        return self._arr

    def __len__(self) -> int:
        return len(self.rows)

    def row(self, i: int) -> Dict[str, Any]:
        return self.rows[i]

    def vectors(self, i: int) -> np.ndarray:
        r = self.rows[i]
        off = int(r["offset"]) // 2          # element offset for fp16
        n, d = int(r["n"]), int(r["dim"])
        return np.asarray(self.arr[off:off + n * d], dtype=np.float32).reshape(n, d)

    def index_of_id(self, key: str) -> Optional[int]:
        return self.by_id.get(str(key))


# --------------------------------------------------------------------------- #
# the shared input builder
# --------------------------------------------------------------------------- #
def build_acoustic_inputs(
    tokenizer,
    messages: Sequence[Dict[str, str]],
    ac_row: int,
    store: AcousticStore,
    max_length: int = 2048,
    disable_thinking: bool = True,
    train: bool = True,
) -> Dict[str, Any]:
    """Tokenise one row and attach the acoustic vectors of its hypothesis span.

    `train=True` renders the FULL conversation and labels every token, matching
    the baseline trainer (which used DataCollatorForLanguageModeling over the whole
    sequence).  `train=False` renders the prompt only and opens the assistant turn,
    for generation.  The acoustics are located the same way in both cases, which is
    the whole point of sharing this function.

    Returns input_ids, attention_mask, labels, acoustic, acoustic_start, acoustic_len.
    """
    template_kwargs: Dict[str, Any] = {}
    if disable_thinking:
        template_kwargs["enable_thinking"] = False

    if train:
        prompt = tokenizer.apply_chat_template(
            list(messages), tokenize=False, add_generation_prompt=False, **template_kwargs)
    else:
        user_msgs = [m for m in messages if m.get("role") != "assistant"]
        prompt = tokenizer.apply_chat_template(
            user_msgs, tokenize=False, add_generation_prompt=True, **template_kwargs)

    enc = tokenizer(prompt, return_offsets_mapping=True, add_special_tokens=False)
    input_ids = list(enc["input_ids"])[:max_length]
    offsets = list(enc["offset_mapping"])[:max_length]
    labels = list(input_ids) if train else [-100] * len(input_ids)

    acoustic = torch.zeros(0, store.dim)
    start, alen = 0, 0
    top1 = store.row(ac_row).get("top1") or ""
    if top1:
        sp = locate_span(prompt, offsets, top1)
        if sp is not None:
            s, nc = sp
            n_clip = min(nc, max(0, len(prompt) - s))
            mapping = chars_to_tokens(offsets, s, n_clip)
            mapping = {t: js for t, js in mapping.items() if t < len(input_ids)}
            if mapping:
                vecs = store.vectors(ac_row)
                t0, t1 = min(mapping), max(mapping)
                t1 = min(t1, len(input_ids) - 1)
                out = torch.zeros(t1 - t0 + 1, store.dim)
                for t, js in mapping.items():
                    if t > t1 or t < t0:
                        continue
                    js = [j for j in js if 0 <= j < vecs.shape[0]]
                    if js:
                        out[t - t0] = torch.from_numpy(vecs[js].mean(axis=0))
                acoustic, start, alen = out, t0, out.size(0)

    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
        "acoustic": acoustic,
        "acoustic_start": start,
        "acoustic_len": alen,
        "prompt_len": len(prompt),
        "span_found": alen > 0,
    }


def collate_acoustic(batch: List[Dict[str, Any]], pad_token_id: int,
                     pad_side: str = "right") -> Dict[str, Any]:
    """Pad a batch and stack the acoustic tensors.

    `pad_side` matters: TRAINING pads on the right, but batched decoder-only
    GENERATION must pad on the LEFT, otherwise the model continues from pad tokens
    and every output is garbage.  When padding on the left the whole sequence shifts
    right, so `acoustic_start` has to shift with it -- forgetting that would attach
    each character's acoustics to the wrong token.
    """
    max_len = max(len(b["input_ids"]) for b in batch)
    max_ac = max((b["acoustic"].size(0) for b in batch), default=0)
    dim = batch[0]["acoustic"].size(1) if batch[0]["acoustic"].numel() else 0

    ids, mask, labels, ac, starts, lens = [], [], [], [], [], []
    for b in batch:
        n = len(b["input_ids"])
        pad = max_len - n
        a = b["acoustic"]
        if dim and a.numel():
            pad_a = torch.zeros(max_ac - a.size(0), dim)
            ac.append(torch.cat([a, pad_a], 0))
            lens.append(b["acoustic_len"])
        else:
            ac.append(torch.zeros(max_ac, dim or 512))
            lens.append(0)

        if pad_side == "left":
            ids.append([pad_token_id] * pad + b["input_ids"])
            mask.append([0] * pad + [1] * n)
            labels.append([-100] * pad + b["labels"])
            starts.append(b["acoustic_start"] + pad)
        else:
            ids.append(b["input_ids"] + [pad_token_id] * pad)
            mask.append([1] * n + [0] * pad)
            labels.append(b["labels"] + [-100] * pad)
            starts.append(b["acoustic_start"])

    return {
        "input_ids": torch.tensor(ids, dtype=torch.long),
        "attention_mask": torch.tensor(mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "acoustic": torch.stack(ac) if ac else torch.zeros(0),
        "acoustic_start": torch.tensor(starts, dtype=torch.long),
        "acoustic_len": torch.tensor(lens, dtype=torch.long),
    }
