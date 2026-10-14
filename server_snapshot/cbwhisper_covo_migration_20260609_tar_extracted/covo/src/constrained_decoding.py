"""SEAL-style constrained decoding over the acoustically-grounded character set.

RESULTS section 17 established WHERE the information is: at the frames where
CB-SenseVoice's top-1 is wrong, the correct character sits inside the model's own
top-5 for 87.5% of errors.  The acoustic-conditioned model is trained to read that
evidence; this module is the second half of the SEAL recipe (arXiv 2501.08421),
**constrained decoding**, which bounds what the language model may emit so that it
cannot invent characters the acoustics never proposed.

Why it is worth having even though the unconstrained arm is the primary experiment:
the earlier text-only SFT arm emitted off-pool output on 15.3% of rows, and that
hallucination was one of the direct causes of its failure.  A decoder that can only
choose among acoustically plausible characters makes that failure mode structurally
impossible instead of merely discouraged by the prompt.

Mechanism
---------
The model replies `{"text":"<corrected sentence>"}`.  Once generation is inside the
text value, output position p may only emit a character from

    { hypothesis[p] } u { the CTC top-k alternatives at p }

and p advances with each character emitted.  Substitution is therefore free within
the acoustically plausible set and impossible outside it.  Insertions and deletions
are not expressible, which matches the measurement that only 2625 of the 2956
character errors are addressable by per-character decisions -- the other 331 are
indels such a mechanism cannot repair.

Outside the text value, and once the sentence has run past the hypothesis, nothing
is masked: the constraint must never be able to trap the model in an unfinishable
sequence.

The processor is deliberately STATELESS: it decodes the ids it is given on every
call and locates the value by the last occurrence of the opening marker.  That
makes it correct whether `input_ids` contains the prompt plus the continuation (the
ordinary case) or only the continuation (what happens when the wrapper generates
from `inputs_embeds`, as this project does) -- and it avoids assuming which.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch

try:
    from transformers import LogitsProcessor
except Exception:  # pragma: no cover - keeps the module importable without transformers
    class LogitsProcessor:  # type: ignore
        pass


NEG_INF = float("-inf")

# Tolerant of whitespace after the colon, and of the key being quoted or not, because
# the marker has to be recognised in whatever the MODEL emits -- the prompt is not
# visible to a logits processor under inputs_embeds generation.
_VALUE_OPEN = re.compile(r'"?text"?\s*:\s*"')


def build_char_to_ids(tokenizer, chars: Sequence[str]) -> Dict[str, List[int]]:
    """Map each character to the token id(s) that can start it.

    Only the FIRST token of each character's encoding is kept: the mask acts on the
    next-token distribution, and a multi-token character is reached by emitting its
    first token, after which that character is no longer the target.  This is exact
    for the CJK case that dominates here, where one character is one token (verified
    elsewhere in this project: 15 characters -> 15 tokens).
    """
    out: Dict[str, List[int]] = {}
    for ch in set(chars):
        if not ch:
            continue
        ids = tokenizer.encode(ch, add_special_tokens=False)
        if ids:
            out[ch] = [int(ids[0])]
    return out


def load_allowed_sets(path: str, top_k: int = 5) -> Dict[str, List[List[str]]]:
    """Read per-position acoustically-grounded character sets.

    Input jsonl, one object per utterance:
        {"id": str, "top1": str, "topk": [[c1, c2, ...], ...]}
    where `topk[p]` lists the acoustic model's alternatives at the emission frame of
    character p of `top1` (argmax excluded), best first.

    Returns id -> per character, the allowed set (hypothesis character first).
    """
    out: Dict[str, List[List[str]]] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        top1 = r.get("top1") or ""
        alts = r.get("topk") or []
        sets: List[List[str]] = []
        for p, ch in enumerate(top1):
            s = [ch]
            if p < len(alts):
                for a in (alts[p] or [])[:top_k]:
                    if a and a not in s:
                        s.append(a)
            sets.append(s)
        out[str(r["id"])] = sets
    return out


class AcousticConstrainedLogitsProcessor(LogitsProcessor):
    """Mask every next-token candidate outside the acoustically plausible set."""

    def __init__(
        self,
        allowed_sets: Sequence[Sequence[Sequence[str]]],
        char_to_ids: Dict[str, List[int]],
        tokenizer,
        eos_token_id: Optional[int] = None,
        extra_allowed_ids: Optional[Sequence[int]] = None,
        open_marker: str = '"text":"',
        first_step_extra_ids: Optional[Sequence[int]] = None,
        safe_chars: Optional[Sequence[str]] = None,
    ):
        self.allowed_sets = allowed_sets
        self.char_to_ids = char_to_ids
        self.tokenizer = tokenizer
        self.eos_token_id = eos_token_id
        # tokens that must always remain reachable: closing quote, braces, newline
        self.extra_allowed_ids = set(int(t) for t in (extra_allowed_ids or []))
        if eos_token_id is not None:
            self.extra_allowed_ids.add(int(eos_token_id))
        self.open_marker = open_marker
        self.first_step_extra_ids = set(int(t) for t in (first_step_extra_ids or []))
        # Characters that encode to EXACTLY one token.  The mask admits only a
        # character's first token, so if an allowed character needs more than one
        # token the emitted token decodes to an incomplete character, `_consumed`
        # fails to advance, and the same position is masked again -- the model can
        # be trapped re-emitting a token it cannot complete.  10.7% of the distinct
        # characters in the ST-CMDS sets are multi-token, so this is not a corner
        # case: at any position whose allowed set contains one, release instead.
        self.safe_chars = set(safe_chars) if safe_chars is not None else None

    # ---------------------------------------------------------------- helpers
    def _consumed(self, text: str) -> Optional[int]:
        """Characters generated inside the text value, or None if outside it.

        Matched with a REGEX tolerant of whitespace between the colon and the opening
        quote.  This matters more than it looks: under `inputs_embeds` generation the
        logits processor receives ONLY the generated tokens -- the prompt never
        appears -- so the marker can only ever come from the model's own output.  If
        it emits `"text": "` with a space and we required exactly `"text":"`, the
        constraint would silently never engage and the arm would look identical to
        the unconstrained one.
        """
        m = None
        for m in _VALUE_OPEN.finditer(text):     # last match wins
            pass
        if m is None:
            return None
        rest = text[m.end():]
        if '"' in rest:                          # the value has already been closed
            return None
        return len(rest)

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        scores = scores.clone()
        vocab = scores.size(-1)
        for b in range(input_ids.size(0)):
            # input_ids may or may not include the prompt; decoding it whole is
            # correct in both cases because we search for the LAST marker.
            text = self.tokenizer.decode(input_ids[b].tolist(), skip_special_tokens=True)
            consumed = self._consumed(text)
            if consumed is None:
                continue
            sets = self.allowed_sets[b] if b < len(self.allowed_sets) else []
            if consumed >= len(sets):
                continue                     # past the hypothesis: release the constraint
            allowed_ids = set()
            for ch in sets[consumed]:
                allowed_ids.update(self.char_to_ids.get(ch, []))
            if not allowed_ids:
                continue                     # nothing usable: stay unconstrained
            if self.safe_chars is not None and any(
                    ch not in self.safe_chars for ch in sets[consumed]):
                continue                     # multi-token char here could trap the model
            allowed_ids |= self.extra_allowed_ids
            if consumed == 0:
                allowed_ids |= self.first_step_extra_ids

            idx = torch.tensor(sorted(i for i in allowed_ids if 0 <= i < vocab),
                               device=scores.device, dtype=torch.long)
            if idx.numel() == 0:
                continue
            mask = torch.full_like(scores[b], NEG_INF)
            mask[idx] = scores[b][idx]
            scores[b] = mask
        return scores
