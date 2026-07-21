"""CTC prefix-beam decoding with lightweight contextual hotword biasing."""

from __future__ import annotations

import math
import os
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch


NEG_INF = -float("inf")
_FAST_DECODER = None
_FAST_DECODER_ERROR: Exception | None = None
_FAST_DECODER_LOCK = threading.Lock()


def _logadd(*values: float) -> float:
    finite = [float(value) for value in values if value != NEG_INF]
    if not finite:
        return NEG_INF
    maximum = max(finite)
    return maximum + math.log(sum(math.exp(value - maximum) for value in finite))


@dataclass(frozen=True)
class CTCCandidate:
    token_ids: Tuple[int, ...]
    acoustic_score: float
    search_score: float
    hotword_score: float


class ContextualHotwordScorer:
    """Rewards only token extensions that advance a detected hotword prefix."""

    def __init__(
        self,
        entries: Iterable[Tuple[Sequence[int], float]],
        token_weight: float,
        completion_weight: float,
    ) -> None:
        self.entries = [
            (tuple(int(token) for token in tokens), max(0.0, float(score)))
            for tokens, score in entries
            if len(tokens) > 0
        ]
        self.token_weight = float(token_weight)
        self.completion_weight = float(completion_weight)
        self.token_ids = {token for tokens, _ in self.entries for token in tokens}

    @staticmethod
    def _progress(prefix: Tuple[int, ...], hotword: Tuple[int, ...]) -> int:
        upper = min(len(prefix), len(hotword))
        for size in range(upper, 0, -1):
            if prefix[-size:] == hotword[:size]:
                return size
        return 0

    def extension_score(self, prefix: Tuple[int, ...], token: int) -> float:
        extended = prefix + (int(token),)
        best = 0.0
        for hotword, confidence in self.entries:
            old_progress = self._progress(prefix, hotword)
            new_progress = self._progress(extended, hotword)
            if new_progress <= old_progress:
                continue
            reward = self.token_weight * confidence / max(len(hotword), 1)
            if new_progress == len(hotword):
                reward += self.completion_weight * confidence
            best = max(best, reward)
        return float(best)


def _load_fast_decoder():
    global _FAST_DECODER, _FAST_DECODER_ERROR
    if _FAST_DECODER is not None:
        return _FAST_DECODER
    if _FAST_DECODER_ERROR is not None:
        return None
    with _FAST_DECODER_LOCK:
        if _FAST_DECODER is not None:
            return _FAST_DECODER
        if _FAST_DECODER_ERROR is not None:
            return None
        try:
            from torch.utils.cpp_extension import load

            source = Path(__file__).with_name("sensevoice_ctc_fast.cpp")
            python_bin_dir = str(Path(sys.executable).resolve().parent)
            path_entries = os.environ.get("PATH", "").split(os.pathsep)
            if python_bin_dir not in path_entries:
                os.environ["PATH"] = python_bin_dir + os.pathsep + os.environ.get("PATH", "")
            _FAST_DECODER = load(
                name="cb_sensevoice_ctc_fast_v1",
                sources=[str(source)],
                extra_cflags=["-O3", "-std=c++17"],
                verbose=os.getenv("CBW_CTC_BUILD_VERBOSE", "0").lower() in {"1", "true", "yes"},
            )
        except Exception as exc:
            _FAST_DECODER_ERROR = exc
            print(f"[cb-sensevoice][warn] fast CTC decoder unavailable, using Python fallback: {exc}")
            return None
    return _FAST_DECODER


def _ctc_prefix_beam_search_fast(
    log_probs: torch.Tensor,
    beam_size: int,
    token_topk: int,
    blank_id: int,
    hotword_scorer: ContextualHotwordScorer | None,
    excluded_token_ids: Iterable[int],
) -> List[CTCCandidate] | None:
    extension = _load_fast_decoder()
    if extension is None:
        return None
    excluded = sorted({int(token) for token in excluded_token_ids})
    values, indices = torch.topk(log_probs.detach().float(), k=token_topk, dim=-1)
    extra_tokens = {int(blank_id)}
    hotwords: List[List[int]] = []
    confidences: List[float] = []
    token_weight = 0.0
    completion_weight = 0.0
    if hotword_scorer is not None:
        hotwords = [list(tokens) for tokens, _ in hotword_scorer.entries]
        confidences = [float(confidence) for _, confidence in hotword_scorer.entries]
        extra_tokens.update(hotword_scorer.token_ids)
        token_weight = float(hotword_scorer.token_weight)
        completion_weight = float(hotword_scorer.completion_weight)
    extra = sorted(extra_tokens)
    if extra:
        extra_index = torch.tensor(extra, dtype=torch.long, device=log_probs.device)
        extra_values = log_probs.detach().float().index_select(-1, extra_index)
        extra_indices = extra_index.unsqueeze(0).expand(log_probs.size(0), -1)
        values = torch.cat((values, extra_values), dim=-1)
        indices = torch.cat((indices, extra_indices), dim=-1)
    rows = extension.decode(
        values.cpu().tolist(),
        indices.cpu().tolist(),
        int(beam_size),
        int(blank_id),
        hotwords,
        confidences,
        token_weight,
        completion_weight,
        excluded,
    )
    return [
        CTCCandidate(
            token_ids=tuple(int(token) for token in tokens),
            acoustic_score=float(acoustic),
            search_score=float(search),
            hotword_score=float(hotword),
        )
        for tokens, acoustic, search, hotword in rows
    ]


def ctc_prefix_beam_search(
    log_probs: torch.Tensor,
    beam_size: int,
    token_topk: int,
    blank_id: int = 0,
    hotword_scorer: ContextualHotwordScorer | None = None,
    excluded_token_ids: Iterable[int] = (),
) -> List[CTCCandidate]:
    """Decode one ``[time, vocab]`` CTC log-probability matrix.

    Search scores contain the contextual bonus, while acoustic scores are
    accumulated separately and remain suitable for downstream reranking.
    """

    if log_probs.dim() != 2:
        raise ValueError(f"expected [time, vocab] CTC log probabilities, got {tuple(log_probs.shape)}")
    beam_size = max(1, int(beam_size))
    token_topk = max(1, min(int(token_topk), int(log_probs.size(-1))))
    excluded = {int(token) for token in excluded_token_ids}
    backend = os.getenv("CBW_CTC_DECODER", "cpp").strip().lower()
    if backend not in {"python", "py", "reference"}:
        fast = _ctc_prefix_beam_search_fast(
            log_probs=log_probs,
            beam_size=beam_size,
            token_topk=token_topk,
            blank_id=blank_id,
            hotword_scorer=hotword_scorer,
            excluded_token_ids=excluded,
        )
        if fast is not None:
            return fast
    beams: Dict[Tuple[int, ...], Tuple[float, float, float, float]] = {
        (): (0.0, NEG_INF, 0.0, NEG_INF)
    }

    cpu_log_probs = log_probs.detach().float().cpu()
    frame_values, frame_indices = torch.topk(cpu_log_probs, k=token_topk, dim=-1)
    for time_idx, (values, indices) in enumerate(zip(frame_values, frame_indices)):
        frame = {int(token): float(value) for token, value in zip(indices.tolist(), values.tolist())}
        frame[int(blank_id)] = float(cpu_log_probs[time_idx, int(blank_id)].item())
        if hotword_scorer is not None:
            for token in hotword_scorer.token_ids:
                if 0 <= token < log_probs.size(-1) and token not in frame:
                    frame[token] = float(cpu_log_probs[time_idx, token].item())

        next_beams: Dict[Tuple[int, ...], Tuple[float, float, float, float]] = {}
        for prefix, (search_blank, search_nonblank, acoustic_blank, acoustic_nonblank) in beams.items():
            search_total = _logadd(search_blank, search_nonblank)
            acoustic_total = _logadd(acoustic_blank, acoustic_nonblank)
            for token, token_log_prob in frame.items():
                token = int(token)
                token_log_prob = float(token_log_prob)
                if token == int(blank_id):
                    old = next_beams.get(prefix, (NEG_INF, NEG_INF, NEG_INF, NEG_INF))
                    next_beams[prefix] = (
                        _logadd(old[0], search_total + token_log_prob),
                        old[1],
                        _logadd(old[2], acoustic_total + token_log_prob),
                        old[3],
                    )
                    continue
                if token in excluded:
                    continue

                last_token = prefix[-1] if prefix else None
                if token == last_token:
                    old = next_beams.get(prefix, (NEG_INF, NEG_INF, NEG_INF, NEG_INF))
                    next_beams[prefix] = (
                        old[0],
                        _logadd(old[1], search_nonblank + token_log_prob),
                        old[2],
                        _logadd(old[3], acoustic_nonblank + token_log_prob),
                    )
                    extended = prefix + (token,)
                    bonus = hotword_scorer.extension_score(prefix, token) if hotword_scorer is not None else 0.0
                    old_ext = next_beams.get(extended, (NEG_INF, NEG_INF, NEG_INF, NEG_INF))
                    next_beams[extended] = (
                        old_ext[0],
                        _logadd(old_ext[1], search_blank + token_log_prob + bonus),
                        old_ext[2],
                        _logadd(old_ext[3], acoustic_blank + token_log_prob),
                    )
                    continue

                extended = prefix + (token,)
                bonus = hotword_scorer.extension_score(prefix, token) if hotword_scorer is not None else 0.0
                old_ext = next_beams.get(extended, (NEG_INF, NEG_INF, NEG_INF, NEG_INF))
                next_beams[extended] = (
                    old_ext[0],
                    _logadd(old_ext[1], search_total + token_log_prob + bonus),
                    old_ext[2],
                    _logadd(old_ext[3], acoustic_total + token_log_prob),
                )

        beams = dict(
            sorted(
                next_beams.items(),
                key=lambda item: _logadd(item[1][0], item[1][1]),
                reverse=True,
            )[:beam_size]
        )

    candidates = []
    for prefix, (search_blank, search_nonblank, acoustic_blank, acoustic_nonblank) in beams.items():
        search_score = _logadd(search_blank, search_nonblank)
        acoustic_score = _logadd(acoustic_blank, acoustic_nonblank)
        candidates.append(
            CTCCandidate(
                token_ids=prefix,
                acoustic_score=float(acoustic_score),
                search_score=float(search_score),
                hotword_score=float(search_score - acoustic_score),
            )
        )
    candidates.sort(key=lambda candidate: candidate.search_score, reverse=True)
    return candidates
