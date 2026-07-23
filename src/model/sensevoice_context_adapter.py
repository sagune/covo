"""Trainable frame-level contextual biasing for SenseVoice CTC logits."""

from __future__ import annotations

import math
import zlib
from typing import Sequence

import torch
import torch.nn.functional as F
from pypinyin import Style, lazy_pinyin
from torch import nn


def pinyin_bucket_ids(text: str, token_count: int, bucket_count: int = 512) -> list[int]:
    """Map a phrase's syllables to stable hashed ids aligned to ASR tokens."""
    syllables = lazy_pinyin(
        str(text),
        style=Style.TONE3,
        neutral_tone_with_five=True,
        errors="ignore",
    )
    ids = [1 + zlib.crc32(item.encode("utf-8")) % max(1, bucket_count - 1) for item in syllables]
    if not ids:
        ids = [0]
    if len(ids) < token_count:
        ids.extend([ids[-1]] * (token_count - len(ids)))
    return ids[:token_count]


class SenseVoiceContextAdapter(nn.Module):
    """Add acoustically conditioned residuals to selected CTC token logits.

    The frozen CTC classifier rows serve as token representations. Only the
    two small projections and three calibration scalars are trainable.
    """

    def __init__(self, hidden_size: int = 512, projection_size: int = 128) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.projection_size = int(projection_size)
        self.audio_projection = nn.Linear(self.hidden_size, self.projection_size, bias=False)
        self.token_projection = nn.Linear(self.hidden_size, self.projection_size, bias=False)
        self.log_scale = nn.Parameter(torch.tensor(-1.5))
        self.log_temperature = nn.Parameter(torch.tensor(2.0))
        self.similarity_offset = nn.Parameter(torch.tensor(0.0))

    def config(self) -> dict[str, int]:
        return {"hidden_size": self.hidden_size, "projection_size": self.projection_size}

    @staticmethod
    def _unique_context(
        token_ids: torch.Tensor,
        confidences: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        unique_ids, inverse = torch.unique(token_ids.long(), sorted=True, return_inverse=True)
        unique_confidences = confidences.new_zeros(unique_ids.numel())
        if hasattr(unique_confidences, "scatter_reduce_"):
            unique_confidences.scatter_reduce_(0, inverse, confidences, reduce="amax", include_self=False)
        else:  # pragma: no cover - compatibility with older PyTorch
            for index in range(unique_ids.numel()):
                unique_confidences[index] = confidences[inverse == index].max()
        return unique_ids, unique_confidences

    def forward(
        self,
        encoder_hidden: torch.Tensor,
        base_log_probs: torch.Tensor,
        ctc_token_weights: torch.Tensor,
        context_token_ids: torch.Tensor | Sequence[int],
        context_confidences: torch.Tensor | Sequence[float] | None = None,
        return_position_logits: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if encoder_hidden.dim() != 3 or base_log_probs.dim() != 3:
            raise ValueError("encoder_hidden and base_log_probs must be [batch,time,dim]")
        if encoder_hidden.shape[:2] != base_log_probs.shape[:2]:
            raise ValueError("encoder_hidden and base_log_probs time dimensions differ")

        token_ids = torch.as_tensor(context_token_ids, dtype=torch.long, device=encoder_hidden.device)
        if token_ids.numel() == 0:
            if return_position_logits:
                empty_logits = encoder_hidden.new_full(encoder_hidden.shape[:2], -20.0)
                return base_log_probs, empty_logits
            return base_log_probs
        if context_confidences is None:
            confidences = torch.ones(token_ids.numel(), dtype=encoder_hidden.dtype, device=encoder_hidden.device)
        else:
            confidences = torch.as_tensor(
                context_confidences,
                dtype=encoder_hidden.dtype,
                device=encoder_hidden.device,
            )
        if confidences.numel() != token_ids.numel():
            raise ValueError("one confidence is required for each context token")

        valid = (token_ids >= 0) & (token_ids < base_log_probs.size(-1))
        token_ids = token_ids[valid]
        confidences = confidences[valid].clamp(0.0, 1.0)
        if token_ids.numel() == 0:
            if return_position_logits:
                empty_logits = encoder_hidden.new_full(encoder_hidden.shape[:2], -20.0)
                return base_log_probs, empty_logits
            return base_log_probs
        token_ids, confidences = self._unique_context(token_ids, confidences)

        token_vectors = ctc_token_weights.index_select(0, token_ids).to(encoder_hidden.dtype)
        audio_queries = F.normalize(self.audio_projection(encoder_hidden), dim=-1)
        token_keys = F.normalize(self.token_projection(token_vectors), dim=-1)
        similarities = torch.matmul(audio_queries, token_keys.transpose(0, 1))
        temperature = self.log_temperature.exp().clamp(1.0, 100.0)
        activation_logits = (similarities - self.similarity_offset) * temperature
        activation = torch.sigmoid(activation_logits)
        residual = F.softplus(self.log_scale) * activation * confidences.view(1, 1, -1)

        adapted = base_log_probs.clone()
        adapted.index_add_(2, token_ids, residual)
        adapted = F.log_softmax(adapted, dim=-1)
        if return_position_logits:
            position_logits = (
                activation_logits + confidences.clamp_min(1e-4).log().view(1, 1, -1)
            ).amax(dim=-1)
            return adapted, position_logits
        return adapted


class SenseVoicePhraseContextAdapter(nn.Module):
    """Fuse complete grapheme-pinyin hotword phrases into SenseVoice frames."""

    expects_phrases = True

    def __init__(
        self,
        hidden_size: int = 512,
        projection_size: int = 128,
        num_heads: int = 4,
        num_context_layers: int = 2,
        max_phrase_tokens: int = 32,
        pinyin_buckets: int = 512,
        dropout: float = 0.1,
        monotonic_phrase_activation: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.projection_size = int(projection_size)
        self.num_heads = int(num_heads)
        self.num_context_layers = int(num_context_layers)
        self.max_phrase_tokens = int(max_phrase_tokens)
        self.pinyin_buckets = int(pinyin_buckets)
        self.dropout = float(dropout)
        self.monotonic_phrase_activation = bool(monotonic_phrase_activation)

        self.audio_projection = nn.Linear(self.hidden_size, self.projection_size, bias=False)
        self.token_projection = nn.Linear(self.hidden_size, self.projection_size, bias=False)
        self.position_embedding = nn.Embedding(self.max_phrase_tokens, self.projection_size)
        self.pinyin_embedding = nn.Embedding(self.pinyin_buckets, self.projection_size, padding_idx=0)
        context_layer = nn.TransformerEncoderLayer(
            d_model=self.projection_size,
            nhead=self.num_heads,
            dim_feedforward=self.projection_size * 4,
            dropout=self.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.phrase_encoder = nn.TransformerEncoder(context_layer, num_layers=self.num_context_layers)
        self.cross_attention = nn.MultiheadAttention(
            self.projection_size,
            self.num_heads,
            dropout=self.dropout,
            batch_first=True,
        )
        self.context_output = nn.Sequential(
            nn.LayerNorm(self.projection_size),
            nn.Linear(self.projection_size, self.hidden_size, bias=False),
        )
        self.log_hidden_scale = nn.Parameter(torch.tensor(-2.5))
        self.log_token_scale = nn.Parameter(torch.tensor(-2.0))
        self.log_temperature = nn.Parameter(torch.tensor(2.0))
        self.similarity_offset = nn.Parameter(torch.tensor(0.0))

    def config(self) -> dict:
        return {
            "hidden_size": self.hidden_size,
            "projection_size": self.projection_size,
            "num_heads": self.num_heads,
            "num_context_layers": self.num_context_layers,
            "max_phrase_tokens": self.max_phrase_tokens,
            "pinyin_buckets": self.pinyin_buckets,
            "dropout": self.dropout,
            "monotonic_phrase_activation": self.monotonic_phrase_activation,
        }

    @staticmethod
    def _monotonic_phrase_logits(
        token_alignment_logits: torch.Tensor,
        phrase_spans: list[tuple[int, int]],
    ) -> torch.Tensor:
        """Score complete phrases as ordered diagonals in frame-token affinity."""
        batch_size, frame_count, _ = token_alignment_logits.shape
        phrase_rows = []
        for start, end in phrase_spans:
            phrase = token_alignment_logits[:, :, start:end]
            token_count = end - start
            rate_rows = []
            for rate in (1, 2, 3):
                shifted_tokens = []
                center = (token_count - 1) / 2.0
                for token_index in range(token_count):
                    offset = int(round((token_index - center) * rate))
                    shifted = phrase.new_full((batch_size, frame_count), -20.0)
                    if 0 <= offset < frame_count:
                        shifted[:, :frame_count - offset] = phrase[:, offset:, token_index]
                    elif -frame_count < offset < 0:
                        shifted[:, -offset:] = phrase[:, :frame_count + offset, token_index]
                    shifted_tokens.append(shifted)
                rate_rows.append(torch.stack(shifted_tokens, dim=-1).mean(dim=-1))
            completion = torch.stack(rate_rows, dim=-1).amax(dim=-1)
            radius = max(1, token_count * 2)
            completion = F.max_pool1d(
                completion.unsqueeze(1),
                kernel_size=radius * 2 + 1,
                stride=1,
                padding=radius,
            ).squeeze(1)
            phrase_rows.append(completion)
        return torch.stack(phrase_rows, dim=-1)

    def _encode_phrases(
        self,
        ctc_token_weights: torch.Tensor,
        context_phrases: Sequence[Sequence[int]],
        context_pinyin_ids: Sequence[Sequence[int]] | None,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[tuple[int, int]]]:
        phrases = []
        pinyin_rows = []
        valid_token_ids = []
        for phrase_index, phrase in enumerate(context_phrases):
            tokens = [
                int(token)
                for token in phrase[: self.max_phrase_tokens]
                if 0 <= int(token) < int(ctc_token_weights.size(0))
            ]
            if not tokens:
                continue
            pinyin = list(context_pinyin_ids[phrase_index]) if context_pinyin_ids is not None else []
            if len(pinyin) < len(tokens):
                pinyin.extend([0] * (len(tokens) - len(pinyin)))
            phrases.append(tokens)
            pinyin_rows.append(pinyin[: len(tokens)])
            valid_token_ids.extend(tokens)

        if not phrases:
            empty = torch.empty(0, self.projection_size, dtype=dtype, device=device)
            return empty, torch.empty(0, dtype=torch.long, device=device), empty, []

        max_length = max(len(phrase) for phrase in phrases)
        phrase_tensor = torch.zeros(len(phrases), max_length, dtype=torch.long, device=device)
        pinyin_tensor = torch.zeros_like(phrase_tensor)
        padding_mask = torch.ones(len(phrases), max_length, dtype=torch.bool, device=device)
        for index, (phrase, pinyin) in enumerate(zip(phrases, pinyin_rows)):
            length = len(phrase)
            phrase_tensor[index, :length] = torch.tensor(phrase, dtype=torch.long, device=device)
            pinyin_tensor[index, :length] = torch.tensor(pinyin, dtype=torch.long, device=device).clamp(
                0, self.pinyin_buckets - 1
            )
            padding_mask[index, :length] = False

        token_vectors = ctc_token_weights.index_select(0, phrase_tensor.reshape(-1)).reshape(
            len(phrases), max_length, self.hidden_size
        ).to(dtype)
        positions = torch.arange(max_length, device=device).unsqueeze(0)
        phrase_hidden = (
            self.token_projection(token_vectors)
            + self.position_embedding(positions)
            + self.pinyin_embedding(pinyin_tensor)
        )
        phrase_hidden = self.phrase_encoder(phrase_hidden, src_key_padding_mask=padding_mask)

        memory_rows = []
        spans = []
        offset = 0
        for index, phrase in enumerate(phrases):
            length = len(phrase)
            memory_rows.append(phrase_hidden[index, :length])
            spans.append((offset, offset + length))
            offset += length
        memory = torch.cat(memory_rows, dim=0)
        token_ids = torch.tensor(valid_token_ids, dtype=torch.long, device=device)
        return memory, token_ids, phrase_hidden, spans

    def forward(
        self,
        encoder_hidden: torch.Tensor,
        base_log_probs: torch.Tensor,
        ctc_token_weights: torch.Tensor,
        context_phrases: Sequence[Sequence[int]],
        context_confidences: torch.Tensor | Sequence[float] | None = None,
        context_pinyin_ids: Sequence[Sequence[int]] | None = None,
        ctc_token_bias: torch.Tensor | None = None,
        return_auxiliary: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if encoder_hidden.dim() != 3 or base_log_probs.dim() != 3:
            raise ValueError("encoder_hidden and base_log_probs must be [batch,time,dim]")
        if encoder_hidden.shape[:2] != base_log_probs.shape[:2]:
            raise ValueError("encoder_hidden and base_log_probs time dimensions differ")

        memory, token_ids, _, phrase_spans = self._encode_phrases(
            ctc_token_weights,
            context_phrases,
            context_pinyin_ids,
            encoder_hidden.dtype,
            encoder_hidden.device,
        )
        if token_ids.numel() == 0:
            if return_auxiliary:
                frame_logits = encoder_hidden.new_empty((*encoder_hidden.shape[:2], 0))
                phrase_logits = encoder_hidden.new_empty((encoder_hidden.size(0), 0))
                return base_log_probs, frame_logits, phrase_logits
            return base_log_probs

        if context_confidences is None:
            phrase_confidences = encoder_hidden.new_ones(len(phrase_spans))
        else:
            phrase_confidences = torch.as_tensor(
                context_confidences,
                dtype=encoder_hidden.dtype,
                device=encoder_hidden.device,
            )[: len(phrase_spans)].clamp(0.0, 1.0)
        if phrase_confidences.numel() != len(phrase_spans):
            raise ValueError("one confidence is required for each context phrase")

        token_confidences = torch.cat(
            [phrase_confidences[index].expand(end - start) for index, (start, end) in enumerate(phrase_spans)]
        )
        audio_queries = self.audio_projection(encoder_hidden)
        attended, _ = self.cross_attention(
            audio_queries,
            memory.unsqueeze(0).expand(encoder_hidden.size(0), -1, -1),
            (memory * token_confidences.unsqueeze(-1)).unsqueeze(0).expand(encoder_hidden.size(0), -1, -1),
            need_weights=False,
        )
        hidden_scale = F.softplus(self.log_hidden_scale)
        adapted_hidden = encoder_hidden + hidden_scale * self.context_output(attended)
        adapted_logits = F.linear(adapted_hidden, ctc_token_weights, ctc_token_bias)

        normalized_queries = F.normalize(audio_queries, dim=-1)
        normalized_memory = F.normalize(memory, dim=-1)
        temperature = self.log_temperature.exp().clamp(1.0, 100.0)
        token_alignment_logits = (
            torch.matmul(normalized_queries, normalized_memory.transpose(0, 1))
            - self.similarity_offset
        ) * temperature
        if self.monotonic_phrase_activation:
            frame_phrase_logits = self._monotonic_phrase_logits(token_alignment_logits, phrase_spans)
            phrase_activation = torch.sigmoid(frame_phrase_logits)
            attended = attended * phrase_activation.amax(dim=-1, keepdim=True)
            adapted_hidden = encoder_hidden + hidden_scale * self.context_output(attended)
            adapted_logits = F.linear(adapted_hidden, ctc_token_weights, ctc_token_bias)
            token_phrase_activation = torch.cat(
                [
                    phrase_activation[:, :, phrase_index:phrase_index + 1].expand(-1, -1, end - start)
                    for phrase_index, (start, end) in enumerate(phrase_spans)
                ],
                dim=-1,
            )
        else:
            frame_phrase_logits = torch.stack(
                [token_alignment_logits[:, :, start:end].amax(dim=-1) for start, end in phrase_spans],
                dim=-1,
            )
            token_phrase_activation = 1.0
        token_residual = (
            F.softplus(self.log_token_scale)
            * torch.sigmoid(token_alignment_logits)
            * token_confidences.view(1, 1, -1)
            * token_phrase_activation
        )
        adapted_logits.index_add_(2, token_ids, token_residual)
        adapted_log_probs = F.log_softmax(adapted_logits, dim=-1)

        if return_auxiliary:
            phrase_logits = torch.logsumexp(frame_phrase_logits, dim=1) - math.log(
                max(1, frame_phrase_logits.size(1))
            )
            return adapted_log_probs, frame_phrase_logits, phrase_logits
        return adapted_log_probs


def load_context_adapter(path: str, map_location: str | torch.device = "cpu") -> nn.Module:
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    adapter_type = checkpoint.get("adapter_type", "frame") if isinstance(checkpoint, dict) else "frame"
    if adapter_type == "phrase":
        adapter = SenseVoicePhraseContextAdapter(**config)
    else:
        adapter = SenseVoiceContextAdapter(**config)
    state = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    adapter.load_state_dict(state, strict=True)
    return adapter
