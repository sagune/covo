"""Trainable frame-level contextual biasing for SenseVoice CTC logits."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn


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


def load_context_adapter(path: str, map_location: str | torch.device = "cpu") -> SenseVoiceContextAdapter:
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    adapter = SenseVoiceContextAdapter(**config)
    state = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    adapter.load_state_dict(state, strict=True)
    return adapter
