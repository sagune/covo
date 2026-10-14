"""Text-conditioned acoustic residual; no recognizer head or output repair."""
from dataclasses import asdict, dataclass
import math

import torch
from torch import nn


@dataclass(frozen=True)
class ConnectorConfig:
    hidden_size: int
    audio_size: int = 512
    width: int = 256
    heads: int = 4
    pool: int = 4
    version: str = "text_acoustic_residual_v1"


def pool_frames(frames, factor=4):
    if frames.ndim != 2 or len(frames) == 0 or factor < 1:
        raise ValueError("Expected nonempty [time, channels] audio and positive pool factor")
    if not torch.isfinite(frames).all():
        raise ValueError("Nonfinite audio features")
    n = len(frames) // factor
    parts = [frames[:n * factor].reshape(n, factor, -1).mean(1)] if n else []
    if len(frames) % factor:
        parts.append(frames[n * factor:].mean(0, keepdim=True))
    return torch.cat(parts)


def positions(length, width, device):
    if width % 2:
        raise ValueError("Position width must be even")
    t = torch.arange(length, device=device, dtype=torch.float32)[:, None]
    freq = torch.exp(torch.arange(0, width, 2, device=device).float() * (-math.log(10000.) / width))
    result = torch.empty(length, width, device=device)
    result[:, 0::2] = torch.sin(t * freq)
    result[:, 1::2] = torch.cos(t * freq)
    return result


class TextAcousticResidual(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        d = config.width
        if d % config.heads or d % 2:
            raise ValueError("width must be even and divisible by heads")
        self.text_projection = nn.Sequential(nn.LayerNorm(config.hidden_size), nn.Linear(config.hidden_size, d))
        self.context = nn.TransformerEncoderLayer(d, config.heads, 2 * d, dropout=0.,
                                                  batch_first=True, norm_first=True, activation="gelu")
        self.audio_projection = nn.Sequential(nn.LayerNorm(config.audio_size), nn.Linear(config.audio_size, d))
        self.cross_attention = nn.MultiheadAttention(d, config.heads, dropout=0., batch_first=True)
        self.output_norm = nn.LayerNorm(d)
        self.output = nn.Linear(d, config.hidden_size, bias=False)
        # First step trains this projection; upstream attention gradients appear after it moves.
        nn.init.zeros_(self.output.weight)

    def forward(self, text_embeddings, frames):
        """Return a delta, not replacement embeddings. No target/reference is accepted."""
        c = self.config
        if text_embeddings.ndim != 2 or not len(text_embeddings) or text_embeddings.shape[1] != c.hidden_size:
            raise ValueError("Expected nonempty [top1_tokens, hidden_size]")
        if frames.ndim != 2 or frames.shape[1] != c.audio_size:
            raise ValueError("Wrong audio feature dimension")
        if not torch.isfinite(text_embeddings).all():
            raise ValueError("Nonfinite text embeddings")
        audio = pool_frames(frames.float(), c.pool)
        q = self.text_projection(text_embeddings.float())
        q = self.context((q + positions(len(q), c.width, q.device)).unsqueeze(0))
        a = self.audio_projection(audio)
        a = (a + positions(len(a), c.width, a.device)).unsqueeze(0)
        value, _ = self.cross_attention(q, a, a, need_weights=False)
        delta = self.output(self.output_norm(value[0]))
        if not torch.isfinite(delta).all():
            raise FloatingPointError("Nonfinite acoustic residual")
        return delta.to(text_embeddings.dtype)

    def configuration(self):
        return asdict(self.config)
