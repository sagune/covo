import torch
import torch.nn as nn
from transformers import ResNetConfig, ResNetModel


class _SE1d(nn.Module):
    """Squeeze-and-Excitation for 1D temporal features."""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Conv1d(channels, hidden, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.fc(self.avg_pool(x))
        return x * scale


class _FreqAttentionPool(nn.Module):
    """
    Frequency attention pooling:
    learn frame-wise weights over frequency bins instead of uniform mean.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.proj = nn.Conv2d(channels, 1, kernel_size=1, bias=True)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, F, T]
        weights = self.softmax(self.proj(x))  # [B, 1, F, T]
        return (x * weights).sum(dim=2)  # [B, C, T]


class _TCResBlock(nn.Module):
    """TC-ResNet style temporal residual block: Conv1D(k=9) x2 + residual shortcut."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        dilation: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.conv1_3 = nn.Conv1d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=False
        )
        self.conv1_5 = nn.Conv1d(
            in_channels, out_channels, kernel_size=5, stride=stride, padding=2 * dilation, dilation=dilation, bias=False
        )
        self.conv1_9 = nn.Conv1d(
            in_channels, out_channels, kernel_size=9, stride=stride, padding=4 * dilation, dilation=dilation, bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2_3 = nn.Conv1d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=dilation, dilation=dilation, bias=False
        )
        self.conv2_5 = nn.Conv1d(
            out_channels, out_channels, kernel_size=5, stride=1, padding=2 * dilation, dilation=dilation, bias=False
        )
        self.conv2_9 = nn.Conv1d(
            out_channels, out_channels, kernel_size=9, stride=1, padding=4 * dilation, dilation=dilation, bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.se = _SE1d(out_channels, reduction=8)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(p=dropout)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        x = (self.conv1_3(x) + self.conv1_5(x) + self.conv1_9(x)) / 3.0
        x = self.bn1(x)
        x = self.relu(x)
        x = self.drop(x)
        x = (self.conv2_3(x) + self.conv2_5(x) + self.conv2_9(x)) / 3.0
        x = self.bn2(x)
        x = self.se(x)
        x = x + residual
        x = self.relu(x)
        return x


class _TCResNetBackbone(nn.Module):
    def __init__(self, in_channels: int, channels: int, num_blocks: int, dropout: float):
        super().__init__()
        # 2D stem extracts local tf patterns, then frequency is aggregated and temporal 1D residual blocks are applied.
        stem_channels = max(32, channels // 2)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, stem_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(stem_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(stem_channels, channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.freq_pool = _FreqAttentionPool(channels)
        blocks = []
        in_c = channels
        dilation_cycle = [1, 2, 4, 8]
        stage = 0
        for i in range(num_blocks):
            # Downsample a bit less aggressively than before, keep more temporal resolution.
            downsample = i > 0 and (i % 6 == 0)
            if downsample:
                stage += 1
            out_c = channels * min(2 ** stage, 4)
            stride = 2 if downsample else 1
            dilation = dilation_cycle[i % len(dilation_cycle)]
            blocks.append(
                _TCResBlock(
                    in_channels=in_c,
                    out_channels=out_c,
                    stride=stride,
                    dilation=dilation,
                    dropout=dropout,
                )
            )
            in_c = out_c
        self.blocks = nn.ModuleList(blocks)
        self.out_channels = in_c
        self.out_norm = nn.BatchNorm1d(self.out_channels)
        self.out_act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        # Frequency-aware aggregation, keeps discriminative phonetic cues better than plain mean.
        x = self.freq_pool(x)  # [B, C, T]
        for block in self.blocks:
            x = block(x)
        x = self.out_norm(x)
        x = self.out_act(x)
        x = x.unsqueeze(2)  # [B, C, 1, T]
        return x


class _SimpleConfig:
    """Minimal config shim for compatibility with existing code paths."""

    def __init__(self, num_channels: int, num_labels: int, hidden_size: int, backbone: str):
        self.num_channels = num_channels
        self.num_labels = num_labels
        self.hidden_sizes = [hidden_size]
        self.backbone = backbone


class Resnet(torch.nn.Module):
    def __init__(
        self,
        num_channels: int,
        num_classes: int,
        backbone: str = "resnet",
        tcresnet_channels: int = 64,
        tcresnet_blocks: int = 6,
        tcresnet_dropout: float = 0.1,
    ):
        super().__init__()
        self.backbone = str(backbone).lower()

        if self.backbone in {"resnet", "resnet50"}:
            self.config = ResNetConfig()
            self.config.num_channels = num_channels
            self.config.num_labels = num_classes
            self.feature_extractor = ResNetModel(self.config)
            self.feature_dim = int(self.config.hidden_sizes[-1])
        elif self.backbone in {"tcresnet", "tc-resnet"}:
            self.feature_extractor = _TCResNetBackbone(
                in_channels=num_channels,
                channels=int(tcresnet_channels),
                num_blocks=int(tcresnet_blocks),
                dropout=float(tcresnet_dropout),
            )
            self.feature_dim = int(self.feature_extractor.out_channels)
            self.config = _SimpleConfig(
                num_channels=num_channels,
                num_labels=num_classes,
                hidden_size=self.feature_dim,
                backbone="tcresnet",
            )
        else:
            raise ValueError(f"Unsupported KWS backbone: {backbone}")

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(in_features=self.feature_dim, out_features=self.config.num_labels, bias=True),
        )

    def extract_feature_map(self, input_features: torch.Tensor) -> torch.Tensor:
        if self.backbone in {"resnet", "resnet50"}:
            out = self.feature_extractor(pixel_values=input_features)
            feature_map = getattr(out, "last_hidden_state", None)
            if feature_map is None:
                feature_map = out[0]
            return feature_map
        return self.feature_extractor(input_features)

    def extract_features(self, input_features: torch.Tensor):
        feature_map = self.extract_feature_map(input_features)
        pooled = self.pool(feature_map)
        features = torch.flatten(pooled, start_dim=1, end_dim=-1)
        return features, feature_map

    def forward(self, input_features: torch.Tensor):
        features, _ = self.extract_features(input_features)
        logits = self.classifier(features)
        return logits
