import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.pointwise_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.depthwise_conv = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=out_channels,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.pointwise_conv(x))
        x = F.relu(self.depthwise_conv(x))
        return x


class StackedDepthwiseSeparableConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_layers: int,
        out_channels: int,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            DepthwiseSeparableConvBlock(
                in_channels if i == 0 else out_channels,
                out_channels,
                kernel_size,
            )
            for i in range(num_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class TransformerEncoderBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        feedforward_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout,
        )
        self.feedforward = nn.Sequential(
            nn.Linear(embed_dim, feedforward_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, embed_dim),
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        attention_output, _ = self.self_attention(
            x,
            x,
            x,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = self.norm1(x + self.dropout(attention_output))
        feedforward_output = self.feedforward(x)
        x = self.norm2(x + self.dropout(feedforward_output))
        return x


class HybridTransformerEncoder(nn.Module):
    def __init__(
        self,
        num_transformer_layers: int,
        embed_dim: int,
        num_heads: int,
        feedforward_dim: int,
        in_channels: int,
        height: int,
        width: int,
        num_conv_layers: int,
        reduced_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.reduced_dim = reduced_dim
        self.height = height
        self.width = width

        self.feature_extractor = StackedDepthwiseSeparableConv(
            in_channels=in_channels,
            num_layers=num_conv_layers,
            out_channels=reduced_dim,
        )
        self.feature_bn = nn.BatchNorm2d(reduced_dim)
        self.position_embedding = nn.Embedding(height * width, embed_dim)

        self.transformer_layers = nn.ModuleList([
            TransformerEncoderBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                feedforward_dim=feedforward_dim,
                dropout=dropout,
            )
            for _ in range(num_transformer_layers)
        ])

    def forward(
        self,
        x: torch.Tensor,
        position_encoding: torch.Tensor,
        key_padding_mask: torch.Tensor = None,
        use_position_encoding: bool = True,
    ) -> torch.Tensor:
        batch_size, _, _, _ = x.shape

        x = self.feature_extractor(x)
        x = self.feature_bn(x)
        x = x.view(batch_size, self.reduced_dim, -1).permute(0, 2, 1)

        if use_position_encoding:
            position_ids = position_encoding.view(batch_size, -1).long()
            position_embedding = self.position_embedding(position_ids)
            x = x + position_embedding

        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.view(batch_size, -1)

        for layer in self.transformer_layers:
            x = layer(x, key_padding_mask=key_padding_mask)

        return x


class PretrainingHead(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        height: int,
        width: int,
        out_channels: int = 1,
    ):
        super().__init__()
        self.height = height
        self.width = width
        self.classifier = nn.Conv2d(embed_dim, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        x = x.permute(0, 2, 1).view(batch_size, -1, self.height, self.width)
        x = self.classifier(x).permute(0, 2, 3, 1).squeeze(-1)
        return x


class SupervisedClassificationHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        height: int,
        width: int,
        num_classes: int = 5,
    ):
        super().__init__()
        self.height = height
        self.width = width

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc = nn.Linear(256 * (width // 2) * (height // 2), num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        x = x.permute(0, 2, 1).view(batch_size, -1, self.height, self.width)
        x = self.conv_block(x)
        x = x.reshape(batch_size, -1)
        return self.fc(x)


class CrownBERT(nn.Module):
    def __init__(
        self,
        num_transformer_layers: int,
        embed_dim: int,
        num_heads: int,
        feedforward_dim: int,
        in_channels: int,
        height: int,
        width: int,
        num_conv_layers: int,
        reduced_dim: int,
        out_channels: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = HybridTransformerEncoder(
            num_transformer_layers=num_transformer_layers,
            embed_dim=embed_dim,
            num_heads=num_heads,
            feedforward_dim=feedforward_dim,
            in_channels=in_channels,
            height=height,
            width=width,
            num_conv_layers=num_conv_layers,
            reduced_dim=reduced_dim,
            dropout=dropout,
        )
        self.pretraining_head = PretrainingHead(
            embed_dim=embed_dim,
            height=height,
            width=width,
            out_channels=out_channels,
        )

    def forward(
        self,
        x: torch.Tensor,
        position_encoding: torch.Tensor,
        key_padding_mask: torch.Tensor = None,
        use_position_encoding: bool = True,
    ) -> torch.Tensor:
        x = self.encoder(
            x=x,
            position_encoding=position_encoding,
            key_padding_mask=key_padding_mask,
            use_position_encoding=use_position_encoding,
        )
        return self.pretraining_head(x)
