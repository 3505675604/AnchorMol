
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiModalFusionHead(nn.Module):
    """
    Fuse three modalities of features and output classification logits (fixed version).
    Supports mean, concatenation MLP, and attention-based fusion.
    """
    def __init__(self, input_dim: int = 512, fusion_type: str = 'concat_mlp', dropout_rate: float = 0.2):
        super().__init__()
        self.fusion_type = fusion_type

        if fusion_type == 'mean':
            # Mean fusion: average the three feature vectors
            self.classifier = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.ReLU(),  # Not using inplace
                nn.Dropout(dropout_rate),
                nn.Linear(256, 128),
                nn.ReLU(),  # Not using inplace
                nn.Dropout(dropout_rate),
                nn.Linear(128, 1)
            )
        elif fusion_type == 'concat_mlp':
            # Concatenate all three features and use an MLP
            self.classifier = nn.Sequential(
                nn.Linear(input_dim * 3, 768),
                nn.ReLU(),  # Not using inplace
                nn.Dropout(dropout_rate * 1.5),
                nn.Linear(768, 256),
                nn.ReLU(),  # Not using inplace
                nn.Dropout(dropout_rate),
                nn.Linear(256, 1)
            )
        elif fusion_type == 'attention':
            # Attention fusion: learn attention weights for each modality
            self.attention = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.Tanh(),
                nn.Linear(128, 1, bias=False)
            )
            self.classifier = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.ReLU(),  # Not using inplace
                nn.Dropout(dropout_rate),
                nn.Linear(256, 1)
            )

    def forward(self, z1: torch.Tensor, z2: torch.Tensor, z3: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for feature fusion.
        Args:
            z1, z2, z3: Feature tensors from three modalities, shape [B, input_dim]
        Returns:
            Logits tensor of shape [B]
        """
        # Use clone to avoid modifying the original features
        z1 = z1.clone()
        z2 = z2.clone()
        z3 = z3.clone()

        if self.fusion_type == 'mean':
            z_fused = (z1 + z2 + z3) / 3
            logits = self.classifier(z_fused)
        elif self.fusion_type == 'concat_mlp':
            z_fused = torch.cat([z1, z2, z3], dim=1)
            logits = self.classifier(z_fused)
        elif self.fusion_type == 'attention':
            # Stack features and apply attention mechanism
            features = torch.stack([z1, z2, z3], dim=1)
            attn_scores = self.attention(features)
            attn_weights = F.softmax(attn_scores, dim=1)
            z_fused = torch.sum(features * attn_weights, dim=1)
            logits = self.classifier(z_fused)

        return logits.squeeze(-1)
