import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    """Three-layer MLP projection head with BatchNorm and Dropout."""

    def __init__(self, input_dim=512, hidden_dim=1024, output_dim=512, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


class ContrastiveModel(nn.Module):
    """Unified structure-image contrastive model.

    Both modalities share the same input dimension (512-dim projected embeddings
    from Stage 1). Separate projection heads align structure and image spaces.

    Args:
        input_dim:  input dimension for both projectors (default 512)
        hidden_dim: hidden dimension of projection heads
        output_dim: output dimension of projection heads
        dropout:    dropout rate in projection heads
    """

    def __init__(self, input_dim=512, hidden_dim=1024, output_dim=512, dropout=0.1):
        super().__init__()
        self.struct_projector = ProjectionHead(input_dim, hidden_dim, output_dim, dropout)
        self.img_projector = ProjectionHead(input_dim, hidden_dim, output_dim, dropout)

    def forward(self, struct_embeds, img_embeds):
        struct_proj = self.struct_projector(struct_embeds)
        img_proj = self.img_projector(img_embeds)
        return struct_proj, img_proj
