import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    """Two-layer MLP projection head with BatchNorm."""

    def __init__(self, input_dim, hidden_dim=1024, output_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


class ContrastiveModel(nn.Module):
    """Two-tower contrastive model for drug-target alignment.

    Args:
        drug_dim: input dimension of drug embeddings (e.g. 384 for ChemBERTa)
        target_dim: input dimension of target embeddings (e.g. 1280 for ESM-2 650M)
        hidden_dim: hidden dimension of projection heads
        output_dim: output dimension of projection heads
    """

    def __init__(self, drug_dim, target_dim, hidden_dim=1024, output_dim=512):
        super().__init__()
        self.drug_projector = ProjectionHead(drug_dim, hidden_dim, output_dim)
        self.target_projector = ProjectionHead(target_dim, hidden_dim, output_dim)

    def forward(self, drug_embeds, target_embeds):
        drug_proj = self.drug_projector(drug_embeds)
        target_proj = self.target_projector(target_embeds)
        return drug_proj, target_proj
