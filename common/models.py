import torch
import torch.nn as nn

class MLPBackbone(nn.Module):

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        output_dim: int = 2,
        n_hidden: int = 3,
    ):
        super().__init__()

        layers = []

        in_dim = input_dim

        for _ in range(n_hidden):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.SiLU())
            in_dim = hidden_dim

        layers.append(nn.Linear(hidden_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class ScoreMLP(nn.Module):
    """
    Learns

        s(x)

    where x ∈ R².
    """

    def __init__(
        self,
        hidden_dim=128,
        n_hidden=3,
    ):
        super().__init__()

        self.backbone = MLPBackbone(
            input_dim=2,
            hidden_dim=hidden_dim,
            output_dim=2,
            n_hidden=n_hidden,
        )

    def forward(self, x):
        return self.backbone(x)

class SigmaEmbedding(nn.Module):
    """
    Small MLP embedding for the noise level sigma.
    """

    def __init__(
        self,
        embedding_dim=16,
    ):
        super().__init__()

        self.embedding = nn.Sequential(
            nn.Linear(1, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def forward(self, sigma):

        if sigma.ndim == 0:
            sigma = sigma.unsqueeze(0)

        if sigma.ndim == 1:
            sigma = sigma.unsqueeze(-1)

        return self.embedding(sigma)

class NoiseConditionalScoreMLP(nn.Module):

    def __init__(
        self,
        hidden_dim=128,
        sigma_embedding_dim=16,
        n_hidden=3,
    ):
        super().__init__()

        self.sigma_embedding = SigmaEmbedding(
            sigma_embedding_dim
        )

        self.backbone = MLPBackbone(
            input_dim=2 + sigma_embedding_dim,
            hidden_dim=hidden_dim,
            output_dim=2,
            n_hidden=n_hidden,
        )

    def forward(
        self,
        x,
        sigma,
    ):

        sigma = self.sigma_embedding(sigma)

        x = torch.cat(
            [
                x,
                sigma,
            ],
            dim=-1,
        )

        return self.backbone(x)

class ConditionalScoreMLP(nn.Module):

    def __init__(
        self,
        n_classes,
        hidden_dim=128,
        sigma_embedding_dim=16,
        label_embedding_dim=16,
        n_hidden=3,
    ):
        super().__init__()

        self.sigma_embedding = SigmaEmbedding(
            sigma_embedding_dim
        )

        self.label_embedding = nn.Embedding(
            n_classes,
            label_embedding_dim,
        )

        self.backbone = MLPBackbone(
            input_dim=(
                2
                + sigma_embedding_dim
                + label_embedding_dim
            ),
            hidden_dim=hidden_dim,
            output_dim=2,
            n_hidden=n_hidden,
        )

    def forward(
        self,
        x,
        sigma,
        labels,
    ):

        sigma = self.sigma_embedding(sigma)

        labels = self.label_embedding(labels)

        x = torch.cat(
            [
                x,
                sigma,
                labels,
            ],
            dim=-1,
        )

        return self.backbone(x)