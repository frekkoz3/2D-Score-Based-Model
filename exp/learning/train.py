import torch

import exp.learning.config as config

from common.datasets import Gaussian2D, GaussianMixture2D
from common.models import NoiseConditionalScoreMLP
from common.noise import GeometricNoiseSchedule
from common.losses import denoising_score_matching_loss

DEVICE = config.DEVICE


def main():

    dataset = GaussianMixture2D(
        means=config.GAUSSIAN_MEAN,
        std=config.GAUSSIAN_STD,
    )

    noise_schedule = GeometricNoiseSchedule(
        sigma_min=config.SIGMA_MIN,
        sigma_max=config.SIGMA_MAX,
        n_levels=config.N_LEVELS,
        device=DEVICE,
    )

    model = NoiseConditionalScoreMLP(
        hidden_dim=config.HIDDEN_DIM,
        sigma_embedding_dim=config.SIGMA_EMBEDDING_DIM,
        n_hidden=config.N_HIDDEN,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.LR,
    )

    print("Starting training...")

    model.train()

    for epoch in range(config.EPOCHS):

        x = dataset.sample(config.BATCH_SIZE).to(DEVICE)

        sigma = noise_schedule.sample(config.BATCH_SIZE)

        loss = denoising_score_matching_loss(
            model,
            x,
            sigma,
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        if epoch % 100 == 0:

            print(
                f"[{epoch:05d}/{config.EPOCHS}] "
                f"Loss: {loss.item():.6f}"
            )

    torch.save(
        model.state_dict(),
        f"{config.BASE_PATH}model.pt",
    )

    print("Training completed.")


if __name__ == "__main__":
    main()