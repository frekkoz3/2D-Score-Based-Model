from common.utils import perturb

def denoising_score_matching_loss(
    model,
    x,
    sigma,
    weighted=True
):

    x_noisy, noise = perturb(x, sigma)

    score = model(x_noisy, sigma)

    target = -noise / sigma[:, None]

    error = score - target

    loss = error.square().sum(dim=-1)

    if weighted:
        loss = loss * sigma.square()

    loss = loss.mean()

    return loss