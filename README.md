# 2D Score-Based Model

This repository contains a collection of small, educational experiments aimed at understanding Score-Based Generative Models from first principles.

Rather than working with high-dimensional image datasets, all experiments are built on 2D probability distributions. Working in two dimensions allows us to visualize every component of the learning process: the data distribution, the learned score field, and the sampling trajectories.

Again, the goal of this repository is understanding, not providing a state of the art implementation.

## Experiments

There will be 3 main experiments in the repository:

1. Learning the Score
2. Unconditional Sampling
3. Conditional Sampling

Each experiment will have its own documentation, both theoretical and practical. Here we just introduce them.

### 1. Learning the Score

The goal is to learn the score function of a 2D probability distribution.

In this experiment we train a neural network using Denoising Score Matching to approximate:

$$
\nabla_x \log p(x).
$$

The learned vector field is then visualized over the 2D space and compared with the underlying data distribution.

### 2. Unconditional Sampling

The goal is to sample using the learned score.

Starting from random Gaussian noise, we iteratively move samples according to the learned score using Langevin Dynamics.

This experiment illustrates how following the score field gradually transforms noise into samples from the target distribution.

### 3. Conditional Sampling

The goal is to sample conditioned on additional informations.

This experiment investigates how conditioning modifies the score field and guides the sampling process toward specific regions of the probability distribution.

We will use the simplest conditioning strategy: labelling.

## Repository Structure

```bash
├── README.md
├── requirements.txt
├── common/
    ├── datasets.py             # probability distributions
    ├── models.py               # score networks
    ├── noise.py                # forward corruption process
    ├── losses.py               # DSM losses
    ├── sampling.py             # Langevin and utilities
    ├── utils.py                # Utilities
    └── visualization.py        # plotting helpers
├── exp/
    ├── learning/
        ├── README.md
        ├── train.py
        └── visualize.py
    ├── unconditional_sampling/
        ├── README.md
        └── sample.py
    └── conditional_sampling/
        ├── README.md
        └── sample.py
└── resources/                  
```

## Requirements

The repository is implemented in `Python 3.11`. Follow the simple instructions below in order to have everything setup to run the experiments.

```bash
# Window version
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

```bash
# Linux version
py -3.11 -m venv .venv
source .venv\bin\Activate
pip install -r requirements.txt
```

## References

Even if all the experiment will be explored and described on the theoretical side, we suggest also the reading of the following papers:

* Song & Ermon (2019), Generative Modeling by Estimating Gradients of the Data Distribution
* Song et al. (2021), Score-Based Generative Modeling through Stochastic Differential Equations
* Hyvärinen (2005), Estimation of Non-Normalized Statistical Models by Score Matching
