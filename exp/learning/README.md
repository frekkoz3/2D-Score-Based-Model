# Learning the Score of a 2D Distribution

This experiment is the first step towards understanding **score-based generative models**.

The goal is intentionally simple:

> Can a neural network learn the gradient of the logarithm of a probability density?

To answer this question, we consider simple **2D probability distributions**, whose score function can be visualized directly.

---

## The Score Function

Let

$$
p(x)
$$

be a probability density.

The **score** of the distribution is defined as

$$
\nabla_x \log p(x).
$$

Intuitively, the score is a vector field that, for every point \(x\), points towards regions of higher probability density.

For example, for a standard Gaussian

$$
p(x)=\mathcal N(0,I),
$$

the score has the simple analytical expression

$$
\nabla_x\log p(x)
=
-x.
$$

Every vector points towards the center of the distribution.

---

## Why Learn the Score?

Learning the probability density directly is difficult.

The density

$$
p(x)
$$

must be normalized, which is generally intractable in high dimensions.

The score,

$$
\nabla\log p(x),
$$

does not depend on the normalization constant.

This makes it much easier to estimate.

Moreover, once the score is known, it can be used to generate new samples through **Langevin Dynamics**, which will be explored in the next experiment.

---

## Denoising Score Matching

Unfortunately, the true score is unknown for real datasets.

Instead, we rely on **Denoising Score Matching (DSM)**.

Given a clean sample

$$
x,
$$

we corrupt it with Gaussian noise

$$
\tilde x
=
x+\sigma\varepsilon,
\qquad
\varepsilon\sim\mathcal N(0,I).
$$

The neural network receives the noisy sample

$$
(\tilde x,\sigma)
$$

and predicts the score of the noisy distribution.

The remarkable result of Vincent (2011) shows that the optimal target is

$$
-\frac{\varepsilon}{\sigma}.
$$

Therefore, the training objective becomes

$$
\mathcal L
=
\mathbb E
\left[
\left\|
s_\theta(\tilde x,\sigma)
+
\frac{\varepsilon}{\sigma}
\right\|^2
\right].
$$

In practice, we use the weighted version introduced by Song & Ermon (2019),

$$
\mathcal L
=
\mathbb E
\left[
\sigma^2
\left\|
s_\theta(\tilde x,\sigma)
+
\frac{\varepsilon}{\sigma}
\right\|^2
\right],
$$

which balances the contribution of different noise levels.

---

## Noise Conditioning

Instead of learning a single score field, the network learns a family of score fields,

$$
s_\theta(x,\sigma),
$$

one for each noise level.

Large values of $\sigma$ correspond to heavily smoothed distributions, while small values recover increasingly fine details.

This multi-scale representation is what later enables **Annealed Langevin Dynamics**.

---

## Model

The score network is a simple Multi-Layer Perceptron (MLP).

The inputs are

- the noisy point $x$,
- the noise level $\sigma$.

Rather than feeding $\sigma$ directly to the network, it is first transformed through a **sinusoidal embedding**, exactly as done in modern diffusion models.

The embedded noise level is then concatenated with the input coordinates before being processed by the MLP.

---

## Training Procedure

Each optimization step follows the same pipeline:

1. Sample points from the target distribution.
2. Sample a noise level from a geometric noise schedule.
3. Corrupt the samples with Gaussian noise.
4. Predict the score using the neural network.
5. Compute the Denoising Score Matching loss.
6. Update the model parameters using Adam.

---

## Visualizations

This experiment produces several visualizations.

---

### Data Distribution

Scatter plot of the training samples.

---

### Ground Truth Score

The analytical score field of the distribution.

This is available only for distributions whose density is known analytically (e.g. Gaussian distributions).

---

### Learned Score

The vector field predicted by the neural network after training.

---

### Score Comparison

By comparing the learned score field with the analytical one, we can directly evaluate whether the network has successfully learned the geometry of the probability distribution.
