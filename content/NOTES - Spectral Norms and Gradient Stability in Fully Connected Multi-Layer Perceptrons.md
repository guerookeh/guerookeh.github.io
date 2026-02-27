---
draft: true
---
Many practitioners face such troubles when choosing to reinvent the wheel and construct their own neural networks while ignoring professors' advice on picking pre-trained, carefully constructed models available in your choice of deep learning library. As a professor of mine I recall described it: "there's already enough graduate students doing graduate student descent to get you the best-performing, generalizable, off-the-shelf models". 

Frustration eventually leads to resorting on brute-force approaches to find a model with good performance on a given dataset, which involves varying persistently across models with differing inherent assumptions about the data being trained on, different shuffles of training-validation-testing splits of the data, different opinionated model architectures, so on and so forth. It only goes to show that there are many intricate variables involved in the construction and training of machine learning models. It is a "black-box" at the end of the day (somewhat). 

This is precisely because we are met by a combinatorial explosion of controllable knobs. Data splits, preprocessing, architectures, initializations, activations, optimizers, learning rates, regularizations, training protocols, etc. In that vast sea of configurations, it is nontrivial to identify a single mechanism behind unstable training

Nevertheless, one thing we can control is finding suitable, mathematical proofs about how exactly our models can behave. 

- “In an information-geometry sense, contraction/ill-conditioning reduces the effective Fisher information (or curvature) seen by early-layer parameters, slowing learning.”

***

#### Notation Preamble

In the schematic chain-rule products below, we suppress Jacobian/transpose bookkeeping for readability. Since layout conventions vary across texts, we state the convention we use on the specific affine map
$$
a_l = W_l z_{l-1} + b_l,
\qquad
a_l\in\mathbb{R}^{d_l},\; z_{l-1}\in\mathbb{R}^{d_{l-1}},\; W_l\in\mathbb{R}^{d_l\times d_{l-1}},\; b_l\in\mathbb{R}^{d_l}.
$$
The corresponding Jacobian with respect to $z_{l-1}$ is
$$
\frac{\partial a_l}{\partial z_{l-1}} = W_l \in \mathbb{R}^{d_l\times d_{l-1}}.
$$

**Column-gradient convention.** Throughout, we interpret $\frac{\partial \ell}{\partial z_{l-1}}$ and $\frac{\partial \ell}{\partial a_l}$ as column vectors. Under this convention, backpropagation propagates gradients by multiplying by the transpose of the local Jacobian:
$$
\frac{\partial \ell}{\partial z_{l-1}}
=
\left(\frac{\partial a_l}{\partial z_{l-1}}\right)^{T}\frac{\partial \ell}{\partial a_l}
=
W_l^{T}\frac{\partial \ell}{\partial a_l},
$$
where $\frac{\partial \ell}{\partial a_l}\in\mathbb{R}^{d_l}$ and therefore $\frac{\partial \ell}{\partial z_{l-1}}\in\mathbb{R}^{d_{l-1}}$.

**Row-gradient convention.** Alternatively, one can interpret $\frac{\partial \ell}{\partial a_l}$ and $\frac{\partial \ell}{\partial z_{l-1}}$ as row vectors, in which case the same propagation step can be written without an explicit transpose:
$$
\frac{\partial \ell}{\partial z_{l-1}}
=
\frac{\partial \ell}{\partial a_l}\,\frac{\partial a_l}{\partial z_{l-1}}
=
\frac{\partial \ell}{\partial a_l}\,W_l,
$$
where $\frac{\partial \ell}{\partial a_l}\in\mathbb{R}^{1\times d_l}$ and $\frac{\partial \ell}{\partial z_{l-1}}\in\mathbb{R}^{1\times d_{l-1}}$.

In either case, the numerical gradient information is the same; only the layout (row vs. column) and the transpose bookkeeping differ.


























































