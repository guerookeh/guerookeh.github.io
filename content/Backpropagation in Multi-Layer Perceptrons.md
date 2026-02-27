---
title: Backpropagation in Multi-Layer Perceptrons
showDate: true
showReadingTime: true
draft: false
tags:
  - "#deep-learning"
  - "#multi-layer-perceptrons"
  - neural-networks
  - dynamic-programming
---
**I explore and develop backpropagation for multi-layer perceptrons from first principles: model definition, empirical risk, forward/backward computation graphs, an induction-based derivation of the error-signal recursion, and the resulting algorithms and complexity; explicitly identifying the intermediates the backward pass depends on.**
***

We first start by fixing the notation for a multi-layer perceptron and the learning objective it is trained on. Next, we will describe the forward pass as a sequence of intermediate pre-activations and activations, view the network as a computation graph, and derive the backward pass by applying the chain rule in a consistent convention. This leads to the standard backpropagation recursion for the layer-wise error signals, from which the parameter gradients follow immediately. Finally, we translate the identities into explicit forward/backward (backpropagation) algorithms and discuss what must be cached (or recomputed) for efficiency, alongside their resulting time complexities. We finish off with some examples by substituting some common output layers and losses.

## Definition of a Multi-Layer Perceptron
***
A [multi-layer perceptron](https://en.wikipedia.org/wiki/Multilayer_perceptron) is a [feedforward](https://en.wikipedia.org/wiki/Feedforward_neural_network) [neural network](https://en.wikipedia.org/wiki/Neural_network_\(machine_learning\)) that consists of layers of fully connected neurons, where every neuron in one layer is connected to every neuron in the subsequent layer, with non-linear activation functions. They are also known as deep feedforward neural networks, since they contain multiple hidden layers between input and output; shallow networks usually have only a single hidden layer.

In general, multi-layer perceptrons can approximate complex, non-linear functions by composing affine transformations with non-linear activation functions across layers. In contrast to its [perceptron](https://en.wikipedia.org/wiki/Perceptron) predecessor, which is limited to learning a linear decision boundary in the input space, multi-layer perceptrons are able to represent non-linear decision regions by applying the previous two transformations repeatedly. 

### Affine Transformations and Why Biases (Might) Help

Before the formal definition, it is useful to understand that a layer has a linear and a shift component. An affine map is exactly that: a linear map plus a translation:
$$
T(x)=Ax+b
$$
where $A$ is the linear component and $b$ is a constant offset. The difference here from a purely linear map $G(x)=Ax$, is that setting $b=0$ forces $T(0)=0$ (i.e., the origin is fixed). When talking about neural networks, this shows up immediately: a single unit computes a pre-activation $a=w^Tx+b$, and a decision boundary $a=0$ is the hyperplane
$$
w^Tx+b=0
$$
Without a bias term, this becomes $w^Tx=0$, which is strictly constrained passing through the origin. Allowing $b\neq 0$ lets the hyperplane translate, removing an artificial geometric constraint and making the layer strictly more flexible.

### Nested Form Without Biases

Formally, a multi-layer perceptron is completely defined by the following mathematical function
$$
\hat{y}\coloneqq f_{\theta}(x)= \psi\!\Big(W_L\,\phi\!\big(W_{L-1}\,\phi\!\big(W_{L-2}\,\cdots\phi(W_1x)\cdots\big)\big)\Big),
$$
where $x\in\mathbb{R}^{d_0}$ is a $d_0$-dimensional input vector, $\hat{y}\in\mathbb{R}^{d_L}$ is a $d_L$-dimensional output vector, $\phi(\cdot)$ is the non-linear activation function, $\psi(\cdot)$ is the output function, and $\theta \coloneqq \{W_1, W_2, \dots, W_L\}$ is the collection of learnable model parameters, where the $l$-th layer of the network is parameterized by the weight matrix $W_l\in\mathbb{R}^{d_l\times d_{l-1}}$.

Let $d_l\in \mathbb{N}$ for all $l\in\{0,\dots, L\}$. The number of layers $L\in \mathbb{N}$ in our network is defined as its depth, and the width of layer $l$ is defined as the number of neurons in that layer, namely $d_l$. We refer to $l=0$ as the input layer, to layers $l\in\{1,\dots,L-1\}$ as the hidden layers, and to $l=L$ as the output layer. Note that the depth includes the output layer, but not the input layer (conventions vary; here depth means number of parameterized layers). Depth and hidden-layer widths are variable architecture hyperparameters, with the exception of $d_0$ and $d_L$ since these are fixed by the input and output dimensions of the approximation task.

### Nested Form With Biases

Although not strictly necessary in a multi-layer perceptron, bias parameters are typically included to allow each layer to implement an affine transformation, yielding,
$$
\hat{y}\coloneqq f_{\theta}(x)= \psi\!\Big(W_L\,\phi\!\big(W_{L-1}\,\phi\!\big(W_{L-2}\,\cdots\phi(W_1x+b_1)+b_2\cdots\big)+b_{L-1}\big)+b_L\Big),
$$
with the collection of learnable model parameters $\theta \coloneqq \{(W_l,b_l)\}_{l=1}^L= \{W_1,\dots,W_L,b_1,\dots,b_L\}$, where the $l$-th layer of the network is parameterized by the weight matrix $W_l\in\mathbb{R}^{d_l\times d_{l-1}}$ and the bias vector $b_l\in\mathbb{R}^{d_l}$.

### Layer-wise Recursive Form With Biases

Equivalently, this fully nested expression can be written in recursive (layer-wise) form. Let $z_0\coloneqq x$. For each layer $l\in[L]=\{1,\dots,L\}$, define the pre-activations and (hidden-layer) activations by
$$
a_l = W_l z_{l-1} + b_l,\qquad z_l = \phi(a_l)\quad\text{for }l=1,\dots,L-1,
$$
and define the network output via a (possibly different) output function $\psi$ as
$$
\hat{y}\coloneqq f_\theta(x)=\psi(a_L)=z_L.
$$
Here $\psi$ is task-dependent (e.g., $\psi=\mathrm{id}$ for regression or $\psi=\sigma$ for binary classification), and the learnable parameters are $\theta=\{(W_l,b_l)\}_{l=1}^L$. 

For the remainder of this post we will work with this layer-wise recursive form with biases and additionally, for the sake of simplicity, assume $\psi=\phi$, so that $\hat{y}=z_L=\phi(a_L)$.

![[MLPFig1.png]]
*Fig 1.  A multi-layer perceptron with input $x\in\mathbb{R}^2$ and scalar output $z_L=\hat{y}\in\mathbb{R}$. Each layer computes a pre-activation $a_l=W_lz_{l-1}+b_l$ (an affine transformation of the previous layer's activations) represented by $\Sigma$, followed by an element-wise non-linearity $z_l=\phi(a_l)$ represented by $\phi$. For example, with ReLU, $\phi(t)=\max(0, t)$, so $(z_l)_j=\max\left(0, (a_l)_j\right)$ for each neuron $j$.*

## Objective and Empirical Risk
***

We consider a supervised learning problem with input space $\mathcal{X}\subseteq\mathbb{R}^{d_0}$ and output space $\mathcal{Y}$, where $\mathcal{Y}$ depends on the approximation task (e.g., $\mathbb{R}^{d_L}$ for regression targets, a finite label set for classification, etc.) Let $\mathcal{P}$ denote the (unknown) joint data-generating distribution on $\mathcal{X}\times\mathcal{Y}$, and suppose we observe a dataset of $n$ samples
$$
\mathcal{D}=\{(x_i,y_i)\}_{i=1}^n \subseteq \mathcal{X}\times \mathcal{Y},
$$
drawn independently and identically distributed (i.i.d.) from $\mathcal{P}$. The i.i.d assumption is a strong assumption since, in practice, datasets often exhibit dependence or distribution shift over time or domains; but it still provides the right conceptual baseline for a brief but important visit to the idea of empirical risk.

### Population Risk

Given parameters $\theta$, the network $f_\theta: \mathcal{X}\rightarrow \mathbb{R}^{d_L}$ maps an input $x$ to a prediction $f_\theta(x)$. Let $\mathcal{L}:\mathbb{R}^{d_L}\times\mathcal{Y}\rightarrow \mathbb{R}$ be a per-sample loss that assigns a scalar cost to predicting $f_\theta(x)$ when the observed label is $y$. The **population risk** is
$$
\mathcal{R}(\theta)\coloneqq \mathbb{E}_{(X,Y)\sim \mathcal{P}}\left[ \mathcal{L}(f_\theta(X),Y) \right].
$$
Conceptually, $\mathcal{R}(\theta)$ is the model's average loss on samples drawn from the true joint distribution $\mathcal{P}$; equivalently, it averages the loss over all possible input-output pairs, weighted by their probability under $\mathcal{P}$.

### Empirical Risk

In practice, $\mathcal{P}$ is unknown, so $\mathcal{R}(\theta)$ cannot be evaluated exactly. So instead, we work with the empirical counterpart. The **empirical risk** is
$$
\hat{\mathcal{R}}_n(\theta)\coloneqq \frac{1}{n}\sum_{i=1}^n \mathcal{L}(f_\theta(x_i), y_i).
$$
A standard learning procedure chooses parameters by (approximately) minimizing $\hat{\mathcal{R}}_n(\theta)$,
$$
\hat{\theta}\in\operatorname{argmin}_\theta\hat{\mathcal{R}}_n(\theta),
$$
typically using a gradient-based optimizer (e.g., SGD or Adam variants, take your pick). Since these methods require gradients, the central computational problem is to efficiently compute $\frac{\partial\hat{\mathcal{R}}_n(\theta)} {\partial \theta}$. 

The connection to backpropagation can be made more explicit by defining the per-sample objective
$$
\ell_i(\theta)\coloneqq \mathcal{L}(f_\theta(x_i), y_i), \quad \text{s.t.} \space \space \hat{\mathcal{R}}_n(\theta)=\frac{1}{n}\sum_{i=1}^n\ell_i(\theta).
$$
Finally, if we differentiate this average we get
$$
\frac{\partial\hat{\mathcal{R}}_n(\theta)}{\partial \theta}=\frac{1}{n}\sum_{i=1}^n \frac{\partial \ell_i(\theta)}{\partial \theta}.
$$
Therefore, it suffices to derive $\frac{\partial \ell_i(\theta)}{\partial \theta}$ for a single example $(x,y)$; gradients for a mini-batch, as with mini-batched SGD, are obtained by averaging the per-sample gradients over the batch.

## Forward and Backward Pass
***

### Forward Pass

A forward pass in the multi-layer perceptron is defined as passing an input $x\in\mathbb{R}^{d_0}$ through the form above in order to compute $\hat{y}=z_L$, which is both the last layer's activation and the overall output of the network. For a given intermediate layer $l$, notice its computation of the activation $z_l$ gets passed on as an input to the next layer's computation of the pre-activation $a_{l+1}$. 
 
More explicitly, the forward pass computes and (conceptually) produces the sequence of intermediate quantities
$$
(x=z_0)\space\longrightarrow\space (a_1,z_1)\space\longrightarrow\space (a_2,z_2)\space\longrightarrow\space\dots\space\longrightarrow\space(a_L,\hat{y}),
$$
where each pair $(a_l,z_l)$ for a given layer $l$ is obtained via
$$
a_l=W_lz_{l-1}+b_l,\quad z_l=\phi(a_l)\space\space\space \text{for }l=1,\dots, L, \quad \hat{y}=z_L.
$$
We can visualize the forward pass more easily as a computation graph in Figure 2 below.

![[MLPFig2.png]]
*Fig 2.  Forward-pass computation graph based on the multi-layer perceptron from Fig 1. The forward pass computes the pre-activations and activations from left to right. Unlike the standard neuron-and-layer visualization, this view naturally extends to illustrating backpropagation gradients. Note that $a_l$'s computation is not entirely elementary as it simplifies both matrix multiplication and addition as a single operation, but the abstraction still works.*

### Backward Pass

For the backward pass, we fix a single example $(x,y)\sim\mathcal{P}$. We first perform a forward pass through the network with $x\in\mathbb{R}^{d_0}$ as shown above, producing the intermediate pre-activations and activations $\{(a_l,z_l)\}_{l=1}^L$ and yielding the final prediction $z_L=f_\theta(x)$. We then define the corresponding per-sample scalar loss as
$$
\ell(\theta)\coloneqq \mathcal{L}(z_L, y)
$$
The backward pass is the computation of the gradient of $\ell(\theta)$ with respect to all parameters $\theta=\{(W_l,b_l)\}_{l=1}^L$, written as $\frac{\partial \ell(\theta)}{\partial \theta}$. We can make this fully explicit by writing the collection of parameter gradients that the backward pass should return
$$
\frac{\partial \ell(\theta)}{\partial \theta} =\biggl\{ \frac{\partial\ell(\theta)}{\partial W_l}\in\mathbb{R}^{d_l\times d_{l-1}}, \space \frac{\partial \ell(\theta)}{\partial b_l}\in\mathbb{R}^{d_l} \biggr\}_{l=1}^L.
$$
Since the network is a composition of layer-wise maps of affine and non-linear transformations, $a_l$ and $z_l$ for a given layer $l$, respectively, these gradients are obtained by applying the chain rule through the computation graph defined by the transformations
$$
a_l=W_lz_{l-1}+b_l,\quad z_l=\phi(a_l)\space\space\space \text{for }l=1,\dots, L, \quad \hat{y}=z_L
$$
Recall the scalar chain rule: if $h(x)=f(g(x))$, a composition of two differentiable functions $f$ and $g$ for every $x$, then $h'(x)=f'(g(x))g'(x)$. In a multi-layer perceptron, the loss $\ell(\theta)=\mathcal{L}(z_L, y)$ depends on the parameters only through a nested composition of the above layer-wise maps. As a result, computing a gradient such as $\partial \ell/ \partial W_l$ amounts to repeatedly applying the chain rule through the sequence
$$
W_l\space\rightarrow\space a_l\space\rightarrow\space z_l\space\rightarrow\space a_{l+1}\space\rightarrow\space z_{l+1} \space\rightarrow\space \dots \space\rightarrow\space z_L \rightarrow \ell.
$$
A conceptually straightforward way to proceed is to expand the chain rule separately for each parameter matrix/vector. For instance,
$$
\frac{\partial\ell}{\partial W_l}=\frac{\partial \ell}{\partial z_L}\frac{\partial z_L}{\partial a_L}\frac{\partial a_L}{\partial z_{L-1}}\dots \frac{\partial z_l}{\partial a_l}\frac{\partial a_l}{\partial W_l}
$$
and similarly for $\partial \ell/\partial b_l$ (for now we suppress transpose/layout details; we fix a consistent convention later in [[#Column-Gradient Convention]]). We can visualize these repeated chain-rule expansions and the shared subcomputations they contain by annotating the forward computation graph (Fig 2.) with the backward-pass gradients, yielding Fig 3.

![[MLPFig3.png]]
*Fig 3.  Forward and backward pass computation graph based on the multi-layer perceptron from Fig 1. The forward pass computes pre-activations and activations from left to right. The backward pass propagates the gradient from $\ell$ back to earlier layers from right to left by associating each edge with its local Jacobian (e.g., $z_3\rightarrow a_4$'s local Jacobian is $\partial a_4/\partial z_3$).*

We reiterate that the goal of the backward computation is to assemble the parameter gradients $\frac{\partial \ell(\theta)}{\partial \theta}$. In the computation graph view (Fig 3.), each component gradient, such as $\partial \ell / \partial W_l$ or $\partial \ell / \partial b_l$, is obtained by propagating derivatives from $\ell$ back to the parameter node using local derivative factors on each edge, which we call local Jacobians, exactly as prescribed by the chain rule. Equivalently, these gradients can be written as products of local Jacobians along the relevant graph paths.

## Efficient Backward Passes via Backpropagation
***

It is worth contemplating Fig.3 because it makes the structure of the computation explicit: the forward pass induces a directed acyclic graph (DAG) of intermediate quantities, and the backward pass amounts to propagating derivatives backwards along this same DAG. If we were to expand the chain rule separately for each parameter matrix/vector, we would repeatedly recompute the same downstream derivative factors.

**This is exactly the kind of setting where a naïve recursive computation is wasteful.** For example, the derivative of $\ell$ with respect to an intermediate activation $z_l$ (or pre-activation $a_l$) appears in the gradients of all parameters in earlier layers. In other words, many chain rule expansions share large common subcomputations.

To make this redundancy explicit without writing the same long products repeatedly, it is helpful to isolate the derivative of the loss with respect to a layer's pre-activation. Define the layer-wise intermediate derivative, which we denote as the layer-wise *error signal*, as
$$
\delta_l \;\coloneqq\; \frac{\partial \ell}{\partial a_l}\in\mathbb{R}^{d_l}.
$$
This quantity captures *all* downstream dependence of the loss on layer $l$ through the remainder of the network. Once $\delta_l$ is known, the gradients with respect to the parameters at layer $l$ factor into a local piece that depends only on the affine map $a_l=W_l z_{l-1}+b_l$:
$$
\frac{\partial \ell}{\partial W_l}
=
\left(\frac{\partial a_l}{\partial W_l}\right)^{T}\delta_l,
\qquad
\frac{\partial \ell}{\partial b_l}
=
\left(\frac{\partial a_l}{\partial b_l}\right)^{T}\delta_l.
$$
Backpropagation is the standard algorithm that exploits this structure: it computes $\delta_L,\delta_{L-1},\dots,\delta_1$ in a single backward sweep and reuses them to form all parameter gradients, rather than recomputing the same downstream derivatives separately for each parameter. This reuse follows the same principle as dynamic programming: solve each subproblem once, cache its result, and assemble the final answer from these cached pieces.

### Deriving Backpropagation via Induction

We now derive the backward recursion for the layer-wise error signal
$$
\delta_l \;\coloneqq\; \frac{\partial \ell}{\partial a_l},
$$
and show how it produces the parameter gradients $\partial \ell/\partial W_l$ and $\partial \ell/\partial b_l$. The derivation is agnostic to the specific choice of loss $\mathcal{L}$ and activation function $\phi$, assuming they are differentiable (or differentiable almost everywhere, e.g., for ReLU at $x=0$, pick a subgradient).

#### Column-Gradient Convention 

> Earlier chain rule expansions were intentionally schematic and did not track transpose/layout conventions. From this point onward we use a single convention so that every expression is shape-consistent.

We use the **column-gradient convention**. We represent activations and error signals as column vectors
$$
z_l\in\mathbb{R}^{d_l}, \quad a_l\in\mathbb{R}^{d_l},\quad \delta_l=\frac{\partial \ell}{\partial a_l}\in\mathbb{R}^{d_l}.
$$
For any column vector $v\in\mathbb{R}^m$, the gradient $\frac{\partial \ell}{\partial v}$ is also a column vector in $\mathbb{R}^m$. 

For a map between vectors $u=g(v)$ with $u\in\mathbb{R}^m$, $v\in\mathbb{R}^n$, we write the Jacobian as
$$
\frac{\partial u}{\partial v}\in\mathbb{R}^{m\times n},\quad \left(\frac{\partial u}{\partial v}\right)_{ij}=\frac{\partial u_i}{\partial v_j}
$$
Then the chain rule takes the following form
$$
\frac{\partial \ell}{\partial v}=\left(\frac{\partial u}{\partial v}\right)^T\frac{\partial \ell}{\partial u}
$$
In what follows we only apply the convention to affine maps and element-wise nonlinearities. These two rules generate all backward equations in the induction. 

Applying it to the affine map $a_l=W_lz_{l-1}+b_l$, we have
$$
\frac{\partial a_l}{\partial z_{l-1}}=W_l\in\mathbb{R}^{d_l\times d_{l-1}},\space\frac{\partial\ell}{\partial a_l}\in\mathbb{R}^{d_l}\quad\Rightarrow \quad \frac{\partial \ell}{\partial z_{l-1}}=\left(\frac{\partial a_l}{\partial z_{l-1}}\right)^T\frac{\partial \ell}{\partial a_l}=W_l^T\frac{\partial \ell}{\partial a_l}
$$
Applied to the element-wise activation $z_l=\phi(a_l)$, the Jacobian is diagonal
$$
\frac{\partial z_l}{\partial a_l}=\text{diag}(\phi'(a_l))\quad\Rightarrow\quad \frac{\partial \ell}{\partial a_l}=\text{diag}(\phi'(a_l))\frac{\partial \ell}{\partial z_l}=\frac{\partial \ell}{\partial z_l}\odot\phi'(a_l).
$$
Finally, gradients with respect to the parameters keep the parameter's shape
$$
\frac{\partial \ell}{\partial W_l}\in\mathbb{R}^{d_l\times d_{l-1}},\quad\frac{\partial \ell}{\partial b_l}\in\mathbb{R}^{d_l}
$$

#### Claim

For each layer $l$, define the error signal
$$
\delta_l \;\coloneqq\; \frac{\partial \ell}{\partial a_l}\in\mathbb{R}^{d_l}.
$$
Then the output-layer value is
$$
\delta_L = \frac{\partial \mathcal{L}(z_L,y)}{\partial z_L}\odot \phi'(a_L),
$$
and for $l=L-1,\dots,1$ the error signals satisfy the recursion
$$
\delta_l = \left(W_{l+1}^{T}\delta_{l+1}\right)\odot \phi'(a_l).
$$
where $\odot$ denotes the element-wise (Hadamard) product. Moreover, the parameter gradients are
$$
\frac{\partial \ell}{\partial W_l}=\delta_l z_{l-1}^{T},
\qquad
\frac{\partial \ell}{\partial b_l}=\delta_l.
$$

#### Lemmas

**Lemma A (activation step):** Since $z_l=\phi(a_l)$ is an element-wise operation, the chain rule gives
$$
\frac{\partial \ell}{\partial a_l}=\left(\frac{\partial z_l}{\partial a_l}\right)
^T\frac{\partial \ell}{\partial z_l}=\frac{\partial \ell}{\partial z_l}\odot \phi'(a_l)
$$
**Proof:** For $z_l=\phi(a_l)$ applied element-wise, each coordinate satisfies
$$
(z_l)_j=\phi((a_l)_j)\space \Rightarrow \space \frac{\partial (z_l)_j}{\partial (a_l)_j}=\phi'((a_l)_j).
$$
And since $(z_l)_j$ depends only on $(a_l)_j$, then for $i\neq j$,
$$
\frac{\partial (z_l)_i}{\partial (a_l)_j}=0
$$
So there is no mixing across coordinates. Additionally, this means that $\partial z_l / \partial a_l$ is a Jacobian
$$
J_\phi(a_l)
\;=\;
\frac{\partial z_l}{\partial a_l}
\;=\;
\operatorname{diag}\!\big(\phi'(a_l)\big)
\;=\;
\begin{bmatrix}
\phi'\!\big((a_l)_1\big) & 0 & \cdots & 0 \\
0 & \phi'\!\big((a_l)_2\big) & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & \phi'\!\big((a_l)_{d_l}\big)
\end{bmatrix}
$$
a diagonal matrix whose diagonal entries are $\phi'\left((a_l)_j\right)$. So multiplying by a diagonal Jacobian is equivalent to an element-wise product
$$
\begin{aligned}
\frac{\partial \ell}{\partial a_l}
&=
\left(\frac{\partial z_l}{\partial a_l}\right)^{T}\frac{\partial \ell}{\partial z_l}
&&\text{(chain rule; column-gradient convention)}\\
&=
\operatorname{diag}\!\big(\phi'(a_l)\big)\,\frac{\partial \ell}{\partial z_l}
&&\text{(since $\frac{\partial z_l}{\partial a_l}=\operatorname{diag}(\phi'(a_l))$)}\\
&=
\frac{\partial \ell}{\partial z_l}\odot \phi'(a_l)
&&\text{(diagonal multiplication $\Leftrightarrow$ Hadamard product).}
\end{aligned}
$$

**Lemma B (affine step):** Since $a_{l+1}=W_{l+1}z_l +b_{l+1}$, then
$$
\begin{aligned}
\frac{\partial \ell}{\partial z_l}
&=
\left(\frac{\partial a_{l+1}}{\partial z_l}\right)^{T}\frac{\partial \ell}{\partial a_{l+1}}
&&\text{(chain rule; column-gradient convention)}\\
&=
W_{l+1}^{T}\frac{\partial \ell}{\partial a_{l+1}}
&&\text{(since $\frac{\partial a_{l+1}}{\partial z_l}=W_{l+1}$).}
\end{aligned}
$$

#### Base case ($l=L$)

By definition, $\ell(\theta)=\mathcal{L}(z_L,y)$, hence
$$
\frac{\partial \ell}{\partial z_L}
=
\frac{\partial \mathcal{L}(z_L,y)}{\partial z_L}.
$$
At the output layer we have $z_L=\phi(a_L)$, so applying the activation step (Lemma A) at layer $L$ yields
$$
\begin{aligned}
\delta_L
\;\coloneqq\;
\frac{\partial \ell}{\partial a_L}
&=
\frac{\partial \ell}{\partial z_L}\odot \phi'(a_L)
&&\text{(Lemma A with $z_L=\phi(a_L)$)}\\
&=
\frac{\partial \mathcal{L}(z_L,y)}{\partial z_L}\odot \phi'(a_L)
&&\text{(substitute $\frac{\partial \ell}{\partial z_L}=\frac{\partial \mathcal{L}}{\partial z_L}$).}
\end{aligned}
$$
This proves the required initialization at the output layer. Note that it depends on the choice of loss and output map.

#### Inductive step ($1\leq l<L$)

Assume $\delta_{l+1}=\frac{\partial \ell}{\partial a_{l+1}}$ has been computed. We show that
$$
\delta_l=\frac{\partial \ell}{\partial a_l}
=
\left(W_{l+1}^{T}\delta_{l+1}\right)\odot \phi'(a_l).
$$
First, propagate derivatives through the affine map $a_{l+1}=W_{l+1}z_l+b_{l+1}$ by applying the affine step (Lemma B):
$$
\begin{aligned}
\frac{\partial \ell}{\partial z_l}
&=
W_{l+1}^{T}\frac{\partial \ell}{\partial a_{l+1}}
&&\text{(Lemma B with $a_{l+1}=W_{l+1}z_l+b_{l+1}$)}\\
&=
W_{l+1}^{T}\delta_{l+1}
&&\text{(definition of $\delta_{l+1}$).}
\end{aligned}
$$
Next, propagate through the activation $z_l=\phi(a_l)$ by applying the activation step (Lemma A):
$$
\begin{aligned}
\delta_l
\;\coloneqq\;
\frac{\partial \ell}{\partial a_l}
&=
\frac{\partial \ell}{\partial z_l}\odot \phi'(a_l)
&&\text{(Lemma A with $z_l=\phi(a_l)$)}\\
&=
\left(W_{l+1}^{T}\delta_{l+1}\right)\odot \phi'(a_l)
&&\text{(substitute $\frac{\partial \ell}{\partial z_l}=W_{l+1}^{T}\delta_{l+1}$).}
\end{aligned}
$$
This proves the recursion $\delta_l=(W_{l+1}^{T}\delta_{l+1})\odot \phi'(a_l)$. 

#### Parameter gradients

It remains to show that, once $\delta_l=\partial \ell / \partial a_l$ is known, the gradients with respect to the parameters at layer $l$ are
$$
\frac{\partial \ell}{\partial W_l}=\delta_l z_{l-1}^{T},
\qquad
\frac{\partial \ell}{\partial b_l}=\delta_l.
$$
These follow from differentiating the local affine map $a_l=W_lz_{l-1}+b_l$.

##### Bias gradients

Since $\frac{\partial a_l}{\partial b_l}=I_{d_l}$, the column-gradient chain rule gives
$$
\frac{\partial \ell}{\partial b_l}
=
\left(\frac{\partial a_l}{\partial b_l}\right)^{T}\frac{\partial \ell}{\partial a_l}
=
I_{d_l}\,\delta_l
=
\delta_l.
$$

##### Weight gradients

For each $i\in\{1,\dots,d_l\}$ and $j\in\{1,\dots,d_{l-1}\}$, the $i$-th coordinate of $a_l$ is
$$
(a_l)_i=\sum_{j=1}^{d_{l-1}}(W_l)_{ij}(z_{l-1})_j+(b_l)_i.
$$
Differentiating with respect to the single entry $(W_l)_{ij}$ yields
$$
\frac{\partial (a_l)_i}{\partial (W_l)_{ij}}=(z_{l-1})_j.
$$
Additionally note that, for $k\neq i$, $(a_l)_k$ does not depend on $(W_l)_{ij}$ except through its own row, so
$$
\frac{\partial (a_l)_k}{\partial (W_l)_{ij}}
=
0
\qquad
(k\neq i).
$$
Now, since $\ell$ depends on $(W_l)_{ij}$ through $a_l$, the chain rule gives
$$
\frac{\partial \ell}{\partial (W_l)_{ij}}=\sum_{k=1}^{d_l}\frac{\partial \ell}{\partial (a_l)_k}\frac{\partial (a_l)_k}{\partial (W_l)_{ij}}.
$$
Having previously shown that only the $k=i$ term survives when differentiating with respect to single entry $(W_l)_{ij}$, then
$$
\frac{\partial \ell}{\partial (W_l)_{ij}}
=
\frac{\partial \ell}{\partial (a_l)_i}\,
\frac{\partial (a_l)_i}{\partial (W_l)_{ij}}.
$$
In turn, substituting $\frac{\partial (a_l)_i}{\partial (W_l)_{ij}}=(z_{l-1})_j$ gives
$$
\frac{\partial \ell}{\partial (W_l)_{ij}}
=
\frac{\partial \ell}{\partial (a_l)_i}\,(z_{l-1})_j.
$$
Finally, by definition $\delta_l=\frac{\partial \ell}{\partial a_l}$, so $(\delta_l)_i=\frac{\partial \ell}{\partial (a_l)_i}$, hence
$$
\frac{\partial \ell}{\partial (W_l)_{ij}}
=
(\delta_l)_i\,(z_{l-1})_j.
$$
This is exactly the $(i,j)$-th entry of the outer product $\delta_l z_{l-1}^{T}$. To see this explicitly, write
$$
\delta_l=
\begin{bmatrix}
(\delta_l)_1\\
(\delta_l)_2\\
\vdots\\
(\delta_l)_{d_l}
\end{bmatrix},
\qquad
z_{l-1}^T=
\begin{bmatrix}
(z_{l-1})_1 & (z_{l-1})_2 & \cdots & (z_{l-1})_{d_{l-1}}
\end{bmatrix}.
$$
Then their outer product is the matrix
$$
\delta_l z_{l-1}^T
=
\begin{bmatrix}
(\delta_l)_1(z_{l-1})_1 & (\delta_l)_1(z_{l-1})_2 & \cdots & (\delta_l)_1(z_{l-1})_{d_{l-1}}\\
(\delta_l)_2(z_{l-1})_1 & (\delta_l)_2(z_{l-1})_2 & \cdots & (\delta_l)_2(z_{l-1})_{d_{l-1}}\\
\vdots & \vdots & \ddots & \vdots\\
(\delta_l)_{d_l}(z_{l-1})_1 & (\delta_l)_{d_l}(z_{l-1})_2 & \cdots & (\delta_l)_{d_l}(z_{l-1})_{d_{l-1}}
\end{bmatrix}.
$$
In particular, its $(i,j)$-th entry is
$$
(\delta_l z_{l-1}^{T})_{ij}
=
(\delta_l)_i\,(z_{l-1})_j.
$$
Therefore, since we have shown entry-wise that
$$
\frac{\partial \ell}{\partial (W_l)_{ij}}=(\delta_l)_i\,(z_{l-1})_j,
$$
collecting all entries yields the matrix identity
$$
\frac{\partial \ell}{\partial W_l}
=
\delta_l z_{l-1}^{T}.
$$
### Derived Identities

We have established the backpropagation recursion for the layer-wise error signals and the corresponding parameter gradient formulas:
$$
\delta_L=\frac{\partial \mathcal{L}(z_L,y)}{\partial z_L}\odot \phi'(a_L),
\qquad
\delta_l=(W_{l+1}^T\delta_{l+1})\odot \phi'(a_l)\quad (l=L-1,\dots,1),
$$
and
$$
\frac{\partial \ell}{\partial W_l}=\delta_l z_{l-1}^{T},
\qquad
\frac{\partial \ell}{\partial b_l}=\delta_l.
$$
Therefore, all parts of the original claim hold. We now present these identities as an algorithmic procedure to perform a forward and backward pass efficiently.

## Forward Propagation and Backpropagation Algorithms
***
The previous derivation implies a simple computational procedure for evaluating $\frac{\partial \ell}{\partial W_l}$ and $\frac{\partial \ell}{\partial b_l}$ for all layers in one backward sweep. We first show the pseudocode for the forward propagation of a single example $(x,y)$. We also continue to assume $\psi=\phi$ at the output layer.

```python
Algorithm 1: Forward Propagation (cache per-layer intermediates)
---------------------------------------------------------------
Inputs:
    x = z[0] ∈ ℝ^{d_0}

    parameters:
        for l = 1..L:
            W_l ∈ ℝ^{d_l×d_{l-1}}
            b_l ∈ ℝ^{d_l}

Cache (written by this algorithm):
    z[0] ∈ ℝ^{d_0}
    for l = 1..L:
        a[l] = W_l z[l-1] + b_l ∈ ℝ^{d_l}
        z[l] = φ(a[l]) ∈ ℝ^{d_l}                # assumes ψ = φ at output

Outputs:
    ŷ = z[L] ∈ ℝ^{d_L}
    cached {a[1..L], z[0..L]}
---------------------------------------------------------------
1. z[0] ← x

for l = 1..L do:
    a[l] ← W_l z[l-1] + b_l                      # a[l] ∈ ℝ^{d_l}
    z[l] ← φ(a[l])                               # z[l] ∈ ℝ^{d_l}
end for

ŷ ← z[L]
return ŷ, {a[1..L], z[0..L]}
```

The cached values are not coincidental. Recall the backward sweep requires (i) $z_{l-1}$ to form the outer product $\delta_l z_{l-1}^T$ for the weight gradients, and (ii) $a_l$ (or equivalently, $\phi'(a_l)$) to apply the element-wise factor $\phi'(a_l)$ in the recursion for $\delta_l$. Thus, for an efficient backward pass, we must either cache these forward-pass intermediates or recompute them ad hoc. 

Note that in the following loop, we compute $\delta_{l-1}$ from $\delta_l$ using $W_l$, which is the same as the recursion from $l\rightarrow l+1$ after reindexing.

```python
Algorithm 2: Backpropagation (Backwards Propagation) (cache per-layer error signals)
----------------------------------------------------------------
Inputs:
    y ∈ 𝒴
    ŷ = z[L] ∈ ℝ^{d_L}

    cached forward-pass values:
        z[0] ∈ ℝ^{d_0}
        for l = 1..L:
            a[l] ∈ ℝ^{d_l}
            z[l] ∈ ℝ^{d_l}

    parameters:
        for l = 1..L:
            W_l ∈ ℝ^{d_l×d_{l-1}}
            b_l ∈ ℝ^{d_l}

Cache (written by this algorithm, optional):
    for l = 1..L:
        δ[l] = ∂ℓ/∂a[l] ∈ ℝ^{d_l}

Outputs:
    for l = 1..L:
        ∂ℓ/∂W_l ∈ ℝ^{d_l×d_{l-1}}
        ∂ℓ/∂b_l ∈ ℝ^{d_l}
----------------------------------------------------------------
ℓ ← 𝓛(ŷ, y)                                   # ℓ ∈ ℝ

# Initialize output-layer error signal
dL_dzL ← ∂𝓛(z[L], y) / ∂z[L]                        # dL_dzL ∈ ℝ^{d_L}
δ[L] ← dL_dzL ⊙ φ'(a[L])                            # δ[L] ∈ ℝ^{d_L}

# Backward sweep
for l = L..1 do:
    ∂ℓ/∂W_l ← δ[l] z[l-1]^T                     # (d_l×1)(1×d_{l-1}) = d_l×d_{l-1}
    ∂ℓ/∂b_l ← δ[l]                              # d_l

    if l > 1 then:
        δ[l-1] ← (W_l^T δ[l]) ⊙ φ'(a[l-1])      
	    # δ[l-1] ∈ ℝ^{d_{l-1}}, since W_l^T δ[l]: (d_{l-1}×d_l)(d_l) = d_{l-1},
        # then Hadamard with φ'(a[l-1]) ∈ ℝ^{d_{l-1}}
    end if
end for

return {∂ℓ/∂W_l, ∂ℓ/∂b_l}_{l=1}^L  (and optionally δ[1..L])
```

Given the cached forward propagation intermediates, backpropagation computes the error signals $\{\delta_l\}$ and returns the full set of parameter gradients $\{\partial \ell/\partial W_l,\partial \ell/\partial b_l\}_{l=1}^L$. Note that the only loss-specific component is the output gradient $\partial \mathcal{L}(z_L,y)/\partial z_L$; similarly, if it were to differ, the only output map-specific component is $\psi'(a_L)$. Once these quantities are available, the remaining steps are purely determined by the network's layer-wise structure.

Finally, the reason for why the error signal cache in backpropagation is optional is because, at every layer-wise error signal computation, you only need the subsequent layer's error signal. Instead of storing all layer-wise error signals, we can instead use a temporal variable to store the current layer's error signal. This allows the next iteration of the backward sweep to compute the relative previous layer's error signal; recursively applying this idea to the newly computed error signal allows us to store a single error signal throughout the backpropagation algorithm rather than all of them.

## Time Complexity Analysis
***
Throughout this section, time complexity is measured in terms of arithmetic operations. We treat data movement, allocation, and writing outputs to memory as constant time bookkeeping unless it contributes additional arithmetic. Under this convention, statements such as "$\partial \ell / \partial b_l \leftarrow \delta_l$" incur no extra arithmetic beyond what is already required to compute $\delta_l$. 

### Forward Propagation

In the forward propagation, we iterate over $L$ layers of affine and non-linear transformations. Per layer $l$, each $a_l=W_lz_{l-1}+b_l$ is a dense matrix-vector multiply and bias addition with cost $O(d_ld_{l-1})$ and $O(d_l)$, respectively; followed by an element-wise, non-linear transformation $z_l=\phi(a_l)$ with cost $O(d_l)$. So per layer, we have that the overall time complexity is
$$
T_{\text{fwd}, l}=O(d_ld_{l-1}+d_l)
$$
Summed over layers,
$$
T_{\text{fwd}}=O\left(\sum_{l=1}^L d_ld_{l-1}+d_l\right)=O(|\theta|).
$$
where $|\theta|$ is the total number of parameters.

### Backpropagation

In backpropagation, we iterate over $L$ layers of parameter gradient computation and error-signal propagation. For a given layer $l$ (i) in the parameter gradient computation, we compute an outer product $\frac{\partial \ell}{\partial W_l}=\delta_l z_{l-1}^T$ for the weight gradient with cost $O(d_ld_{l-1})$; we can assume the writing/copying operation $\frac{\partial \ell}{\partial b_l}=\delta_l$ for the bias gradient under our complexity model is constant time, (ii) in the error-signal propagation for $l>1$, $W_l^T\delta_l$ is a dense matrix-vector multiply with cost $O(d_ld_{l-1})$ and its Hadamard product with $\phi'(a_{l-1})$ is of cost $O(d_{l-1})$ since it is an element-wise operation. So per layer, we have that the overall time complexity is
$$
T_{\text{bwd}, l}=O(d_ld_{l-1})+O(d_ld_{l-1})+O(d_{l-1})=O(d_ld_{l-1}+d_l).
$$
Summing over layers,
$$
T_{\text{bwd}}=O\left( \sum_{l=1}^L d_ld_{l-1}+d_l \right)=O(|\theta|)
$$
where $|\theta|$ is the total number of parameters. 

### Discussion

We have shown that **both the forward and backpropagation algorithms scale linearly (up to constant factors) in the total number of parameters.**
$$
T_{\text{fwd}}=T_{\text{bwd}}=O(|\theta|)
$$
As previously mentioned, the purpose of the induction exercise was precisely to obtain a linear runtime algorithm for the backward pass: the backpropagation algorithm. Alternatively, consider the naïve chain rule expansion algorithm as a substitute to the backpropagation algorithm. Consider computing the error signal at layer $l$ directly from the definition
$$
\delta_{l}\coloneqq \frac{\partial \ell}{\partial a_{l}}.
$$
We can see that a full chain expansion of the above yields
$$
\begin{aligned}
\frac{\partial \ell}{\partial a_{l}}
&=
\left(\frac{\partial z_{l}}{\partial a_{l}}\right)^T
\left(\frac{\partial a_{l+1}}{\partial z_{l}}\right)^T
\left(\frac{\partial z_{l+1}}{\partial a_{l+1}}\right)^T
\left(\frac{\partial a_{l+2}}{\partial z_{l+1}}\right)^T
\cdots
\left(\frac{\partial z_L}{\partial a_L}\right)^T
\frac{\partial \ell}{\partial z_L}.
\end{aligned}
$$
Now substitute the local Jacobians for a multi-layer perceptron as derived in the [[#Lemmas]]
$$
\left(\frac{\partial a_{k+1}}{\partial z_{k}}\right)^T=W_{k+1}^T, \quad \left( \frac{\partial z_{k}}{\partial a_{k}} \right)^T=\text{diag}\left( \phi'(a_{k})\right),
$$
which yields the expanded chain rule expression
$$
\begin{aligned}
\delta_{l}
&=
\operatorname{diag}\!\big(\phi'(a_{l})\big)\,
W_{l+1}^T\,
\operatorname{diag}\!\big(\phi'(a_{l+1})\big)\,
W_{l+2}^T\,
\cdots\,
W_L^T\,
\operatorname{diag}\!\big(\phi'(a_L)\big)\,
\frac{\partial \ell}{\partial z_L}.
\end{aligned}
$$
Each diagonal Jacobian $\text{diag}(\phi'(a_k))$ times a vector is an element-wise scaling with cost $O(d_k)$. Each multiplication by $W_k^T\in\mathbb{R}^{d_{k-1}\times d_k}$  is a dense matrix-vector multiply with cost $O(d_kd_{k-1})$. Therefore, computing $\delta_{l}$ directly from this expanded expression is of time complexity
$$
T(\delta_{l}\ \text{from scratch})
=
O\!\left(
\sum_{k={l+1}}^{L} d_k d_{k-1}+d_k
\right)
\leq
O\!\left(
\sum_{k=1}^{L} d_k d_{k-1}+d_k
\right)
=
O\!\left(
|\theta|
\right).
$$
Once $\delta_{l}$ is available, we must still form the weight gradient via the outer product $\partial \ell / \partial W_l = \delta_l z_{l-1}^T$ which is $O(d_ld_{l-1})$. Thus, the total per-layer time complexity in the naïve strategy is
$$
T_{\text{naïve-bwd}, l}=O(|\theta|)+O(d_ld_{l-1})=O(|\theta|),
$$
and summing over layers yields the overall upper bound
$$
T_{\text{naïve-bwd}}=\sum_{l=1}^L T_{\text{naïve-bwd}, l}=O(L|\theta|).
$$
This is an intentionally coarse upper bound, as it treats every $\delta_l$ recomputation as traversing the entire remaining network each time. Regardless, it captures the essence of what backpropagation compensates for: without maintaining intermediate error signals, the backward computation can scale quadratically in the depth $L$.  For example, with constant width $d$ throughout the network, then $|\theta|=O(Ld^2)$ and the naïve bound becomes $O(L^2d^2)$, a quadratic in $L$.

It is also worth noting that, as shown in the chain expansion above, the activation derivative $\phi'(a_{l})$ can be viewed as a diagonal Jacobian $\text{diag}(\phi'(a_{l}))\in\mathbb{R}^{d_{l}\times d_{l}}$ as previously shown in [[#Lemmas]]. In that notation, the error signal update can be written as
$$
\delta_{l}=\text{diag}(\phi'(a_{l}))W_{l+1}^T\delta_{l+1}
$$
One should not explicitly form the dense product $\text{diag}(\phi'(a_{l}))W_{l+1}^T$. If the diagonal structure is ignored and $\text{diag}(\phi'(a_{l}))$ is treated as a generic dense matrix, then computing this intermediate via a dense matrix-matrix multiplication would cost $O(d^2_{l-1}d_l)$ time. Instead, exploiting diagonality is equivalent to an element-wise scaling
$$
\delta_{l}=(W_{l+1}^T\delta_{l+1})\odot \phi'(a_{l})
$$
which avoids ever materializing the diagonal Jacobian.

## Common Output Layers and Losses for Closed-Form $\delta_L$
***
From our [[#Derived Identities]], we can see how the closed form of the last layer's error signal initialization is dependent on the choice of the function at the output layer and the loss. Throughout this post, we have assumed that the output map was the same as the activation map, namely $\psi(\cdot)=\phi(\cdot)$. We will now explore different initializations of $\delta_L$ given common output layers and losses.

### General Output-Layer Identity

Up to this point we have assumed that the output map is element-wise (in particular, $\psi=\phi$, so that its Jacobian is diagonal and the output layer initialization reduces to a Hadamard product. But this simplification relies on the assumption that each entry-wise output $(z_L)_i$ depends only on the corresponding entry-wise input $(a_L)_i$. So what if it didn't?

If the output map $\psi$ mixes coordinates, such as with $\text{softmax}$ or normalization, then for some $i\neq j$, $(a_L)_j$ may influence $(z_L)_i$, and off-diagonal terms in the Jacobian need not be zero. In that case we should fall back on the fully general chain rule form.

Let
$$
z_L=\psi(a_L)\in\mathbb{R}^{d_L}, \qquad \ell\coloneqq \mathcal{L}(z_L,y).
$$
Then the output-layer error signal initialization is
$$
\delta_L \;\coloneqq\; \frac{\partial \ell}{\partial a_L}
=
\left(\frac{\partial z_L}{\partial a_L}\right)^T
\frac{\partial \mathcal{L}(z_L,y)}{\partial z_L}
=
J_\psi(a_L)^T\frac{\partial \mathcal{L}(z_L,y)}{\partial z_L},
$$
where $J_\psi(a_L)=\frac{\partial z_L}{\partial a_L}\in\mathbb{R}^{d_L\times d_L}$ is the Jacobian of the output map. **This identity will be our template for all closed-form initializations of $\delta_L$ below.**

### Regression: Identity Output and Squared Error

Take $\psi=\text{id}$, so $z_L=a_L$. Let the per-example loss be the half-squared error:
$$
\mathcal{L}(z_L, y)=\frac{1}{2}||z_L-y||_2^2=\frac{1}{2}\sum_{k=1}^{d_L}(z_k-y_k)^2
$$
We must now compute $\frac{\partial \mathcal{L}(z_L, y)}{\partial z_L}$. For each coordinate $k$,
$$
\frac{\partial \mathcal{L}(z_L,y)}{\partial z_k}=\frac{\partial}{\partial z_k}\left( \frac{1}{2}(z_k-y_k)^2 \right)=(z_k-y_k)
$$
Stacking the coordinates, we get
$$
\frac{\partial \mathcal{L}(z_L, y)}{\partial z_L}=z_L-y
$$
Since $z_L=a_L$, the Jacobian is $J_\psi(a_L)=I$, therefore
$$
\delta_L=J_\psi(a_L)^T\frac{\partial \mathcal{L}(z_L, y)}{\partial z_L}=I(z_L-y)=z_L-y
$$
(If you use $||z-y||_2^2$ without the $\frac{1}{2}$, you get $\delta_L=2(z_L-y)$.)

### Binary Classification: Sigmoid Output and Binary Cross-Entropy Loss

Let $\psi=\sigma$ be applied element-wise at the output layer, so that $z_L=\sigma(a_L)$. Let the per-example binary cross-entropy loss be
$$
\mathcal{L}(z_L,y)=-[y\log z_L+(1-y)\log(1-z_L)].
$$
We must now compute $\frac{\partial \mathcal{L}(z_L, y)}{\partial z_L}$. Differentiating with respect to $z_L$ gives
$$
\begin{align*}
\frac{\partial \mathcal{L}(z_L, y)}{\partial z_L}
&=
\frac{\partial}{\partial z_L}\left(-y\log z_L-(1-y)\log(1-z_L)\right)\\
&=
-\frac{y}{z_L}-(1-y)\cdot\frac{-1}{1-z_L}\\
&=
-\frac{y}{z_L}+\frac{1-y}{1-z_L}\\
&=
\frac{z_L-y}{z_L(1-z_L)}.
\end{align*}
$$
We must now compute $\frac{\partial z_L}{\partial a_L}$. For sigmoid,
$$
\frac{\partial z_L}{\partial a_L}=\sigma'(a_L)=z_L(1-z_L).
$$
Since the output map is element-wise, $J_\psi(a_L)=\operatorname{diag}(\sigma'(a_L))$, and thus the general output-layer identity reduces to an element-wise product:
$$
\delta_L
=
J_\psi(a_L)^T\frac{\partial \mathcal{L}(z_L, y)}{\partial z_L}
=
\frac{\partial \mathcal{L}(z_L, y)}{\partial z_L}\odot \sigma'(a_L).
$$
Substituting the derivatives yields (entry-wise)
$$
\delta_L
=
\frac{z_L-y}{z_L(1-z_L)}\odot z_L(1-z_L)
=
z_L-y.
$$
Therefore, for sigmoid output with binary cross-entropy loss, the output-layer error signal initialization is $\delta_L=z_L-y$.

### Multi-class Classification: Softmax Output and Cross-Entropy Loss  

Take $\psi=\mathrm{softmax}$, so for $a_L\in\mathbb{R}^{d_L}$,  
$$  
(z_L)_k=\frac{e^{(a_L)_k}}{\sum_{j=1}^{d_L}e^{(a_L)_j}},\qquad k=1,\dots,d_L,  
$$
and let the per-example cross-entropy loss be  
$$  
\mathcal{L}(z_L,y)=-\sum_{k=1}^{d_L} y_k\log (z_L)_k,  
\qquad \text{(typically } \sum_{k=1}^{d_L} y_k=1\text{)}.  
$$
Since $\psi$ mixes coordinates, we use the general output-layer identity  
$$  
\delta_L  
=  
J_\psi(a_L)^T\frac{\partial \mathcal{L}(z_L,y)}{\partial z_L}.  
$$
We first compute $\frac{\partial \mathcal{L}(z_L,y)}{\partial z_L}$. For each coordinate $k$,  
$$  
\frac{\partial \mathcal{L}(z_L,y)}{\partial (z_L)_k}  
=  
\frac{\partial}{\partial (z_L)_k}\left(-\sum_{i=1}^{d_L} y_i\log (z_L)_i\right)  
=  
-\frac{y_k}{(z_L)_k}.  
$$
Stacking the coordinates,  
$$  
\frac{\partial \mathcal{L}(z_L,y)}{\partial z_L}  
=  
-\;y\oslash z_L,  
$$
where $\oslash$ denotes element-wise division.  
  
Then, we compute $J_\psi(a_L)=\frac{\partial z_L}{\partial a_L}$. For each pair $(i,j)$,  
$$  
\frac{\partial (z_L)_i}{\partial (a_L)_j}  
=  
\frac{\partial}{\partial (a_L)_j}\left(\frac{e^{(a_L)_i}}{\sum_{k}e^{(a_L)_k}}\right)  
=  
(z_L)_i\left(\mathbf{1}\{i=j\}-(z_L)_j\right).  
$$
Equivalently, in matrix form,  
$$  
J_\psi(a_L)  
=  
\operatorname{diag}(z_L)-z_L z_L^T.  
$$
Since this matrix is symmetric, $J_\psi(a_L)^T=J_\psi(a_L)$.  
  
Finally, we form the Jacobian–vector product. Let $g\coloneqq \frac{\partial \mathcal{L}}{\partial z_L}=-\,y\oslash z_L$. Then  
$$  
\delta_L  
=  
(\operatorname{diag}(z_L)-z_L z_L^T)\,g  
=  
\operatorname{diag}(z_L)g - z_L(z_L^T g).  
$$
Compute each term. First,  
$$  
\operatorname{diag}(z_L)g  
=  
\operatorname{diag}(z_L)\left(-\,y\oslash z_L\right)  
=  
\left(-\,y\oslash z_L\right)\odot z_L
=  
-\,y.  
$$
Next,  
$$  
z_L^T g  
=  
z_L^T\left(-\,y\oslash z_L\right)  
=  
-\sum_{k=1}^{d_L} y_k  
=  
-1,  
$$
so  
$$  
-\,z_L(z_L^T g)= -\,z_L(-1)=z_L.  
$$
Therefore,  
$$  
\delta_L = -y + z_L = z_L-y.  
$$
Notice that $\psi=\mathrm{softmax}$ mixes coordinate, as noted in [[#General Output-Layer Identity]].

### Output Normalization: $\ell_2$-Normalized Output and Squared Error  
  
Another popular non-element-wise output map is $\ell_2$ normalization, used when one wants the output to lie on the unit sphere. Take  
$$  
z_L=\psi(a_L)=\frac{a_L}{\|a_L\|_2}\in\mathbb{R}^{d_L},  
\qquad r\coloneqq \|a_L\|_2=\sqrt{a_L^Ta_L},  
$$
and let the per-example loss be the half-squared error to a target $y\in\mathbb{R}^{d_L}$:  
$$  
\mathcal{L}(z_L,y)=\frac{1}{2}\|z_L-y\|_2^2.  
$$
Since $\psi$ mixes coordinates through the shared norm $r$, we use the general output-layer identity  
$$  
\delta_L  
=  
J_\psi(a_L)^T\frac{\partial \mathcal{L}(z_L,y)}{\partial z_L}.  
$$
We first compute $\frac{\partial \mathcal{L}(z_L,y)}{\partial z_L}$. As in the regression case,  
$$  
\frac{\partial \mathcal{L}(z_L,y)}{\partial z_L}=z_L-y.  
$$
We then compute $J_\psi(a_L)=\frac{\partial z_L}{\partial a_L}$. Since
$$
r=\|a_L\|_2=\sqrt{a_L^Ta_L},
$$
each coordinate derivative is
$$
\frac{\partial r}{\partial (a_L)_j}
=
\frac{1}{2}(a_L^Ta_L)^{-1/2}\cdot 2(a_L)_j
=
\frac{(a_L)_j}{r},
$$
so stacking coordinates gives
$$
\frac{\partial r}{\partial a_L}=\frac{a_L}{r}.
$$
Now fix indices $(i,j)$ and write $(z_L)_i=\frac{(a_L)_i}{r}$. By the quotient rule,
$$
\frac{\partial (z_L)_i}{\partial (a_L)_j}
=
\frac{\partial}{\partial (a_L)_j}\left(\frac{(a_L)_i}{r}\right)
=
\frac{\mathbf{1}\{i=j\}\,r-(a_L)_i\,\frac{\partial r}{\partial (a_L)_j}}{r^2}
=
\frac{\mathbf{1}\{i=j\}\,r-(a_L)_i\,\frac{(a_L)_j}{r}}{r^2}.
$$
Simplifying,
$$
\frac{\partial (z_L)_i}{\partial (a_L)_j}
=
\frac{1}{r}\mathbf{1}\{i=j\}-\frac{(a_L)_i(a_L)_j}{r^3}.
$$
Collecting these entries yields the Jacobian
$$
J_\psi(a_L)
=
\frac{1}{r}I-\frac{1}{r^3}a_La_L^T
=
\frac{1}{\|a_L\|_2}\left(I-\frac{a_La_L^T}{\|a_L\|_2^2}\right).
$$
Since $z_L=a_L/r$, we have $\frac{a_La_L^T}{r^2}=z_Lz_L^T$, so equivalently
$$
J_\psi(a_L)=\frac{1}{\|a_L\|_2}\left(I-z_Lz_L^T\right).
$$
Finally, substituting into the general output-layer identity gives the output-layer error signal initialization
$$
\delta_L
=
J_\psi(a_L)^T\frac{\partial \mathcal{L}(z_L,y)}{\partial z_L}
=
\frac{1}{\|a_L\|_2}\left(I-z_Lz_L^T\right)(z_L-y).
$$
Notice that the Jacobian is exactly the orthogonal projection matrix onto the tangent space of the unit sphere at $z_L$, you can recognize it from its familiar structure $P\coloneqq I-z_Lz_L^T$.

## Extensions and Further Directions
***
The derivation above is easiest to read for the standard, vanilla multi-layer perceptron block that is known as $a_l=W_lz_{l-1}+b_l$, $z_l=\phi(a_l)$, but the same chain-rule view can extend to much richer, intricate, and modern architectures with much more elaborate analyses.

One natural extension of this idea is to insert non-element-wise maps between layers: it is known that most networks contain some kind of layer normalizations to stabilize gradient flow, or have some kind of attention-like mixing, or other coordinate-coupling transforms of the like. In this case, the only structural change is that the activation step we derived in [[#Lemmas]] becomes a Jacobian-vector product $J_{\psi_l}(a_l)^Tv$ rather than a Hadamard factor. 

Another idea is that training is almost never done per-example or in full, it is typically done in mini-batches of size $B$, both for statistical and computational efficiency. The identities derived above are structurally unchanged, but the per-example vectors become matrices whose columns index examples.

On the implementation side, it is very clear and instructive to see how to code the full forward and backward pass natively in the programming language of your choice (probably using some kind of scientific computing library to avoid numerical analysis hassle, e.g., `numpy` in Python). Doing the careful caching and the shape checks to see exactly what reverse-mode differentiation is doing under the hood leads to a better foundation for understanding reverse-mode AD/auto-differentiation systems, which can be viewed as automating the same computational-graph chain rule bookkeeping we showed in [[#Forward and Backward Pass]] at scale.

Finally, once Jacobians are explicit, they provide a clean lens for diagnosing instability. Exploding and vanishing gradients correspond to repeated multiplication by ill-conditioned Jacobian factors across layers during backpropagation, and techniques such as renormalization can be seen as mitigators of this effect by controlling the spectral norm (in particular, the spectral norm $||J||_2=\sigma_\text{max}(J)$ of the effective backpropagation operator; this is the long product of Jacobian matrices from the output gradient to the layer-$l$ error signal $\delta_l$). 

In fact, it is precisely the extension into gradient instability in multi-layer perceptrons that motivated me to write this particular post in this first place. Hopefully, with a foundation set in stone, I'll be able to post about this particular topic in detail and derive some interesting results soon enough!

## Symbols and Identities (Reference)  
***
$$  
\begin{array}{ll}  
\textbf{Core objects (single example)} & \\ \hline  
L & \text{number of parameterized layers (depth; excludes input layer)}\\  
d_l & \text{width of layer } l\ \text{(number of units)}\\  
x & x\in\mathbb{R}^{d_0}\ \text{(input vector)}\\  
y & y\in\mathcal{Y}\ \text{(target/label)}\\  
\theta & \theta=\{(W_l,b_l)\}_{l=1}^L\ \text{(parameter set)}\\  
W_l & W_l\in\mathbb{R}^{d_l\times d_{l-1}}\ \text{(weights at layer } l)\\  
b_l & b_l\in\mathbb{R}^{d_l}\ \text{(biases at layer } l)\\  
\phi & \text{element-wise activation function}\\  
\phi'(a_l) & \phi'(a_l)\in\mathbb{R}^{d_l}\ \text{(element-wise derivative at } a_l)\\  
\mathcal{L}(\hat y,y) & \text{per-example scalar loss}\\  
\ell & \ell \coloneqq \mathcal{L}(z_L,y)\in\mathbb{R}\ \text{(shorthand)}\\[6pt]  
  
\textbf{Forward-pass intermediates} & \\ \hline  
z_0 & z_0\coloneqq x\in\mathbb{R}^{d_0}\ \text{(input-as-activation)}\\  
a_l & a_l=W_l z_{l-1}+b_l\in\mathbb{R}^{d_l}\ \text{(pre-activation)}\\  
z_l & z_l=\phi(a_l)\in\mathbb{R}^{d_l}\ \text{(activation)}\\  
\hat y & \hat y=z_L\in\mathbb{R}^{d_L}\ \text{(output; assuming } \psi=\phi \text{ earlier for simplicitly})\\[6pt]  
  
\textbf{Backward-pass quantities} & \\ \hline  
\delta_l & \delta_l \coloneqq \frac{\partial \ell}{\partial a_l}\in\mathbb{R}^{d_l}\ \text{(error signal)}\\  
\frac{\partial \ell}{\partial z_l} & \frac{\partial \ell}{\partial z_l}\in\mathbb{R}^{d_l}\ \text{(loss grad w.r.t. } z_l)\\  
\frac{\partial \ell}{\partial W_l} & \frac{\partial \ell}{\partial W_l}\in\mathbb{R}^{d_l\times d_{l-1}}\ \text{(loss grad w.r.t. } W_l)\\  
\frac{\partial \ell}{\partial b_l} & \frac{\partial \ell}{\partial b_l}\in\mathbb{R}^{d_l}\ \text{(loss grad w.r.t. } b_l)\\  
\odot & \text{Hadamard (element-wise) product}\\
\oslash & \text{Hadamard (element-wise) division}

\end{array}  
$$

$$  
\textbf{Key identities}\qquad \text{(single example; column-gradient convention)}  
$$
$$  
\begin{aligned}  
\delta_l  
&\coloneqq  
\frac{\partial \ell}{\partial a_l}  
\in\mathbb{R}^{d_l}.\\[6pt]  
\delta_L  
&=  
\frac{\partial \mathcal{L}(z_L,y)}{\partial z_L}\odot \phi'(a_L).\\[6pt]  
\delta_l  
&=  
\left(W_{l+1}^{T}\delta_{l+1}\right)\odot \phi'(a_l),  
\qquad l=L-1,\dots,1.\\[8pt]  
\frac{\partial \ell}{\partial W_l}  
&=  
\delta_l\, z_{l-1}^{T},  
\qquad  
\frac{\partial \ell}{\partial b_l}  
=  
\delta_l,  
\qquad l=1,\dots,L.\\[8pt]  
\text{cache }  
&\{\,z_{l-1},\, a_l\,\}_{l=1}^{L}  
\ \ \text{(or } \{\,z_{l-1},\, \phi'(a_l)\,\}_{l=1}^{L}\text{).}  
\end{aligned}  
$$
## References 
***

1. Han Zhao. (2025). Note on Backpropagation. *CS442, University of Illinois Urbana-Champaign*
2. Daniel A. Roberts, Sho Yaida, Boris Hanin. (2021). Neural Networks. *The Principles of Deep Learning Theory,* 37-47
