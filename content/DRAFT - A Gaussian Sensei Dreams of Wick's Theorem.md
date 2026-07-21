---
title: "A Gaussian Sensei Dreams of Wick's Theorem"
published: 2026-07-03
created: 2026-07-03
modified: 2026-07-03
description: "A detailed derivation of Wick's theorem, building from univariate Gaussian moments to the pairing structure behind the multivariate formula."
showDate: true
showReadingTime: true
draft: false
tags:
  - probability
  - statistics
  - deep-learning
  - gaussian-distributions
  - combinatorics
---
The aim of this post is for me to try and intuitively tackle the Gaussian integral section of the pretraining section of *The Principles of Deep Learning Theory*, particularly the grand result: a derivation of Wick's theorem. 
***
## Abstract

We derive Wick's theorem first for a single Gaussian random variable, also called the univariate Gaussian case, and then extend it to Gaussian random vectors, also called the multivariate Gaussian case. In probability theory, Wick's theorem, also known as Isserlis's theorem, is a formula for computing higher-order moments of a multivariate Gaussian random vector in terms of the entries of its covariance matrix.[^isserlis-theorem]

Let $\mathbf{X}=(X_1, \dots, X_n)$ be a multivariate Gaussian random vector. We can decompose it into
$$
(X_1, \dots, X_n)=(\mu_1+Z_1, \dots \mu_n+Z_n)
$$
where $\mu_i=\mathbb E[X_i]$ is the $i$-th component of the mean vector, and $\mathbf Z=(Z_1,\dots,Z_n)$ is a zero-mean multivariate Gaussian random vector.

Given the previous set up, here is the statement of Wick's theorem.

If $\mathbf{Z}=(Z_1, \dots, Z_n)$ is a zero-mean multivariate Gaussian random vector and $(i_1,\dots,i_{2m})\in\{1,\dots,n\}^{2m}$ is a tuple of selected coordinate indices, then
$$ 
\mathbb{E}[Z_{i_1}Z_{i_2}\cdots Z_{i_{2m}}]
=
\sum_{p\in P^2_{2m}}\prod_{\{\alpha,\beta\}\in p} \mathbb{E}[Z_{i_\alpha} Z_{i_\beta}]
=
\sum_{p\in P^2_{2m}}\prod_{\{\alpha,\beta\}\in p} \operatorname{Cov}(Z_{i_\alpha}, Z_{i_\beta}) 
 $$
where the sum is over all pairings of the positions $\{1, \dots, 2m\}$, i.e., all the distinct ways of partitioning $\{1, \dots, 2m\}$ into pairs $\{\alpha, \beta\}$, and the product is over pairs contained in $p$. 

If instead the joint moment contains an odd number of factors, then no pairing of all selected positions exists, and therefore $\mathbb{E}[Z_{i_1}Z_{i_2}\cdots Z_{i_{2m+1}}]=0$.   

### Centering the Gaussian Vector

For the rest of the derivation, Wick's theorem will be applied to the centered coordinates
$$
Z_i=X_i-\mu_i
$$
**Therefore, whenever we compute higher-order moments using Wick's theorem, it is important to note that we are working with the zero-mean multivariate Gaussian random vector $\mathbf{Z}$, or alternatively univariate Gaussian random variable, rather than directly with $\mathbf{X}$**. Moments of the original variables $X_i$ can be obtained afterward by substituting $X_i=\mu_i+Z_i$ and expanding.

***
## Part I: Univariate Gaussian Case

Before deriving Wick's theorem for multivariate Gaussian random vectors, we first study the univariate case. We review the Gaussian density, the definition of expectations and moments, and the role of the MGF. We then derive the moments of a centered univariate Gaussian and show how the resulting formula already contains the pairing structure that appears in Wick's theorem.

### 1.1 — Gaussian Density

Let $Z\sim\mathcal{N}(0, \sigma^2)$ be a zero-mean Gaussian random variable with variance $\sigma^2$ as per [[#Centering the Gaussian Vector]]. The probability density function of $Z$ is then[^normal-density]
$$
f_Z(z)=\frac{1}{\sqrt{2\pi \sigma^2}}\exp\left( -\frac{z^2}{2\sigma^2} \right).
$$
Note that the constant $\frac{1}{\sqrt{2\pi\sigma^2}}$ is chosen such that the density integrates to $1$ as follows
$$
\int_{-\infty}^{\infty}f_Z(z)\,dz =1.
$$
### 1.2 — Expectations and Moments

By the Law of the Unconscious Statistician (LOTUS), if $Z$ is a continuous random variable with density $f_Z$, then the expectation of a suitable function $g(Z)$ is defined as[^lotus]
$$
\mathbb{E}[g(Z)]=\int_{-\infty}^{\infty} g(z)f_Z(z)\, dz.
$$
We consider the special case where $g$ is defined by $g(z)=z^k$, so that $g(Z)=Z^k$. This gives us what is called the $k$-th moment of $Z$:
$$
\mathbb{E}[Z^k]=\int_{-\infty}^{\infty} z^k f_Z(z)\, dz.
$$
Note that $k=1$ results in the first moment $\mathbb{E}[Z]$, which is simply just the mean of $Z$. We also know that $\mathbb{E}[Z]=0$ from [[#1.1 — Gaussian Density]].[^normal-mean] On the other hand, $k=2$ results in the second moment $\mathbb{E}[Z^2]$, which is just the variance of $Z$ since
$$
\operatorname{Var}(Z)=\mathbb{E}[(Z-\mathbb{E}[Z])^2]=\mathbb{E}[(Z-0)^2]=\mathbb{E}[Z^2].
$$
More generally, the $k$-th moment $\mathbb{E}[Z^k]$ measures the expected value of the $k$-th power of $Z$. However, these higher moments are generally more tedious to evaluate directly as $k$ grows. Wick's theorem gives us a shortcut to computing these higher-order Gaussian moments without evaluating a new integral each time.

### 1.3 — Moment-Generating Functions

A remark by a statistics professor that I hadn't recalled until now was that MGFs are immensely powerful theoretically, but practically rather restricted. This derivation makes use of them as they greatly simplify the derivation. Otherwise, directly computing higher moments from the integral definition is possible by a repeated integration-by-parts method, but quickly becomes cumbersome. MGFs provide a more accessible route to computing these.

Without going into too much detail, a moment-generating function (MGF) does exactly what its name suggests: it generates the moments of a distribution. The MGF of a random variable $X$ is defined by[^moment-generating-function]
$$
M_X(t)\coloneqq \mathbb{E}[e^{tX}], \quad t\in\mathbb{R}.
$$
Recall that the Maclaurin series for the exponential function is
$$
e^x=\sum_{n=0}^\infty \frac{x^n}{n!}.
$$
Expanding the Maclaurin series for the above natural exponential $e^{tX}$ gives
$$
e^{tX}=\sum_{n=0}^{\infty} \frac{(tX)^n}{n!}=1+tX+\frac{t^2X^2}{2!}+\frac{t^3X^3}{3!}+\dots +\frac{t^nX^n}{n!}+\dots.
$$
Taking the expectation gives
$$
M_X(t)=\mathbb{E}[e^{tX}]=\mathbb{E}\left[ 1+tX+\frac{t^2X^2}{2!}+\frac{t^3X^3}{3!}+\dots + \frac{t^n X^n}{n!}+\dots \right].
$$
Under conditions that are satisfied for Gaussian random variables, we may exchange the expectation with the infinite series (see [[#Appendix A — Exchanging an Infinite Sum and an Expectation]]) and write
$$
M_X(t)=\mathbb{E}[e^{tX}]=1+t\mathbb{E}[X]+\frac{t^2\mathbb{E}[X^2]}{2!}+\frac{t^3\mathbb{E}[X^3]}{3!}+\dots+\frac{t^n\mathbb{E}[X^n]}{n!}+\cdots
$$
Equivalently, we can express this general expansion as
$$
M_X(t)=\sum_{n=0}^\infty \mathbb{E}[X^n]\frac{t^n}{n!}.
$$
It is this expression that shows why $M_X(t)$ is called an MGF. The coefficients of each power of $t$ contains a moment of $X$, and by differentiating $M_X(t)$ $n$ times with respect to $t$ and then setting to $t=0$, we obtain the $n$-th moment about the origin.
$$
M_X^{(n)}(0)=\mathbb{E}[X^n]
$$

### 1.4 — Wick's Theorem

Let $X\sim\mathcal{N}(\mu, \sigma^2)$, then its MGF is[^normal-mgf]
$$
M_X(t)=\mathbb{E}\left[ e ^{tX} \right]=\exp\left( \mu t+\frac{\sigma^2t^2}{2} \right),\quad t\in\mathbb{R}
$$
A full expansion is worked out in *The Book of Statistical Proofs*. 

In the centered case, $Z\sim \mathcal{N}(0, \sigma^2)$, the MGF reduces to
$$
M_Z(t)=\mathbb{E}[e^{tZ}]=\exp\left( \frac{\sigma^2t^2}{2} \right), \quad t\in\mathbb{R}
$$
Here, $t$ is an auxiliary variable. In PDLT, this auxiliary variable is equivalently referred to as the source term $J$. 

Expanding this MGF as a Maclaurin series
$$
\exp\left( \frac{\sigma^2t^2}{2} \right)=\sum_{m=0}^{\infty} \frac{1}{m!}\left(\frac{\sigma^2t^2}{2}\right)^m
$$
Therefore,
$$
M_Z(t)=\sum_{m=0}^\infty \frac{\sigma^{2m}}{2^mm!}t^{2m}.
$$
The expansion of $M_Z(t)$ contains only even powers of $t$. Therefore, the coefficients of all odd powers are zero, so all odd moments vanish:
$$
\mathbb{E}[Z^{2m+1}]=0.
$$
For even moments, recall the previous general expansion from [[#1.3 Moment-Generating Functions]]. We can apply this general expansion to the centered Gaussian $Z$,
$$
M_Z(t)=\mathbb{E}[e^{tZ}]=\sum_{k=0}^\infty\mathbb{E}[Z^k]\frac{t^k}{k!}
$$
We know that from this expansion, the coefficient of $t^{2m}$ is
$$
\frac{\mathbb{E}[Z^{2m}]}{(2m)!}
$$
We can equate this coefficient of $t^{2m}$ from the general expansion to the explicit Gaussian expansion one since we know that
$$
M_Z(t)=\sum_{k=0}^\infty\mathbb{E}[Z^k]\frac{t^k}{k!}=\sum_{m=0}^\infty \frac{\sigma^{2m}}{2^mm!}t^{2m}.
$$
Therefore,
$$
\frac{\mathbb{E}[Z^{2m}]}{(2m)!}=\frac{\sigma^{2m}}{2^mm!}
$$
Solving for the $2m$-th moment,
$$
\mathbb{E}[Z^{2m}]=\frac{(2m)!}{2^mm!}\sigma^{2m}.
$$
Finally, since 
$$
\begin{align*}
\frac{(2m)!}{2^m m!}&=\frac{(2m)\cdot (2m-1)\cdot (2m-2)\cdots 1}{(2\cdot 2 \cdots 2)\cdot(m\cdot (m-1)\cdot (m-2)\cdots 1)} \\ &=\frac{(2m)\cdot (2m-1)\cdot (2m-2)\cdots 1}{(2m\cdot (2m-2)\cdot (2m-4)\cdots 2)}\\ &=(2m-1)\cdot (2m-3)\cdots 1 \\ &=(2m-1)!!,
\end{align*}
$$
then
$$
\mathbb{E}[Z^{2m}]=(2m-1)!!\sigma^{2m}.
$$
Therefore, our general formula for the $k$-th centered Gaussian moment is:
$$
\mathbb{E}[Z^k]=\begin{cases} (2m-1)!!\sigma^{k} & \text{if } k \text{ is even} \\ 0 & \text{if } k \text{ is odd} \end{cases}
$$
The above is what is known as Wick's theorem in the univariate centered Gaussian case. But the result still feels somewhat cryptic, so we need to ask ourselves:
- What exactly does this expression, and Wick's theorem, tell us intuitively? 
- How does it tie in with the mathematical statement shown in the [[#Abstract]]?

### 1.5 — Combinatorial Intuition for Wick's Theorem

To understand the reasoning for the above expression with respect to the even moments, we first consider $Z^{2m}$ as a product of $2m$ factors:
$$
Z^{2m}=Z\cdot Z\cdot \cdots \cdot Z.
$$

Assign an index to each factor, so that we can distinguish the positions of the factors in the product:
$$
(Z_1,Z_2,Z_3,\dots,Z_{2m}).
$$
In the univariate case, these labels do not represent independent random variables. Rather, they label the $2m$ occurrences of the same centered Gaussian random variable $Z$ inside the product. Although the reason for this distinction might not be entirely clear in the univariate case, it will become clear in the multivariate case; it still helps illustrate our point here.

With this assignment, the product of $2m$ factors above can be written as
$$
Z_{i_1}\cdot Z_{i_2}\cdots Z_{i_{2m}},
$$
where $i_k$ corresponds to the $k$-th selected factor. As one can imagine, many orderings of indices $(i_1,i_2,\dots,i_{2m})$ can be made (with repetition allowed), but in the univariate case this is redundant because every selected factor is still the same random variable $Z$. Again, the idea will be useful in the multivariate case.

A pair is a two-element subset of indices, such as $\{\alpha,\beta\}$. The contribution of this pair is the expected value of the product of the two corresponding factors. In the univariate case, since all factors are occurrences of the same random variable, this contribution is the same for every pair. More precisely, for a pair $\{\alpha,\beta\}$,
$$
\mathbb{E}[Z_\alpha Z_\beta]
=
\mathbb{E}[Z^2]
=
\operatorname{Var}(Z)
=
\sigma^2.
$$

With the necessary previous context, Wick's theorem says that **the expectation of the above product of $2m$ factors $Z_{i_1}\cdot Z_{i_2}\cdots Z_{i_{2m}}$ is obtained by summing over all possible pairings of these $2m$ factors**.

This is a combinatorics problem that can be framed as follows: provide all possible unordered collections of unordered pairs of indices $\{i_1,\dots,i_{2m}\}$ without replacement. A pairing is one such unordered collection of unordered pairs. Visually, a pairing has the form
$$
p=
\left\{
\{\alpha_1,\beta_1\},
\{\alpha_2,\beta_2\},
\dots,
\{\alpha_m,\beta_m\}
\right\}.
$$

The above object is one pairing. We denote by $P^2_{2m}$ the set of all pairings of $\{1,\dots,2m\}$. Equivalently, $P^2_{2m}$ is the set of all partitions of $\{1,\dots,2m\}$ into pairs.

If we have $2m$ indices to pair, then each pairing contains $m$ pairs. The first pair can be made into
$$
2m(2m-1)
$$
ordered pairs, or
$$
\frac{2m(2m-1)}{2!}
$$
unordered pairs, which is what we want. Repeating the process for the second pair gives
$$
\frac{(2m-2)(2m-3)}{2!}
$$
possible unordered pairs. If we continue this process for all $m$ pairs, then we obtain
$$
\frac{2m(2m-1)}{2!}
\cdot
\frac{(2m-2)(2m-3)}{2!}
\cdots
\frac{2\cdot 1}{2!}
=
\frac{(2m)!}{2^m}.
$$

This counts ordered collections of unordered pairs. Since a pairing is an unordered collection of unordered pairs, we must also divide by $m!$ to remove the ordering of the $m$ pairs themselves. Therefore, the number of pairings is
$$
\frac{(2m)!}{2^m m!}
=
(2m-1)!!,
$$
which is exactly the factor found in the general formula above.

Having understood the reasoning for the previous factor, recall Wick's theorem once again: the expectation of the product of the $2m$ factors $Z_{i_1}\cdot Z_{i_2}\cdots Z_{i_{2m}}$ is obtained by summing over all possible pairings of these factors. Knowing that we have $(2m-1)!!$ possible pairings of these $2m$ factors, the only thing left to do is to sum up the contribution from each pairing.

For a fixed pairing $p\in P^2_{2m}$, Wick's theorem associates one second moment to each pair $\{\alpha,\beta\}\in p$. The term corresponding to the pairing $p$ is therefore
$$
\prod_{\{\alpha,\beta\}\in p}
\mathbb E[Z_\alpha Z_\beta].
$$
In the univariate case, every factor in this product is the same:
$$
\mathbb E[Z_\alpha Z_\beta]
=
\mathbb E[Z^2]
=
\sigma^2.
$$
Since a pairing of $2m$ factors contains exactly $m$ pairs, the term corresponding to any fixed pairing is
$$
\prod_{\{\alpha,\beta\}\in p}
\mathbb E[Z_\alpha Z_\beta]
=
\prod_{\{\alpha,\beta\}\in p}
\sigma^2
=
(\sigma^2)^m
=
\sigma^{2m}.
$$
Wick's theorem says that the full moment is obtained by summing these terms over all pairings:
$$
\mathbb E[Z^{2m}]
=
\sum_{p\in P^2_{2m}}
\prod_{\{\alpha,\beta\}\in p}
\mathbb E[Z_\alpha Z_\beta].
$$
Since there are $(2m-1)!!$ pairings and each pairing contributes $\sigma^{2m}$, we obtain
$$
\mathbb E[Z^{2m}]
=
(2m-1)!!\,\sigma^{2m}.
$$

***
## Part II: Multivariate Gaussian Case

We now extend the ideas from the univariate case to multivariate Gaussian random vectors. As in the [[#Univariate Gaussian Case]], we review the same topics before deriving the general Wick's theorem result.

### 2.1 — Gaussian Density

Let $\mathbf{Z}=(Z_1, \dots, Z_n)$ be a zero-mean multivariate Gaussian random vector with covariance matrix $\Sigma=\left(\Sigma_{ij}\right)_{i,j=1}^n$. We write the following
$$
\mathbf{Z}\sim\mathcal{N}(\mathbf{0}, \Sigma)
$$
If $\Sigma$ is nonsingular, then it is symmetric positive definite, and the density of $\mathbf{Z}$ is[^multivariate-normal-density]
$$
f_{\mathbf{Z}}(\mathbf{z})=\frac{1}{\sqrt{\left(2\pi\right)^n \text{det}(\Sigma)}}\exp\left(-\frac{1}{2}\mathbf{z}^T \Sigma^{-1}\mathbf{z}\right).
$$
where $\mathbf{z}=(z_1, \dots, z_n)^T$. This density formula assumes that $\Sigma$ is nonsingular, equivalently positive definite in this covariance-matrix setting. If $\Sigma$ is only positive semidefinite, then the Gaussian distribution can still be defined, but it may be degenerate and need not have a density with respect to Lebesgue measure. Wick's theorem itself does not require nonsingularity.

Note again that the constant is chosen such that the density integrates to $1$ as follows
$$
\int_{\mathbb{R}^n} f_{\mathbf{Z}}(\mathbf{z})\,d\mathbf{z}
=
\int_{-\infty}^{\infty}\cdots\int_{-\infty}^{\infty}
f_\mathbf{Z}(z_1,z_2,\dots,z_n)\,dz_1\,dz_2\cdots dz_n
=1.
$$

### 2.2 — Expectations and Joint Moments

By the Law of the Unconscious Statistician (LOTUS), if $\mathbf{Z}$ is a continuous random vector with density $f_\mathbf{Z}$, then the expectation of a suitable function $g(\mathbf{Z})$ is defined as[^lotus]
$$
\mathbb{E}[g(\mathbf{Z})]=\int_{-\infty}^{\infty}g(\mathbf{z})f_\mathbf{Z}(\mathbf{z})d\mathbf{z}
$$
Now, we take the function $g(\mathbf{z})=z_{i_1}z_{i_2}\cdots z_{i_{2m}}$ for indices $(i_1, \dots, i_{2m}) \in\{1,\dots, n \}$, note that these are not necessarily distinct, meaning repetition is allowed. Therefore, the joint moment of order $2m$ is
$$
\mathbb{E}[Z_{i_1}Z_{i_2}\dots Z_{i_{2m}}]=\int_{-\infty}^{\infty}\dots\int_{-\infty}^{\infty} z_{i_1}z_{i_2}\dots z_{i_{2m}}\frac{1}{\sqrt{(2\pi)^n \text{det}(\Sigma)}}\exp\left( -\frac{1}{2}\mathbf{z}^T\Sigma ^{-1}\mathbf{z} \right)dz_1\dots dz_{n}
$$
In the univariate case, our natural choice was $g(z)=z^k$ since we only had to compute the $k$-th moment $\mathbb{E}[Z^k]$ of our single random variable $Z$. In the multivariate case, the idea of a single $k$-th moment is put aside. Instead, there is instead a family of *joint* moments, particularly one for each choice of indices $(i_1, \dots, i_{2m})$. 

We can generalize the univariate version's product $z^k$ to the multivariate version's as follows
$$
g(\mathbf{z})=z_{i_1}z_{i_2}\cdots z_{i_{2m}}.
$$
The above monomial picks out one coordinate of $\mathbf{z}$ per position in the list, with repeats allowed (e.g., $g(\mathbf{z})=z_1z_1z_3z_3=z_1^2z_3^2$, where $i_1=i_2=1$, $i_3=i_4=3$, so the coordinate-label tuple is $(1, 1, 3, 3)$). It is the choice of the tuple $(i_1, \dots, i_{2m})$ that determines which joint moment $g$ computes. We can see this by taking the expectation of $g(\mathbf{Z})$, which is the function applied to the random vector:
$$
\mathbb{E}[g(\mathbf{Z})]=\mathbb{E}[Z_{i_1}Z_{i_2}\cdots Z_{i_{2m}}],
$$
which is the joint moment of order $2m$.

### 2.3 — Moment-Generating Functions

Recall our review of MGFs from section  [[#1.3 — Moment-Generating Functions]]. Fortunately, the same idea extends directly to random vectors. The MGF of a random vector $\mathbf{X}=(X_1, \dots, X_n)$ is defined by[^moment-generating-function]
$$
M_{\mathbf{X}}(\mathbf{t})\coloneqq \mathbb{E}[e^{\mathbf{t}^T\mathbf{X}}],\quad \mathbf{t}=(t_1, \dots, t_n)\in\mathbb{R}^n,
$$
where $\mathbf{t}^T\mathbf{X}=t_1X_1+t_2X_2+\cdots+t_nX_n$ is simply just the dot product between the auxiliary vector $\mathbf{t}$ and the random vector $\mathbf{X}$. Expanding the exponential as a Maclaurin series, as before, but with $\mathbf{t}^T\mathbf{X}$ replacing $tX$,
$$
e^{\mathbf{t}^T\mathbf{X}}=\sum_{k=0}^{\infty}\frac{(\mathbf{t}^T\mathbf{X})^k}{k!}=1+\mathbf{t}^T\mathbf{X}+\frac{(\mathbf{t}^T\mathbf{X})^2}{2!}+\cdots.
$$
Under conditions that are satisfied for Gaussian random variables, we may again exchange the expectation with the infinite series (see [[#Appendix A — Exchanging an Infinite Sum and an Expectation]]) and write
$$
M_\mathbf{X}(\mathbf{t})=1+\mathbb{E}[\mathbf{t}^T\mathbf{X}]+\frac{\mathbb{E}[(\mathbf{t}^T\mathbf{X})^2]}{2!}+\cdots=\sum_{k=0}^\infty \frac{\mathbb{E}[(\mathbf{t} ^T\mathbf{X})^k]}{k!}
$$
Each term $(\mathbf{t}^T\mathbf{X})^k$ is a sum over products of coordinates, and expanding them via the multinomial theorem allows us to uncover the richer structure of the multivariate case that we need to obtain the individual joint moments we want,  $\mathbb{E}[X_{i_1}X_{i_2}\cdots X_{i_{2m}}]$, which are indexed by tuples of selected coordinates.[^multinomial-theorem]

Fortunately, to extract a particular joint moment, we do not need to expand these multinomials. Instead, we can differentiate them directly, similar to the univariate case. To understand why, we must first differentiate $M_\mathbf{X}(\mathbf{t})=\mathbb{E}[e^{\mathbf{t}^T\mathbf{X}}]$ with respect to a single auxiliary variable $t_j$:
$$
\frac{\partial }{\partial t_j}\mathbb{E}[e^{\mathbf{t}^T\mathbf{X}}]=\mathbb{E}\left[\frac{\partial}{\partial t_j}e^{\mathbf{t} ^T\mathbf{X}}\right]=\mathbb{E}\left[X_je^{\mathbf{t}^T\mathbf{X}}\right],
$$
where the swap of the expected value outside of the derivative is justified by an argument that is analogous to [[#Appendix A — Exchanging an Infinite Sum and an Expectation]], but via the dominated convergence theorem instead.[^differentiate-under-expectation] 

By differentiating $2m$ times with respect to each of $t_{i_1}, t_{i_2}, \dots t_{i_{2m}}$, we end up with
$$
\frac{\partial ^{2m}}{\partial t_{i_1}\partial t_{i_2}\cdots \partial t_{i_{2m}}}e^{\mathbf{t}^T\mathbf{X}}=X_{i_1}X_{i_2}\cdots X_{i_{2m}}e^{\mathbf{t}^T\mathbf{X}}.
$$
Subsequently, evaluating at $\mathbf{t}=0$ sends $\mathbf{e}^{\mathbf{t} ^T\mathbf{X}}\rightarrow e^0=1$, which leaves exactly the target product $X_{i_1}\cdots X_{i_{2m}}$ inside of the expectation:
$$
\frac{\partial^{2m}}{\partial t_{i_1}\partial t_{i_2}\cdots \partial t_{i_{2m}}} M_{\mathbf{X}}(\mathbf{t})\bigg|_{\mathbf{t}=\mathbf{0}} = \mathbb{E}[X_{i_1}X_{i_2}\cdots X_{i_{2m}}].
$$
The above is the direct multivariate generalization of $M_X^{(k)}(0)=\mathbb{E}[X^k]$ from section [[#1.3 — Moment-Generating Functions]]. 

It is a worthwhile exercise to rigorously derive the above formula since we mostly gloss over some of the differentiation and combinatoric details. We work through these details explicitly in [[#Appendix B — Explicit Multinomial Expansion]], showing how the multinomial coefficient produced by expanding $(\mathbf{t}^T\mathbf{X})^{2m}$ exactly cancels the factorials produced by repeated differentiation. We also illustrate an example of applying the above formula.

### 2.4 — Wick's Theorem

Let $\mathbf{X}\sim\mathcal{N}(\mu, \Sigma)$, then its MGF is[^multi-norm-mgf]
$$
M_\mathbf{X}(\mathbf{t})=\mathbb{E}\left[e^{\mathbf{t} ^T\mathbf{X}}\right] = \exp\left( \mathbf{t} ^T\mu+\frac{1}{2}\mathbf{t} ^T\Sigma \mathbf{t} \right), \quad \mathbf{t}\in\mathbb{R}^n.
$$
Again, a full expansion of the above is worked out in *The Book of Statistical Proofs*.

In the centered case, $\mathbf{Z}\sim \mathcal{N}(0, \Sigma)$, the MGF reduces to
$$
M_\mathbf{Z}(\mathbf{t})=\mathbb{E}\left[e ^{\mathbf{t}^T\mathbf{Z}}\right
]=\exp\left( \frac{1}{2}\mathbf{t}^T\Sigma \mathbf{t} \right),\quad \mathbf{t}\in\mathbb{R}^n.
$$
Note that in PDLT, $J^\mu$ is the $\mu$-th source term, which is the same as the auxiliary vector's $i$-th term, $t_{i}$. 

Expanding this MGF as a Maclaurin series
$$
\exp\left( \frac{1}{2}\mathbf{t}^T\Sigma \mathbf{t} \right)=\sum_{m=0}^\infty\frac{1}{m!}\left( \frac{1}{2}\mathbf{t}^T\Sigma \mathbf{t} \right)^m
$$
Therefore, 
$$
M_\mathbf{Z}(\mathbf{t})=\sum_{m=0}^\infty \frac{1}{2 ^m m!}\left( \mathbf{t}^T\Sigma \mathbf{t}\right)^m
$$
Now, expand the inner term of the exponential into a double summation,
$$
M_\mathbf{Z}(\mathbf{t})=\sum_{m=0}^\infty \frac{1}{2 ^m m!}\left( \sum_{\alpha,\beta=1}^n t_\alpha\Sigma_{\alpha\beta}t_\beta\right)^m.
$$
The inner object $\sum_{\alpha,\beta} t_\alpha\Sigma_{\alpha\beta}t_\beta$ is homogeneous of degree $2$ in $\mathbf{t}$. Therefore, its $m$-th power is homogeneous of degree $2m$, so the Gaussian MGF contains only even-degree terms.

Recall from section [[#2.3 — Moment-Generating Functions]] that to compute $\mathbb{E}[Z_{i_1}\cdots Z_{i_k}]$, we differentiate $M_\mathbf{Z}(\mathbf{t})$ once with respect to each of $t_{i_1}, \dots, t_{i_k}$ and then evaluate at $\mathbf{t}=0$. Since the Gaussian MGF contains only even-degree terms, there is no degree-$k$ term for this derivative to extract when $k$ is odd. Therefore,
$$
\mathbb{E}[Z_{i_1}Z_{i_2}\cdots Z_{i_k}]=\frac{\partial ^k}{\partial t_{i_1}\cdots \partial t_{i_k}}M_{\mathbf{Z}}(\mathbf{t})\bigg|_{\mathbf{t}=\mathbf{0}}=0, \quad \text{when } k \text{ is odd}
$$
From the same section above, we know that the general Maclaurin series of the MGF for *any* random vector $\mathbf{X}$ is,[^moment-generating-function]
$$
M_{\mathbf{X}}(\mathbf{t})=\mathbb{E}[e^{\mathbf{t}^T\mathbf{X}}]=\sum_{k=0}^\infty\frac{\mathbb{E}[(\mathbf{t}^T\mathbf{X})^k]}{k!}
$$
On the other hand, we previously obtained the zero-mean Gaussian random vector-specific Maclaurin series of the MGF. 
$$
M_\mathbf{Z}(\mathbf{t})=\sum_{m=0}^\infty \frac{1}{2 ^m m!}\left( \mathbf{t}^T\Sigma \mathbf{t}\right)^m
$$
Both expressions are Maclaurin series of the same function $M_{\mathbf{Z}}(\mathbf{t})$ (if we set $\mathbf{X}=\mathbf{Z}$ in the general formula above), which are the power series expansions of $M_\mathbf{Z}(\mathbf{t})$ centered at $\mathbf{t}=0$. Since a convergent Maclaurin series representation of a function is unique, the coefficients of matching-degree terms in the two expansions must be equivalent.
$$
M_\mathbf{Z}(\mathbf{t})=\sum_{k=0}^\infty\frac{\mathbb{E}[(\mathbf{t}^T\mathbf{Z})^k]}{k!}=\sum_{m=0}^\infty \frac{1}{2 ^m m!}\left( \mathbf{t}^T\Sigma \mathbf{t}\right)^m
$$
For the Gaussian-specific series, we know that this expansion contributes only monomials of total degree $2m$ at its $m$-th term (as established above). This means that the general series has to match its terms at even orders $k=2m$; notice this also immediately reproduces the vanishing odd moments result as well, viewed as a consequence of directly matching both of these series. Equating the degree-$2m$ terms of both series then gives
$$
\frac{\mathbb{E}\left[(\mathbf{t}^T\mathbf{Z})^{2m}\right]}{(2m)!}=\frac{1}{2^mm!}(\mathbf{t}^T\Sigma \mathbf{t})^m,
$$
or equivalently,
$$
\mathbb{E}\left[(\mathbf{t}^T\mathbf{Z})^{2m}\right]=\frac{(2m)!}{2^mm!}(\mathbf{t}^T\Sigma \mathbf{t})^m,
$$
This is a finite and fully explicit polynomial identity in $\mathbf{t}$ now. In order to compute the specific joint moment, $\mathbb{E}[Z_{i_1}Z_{i_2}\cdots Z_{i_{2m}}]$, all that is left to do is differentiate both sides with respect to $t_{i_1}, \dots, t_{i_{2m}}$ and evaluate at $\mathbf{t}=0$. 
$$
\frac{\partial ^{ 2m}}{\partial t_{i_1}\partial t_{i_2}\cdots \partial t_{i_{2m}}}\mathbb{E}\left[(\mathbf{t}^T\mathbf{Z})^{2m}\right]=\frac{(2m)!}{2^mm!}\left[\frac{\partial ^{ 2m}}{\partial t_{i_1}\partial t_{i_2}\cdots \partial t_{i_{2m}}}(\mathbf{t}^T\Sigma \mathbf{t})^m\bigg|_{\mathbf{t}=\mathbf{0}}\right].
$$
Per [[#Appendix B — Explicit Multinomial Expansion]], the left hand side ends up reducing to $(2m)!\mathbb{E}[Z_{i_1}Z_{i_2}\cdots Z_{i_{2m}}]$, therefore,
$$
\boxed{
\mathbb{E}\left[Z_{i_1}Z_{i_2}\cdots Z_{i_{2m}}\right]=\frac{1}{2^mm!}\left[\frac{\partial ^{ 2m}}{\partial t_{i_1}\partial t_{i_2}\cdots \partial t_{i_{2m}}}(\mathbf{t}^T\Sigma \mathbf{t})^m\bigg|_{\mathbf{t}=\mathbf{0}}\right]}.
$$
This derivative formula gives the desired moment, but it does not yet display the pairing structure from the [[#Abstract]]. To recover that form, expand $(\mathbf{t}^T\Sigma\mathbf{t})^m=\left(\sum_{\alpha,\beta=1}^n t_\alpha\Sigma_{\alpha\beta}t_\beta\right)^m$ as follows:
$$
\left(  \sum_{\alpha,\beta=1}^n t_\alpha\Sigma_{\alpha\beta} t_\beta\right)^m=\sum_{\alpha_1,\beta_1=1}^n\cdots \sum_{\alpha_m,\beta_m=1}^n\left(t_{\alpha_1}\Sigma_{\alpha_1 \beta_1}t_{\beta_1}\right)\cdots\left(t_{\alpha_m}\Sigma_{\alpha_m \beta_m}t_{\beta_m}\right).
$$
We can simplify the above summation by multiplying the terms as follows,
$$
\left( \sum_{\alpha,\beta=1}^n t_\alpha \Sigma_{\alpha\beta}t_\beta \right)^m=\sum_{(\alpha_1, \beta_1), \dots, (\alpha_m, \beta_m)\in\{1,\dots, n\}^ 2} \Sigma_{\alpha_1\beta_1}\Sigma_{\alpha_2\beta_2}\cdots\Sigma_{\alpha_m\beta_m} t_{\alpha_1}t_{\beta_1}t_{\alpha_2}t_{\beta_2}\cdots t_{\alpha_m}t_{\beta_m}.
$$
This is a sum of $n^{2m}$ terms, one for each ordered sequence of $m$ index-pairs $(\alpha_1, \beta_1), \dots, (\alpha_m,\beta_m)$ which are drawn from $\{ 1,\dots,n \}^2$. More intuitively, at each slot in each pair, you have $n$ choices, for which if you repeat $2m$ times you get $n^{2m}$. We can substitute this expression back into the general-use formula above.
$$
\frac{1}{2^mm!}\left[\frac{\partial ^{ 2m}}{\partial t_{i_1}\partial t_{i_2}\cdots \partial t_{i_{2m}}}\left(\sum_{(\alpha_1, \beta_1), \dots, (\alpha_m, \beta_m)\in\{1,\dots, n\}^ 2} \Sigma_{\alpha_1\beta_1}\Sigma_{\alpha_2\beta_2}\cdots\Sigma_{\alpha_m\beta_m} t_{\alpha_1}t_{\beta_1}t_{\alpha_2}t_{\beta_2}\cdots t_{\alpha_m}t_{\beta_m}\right)\bigg|_{\mathbf{t}=\mathbf{0}}\right]
$$
We first fix one term of the sum that corresponds to a particular choice of coordinate-label pairs $(\alpha_1, \beta_1), \dots, (\alpha_m, \beta_m)$. The monomial corresponding to it is $t_{\alpha_1}t_{\beta_1}\cdots t_{\alpha_{m}}t_{\beta_m}$, which is a product of $2m$ auxiliary variables drawn with repetition allowed from $\{t_1, \dots, t_n\}$. 

After differentiating with respect to $t_{i_1}t_{i_2}\cdots t_{i_{2m}}$ and then setting $\mathbf{t}=\mathbf{0}$, if the multiset $\{\alpha_1, \beta_1, \dots, \alpha_m, \beta_m\}$ does not exactly equal the multiset $\{i_1, \dots, i_{2m}\}$, then the term is annihilated; this since either some auxiliary variable never appears in the monomial (which kills the term instantly), or some other auxiliary variable gets over or under-differentiated relative to its multiplicity, which leaves either $0$ or a nonzero power of it that vanishes upon setting $\mathbf{t}=0$. On the other hand, if the aforementioned multisets do equal each other, then the monomial is exactly $t_{i_1}t_{i_2}\cdots t_{i_{2m}}$, and differentiating once with respect to each gives coefficient $1$. 

Here, a pairing should be understood as a partition of the positions $\{1,\dots,2m\}$, not as a partition of the coordinate-label tuple $(i_1,\dots,i_{2m})$ itself. The tuple only tells us which Gaussian coordinate sits in each position. Thus, if $\{\alpha,\beta\}$ is one pair in a pairing, it selects the two factors $Z_{i_\alpha}$ and $Z_{i_\beta}$.

Each surviving choice of coordinate-label pairs determines one such partition of the positions $\{1,\dots,2m\}$ into $m$ unordered pairs. But many of these choices give the same positional pairing for two reasons: swapping the two coordinate labels inside any pair does not change $\Sigma_{\alpha_k\beta_k}=\Sigma_{\beta_k\alpha_k}$, giving a $2^m$ redundancy across $m$ pairs; and permuting the $m$ pair labels does not change the product, giving an additional $m!$ redundancy. So every distinct pairing $p$ of the positions is represented by exactly $2^m\cdot m!$ surviving terms, which each contribute the identical value obtained by multiplying $\Sigma_{i_\alpha i_\beta}$ over the positional pairs $\{\alpha,\beta\}\in p$. 

Therefore, after having differentiated and evaluated at $\mathbf{t}=\mathbf{0}$,
$$
\sum_{\text{surviving terms}}\Sigma_{\alpha_1 \beta_1}\cdots \Sigma_{\alpha_m\beta_m}=2^m m!\sum_{p\in P^2_{2m}}\prod_{\{\alpha, \beta\}\in p}\Sigma_{i_\alpha i_\beta}.
$$
where $P^2_{2m}$ is the set of all pairings of the positions $\{1, \dots, 2m\}$ into $m$ unordered pairs, and there are $(2m-1)!!$ such pairings in total (also established in [[#1.5 — Combinatorial Intuition for Wick's Theorem]]). Finally, multiplying by the $\frac{1}{2^m m!}$ prefactor that is present in the general-use moment formula cancels this factor exactly:
$$
\mathbb{E}\left[ Z_{i_1}\cdots Z_{i_{2m}} \right]=\frac{1}{2^m m!}\cdot 2^mm!\sum_{p\in P^2_{2m}} \prod_{\{\alpha,\beta\}\in p}\Sigma_{i_\alpha i_\beta}=\sum_{p\in P^2_{2m}}\prod_{\{\alpha,\beta\}\in p}\Sigma_{i_\alpha i_\beta}.
$$
We can rephrase a bit here to obtain the final version of Wick's theorem as originally shown in the abstract. Since $\mathbf{Z}$ is centered, each entry of the covariance matrix satisfies
$$
\Sigma_{\alpha\beta}
=\operatorname{Cov}(Z_\alpha,Z_\beta)
=\mathbb{E}[Z_\alpha Z_\beta].
$$
To avoid ambiguity when some of the selected coordinate labels repeat, each pairing is viewed as a partition of the positions $\{1,\dots,2m\}$ rather than directly as a partition of the values in the tuple $(i_1,\dots,i_{2m})$. Thus, if $p\in P^2_{2m}$ and $\{\alpha,\beta\}\in p$, then the pair $\{\alpha,\beta\}$ selects the two Gaussian coordinates $Z_{i_\alpha}$ and $Z_{i_\beta}$. Therefore,
$$
\prod_{\{\alpha,\beta\}\in p}\Sigma_{i_\alpha i_\beta}
=
\prod_{\{\alpha,\beta\}\in p}\mathbb{E}[Z_{i_\alpha}Z_{i_\beta}].
$$
Substituting this into the previous pairing sum gives the final multivariate Wick formula:
$$
\boxed{
\begin{aligned}
\mathbb{E}\left[Z_{i_1}Z_{i_2}\cdots Z_{i_{2m}}\right]
&=
\sum_{p\in P^2_{2m}}
\prod_{\{\alpha,\beta\}\in p}
\mathbb{E}\left[Z_{i_\alpha}Z_{i_\beta}\right] \\
&=
\sum_{p\in P^2_{2m}}
\prod_{\{\alpha,\beta\}\in p}
\operatorname{Cov}\left(Z_{i_\alpha},Z_{i_\beta}\right).
\end{aligned}
}
$$
where $P^2_{2m}$ is the set of all pairings of the positions $\{1,\dots,2m\}$. This is the multivariate version of the same pairing structure we saw in the univariate case: **higher even moments of a centered Gaussian are completely determined by sums of products of second moments.**

***
## Appendix

### Appendix A — Exchanging an Infinite Sum and an Expectation

Swapping an expectation with an infinite sum requires justification. In general, the identity $\mathbb{E}[\sum_n a_n(X)]=\sum_n \mathbb{E}[a_n(X)]$ need not hold without additional convergence assumptions.

To be a bit more technical, we say that the sum over the natural numbers can be written as an integral against the counting measure on $\mathbb{N}$. So if $\mu$ is the measure assigning a mass of 1 to each nonnegative integer, then by definition we have that 
$$
\sum_{n=0}^\infty a_n =\int_{\mathbb{N}}a_nd\mu(n).
$$
This is the definition of the Lebesgue integral with respect to counting measure, so, 
$$
\mathbb{E}\left[ \sum_n a_n(X)\right]\quad \text{vs. } \quad \sum_n \mathbb{E}[a_n(X)]
$$
can be considered really as 
$$
\int_{\mathbb{R}}\left( \int_{\mathbb{N}}a_n(x)d\mu(n) \right)f_X(x)dx \quad \text{vs .} \quad \int_{\mathbb{N}}\left( \int_\mathbb{R} a_n(x) f_X(x) dx \right)d\mu(n),
$$
which is now a double integral over the product space $\mathbb{R}\times \mathbb{N}$, with the two factors being the Lebesgue measure weighted by $f_X(x)$ on $\mathbb{R}$, and the counting measure on $\mathbb{N}$. 

The standard sufficient condition to guarantee this swap is absolute convergence of the resulting series. More precisely, the following inequality needs to hold:
$$
\sum_{n=0}^{\infty}\frac{|t|^n}{n!}\mathbb{E}[|X|^n]<\infty
$$
for $t$ in some open interval around 0. This is the condition for [Fubini's theorem](https://en.wikipedia.org/wiki/Fubini's_theorem) (or Tonelli's theorem, for nonnegative terms), which allows us to swap the sum and the expectation as long as the absolute-convergence condition above holds.[^fubini-infinite-sums] 

For Gaussian $X$, this absolute-convergence condition holds. One way to verify it is by applying a ratio test to the resulting moment series; the Gaussian tail decay, of order $e^{-x^2/2}$, is strong enough to control the exponential series near the origin.

### Appendix B — Explicit Multinomial Expansion

The multinomial theorem states that, for any positive integer $d$ and any nonnegative integer $q$, we can expand a sum of $d$ terms raised to the $q$-th power as follows:[^multinomial-theorem]
$$
(x_{1}+x_{2}+\cdots +x_{d})^{q}=\sum _{\begin{array}{c}k_{1}+k_{2}+\cdots +k_{d}=q\\k_{1},k_{2},\cdots ,k_{d}\geq 0\end{array}}{q \choose k_{1},k_{2},\ldots ,k_{d}}x_{1}^{k_{1}}\cdot x_{2}^{k_{2}}\cdots x_{d}^{k_{d}}.
$$
Note that this formulation treats both the coefficient and the variable as a single term. 

In our application, the number of terms is the number of coordinates $d=n$, and the exponent is the even order $q=2m$. We can directly apply this to the dot product of the auxiliary vector and random vector, both $n$-dimensional:
$$
\begin{align*}
(\mathbf{t}^T\mathbf{X})^{2m}&=(t_1X_1+t_2X_2+\cdots+t_nX_n)^{2m} \\
&=\sum _{\begin{array}{c}k_{1}+k_{2}+\cdots +k_{n}=2m\\k_{1},k_{2},\cdots ,k_{n}\geq 0\end{array}}{2m \choose k_{1},k_{2},\ldots ,k_{n}}(t_1X_1)^{k_1}\cdot(t_2X_2)^{k_2}\cdots (t_nX_n)^{k_n}
\end{align*}
$$
When differentiating the MGF $2m$ times with respect to the auxiliary variables $t_{i_1}, \dots t_{i_{2m}}$ corresponding to the selected indices $(i_1, \dots, i_{2m})$, consider that all terms in the infinite series expansion of the MGF, as shown in [[#2.3 — Moment-Generating Functions]], vanish since they are of order less than $2m$. Next, consider that after differentiation, we subsequently set $\mathbf{t}=0$ to the remaining terms of the expression, thus eliminating all terms of order greater than $2m$. Therefore, after differentiation and setting the auxiliary vector be equal to 0, we are left with the only remaining term of order exactly $2m$, which is exactly the joint moment we are targeting. 
$$
\frac{\partial^{2m}}{\partial t_{i_1}\partial t_{i_2}\cdots \partial t_{i_{2m}}} M_{\mathbf{X}}(\mathbf{t})\bigg|_{\mathbf{t}=\mathbf{0}} = \boxed{\frac{\partial^{2m}}{\partial t_{i_1}\partial t_{i_2}\cdots \partial t_{i_{2m}}}\left(\frac{\mathbb{E}[(\mathbf{t}^T\mathbf{X})^{2m}]}{(2m)!}\right)\bigg|_{\mathbf{t}=\mathbf{0}}}
$$
Applying linearity of expectation to the above finite sum then gives,
$$
\frac{\mathbb{E}[(\mathbf{t}^T\mathbf{X})^{2m}]}{(2m)!}=\frac{1}{(2m)!}\sum _{\begin{array}{c}k_{1}+k_{2}+\cdots +k_{n}=2m\\k_{1},k_{2},\cdots ,k_{n}\geq 0\end{array}}{2m \choose k_{1},k_{2},\ldots ,k_{n}} t_1^{k_1}t_2^{k_2}\cdots t_n^{k_n}\cdot\mathbb{E}\left[ X_1^{k_1}X_2^{k_2}\cdots X_n^{k_n} \right]
$$
We now need to differentiate this expression with respect to each of the auxiliary variables $t_{i_1}, \dots, t_{i_{2m}}$ which, again, correspond to our selected indices, and evaluate at $\mathbf{t}=0$. By doing so, only the single term whose exponents $(k_1, \dots, k_n)$ exactly match the multiplicities of $(i_1, \dots, i_{2m})$ is preserved after differentiation and evaluation at $\mathbf{t}=0$; the rest of the terms vanish. For this surviving term, differentiation contributes a factor of $k_1!k_2!\cdots k_n!$, since differentiating $t_j^{k_j}$ with respect to $t_j$ exactly $k_j$ times gives $k_j!$. This previous product of factorials exactly cancels the denominator of the multinomial coefficient
$$
 {2m \choose k_1. \dots, k_n}\cdot k_1!k_2!\cdots k_n!=(2m)!.
$$
Combined with the $\frac{1}{(2m)!}$ prefactor outside of the sum, this gives us an overall coefficient of $1$. Therefore,
$$
\frac{\partial ^{2m}}{\partial t_{i_1}\partial t_{i_2}\cdots \partial t_{i_{2m}}}\left(\frac{\mathbb{E}[(\mathbf{t}^T\mathbf{X})^{2m}]}{(2m)!}\right)\bigg|_{\mathbf{t}=0}=\mathbb{E}\left[X_{i_1}X_{i_2}\cdots X_{i_{2m}}\right]
$$
which subsequently concludes that
$$
\boxed{\frac{\partial ^{2m}}{\partial t_{i_1}\partial t_{i_2}\cdots \partial t_{i_{2m}}} M_\mathbf{X}(\mathbf{t})\bigg|_{t=0}=\mathbb{E}\left[ X_{i_1}X_{i_2}\cdots X_{i_{2m}} \right]}.
$$
To illustrate, consider the following example. Let $n=8$ with selected indices $(i_1, \dots, i_6)=(1, 1, 3, 3, 3, 6)$, where index 1 appears twice, index 3 appears three times, and index 6 appears once; where $2m=6$. This means the joint moment we are targeting is
$$
\mathbb{E}[X_{i_1}X_{i_2}\cdots X_{i_6}]=\mathbb{E}[X_1X_1X_3X_3X_3X_6]=\mathbb{E}[X_1^2X_3^3X_6].
$$
Correspondingly, since we have $n=8$ coordinates, the exponent tuple, which matches one entry per total coordinates, should then be
$$
(k_1, \dots, k_8)=(2,0,3,0,0,1,0,0),
$$
since $k_1=2$ because the first coordinate appears twice, $k_3=3$ because the third coordinate appears three times, and $k_6=1$ since the sixth coordinate appears once; all other ones are zero'ed out since they do not appear (also summing them up equals $6$). So, the $6$th term of the multinomial expansion that corresponds to this exponent tuple is
$$
{6 \choose 2,0,3,0,0,1,0,0}t_1^2t_3^3t_6^1\mathbb{E}[X_1^2X_3^3X_6],
$$
where the multinomial coefficient above is equal to
$$
{6 \choose 2,0,3,0,0,1,0,0}=\frac{6!}{2!0!3!0!0!1!0!0!}=\frac{720}{12}=60
$$
Now, by differentiating the above expression with respect to $t_1$ twice, $t_3$ three times, and $t_6$ once, and then evaluating at $\mathbf{t}=0$, we get
$$
\begin{align*}
\frac{\partial ^6}{\partial t_1^2\partial t_3^3 \partial t_6}{6 \choose 2,0,3,0,0,1,0,0}t_1^2t_3^3t_6^1\mathbb{E}[X_1^2X_3^3X_6]&=60\cdot 2!\cdot 3!\cdot 1\cdot\mathbb{E}[X_1 ^2X_3^3X_6] \\ &=720\cdot \mathbb{E}[X_1^2X_3^3X_6]
\end{align*}
$$
And, by looking closely, $60\times 12 =720=6!=(2m)!$. So this single term contributes exactly $(2m)!\mathbb{E}[X_1^2X_3^3X_6]$, and after dividing by the $\frac{1}{(2m)!}=\frac{1}{6!}$ factor from the original Maclaurin series expansion, the overall coefficient is exactly one, which simply leaves,
$$
\frac{\partial ^6}{\partial t_1^2\partial t_3 ^3 \partial t_6}\left( \frac{\mathbb{E}[(\mathbf{t} ^T\mathbf{X})^6]}{6!} \right) \bigg|_{\mathbf{t}=\mathbf{0}}=\mathbb{E}\left[ X_1^2X_3^3X_6 \right].
$$







[^isserlis-theorem]: *Wikipedia*, "[Isserlis' theorem](https://en.wikipedia.org/wiki/Isserlis%27s_theorem)."
[^normal-density]: *StatProofBook*, "[Probability density function of the normal distribution](https://statproofbook.github.io/P/norm-pdf)."
[^normal-mean]: *StatProofBook*, "[Expected value of the normal distribution](https://statproofbook.github.io/P/norm-mean.html)."
[^lotus]: *Wikipedia*, "[Law of the unconscious statistician](https://en.wikipedia.org/wiki/Law_of_the_unconscious_statistician)."
[^moment-generating-function]: *StatProofBook*, "[Moment-generating function](https://statproofbook.github.io/D/mgf)," and *Wikipedia*, "[Moment-generating function](https://en.wikipedia.org/wiki/Moment_generating_function#Definition)."
[^normal-mgf]: *StatProofBook*, "[Moment-generating function of the normal distribution](https://statproofbook.github.io/P/norm-mgf)."
[^fubini-infinite-sums]: See the Math StackExchange discussion, "[Linearity of expectation for infinite sums](https://math.stackexchange.com/questions/1166994/linearity-of-expectation-for-infinite-sums)," for the sum-expectation interchange viewpoint.
[^multi-norm-mgf]: *StatProofBook*, "[Moment-generating function of the multivariate normal distribution](https://statproofbook.github.io/P/mvn-mgf.html)"
[^multivariate-normal-density]: *StatProofBook*, "[Probability density function of the multivariate normal distribution](https://statproofbook.github.io/P/mvn-pdf.html)."
[^multinomial-theorem]: *Wikipedia*, "[Multinomial theorem](https://en.wikipedia.org/wiki/Multinomial_theorem)."
[^differentiate-under-expectation]: See the Math StackExchange discussion, "[Interchanging expectation value and derivative](https://math.stackexchange.com/questions/2519966/interchanging-expectation-value-and-derivative)," for a dominated-convergence/differentiation-under-the-integral-sign viewpoint.

%%
i think appendix should contain sections on how to obtain the normalization factors for both the univariate and the multivariate cases; i think they're still worth pointing out, especially the latter since i think it's pretty interesting with the eigenbasis stuff

i think we need to have numbers on the right side of the expressions to refer back to. for example, in the even moments mention of the coefficient, we can refer back to the general moment-generating function
%%
%%
in the multivariate case, they are not independent, and they are not identically distributed for the ordered collection of gaussians. they are jointly distributed; they might have dependencies between one another, and they might not have the same variance even though they are all centered at mean/expectation 0
- nevermind

disambiguate terminology between univariate, multivariate, random variable, random vector.

n is used throughout the writing interchangeably, especially for MGF, this needs to be fixed porbably, note the multivariate gaussian mgf using n for the expansion when it's iterating over it and also defined as the amount of auxiliary variables / gaussian coordinates in teh random vector
- hmm nevermind, i think i t should be fine, we can expand and indicate the nth one, since it would be equivalent to the 2mth? no that's not right, that's the number of gaussian coordiantes, 2m is the even number of coordinates being targeted in the joint moment. 
%%

%%
One thing that should also be noted is that the density formula requires $\Sigma$ to be invertible, although Wick's theorem does not require this to be true. Here, the only constraint on $\Sigma$ is that it be positive semidefinite, meaning that $x^T\Sigma x\geq 0$ for all $x$, but the stronger condition that guarantees it invertibility is that it be positive definite, meaning that $x^Tx>0$ for all $x$; note the strict inequality for $x\neq 0$. 
%%


%%
The univariate Gaussian function is $\exp\left(-\frac{z²}{2}\right)$, a *bell-shaped function* which is symmetrically centered on $z=0$ and which tapers off rapidly if $|z|\gg 1$. If we take the integral of the function over the interval $[-\infty, \infty]$, we end up with the following
$$
I_1\equiv \int_{-\infty}^{\infty}\exp\left(-\frac{z²}{2}\right)dz=\sqrt{2\pi} 
$$
where the previous result can be more easily obtained by squaring $I_1$ and then doing a change of variables to polar coordinates. Here, $I_1$ is what is known as the *normalization factor* of the univariate Gaussian probability distribution with unit variance. 

This unit variance is invisible within the exponential's denominator; a more general expression to account for it looks like
$$
I_K=\int_{-\infty}^{\infty} \exp\left( -\frac{z²}{2K} \right)dz=\sqrt{K}\int_{-\infty}^{\infty}\exp\left( -\frac{u²}{2} \right)du
=\sqrt{2\pi K}
$$
where $K>0$ is the variance of the Gaussian distribution. 
%%


%%
First, we note that odd moments vanish. If $k=2m+1$ where $m>0$, then
$$
z^{2m+1}f_Z(z)
$$
is an odd function because $z^{2m+1}$ is odd and $f_Z(z)$ is even. Therefore,
$$
\mathbb{E}[Z^{2m+1}]=\int_{-\infty}^{\infty}z^{2m+1}f_Z(z)\,dz=0.
$$
%%

%%
For the even moments, we first differentiate the moment-generating function once with respect to the auxiliary variable $t$:
$$
\frac{d}{d t}M_Z(t)=\sigma²tM_Z(t).
$$
Then we differentiate the above identity $n$ more times:
$$
\frac{d^{n+1}}{d t^{n+1}}M_Z(t)=\sigma² \frac{d^n}{d t^n}\left(tM_z(t)\right)
$$
The right-hand side can be computed by applying the general Leibniz rule
$$
\begin{align}
\frac{d^n}{d t^{n}}\left(tM_z(t)\right)&=\sum_{k=0}^n \begin{pmatrix}n \\ k\end{pmatrix} \left(\frac{d^{n-k}}{d t^{n-k}}t\right)\left(\frac{d^k}{d t ^k}M_Z(t)\right) \\
&=\begin{pmatrix}n \\ n-1\end{pmatrix}\left( \frac{d}{d t}t \right)\left(\frac{d^{n-1}}{d t^{n-1}}M_z(t)\right) + \begin{pmatrix}n \\ n\end{pmatrix}t\left(\frac{d^{n}}{d t^{n}}M_z(t)\right) \\
&=n\left(\frac{d^{n-1}}{d t^{n-1}}M_z(t)\right)+t\left(\frac{d^{n}}{d t^{n}}M_z(t)\right) \\ &= nM^{(n-1)}_Z(t)+tM_Z^{(n)}(t).
\end{align}
$$
Note that terms corresponding to $k\leq n-2$ disappear since $dt^k/dt=0$. In the last step, we also simplified the notation to make it more digestible; more specifically
$$
\frac{d^{k}}{d t^k}M_Z(t)= M^{(k)}_Z(t), \quad k\in\mathbb{N}_0
$$
Therefore,
$$
M^{(n+1)}_Z(t)=\sigma² \left(nM^{(n-1)}_Z(t)+tM_Z^{(n)}(t)\right)
$$
Setting $t=0$ to the above, the second term vanishes, so
$$
M_Z^{(n+1)}(0)=n\sigma²M_Z^{(n-1)}(0).
$$
The above gives the recursive formula
$$
\mathbb{E}[Z^{n+1}]=n\sigma² \mathbb{E}[Z^{n-1}].
$$
We can cleanly write the recurrence for the $k$-th Gaussian moment by specifying that $k=n+1$. This gives
$$
\mathbb{E}[Z^{k}]=(k-1)\sigma²\mathbb{E}[Z^{k-2}], \quad k\geq 2.
$$
This is a recurrence relation for the Gaussian moments. In an induction proof, the identities
$$
\mathbb{E}[Z^0]=1\quad \text{and}\quad \mathbb{E}[Z]=0
$$
serve as the base cases, while the recurrence gives the inductive step. 

Instead of going the full induction proof route, we'll avoid the rigor and simply just iterate the recurrence to illustrate the general formula. Since we already made clear that odd moments vanish, we set $k=2m$ to represent even moments. Then,
$$
\mathbb{E}[Z^{2m}]=(2m-1)\sigma² \mathbb{E}[Z^{2m-2}].
$$
Applying the same recurrence again to $\mathbb{E}[Z^{2m-2}]$ gives:
$$
\mathbb{E}[Z^{2m-2}]=(2m-3)\sigma^2\mathbb{E}[Z^{2m-4}]
$$
Substituting the above equality,
$$
\mathbb{E}[Z^{2m}]=(2m-1)(2m-3)\sigma^4\mathbb{E}[Z^{2m-4}].
$$
By iterating over the recurrence, and considering the base cases, we obtain
$$
\mathbb{E}[Z^{2m}]=(2m-1)(2m-3)\cdots 3\cdot 1 \sigma ^{2m}.
$$
By definition, 
$$
(2m-1)!! =(2m-1)(2m-3)\cdots 3\cdot 1,
$$
so, we can conclude the general formula is:
$$
\mathbb{E}[Z^{2m}]=(2m-1)!!\sigma^{2m}
.$$
For odd moments, the same recurrence reduces the power by two at each step. Thus,
$$
\mathbb{E}[Z^{2m+1}]=(2m)\sigma^2\mathbb{E}[Z^{2m-1}],
$$
and repeatedly applying the recurrence eventually gives a multiple of the previously mentioned base case $\mathbb{E}[Z]$. Since we know that $Z$ is centered, $\mathbb{E}[Z]=0$; so every odd moment ends up vanishing as previously mentioned:
$$
\mathbb{E}[Z^{2m+1}]=0.
$$

%%
