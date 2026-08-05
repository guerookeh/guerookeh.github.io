---
title: "SCRATCH - Gaussian Wick Notes"
draft: true
tags:
  - probability
  - gaussian-distributions
  - combinatorics
---

i think appendix should contain sections on how to obtain the normalization factors for both the univariate and the multivariate cases; i think they're still worth pointing out, especially the latter since i think it's pretty interesting with the eigenbasis stuff

i think we need to have numbers on the right side of the expressions to refer back to. for example, in the even moments mention of the coefficient, we can refer back to the general moment-generating function

in the multivariate case, they are not independent, and they are not identically distributed for the ordered collection of gaussians. they are jointly distributed; they might have dependencies between one another, and they might not have the same variance even though they are all centered at mean/expectation 0
- nevermind

we acknowledge that the surviving ordered sequences of $m$ coordinate-label pairs can refer to the same contribution, the reason for this is due to two redundancies, 1) within a coordinate-label pair, we can swap the labels via symmetric property of $\Sigma$, and 2) multiplication is commutative, so order in the sequence of pairs needs to be disregarded. the two previous points tell us that, for a fixed unordered collection of $m$ coordinate-label unordered pairs (disambiguate whether pairs are unordered by def.), we have $2^m\cdot m!$ different ordered collections. we can most likely rephrase ordered sequence into ordered collection. refer to the set of ordered collections corresponding to this unordered collection as $P$ and iterate over the pairs as follows
$$
\sum_{p\in P}\prod_{\{a,b\}\in p}\Sigma_{i_ai_b}
$$
every $p$ is an individual ordered collection of ordered pairs. every $p$ contains $m$ pairs, all of which produce the same contribution since multiplication is commutative. we recognize that $a,b$ are the corresponding coordinate labels. 

disambiguate terminology between univariate, multivariate, random variable, random vector.

n is used throughout the writing interchangeably, especially for MGF, this needs to be fixed porbably, note the multivariate gaussian mgf using n for the expansion when it's iterating over it and also defined as the amount of auxiliary variables / gaussian coordinates in teh random vector
- hmm nevermind, i think i t should be fine, we can expand and indicate the nth one, since it would be equivalent to the 2mth? no that's not right, that's the number of gaussian coordiantes, 2m is the even number of coordinates being targeted in the joint moment. 

One thing that should also be noted is that the density formula requires $\Sigma$ to be invertible, although Wick's theorem does not require this to be true. Here, the only constraint on $\Sigma$ is that it be positive semidefinite, meaning that $x^T\Sigma x\geq 0$ for all $x$, but the stronger condition that guarantees it invertibility is that it be positive definite, meaning that $x^Tx>0$ for all $x$; note the strict inequality for $x\neq 0$. 

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

First, we note that odd moments vanish. If $k=2m+1$ where $m>0$, then
$$
z^{2m+1}f_Z(z)
$$
is an odd function because $z^{2m+1}$ is odd and $f_Z(z)$ is even. Therefore,
$$
\mathbb{E}[Z^{2m+1}]=\int_{-\infty}^{\infty}z^{2m+1}f_Z(z)\,dz=0.
$$

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

some unorganized writing

Since every surviving ordered sequence assigns the $2m$ target values $i_1, \dots, i_{2m}$ to the $2m$ slots $a_1, b_1, \dots, a_m, b_m$ in a bijective way, we can equivalently describe a surviving sequence by recording which position in the tuple $(i_1, \dots, i_{2m})$ fills each slot, rather than which value. This turns the counting problem into a purely combinatorial one on the position set $\{1, \dots, 2m\}$, independent of whether the underlying $i_1, \dots, i_{2m}$ values happen to repeat.

A pairing of $\{1, \dots, 2m\}$ into $m$ pairs is an unordered collection of $m$ disjoint, unordered pairs whose union is $\{1, \dots, 2m\}$: 
$$
p=\{\{\alpha_1,\beta_1\},\dots,\{\alpha_m,\beta_m\}\}.
$$
Equivalently, $p$ is a partition of $\{1, \dots, 2m\}$ into blocks of size $2$. Given a pairing $p$, its contribution to the sum above is
$$
\prod_{\{\alpha, \beta\}\in p} \Sigma_{i_\alpha i_\beta},
$$
which is the product of $\Sigma_{i_\alpha i_\beta}$ over each pair $\{\alpha, \beta\}\in p$, using the values $i_\alpha, i_\beta$ found at the two positions that pairing groups together.

Finally, each pairing $p$ corresponds to exactly $2^m \cdot m!$ ordered sequences $(a_1, b_1), \dots, (a_m, b_m)$. There are two reasons why that provide the individual constants that form the product above: 1) swapping two coordinate labels inside any pair in the ordered sequence does not change the covariance value $\Sigma_{a_k b_k}=\Sigma_{b_k a_k}$, meaning that for $m$ pairs there are

are $2^m$ times more ordered sequences that correspond to the same pairing, and 2) 
