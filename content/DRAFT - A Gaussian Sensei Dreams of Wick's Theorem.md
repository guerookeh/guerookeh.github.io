---
title: "A Gaussian Sensei Dreams of Wick's Theorem"
published: 2026-07-03
created: 2026-07-03
modified: 2026-07-03
description: "An intuitive pass through the Gaussian integral section of The Principles of Deep Learning Theory, leading toward Wick's theorem."
showDate: true
showReadingTime: true
draft: true
tags:
  - probability
  - statistics
  - deep-learning
  - gaussian-distributions
---
The aim of this post is for me to try and intuitively tackle the Gaussian integral section of the pretraining section of *The Principles of Deep Learning Theory*, particularly the grand result: a derivation of Wick's theorem. 
***
## Abstract

We derive Wick's theorem first for a single Gaussian random variable, also called the univariate Gaussian case, and then extend it to Gaussian random vectors, also called the multivariate Gaussian case. In probability theory, Wick's theorem, also known as Isserlis's theorem, is a formula for computing higher-order moments of a multivariate Gaussian random vector in terms of the entries of its covariance matrix.

Let $\mathbf{X}=(X_1, \dots, X_n)$ be a multivariate Gaussian random vector. We can decompose it into
$$
(X_1, \dots, X_n)=(\mu_1+Z_1, \dots \mu_n+Z_n)
$$
where $\mu_i=\mathbb E[X_i]$ is the $i$-th component of the mean vector, and $\mathbf Z=(Z_1,\dots,Z_n)$ is a zero-mean multivariate Gaussian random vector.

Given the previous set up, [here is the statement of Wick's theorem from Wikipedia](https://en.wikipedia.org/wiki/Isserlis%27s_theorem). 

> If $\mathbf{Z}=(Z_1, \dots, Z_n)$ is a zero-mean multivariate Gaussian random vector, then
> $$ 
> \mathbb{E}[Z_1 Z_2\cdots Z_n]=\sum_{p\in P²_n}\prod_{\{i,j\}\in p} \mathbb{E}[Z_iZ_j]=\sum_{p\in P²_n}\prod_{\{i,j\}\in p} \text{Cov}(Z_i, Z_j) 
> $$
> where the sum is over all the pairings of $\{1, \dots, n\}$, i.e., all the distinct ways of partitioning $\{1, \dots, n\}$ into pairs $\{i, j\}$, and the product is over pairs contained in $p$. 

Note the above equality is only true if $n=2m$ is even for $m\geq 1$. If $n=2m+1$ is odd, then no pairing of $\{1,\dots, 2m+1\}$ exists, and therefore $\mathbb{E}[Z_1 Z_2 \cdots Z_{2m+1}]=0$.   

### Centering the Gaussian Vector

For the rest of the derivation, Wick's theorem will be applied to the centered coordinates
$$
Z_i=X_i-\mu_i
$$
**Therefore, whenever we compute higher-order moments using Wick's theorem, it is important to note that we are working with the zero-mean multivariate Gaussian random vector $\mathbf{Z}$, or alternatively univariate Gaussian random variable, rather than directly with $\mathbf{X}$**. Moments of the original variables $X_i$ can be obtained afterward by substituting $X_i=\mu_i+Z_i$ and expanding.

***
## Univariate Gaussian Case

Before deriving Wick's theorem for multivariate Gaussian random vectors, we first study the univariate case. We review the Gaussian density, the definition of expectations and moments, and the role of the moment-generating function. We then derive the moments of a centered univariate Gaussian and show how the resulting formula already contains the pairing structure that appears in Wick's theorem.

### Gaussian Density

Let $Z\sim\mathcal{N}(0, \sigma²)$ be a zero-mean Gaussian random variable with variance $\sigma²$ as per [[#Centering the Gaussian Vector]]. The probability density function of $Z$ is then
$$
f_Z(z)=\frac{1}{\sqrt{2\pi \sigma²}}\exp\left( -\frac{z²}{2\sigma²} \right).
$$
Note that the constant $\frac{1}{\sqrt{2\pi\sigma²}}$ is chosen such that the density integrates to $1$ as follows
$$
\int_{-\infty}^{\infty}f_Z(z)\,dz =1.
$$
### Expectations and Moments

By the Law of the Unconscious Statistician (LOTUS), if $Z$ is a continuous random variable with density $f_Z$, then the expectation of a suitable function $g(Z)$ is defined as
$$
\mathbb{E}[g(Z)]=\int_{-\infty}^{\infty} g(z)f_Z(z)\, dz.
$$
We consider the special case where $g$ is defined by $g(z)=z^k$, so that $g(Z)=Z^k$. This gives us what is called the $k$-th moment of $Z$:
$$
\mathbb{E}[Z^k]=\int_{-\infty}^{\infty} z^k f_Z(z)\, dz.
$$
Note that $k=1$ results in the first moment $\mathbb{E}[Z]$, which is simply just the mean of $Z$. We also know that $\mathbb{E}[Z]=0$ from [[#Gaussian Density]]. On the other hand, $k=2$ results in the second moment $\mathbb{E}[Z²]$, which is just the variance of $Z$ since
$$
\operatorname{Var}(Z)=\mathbb{E}[(Z-\mathbb{E}[Z])²]=\mathbb{E}[(Z-0)²]=\mathbb{E}[Z²].
$$
More generally, the $k$-th moment $\mathbb{E}[Z^k]$ measures the expected value of the $k$-th power of $Z$. However, these higher moments are generally more tedious to evaluate directly as $k$ grows. Wick's theorem gives us a shortcut to computing these higher-order Gaussian moments without evaluating a new integral each time.

### Moment-Generating Functions

A remark by a statistics professor that I hadn't recalled until now was that moment generating functions are immensely powerful theoretically, but practically rather restricted. This derivation makes use of them as they greatly simplify the derivation. Otherwise, directly computing higher moments from the integral definition is possible by a repeated integration-by-parts method, but quickly becomes cumbersome. Moment-generating functions provide a more accessible route to computing these.

Without going into too much detail, a moment-generating function does exactly what its name suggests: it generates the moments of a distribution. The moment generating function of a random variable $X$ is defined by
$$
M_X(t)\coloneqq \mathbb{E}[e^{tX}], \quad t\in\mathbb{R}.
$$
Recall that the Maclaurin series for the exponential function is
$$
e^x=\sum_{n=0}^\infty \frac{x^n}{n!}.
$$
Expanding the Maclaurin series for the above natural exponential $e^{tX}$ gives
$$
e^{tX}=\sum_{n=0}^{\infty} \frac{(tX)^n}{n!}=1+tX+\frac{t^2X²}{2!}+\frac{t³X³}{3!}+\dots +\frac{t^nX^n}{n!}+\dots.
$$
Taking the expectation gives
$$
M_X(t)=\mathbb{E}[e^{tX}]=\mathbb{E}\left[ 1+tX+\frac{t²X²}{2!}+\frac{t³X^3}{3!}+\dots + \frac{t^n X^n}{n!}+\dots \right].
$$
By linearity of expectation, and under conditions that are satisfied for Gaussian random variables, we may then write
$$
M_X(t)=\mathbb{E}[e^{tX}]=1+t\mathbb{E}[X]+\frac{t²\mathbb{E}[X²]}{2!}+\frac{t³\mathbb{E}[X³ ]}{3!}+\dots+\frac{t^n\mathbb{E}[X^n]}{n!}+\dots
$$
Equivalently, we can express this general expansion as
$$
M_X(t)=\sum_{n=0}^\infty \mathbb{E}[X^n]\frac{t^n}{n!}.
$$
It is this expression that shows why $M_X(t)$ is called a moment-generating function. The coefficients of each power of $t$ contains a moment of $X$, and by differentiating $M_X(t)$ $n$ times with respect to $t$ and then setting to $t=0$, we obtain the $n$-th moment about the origin.
$$
M_X^{(n)}(0)=\mathbb{E}[X^n]
$$

%% https://en.wikipedia.org/wiki/Moment_generating_function#Definition  %%

### Wick's Theorem

If
$$
X\sim\mathcal{N}(\mu, \sigma²)
$$
then its moment-generating function is
$$
M_X(t)=\exp\left( \mu t+\frac{\sigma²t ²}{2} \right)
$$
where $\mathbb{E}[X]=\mu$ and $\text{Var}(X)=\sigma²$.

%% https://statproofbook.github.io/P/norm-mgf.html    %%

Similarly, in the centered case,
$$
Z\sim \mathcal{N}(0, \sigma²)
$$
this reduces to 
$$
M_Z(t)=\mathbb{E}[e^{tZ}]=\exp\left( \frac{\sigma²t²}{2} \right).
$$
Here, $t$ is an auxiliary variable. In PDLT's physics-based terminology, this auxiliary variable is equivalently referred to as the source term $J$. 

Expanding this moment-generating function as a Maclaurin series gives
$$
\exp\left( \frac{\sigma²t²}{2} \right)=\sum_{m=0}^{\infty} \frac{1}{m!}\left(\frac{\sigma²t²}{2}\right)^m
$$
Therefore,
$$
M_Z(t)=\sum_{m=0}^\infty \frac{\sigma^{2m}}{2^mm!}t^{2m}.
$$
Notice that the expansion of $M_Z(t)$ contains only even powers of $t$. Therefore, all odd moments vanish:
$$
\mathbb{E}[Z^{2m+1}]=0.
$$
For even moments, recall the previous general expansion from [[#Moment-Generating Functions]]. We can apply this general expansion to the centered Gaussian $Z$,
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

### Combinatorial Intuition for Wick's Theorem

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
where $i_k$ corresponds to the $k$-th selected factor. As one can imagine, many orderings of selected indices $\{i_1,i_2,\dots,i_{2m}\}$ can be made, but in the univariate case this is redundant because every selected factor is still the same random variable $Z$. Again, the idea will be useful in the multivariate case.

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

This is a combinatorics problem that can be framed as follows: provide all possible unordered collections of unordered pairs of selected indices $\{i_1,\dots,i_{2m}\}$ without replacement. A pairing is one such unordered collection of unordered pairs. Visually, a pairing has the form
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
## Multivariate Gaussian Case

We now extend the ideas from the univariate case to multivariate Gaussian random vectors. Similarly as in the [[#Univariate Gaussian Case]], we first study the densities, expectations, moments, and moment-generating functions of the multivariate Gaussian. We then derive the moments of the centered multivariate Gaussian and show how the resulting formua




Before deriving Wick's theorem for multivariate Gaussian random vectors, we first study the univariate case. We review the Gaussian density, the definition of expectations and moments, and the role of the moment-generating function. We then derive the moments of a centered univariate Gaussian and show how the resulting formula already contains the pairing structure that appears in Wick's theorem.


***

https://en.wikipedia.org/wiki/Bell-shaped_function
https://en.wikipedia.org/wiki/Isserlis%27s_theorem as a reference to the first mention
https://statproofbook.github.io/P/norm-pdf as a reference to the pdf of univariate gaussian
https://statproofbook.github.io/P/norm-mean.html as a reference to mean of univariate gaussian
https://en.wikipedia.org/wiki/Law_of_the_unconscious_statistician LOTUS justification
https://statproofbook.github.io/D/mgf moment-generating function
https://en.wikipedia.org/wiki/Moment_generating_function#Definition
https://statproofbook.github.io/P/norm-mgf.html mgf of the normal distribution
https://en.wikipedia.org/wiki/General_Leibniz_rule leibniz rule

%%
i think appendix should contain sections on how to obtain the normalization factors for both the univariate and the multivariate cases; i think they're still worth pointing out, especially the latter since i think it's pretty interesting with the eigenbasis stuff

i think we need to have numbers on the right side of the expressions to refer back to. for example, in the even moments mention of the coefficient, we can refer back to the general moment-generating function
%%

in the multivariate case, they are not independent, and they are not identically distributed for the ordered collection of gaussians. they are jointly distributed; they might have dependencies between one another, and they might not have the same variance even though they are all centered at mean/expectation 0
- nevermind

disambiguate terminology between univariate, multivariate, random variable, random vector.

***
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

***
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
