---
title: "Bessel's Correction: Deriving the Unbiased Estimator of Variance"
published: 2025-10-12
created: 2025-10-12
modified: 2025-10-12
description: "A derivation showing how finite-sample bias arises in the naive sample variance, and how Bessel's correction removes it."
draft: false
showDate: true
showReadingTime: true
tags:
  - statistics
  - estimation
---
**A derivation showing how finite-sample bias arises in the naive sample variance, and how Bessel's correction removes it.**
***

Among the classic plug-and-chug formulas that are required to properly conduct nearly every high-school science experiment is the sample variance formula. 
$$
s^2=\frac{1}{n-1}\sum_{i=1}^{n}(x_i-\bar{x})^2
$$
At first glance it's not particularly interesting; given a collection of observations $\{x_1,x_2,\dots,x_n\}$ sampled from a discrete random variable $X$, which we assume has mean $\mu$ and variance $\sigma^2$, it is the sum of the squared deviations from their mean, divided by a seemingly arbitrarily defined constant in the denominator. 

**So what exactly is this constant?**

Before addressing this question, it is worth briefly going over some details. 

## Estimators and the Problem of Finite-Sample Bias

What we are doing is called statistical inference: we try to estimate the unknown parameters mean $\mu$ and variance $\sigma^2$ of an underlying population modeled by a random variable $X$. A collection of observations like above, $\{x_1, x_2, \dots, x_n\}$, are just one finite sample from that population. Estimators, like the sample mean and variance, use these samples in order to estimate the parameters of the population. 

Nevertheless, these estimators are prone to systemic bias due to having only a finite amount of samples, which will cause their expected values to differ from the true population parameters, requiring us to apply corrections in order to make them unbiased estimators.
## Sample Mean as an Unbiased Estimator of the Mean

To understand where that constant comes from, consider what happens when we compute the sample mean $\bar{x}$ used in the computation of the sample variance above, 
$$
\bar{x}=\frac{1}{n}\sum_{i=1}^n x_i
$$
which we use to estimate the population mean 
$$
\mu=\mathbb{E}[X]=\sum_{i=1}^\infty x_i \cdot\Pr(x=x_i)
$$
From a statistical perspective, we model the observations as independent and identically distributed random variables $X_1, X_2, \dots, X_n$, each following the same distribution as $X$; formally, 
$$
X_1, X_2,\dots, X_n\overset{i.i.d}{\sim} X
$$
The sample mean as a random variable itself is then defined as,
$$
\bar{X}=\frac{1}{n}\sum_{i=1}^n X_i
$$
Since each $X_i$ is an independent draw from the same distribution, by linearity of expectation, 
$$
\mathbb{E}[\bar{X}]=\mathbb{E}\left[ \frac{1}{n}\sum_{i=1}^n X_i \right]=\frac{1}{n}\sum_{i=1}^n \mathbb{E}[X_i]=\frac{1}{n}(n\mu)=\mu
$$
Thus the sample mean $\bar{X}$ is what we call **an unbiased estimator of the population mean $\mu$.** 

It is easy to see that if we changed the formula of the sample mean by dividing by any constant other than $n$, the estimator would no longer be unbiased. For example, if we defined, 
$$
\bar{X}=\frac{1}{n-1}\sum_{i=1}^n X_i
$$
then, 
$$
\mathbb{E}[\bar{X}]=\frac{1}{n-1}\sum_{i=1}^n\mathbb{E}[X_i]=\frac{n}{n-1}\mu\neq \mu
$$
meaning that this alternative estimator would actually overestimate the population mean.

Note that this result doesn't make any distributional assumptions, so it holds for any identically and independently distributed data with a finite mean. Although these assumptions change when discussing sampling distributions of variance and standard deviation.

## Naive Sample Variance as a Biased Estimator of the Variance

Suppose our naive guess at defining the sample variance estimator, $s^2_g$, was,
$$
s_g^2=\frac{1}{n}\sum_{i=1}^{n}(x_i-\bar{x})^2
$$
Model the observations as independent and identically distributed random variables following $X$
$$
\begin{align*}
X_1, X_2,\dots, X_n\overset{i.i.d}{\sim} X \\
\bar{X}=\frac{1}{n}\sum_{i=1}^nX_i
\end{align*}
$$
then calculate the expected value of this estimator,
$$
\begin{align*}
	\mathbb{E}[s_g^2]
	&= 
	\mathbb{E}
		\left[
			\frac{1}{n}\sum_{i=1}^n(X_i-\bar{X})^2
		\right] \quad\quad (\text{def. of }s^2)
	\\
	&= 
		\mathbb{E}
			\left[
				\frac{1}{n}\sum_{i=1}^n (X_i^2-2X_i\bar{X}+\bar{X}^2)
			\right] \quad\quad (\text{expand squared term})
	\\
	&= \mathbb{E}
			\left[
				\frac{1}{n} \sum_{i=1}^n X_i^2 -2\bar{X}\frac{1}{n}\sum_{i=1}^nX_i+\frac{1}{n}\sum_{i=1}^n\bar{X}^2  
			\right] \quad\quad (\text{separate the summations})
	\\
	&= \mathbb{E}
			\left[
				\frac{1}{n}\sum_{i=1}^n X_i^2
			\right] 
		-\mathbb{E}
			\left[
				2\bar{X}\frac{1}{n}\sum_{i=1}^n X_i
			\right]
		+ \mathbb{E}
			\left[
				\frac{1}{n}\sum_{i=1}^n\bar{X}^2
			\right] \quad (\text{linearity of }\mathbb{E}[\cdot])
	\\
	&= \frac{1}{n}\sum_{i=1}^n\mathbb{E}[X_i^2]
	   - \mathbb{E} \left[ 2\bar{X}^2 \right]
	   + \frac{1}{n}\sum_{i=1}^n\mathbb{E} [ \bar{X}^2 ] \quad\quad (\text{linearity of }\mathbb{E}[\cdot]\text{ and def. of }\bar{X})
	\\
	&= \mathbb{E}[X^2] - \mathbb{E}[2\bar{X}^2]+\mathbb{E}[\bar{X}^2] \quad\quad (X_i \text{'s are i.i.d, so }\mathbb{E}[X^2_i]=\mathbb{E}[X^2] \text{ for all }i\leq n)
	\\
	&= \mathbb{E}[X^2]-\mathbb{E}[\bar{X}^2] \quad\quad (\text{linearity of }\mathbb{E}[\cdot])
\end{align*}
$$
Note that on Step 5, whenever we say that a collection of random variables are independently and identically distributed, it means that they are 1) independent; knowing one doesn't tell you anything about the other, and 2) identical: each $X_i$ follows same probability distribution as $X$. 

To explain the previous point more formally, $\text{i.i.d}$ is defined as: for every measurable function $g$, 
$$
\mathbb{E}[g(X_i)]=\mathbb{E}(g(X))
$$
for all $i\leq n$ which, in the case above, $g(x)=x^2$; therefore $\mathbb{E}[X_i^2]=\mathbb{E}[X^2]$.

To continue off from the last step in the equality in the previous calculation, recall the definition of the variance $\mathbb{V}[Z]$ of an arbitrary random variable $Z$, 
$$
\mathbb{V}[Z]=\mathbb{E}\left[ (Z-\mathbb{E}[Z])^2 \right]
$$
If we expand the previous expression's inner term by linearity of expectation, 
$$ 
\begin{align*}
	\mathbb{V}[Z]&=\mathbb{E}[(Z-\mathbb{E}[Z])^2] \\
	&= \mathbb{E}[Z^2-2Z\mathbb{E}[Z]+\left(\mathbb{E}[Z]\right)^2]] \\
	&= \mathbb{E}[Z^2]-2\mathbb{E}[Z]\mathbb{E}\left[\mathbb{E}[Z]\right]+\mathbb{E}\left[\left(\mathbb{E}[Z]\right)^2\right] \\
	&= \mathbb{E}[Z^2]-2\left(\mathbb{E}[Z]\right)^2+\left(\mathbb{E}[Z]\right)^2 \\
	&= \mathbb{E}[Z^2]-(\mathbb{E}[Z])^2 
\end{align*}
$$
From following a similar procedure as shown above, we can also show a property of the variance,
$$
\mathbb{V}[cX]=c^2(\mathbb{E}[Z^2]-(\mathbb{E}[Z])^2)=c^2\mathbb{V}[X]
$$
for some constant $c$ by following a similar procedure. This will be useful in a bit.

We then use this result to substitute $\mathbb{E}[Z^2]=\mathbb{V}[Z]+\left( \mathbb{E}[Z] \right)^2$ into the last step in the equality, 
$$
\begin{align*}
	\mathbb{E}[X^2]-\mathbb{E}[\bar{X}^2]
	&=  \left( \mathbb{V}[X]+\left( \mathbb{E}[X] \right)^2 \right)-\left(\mathbb{V}[\bar{X}]+\left( \mathbb{E}[\bar{X}] \right)^2\right)
	\\
	&=\left( \mathbb{V}[X]+\mu^2 \right)-\left(\mathbb{V}[\bar{X}]+\mu^2\right) \quad\quad (\text{def. of }\mathbb{E}[X],\space\mathbb{E}[\bar{X}])
	\\
	&=\mathbb{V}[X]-\mathbb{V}[\bar{X}]
\end{align*}
$$
We know $\mathbb{V}[X]=\sigma^2$, where $\sigma^2$ is the population variance. Now, we need to calculate $\mathbb{V}[\bar{X}]$, 
$$ 
\begin{align*}
\mathbb{V}[\bar{X}]&=\mathbb{V}\left[ \frac{1}{n}\sum_{i=1}^n X_i \right] \\
&= \frac{1}{n^2}\sum_{i=1}^n \mathbb{V}[X_i] \quad\quad \left( \mathbb{V}[cX]=c^2\mathbb{V}[X] \right) \\ &= \frac{1}{n^2}\sum_{i=1}^n \mathbb{V}[X] \quad\quad (X_i's\text{ are i.i.d, so }\mathbb{V}[X_i]=\mathbb{V}[X] \text{ for all } i \leq n) \\ &=\frac{1}{n^2}n\sigma^2 \\ &= \frac{\sigma^2}{n }
\end{align*}
$$
Substituting this back into our final expression gives us, 
$$
\mathbb{V}[X]-\mathbb{V}[\bar{X}]=\sigma^2-\frac{\sigma^2}{n}=\left(1-\frac{1}{n}\right)\sigma^2
$$
Therefore, taking the expectation of our naive sample variance estimator results in, 
$$
\mathbb{E}[s^2_g]=\mathbb{E}\left[\frac{1}{n}\sum_{i=1}^{n}(x_i-\bar{x})^2\right]=\left(1-\frac{1}{n}\right)\sigma^2=\left( \frac{n-1}{n} \right)\sigma^2
$$
which means that our **naive sample variance estimator is biased!** 

More specifically, it is biased by the factor $\left(1-\frac{1}{n}\right)=\frac{n-1}{n}$. If we take its limit to infinity,
$$
\lim_{n\rightarrow\infty}\left(\frac{n-1}{n}\right)=1
$$
it's easy to see that with a large $n$, it should not be a concern. Nevertheless, it is precisely when $n$ is not large that the factor biases our naive sample variance. For example, given $n=10$, the factor would be biased downwards by $10\%$ relative to the population variance. 

**So how exactly do we correct this bias?** By multiplying our naive sample variance estimator by the reciprocal of the bias factor! 
$$
\mathbb{E}\left[\left( \frac{n}{n-1} \right)s_g^2\right]=\mathbb{E}\left[\left( \frac{n}{n-1} \right)\frac{1}{n}\sum_{i=1}^{n}(X_i-\bar{X})^2\right]=\mathbb{E}\left[\frac{1}{n-1}\sum_{i=1}^{n}(X_i-\bar{X})^2\right]=\mathbb{E}[s^2]
$$
This is what is known as [Bessel's correction](https://en.wikipedia.org/wiki/Bessel%27s_correction). It is the use of $n-1$ in the denominator instead of $n$ in the formula for the sample variance, and exactly explains the constant in the denominator! By applying Bessel's correction, we obtain an unbiased estimator of the variance.
$$ 
\mathbb{E}[s^2]=\mathbb{E}\left[ \left( \frac{n}{n-1} \right)s_g^2 \right]=\left( \frac{n}{n-1} \right)\mathbb{E}[s_g^2]=\left( \frac{n}{n-1} \right)\left[\left( \frac{n-1}{n} \right)\sigma^2\right]=\sigma^2
$$
***

**Next post will address the unbiased estimator of the standard deviation, which we might naively assume, yet again, that we can obtain by taking the square root of our unbiased estimator of the variance.** 

**Sadly, Jensen's inequality is there to stop us with a tiny, negligible bias that needs to be accounted for.**

***
## References
1. [Bessel's correction, Gregory Gundersen](https://gregorygundersen.com/blog/2019/01/11/bessel/)
2. [Bessel's correction, Wikipedia](https://en.wikipedia.org/wiki/Bessel%27s_correction)
3. [Variance, Wikipedia](https://en.wikipedia.org/wiki/Variance)
