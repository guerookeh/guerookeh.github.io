---
title: Autoregressive Moving-Average Models (ARMA) —The Baseline Model For Stationary Time Series
draft: true
showDate: true
showReadingTime: true
tags:
  - statistics
  - forecasting
  - time-series
  - stochastic-processes
---


**A look into one of the baseline statistical models for time series forecasting.**
***
### Motivation

A time series is a sequence of data points that are indexed in order. In time series modeling, different models describe how the series might evolve, which can be used to fit the observed data (in-sample) and to forecast future values (out-of-sample). 

One of such models used is the autoregressive-moving-average (ARMA) model, which was first described in [Peter Whittle](https://en.wikipedia.org/wiki/Peter_Whittle_(mathematician))'s 1951 thesis *Hypothesis testing in time series analysis* and later popularized by [George E. P. Box](https://en.wikipedia.org/wiki/George_E._P._Box) and [Gwilym Jenkins](https://en.wikipedia.org/wiki/Gwilym_Jenkins), known for the [Box-Jenkins method](https://en.wikipedia.org/wiki/Box–Jenkins_method). 

The ARMA model is used as a simple yet powerful way to represent stationary time series whose current value depends on a linear combination of both of its own past values (the autoregressive part) and on past residuals (the moving-average part) separately. 

Before attempting to fit an ARMA model to any time series, it is crucial that we first understand the properties that the series must satisfy by starting with the concept of a stochastic process.

***
### Definitions

#### Stochastic Process

Let $\{X_t:t\geq 0\}$ be a stochastic process on a well-defined probability space $(\Omega, \mathcal{F}, \mathbb{P})$. A stochastic process is a collection of random variables $X_t:\Omega\rightarrow \mathbb{R}$ that are indexed by the time parameter $t\in[0,\infty)$. Specifically, we say that $\{X_t:t\geq 0\}$ is a continuous-time stochastic process since $t$ ranges over the real numbers, implying an uncountable collection of random variables. Alternatively, we say that $\{X_t : t\in T\}$ is a discrete-time stochastic process where $T$ is a countable set such as $\mathbb{N}_0$ or $\mathbb{Z}$. 

For example, consider repeatedly flipping a coin that lands heads with probability $p$. Let $t\in\mathbb{N}_0$ index the amount of flips, and let $X_t\sim\text{Bernoulli}(p)$ denote the indicator of heads on the $t$-th flip. For our purposes, $\Omega=\{0,1\}^{\mathbb{N}_0}$ is the set of all infinite sequences $\omega=(\omega_0,\omega_1,\dots)$ with $\omega_t\in\{0,1\}$. Each random variable $X_t:\Omega\rightarrow \{0,1\}$ is then defined by $X_t(\omega)=\omega_t$. Thus, $\{X_t : t\in\mathbb{N}_0 \}$ forms an independent and identically distributed discrete $\text{Bernoulli}(p)$ process. It is also worth noting that we can model a fixed finite number $n$ of flips, for which we would instead write $\Omega=\{0,1\}^n$. 

A stochastic process $\{X_t\}$ describes the random mechanism that could generate different possible sequences, but the time series data we empirically observe, say $\{x_t\}_{t=1}^n$, is one particular realization of that process. 

It is important to note that if the time series data is entirely deterministic, this could be defined as a linear function such as $x_t=2t+1$, then no randomness exists and there is no need for a stochastic description. On the other hand, a time series described by a function such as $x_t=2t+1+\varepsilon$ where $\varepsilon\sim \mathcal{N}(0, 1)$ is a random error term (or noise) would warrant such a description, since even though $2t+1$ is a deterministic component, $\varepsilon$ is a random component. 

With the concept of stochastic process established, we now consider an important property that many time-series models, including ARMA, require: stationarity. 
#### Stationary Process

We say a 

let $\{X_t\}$ be a stochastic process

let $F_X(x_{t_1+\tau},\dots,x_{t_n+\tau})$ represent the cumulative distribution function of the unconditional joint distribution of $\{X_t\}$ at times $t_1+\tau, \dots, t_n+\tau$

then, $\{X_t\}$ is said to be strictly stationary, strongly stationary, or strict-sense stationary if

$$
F_X(x_{t_1+\tau},\dots,x_{t_n+\tau})=F_X(x_{t_1},\dots,x_{t_n})\quad \text{for all }\tau,t_1,\dots,t_n\in\mathbb{R} \text{ and for all } n\in \mathbb{N}_{>0}
$$

since $\tau$ does not affect $F_X(\cdot)$, $F_X$ is independent of time

intuitively, this is saying that if we shift all time points by the same amount $\tau$, the joint probabilistic behavior doesn't change. this rules out possibilities for trends, seasonalities, changing variance, and evolving correlations

the unconditional joint distribution of $\{X_t\}$ at times $t_1+\tau, \dots, t_n+\tau$ can also be written as,
$$
F_X(x_{t_1}, \dots, x_{t_n})=\Pr(X_{t_1}\leq x_{t_1}, X_{t_2}\leq x_{t_2}, \dots, X_{t_n}\leq x_{t_n})
$$

weak or wide-sense stationarity instead require that only the first moment (i.e. the mean) and autocovariance do not vary with respect to the time and that the second moment is finite for all times, any strictly stationary process which has a finite mean and covariance is also WSS

a continuous time random process $\{X_t\}$ which is WSS has the following restrictions on its 
- mean function $m_X(t) \triangleq \mathbb{E}[X_t]$ 
- autocovariance function $K_{XX}(t_1,t_2)\triangleq \mathbb{E}[(X_{t_1}-m_X(t_1))(X_{t_2}-m_X(t_2))]$

the autocovariance function measures how much $X_{t_1}$ and $X_{t_2}$ co-vary, i.e., how the value of the process at one time is linearly related to its value at another. 

diving a bit deeper into this, covariance quantifies the joint variability of two random variables. if large values of one variable tend to correspond to large values of another, the covariance is positive, if large values of one correspond to small values of another, its negative. correlation is the normalized version of covariance, it always lies between -1 and 1 and measure the strength of the linear relationship rather than its magnitude.
$$
\rho_{XX}(t_1,t_2)=\frac{K_{XX}(t_1,t_2)}{\sqrt{K_{XX}(t_1,t_1)}\sqrt{K_{XX}(t_2,t_2)}}
$$thus, $K_{XX}$ encodes the same structural information as correlation, but in physical units of $X_t$, so it's not normalized.

we can show why the autocovariance function is of such a form by deriving it from basic covariance. from two random variables $X$ and $Y$,
$$
\text{Cov}(X,Y)=\mathbb{E}[(X-\mathbb{E}[X])(Y-\mathbb{E}[Y])]
$$
for a process, we simply take $X=X_{t_1}$ and $Y=X_{t_2}$,
$$
K_{XX}(t_1,t_2)=\text{Cov}(X_{t_1}, X_{t_2})
$$
which measures the linear dependence between values of the process at two times

to understand covariance a bit more thoroughly, let's decompose covariance to its standard and most useful form
$$
\begin{align}
\text{Cov}(X,Y)&=\mathbb{E}[XY-X\mathbb{E}[Y]-Y\mathbb{E}[X]+\mathbb{E}[X]\mathbb{E}[Y]] \\
&= \mathbb{E}[XY]-\mathbb{E}[X]\mathbb{E}[Y]
\end{align}
$$
covariance is the expected product of deviations of $X$ and $Y$ from their means,
- if $X$ and $Y$ are both above or both below their means simultaneously, $(X-\mathbb{E}[X])(Y-\mathbb{E}[Y])>0$ 
- if one is above and the other is below, $(X-\mathbb{E}[X])(Y-\mathbb{E}[Y])<0$ 
- taking the expectation of this product tells us the typical direction and magnitude of joint deviations
thus,
- $\text{Cov}(X,Y)>0$ means they move together, a positive linear relationship
- $\text{Cov}(X,Y)<0$ means that they move oppositely, a negative linear relationship
- $\text{Cov}(X,Y)=0$ means no linear relationship, but not necessarily independent
covariance detects only linear dependence 





***
### References
1. [Autoregressive moving-average model, Wikipedia](https://en.wikipedia.org/wiki/Autoregressive_moving-average_model)
2. [Introduction to Random Processes, Probability Course](https://www.probabilitycourse.com/chapter10/10_1_0_basic_concepts.php)
3. 






- why do we want to look at baseline models when there are more complicated?
	- a great way to understand what conditions does the time-series need to meet
		- e.g, stationary timeseries are the main use for these, not nonstationary
	- baseline models start off simpler, more complex methods integrate ARMA that integrate other considerations in order to further improve the forecast
- the AR component of the model
- the MA component of the model
- a consideration for parallels to linear regression
	- what assumptions does linear regression do that might be important to consider here?
- what happens when we join both of them? the intended effect
- limitations, drawbacks of other time-series and model simplicitly limitations
