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
