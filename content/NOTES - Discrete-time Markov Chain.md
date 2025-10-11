---
title: NOTES - Discrete-time Markov Chain
draft: true
tags:
  - "#probability"
  - "#markov-chain"
---
Based off of J.R. Norris' Markov Chains textbook.
## 1.1  Definition and Basic Properties
Let $I$ be a countable set.
Each $i\in I$ is a *state* and $I$ is called the state-space. 
We say that $\lambda=\left( \lambda_i :  i \in I \right)$ is a *measure* on $I$ if $0\leq \lambda_i<\infty$ for all $i\in I$. 
In addition, if the total mass $\sum_{i\in I} \lambda_i=1$, then we call $\lambda$ a *distribution*.

We work throughout with a probability space $\left( \Omega, \mathcal{F}, \mathbb{P} \right)$, for reference as to what this probability space thoroughly means, check [[NOTES - Measure Theory]]. 

A *random variable* $X$ with values in $I$ is a function $X : \Omega \rightarrow I$. 
Suppose we set,
$$ \lambda_i=\Pr(X=i)=\Pr(\{ \omega:X(\omega)=i \}) $$
Then $\lambda$ defines a distribution, the *distribution* of $X$. 
Think of $X$ as modelling a random state which takes the value $i$ with probability $\lambda_i$.



