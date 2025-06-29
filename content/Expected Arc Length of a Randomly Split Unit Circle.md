---
title: Expected Arc Length of a Randomly Split Unit Circle
showDate: true
showReadingTime: true
tags:
  - probability
---

A somewhat elaborate solution to an interesting problem I saw online.
## Problem Statement 
Suppose you have a unit circle whose circumference is split by choosing three uniformly random points. What is the expected length of the arc that contains the point $(1,0)$?

As a personal afterthought, what is the expected length of the arc if we generalized it to $n$ points?

---
## 3-Point Case
Suppose $X_1,X_2,X_3\sim U(0, 2\pi)$, representing three points chosen uniformly at random on the circumference of the unit circle. Consider mapping its circumference onto the interval $[0, 2\pi)$, therefore it suffices to consider the points with the minimum and maximum values of the $X_i$'s to compute the length of the arc that contains $(1,0)$.

Specifically, for any three points $x_1, x_2, x_3\in[0,2\pi)$, we can calculate the length of the arc containing $(1,0)$ as, 
$$ 
\text{min}(x_1,x_2,x_3)+(2\pi-\max(x_1,x_2,x_3))
$$
Since taking the minimum and maximum of the $X_i$'s are random variables themselves, calculating the expected minimum and maximum values of them will yield the result we are looking for.
$$
\begin{align*}
\mathbb{E}[\text{arc length containing } (1,0)] &= \mathbb{E}[\text{min}(X_1, X_2, X_3)]+(2\pi-\mathbb{E}[\text{max}(X_1,X_2,X_3)])
\end{align*}
$$
Suppose $X_{\text{min}}=\text{min}(X_1,X_2,X_3)$ and $X_{\text{max}}=\text{max}(X_1,X_2,X_3)$, representing the minimum and maximum values of $X_i$. Since we want to obtain the expected value for both $X_\text{min}$ and $X_{\text{max}}$ to substitute on the formula above, we must first calculate their cumulative distribution functions (CDFs). Note that $\Pr(X_i\leq x)=\frac{x}{2\pi}$, similarly, $\Pr(X_i>x)=1-\frac{x}{2\pi}$. 
$$
\begin{align*}
F_{X_{\text{min}}}(x)=\Pr(X_{\text{min}}\leq x)&=1-\Pr(X_\text{min}>x) \\
&= 1-\Pr(X_1>x, X_2>x, X_3>x) \\
&= 1-\Pr(X_1>x)\Pr(X_2>x)\Pr(X_3>x) \quad (\text{independent events}) \\
&= 1-\left(1-\frac{x}{2\pi}\right)^3 \\
F_{X_{\text{max}}}(x)=\Pr(X_{\text{max}}\leq x)&=\Pr(X_1\leq x, X_2\leq x, X_3 \leq x) \\
&= \Pr(X_1\leq x)\Pr(X_2\leq x)\Pr(X_3 \leq x) \quad (\text{independent events}) \\
&= \left( \frac{x}{2\pi} \right)^3,
\end{align*}
$$
where $F_{X_{\text{min}}}(x)$ and $F_{X_{\text{min}}}(x)$ are the CDFs of $X_{\text{min}}$ and $X_{\text{max}}$, respectively.

We can calculate the probability distribution functions (PDFs) of $X_{\text{min}}$ and $X_{\text{max}}$ by differentiating their CDFs with respect to $x$.
$$ 
\begin{align*}
f_{X_{\text{min}}}(x)&=\frac{d}{dx}F_{X_{\text{min}}}(x)=\frac{d}{dx}\left( 1- \left( 1-\frac{x}{2\pi} \right)^3 \right)=\frac{3(2\pi-x)^2}{8\pi^3} \\
f_{X_{\text{max}}}(x)&=\frac{d}{dx}F_{X_{\text{max}}}(x)=\frac{d}{dx}\left(\left( \frac{x}{2\pi} \right)^3 \right)=\frac{3x^2}{8\pi^3},
\end{align*}
$$
where $f_{X_{\text{min}}}(x)$ and $f_{X_{\text{max}}}(x)$ are the PDFs of $X_{\min}$ and $X_{\max}$, respectively.

Finally, we can calculate the expected value of $X_{\min}$ and $X_{\max}$ using the calculated PDFs above,
$$
\begin{align*}
\mathbb{E}[X_{\min}]&=\int_0^{2\pi}xf_{X_{\min}}(x)dx=\int_0^{2\pi}x\left( \frac{3(2\pi-x)^2}{8\pi^3} \right)dx=\frac{\pi}{2} \\
\mathbb{E}[X_{\max}]&=\int_0^{2\pi}xf_{X_{\max}}(x)dx=\int_0^{2\pi}x\left( \frac{3x^2}{8\pi^3}\right)dx=\frac{3\pi}{2}
\end{align*}
$$
Substitute these into the formula laid out at the beginning, 
$$
\begin{align*}
\mathbb{E}[\text{arc length containing } (1,0)] &= \mathbb{E}[\text{min}(X_1, X_2, X_3)]+(2\pi-\mathbb{E}[\text{max}(X_1,X_2,X_3)]) \\ 
&= \mathbb{E}[X_{\min}]+(2\pi-\mathbb{E}[X_{\max}]) \\
&= \frac{\pi}{2}+\left( 2\pi-\frac{3\pi}{2} \right) \\
&= \pi
\end{align*}
$$
Therefore, the expected arc length containing $(1,0)$ for the 3-point case is $\pi$.

## $n$-Point Generalization

To make an $n$-point generalization, we similarly follow the 3-point case procedure.

Suppose $X_1, X_2,\dots, X_n\sim U(0, 2\pi)$. We must calculate, 
$$
\begin{align*}
\mathbb{E}[\text{arc length containing } (1,0)] &= \mathbb{E}[\text{min}(X_1, X_2, \dots, X_n)]+(2\pi-\mathbb{E}[\text{max}(X_1,X_2,\dots, X_n)])
\end{align*}
$$
Suppose $X_{\min}=\min(X_1,X_2,\dots, X_n)$ and $X_{\max}=\max(X_1,X_2,\dots, X_n)$. 

Calculating the CDFs of $X_{\min}$ and $X_{\max}$, since $X_i$'s are independent, 
$$
\begin{align*}
F_{X_{\text{min}}}(x)&=\text{(similar procedure as above)} =1-\left(1-\frac{x}{2\pi}\right)^n \\ 
F_{X_{\text{max}}}(x)&=\text{(similar procedure as above)}=\left( \frac{x}{2\pi} \right)^n
\end{align*}
$$
Calculating the PDFs of $X_{\min}$ and $X_{\max}$,
$$
\begin{align*}
f_{X_{\min}}(x)&=\frac{\partial}{\partial x}\left( 1-\left(1-\frac{x}{2\pi}\right)^n \right)=\frac{n(2\pi-x)^{(n-1)}}{(2\pi)^n} \\
f_{X_{\max}}(x)&=\frac{\partial}{\partial x}\left( \left( \frac{x}{2\pi} \right)^n \right)=\frac{nx^{(n-1)}}{(2\pi)^n}
\end{align*}
$$
Calculating the expected value of $X_{\min}$ and $X_{\max}$,
$$ 
\begin{align*}
\mathbb{E}[X_{\min}]&=\int_0^{2\pi}xf_{X_{\min}}(x)dx=\int_0^{2\pi}x\left( \frac{n(2\pi-x)^{(n-1)}}{(2\pi)^n} \right)dx=\frac{2\pi}{n+1} \\
\mathbb{E}[X_{\max}]&=\int_0^{2\pi}xf_{X_{\max}}(x)dx=\int_0^{2\pi}x\left( \frac{nx^{(n-1)}}{(2\pi)^n}\right)dx=\frac{2\pi n}{n+1}
\end{align*}
$$
Substitute these into the formula laid out at the beginning,
$$
\begin{align*}
\mathbb{E}[\text{arc length containing } (1,0)] &= \mathbb{E}[\text{min}(X_1, X_2, \dots, X_n)]+(2\pi-\mathbb{E}[\text{max}(X_1,X_2,\dots, X_n)]) \\
&=\mathbb{E}[X_{\min}]+(2\pi-\mathbb{E}[X_{\max}]) \\
&=\frac{2\pi}{n+1}+\left(2\pi-\frac{2\pi n}{n+1}\right) \\
&=\frac{4\pi}{n+1}
\end{align*}
$$
Therefore, the expected arc length containing $(1,0)$ for the $n$-point generalization is $\frac{4\pi}{n+1}$.

### Alternative Approach

This approach is somewhat redundant considering since it considers both a minimum and a maximum in the interval. An alternative approach which is cleaner and conciser for the $n$-point case is [Hagen Von Eitzen's solution in StackExchange](https://math.stackexchange.com/questions/272927/expected-length-of-arc-in-a-randomly-divided-circle).

This solution considers the arc length from $(1,0)$ to the next point in the interval $[0, 2\pi)$ which is the minimum of the $X_i$'s; visually, it is the upper side of the arc. Since the lower side of the arc applies the exact same procedure, doubling the expected length of the upper side of the arc provides the same answer we found above.

---
## References
1. [Hagen Von Eitzen's solution in StackExchange](https://math.stackexchange.com/questions/272927/expected-length-of-arc-in-a-randomly-divided-circle)
2. [WolframAlpha, derivative and integral work](https://www.wolframalpha.com)







