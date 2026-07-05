---
title: NOTES - Measure Theory
draft: true
tags:
  - measure-theory
---

Notes from [MATH 41021/61021 Measure Theory and Ergodic Theory, Donald Robertson](https://personalpages.manchester.ac.uk/staff/donald.robertson/teaching/23-24/41021/notes/length.html).

---
### 1  Length
When $a\leq b$, the interval $(a,b)$ has a length of $b-a$. For this, we need to investigate the extent to which we can assign a length to other subsets of $\mathbb{R}$.
#### 1.1  Outer Measure
The *power set* of a set is the set consisting of all its subsets. Write $\mathcal{P}(X)$ for the power set of a set $X$. Thus $E\in \mathcal{P}(X)$ is the same as writing $E\subset X$. 

If a set $B\subset \mathbb{R}$, which is a subset of the power set of the real numbers as explained above, is contained within a union of intervals, it would be natural to come up with the following reasoning: the length of $B$ is not larger than the sum of the lengths of the intervals.

The choice of intervals is arbitrary conditioned on that $B$ is contained within them. For this reason, intuitively, we have two cases: the intervals overlap and do not. For example, suppose $B_1=B_2=[0,2]$, where $B_1\subset [ 0,1 )\cup[1,2]$ and $B_2\subset[0,2]\cup[1,2]$, where $B_1$ does not overlap and $B_2$ does. Clearly, $\text{Length}(B_1)=\text{Length}(B_2)=2$, nevertheless the sums of the intervals $\sum_{n=1}^N b_n-a_n$ are $2$ and $3$, respectively. Formally, we can write this as following,
$$ B\subset (a_1,b_1)\cup(a_2,b_2)\cup\dots\cup(a_N,b_N)=\bigcup_{n=1}^N(a_n,b_n)\implies \text{Length}(B)\leq\sum_{n=1}^N b_n-a_n $$
It is worth noting that the notation of the subset above, specifically $\subset$, is used inclusively, meaning that $B$ is contained in, and possibly equal, to the union.

The first major idea is that if we choose certain intervals wisely, then there could be very little discrepancy between the set $B$ and the union of intervals. In that case, the sum could be considered very close to the length of $B$. In the example laid out above, this would be as simple as omitting the latter interval of $B_2$, which would allow for the sums of intervals to be exactly equal to the length of $B_2$. 

This idea is formalized as follows. Define the ***outer measure*** of a set $B\subset \mathbb{R}$ by finding the most efficient way of covering it by a union of intervals. Thus, we define,
$$ \Lambda(B)=\inf \left\{ \sum_{n=1}^\infty b_n-a_n : B\subset \bigcup_{n=1}^\infty (a_n,b_n) \right\}, \quad \text{for all sets } B \subset \mathbb{R} $$
Intuitively, the infimum picks the most efficient way of covering $B$ by intervals. Notice that in the definition it not only allows unions of a finite number of intervals, but unions of an infinite number as well. 

The function $\Lambda$ is defined on $\mathcal{P}(\mathbb{R})$. It assigns a value in $[0,\infty]$ to *every* subset of $\mathbb{R}$, the sum of the length of the intervals chosen which most efficiently covers $B$. It also has the following properties, which are reasonable when thinking of $\lambda$ as assigning length to subsets of $\mathbb{R}$.

**Theorem**  The map $\Lambda:\mathcal{P}(\mathbb{R})\rightarrow[0,\infty]$ has the following properties.
1. $\Lambda(\varnothing)=0$ 
2. $\Lambda(A)\leq \Lambda(B)$ whenever $A\subset B$
3. $\Lambda(\bigcup_{n=1}^N A_n)\leq \sum_{n=1}^N \Lambda (A_n)$ for all $A_n\subset \mathbb{R}$ where $n\in \mathbb{N}$
4. $\Lambda (A-t)=\Lambda (A)$ for all $A\subset \mathbb{R}$, where $A-t=\{ a-t : t\in\mathbb{R} \}$
The properties above state: (1) the length of the empty set is zero, (2) a set that is larger or equal to another will have larger or equal length, (3) the length of the union of sets will be less than or equal to the sum of the length of the individual sets, and (4) the sets $A$ and $A-t$ have the same measure, this is the translation invariance.

A proof of these properties is separate.
#### 1.2  Countable Additivity
The properties proved for $\Lambda$ are reasonable if we think of $\Lambda$ as assigning a length. Here is a further reasonable property that should be expected from such a length-assigning device.

Say that $\Xi:\mathcal{P}(\mathbb{R})\rightarrow [0,\infty]$ is ***countably additive*** if one has,
$$ \Xi \left( \bigcup_{n=1}^\infty A_n \right)=\sum_{n=1}^\infty \Xi (A_n)$$
whenever $n\mapsto A_n$ is a sequence of subsets of $\mathbb{R}$ that is pairwise disjoint.

Recall that a sequence $n\mapsto A_n$ of sets is pairwise disjoint if $A_i\cap A_j = \varnothing$ for all $i\neq j$. Informally, a mapping $\Xi$ from $\mathcal{P}$ to $[0,\infty]$ is countably additive if we can calculate $\Xi(A)$ by breaking up $A$ into countably many pieces $A_1,A_2,\dots$ and summing up the values $\Xi(A_1),\Xi(A_2),\dots$. 

We are now faced with a central question: is $\Lambda$ countably additive?

The answer is: **there is no mapping $\Xi:\mathcal{P}(\mathbb{R})\rightarrow [0,\infty]$ that has all three of the following properties: countably additive, assigns lengths to intervals, and translation invariant.**

In order to proceed with a theory of length, we have to give something up. We insist on translation invariance and countable additivity by abandoning the requirement that $\Lambda$ is defined on $\mathcal{P}(\mathbb{R})$. The next step, therefore, is to try and identify a rich collection $\mathscr{B}$ of subsets of $\mathbb{R}$ so that,
$$\Lambda:\mathscr{B}\rightarrow[0,\infty] $$
is countably additive and translation invariant.

___
### 2  $\sigma$-algebras
We saw there was no function $\Xi$ from $\mathcal{P}(\mathbb{R})$ to $[0,\infty]$ that is countably additive, translation invariant, and assigns intervals their usual lengths. Beholden to those three properties, we continue to allowing ourselves to shrink the domain of $\Xi$ to a specific collection of subsets of $\mathbb{R}$. For this, we need to discuss the abstract properties such collections of subsets will have.

#### 2.1  The Definition

A $\sigma$-algebra on a set $X$ is any collection of subsets of $X$ satisfying abstract rules. For this course specifically, $\sigma$-algebras are used to keep track of the subsets of $X$ whose size we are permitted to calculate.

**Definition**  Fix a set $X$, a set $\mathscr{B}\subset \mathcal{P}(X)$ is a $\sigma$-algebra if it has the following properties,
1. $X\in\mathscr{B}$
2. If $B\in\mathscr{B}$, then $X\backslash B\in \mathscr{B}$.
3. For any sequence $B_1,B_2,B_3,\dots$ of sets in $\mathscr{B}$, the union $B_1\cup B_2\cup B_3\cup \dots$ belongs to $\mathscr{B}$.

An example for an $\sigma$-algebra: suppose we were flipping two fair coins with possibility of either heads or tails in each, the possibilities then, which are more precisely described as the sample space, is $\Omega=\{ (H,H), (H,T), (T,H), (T, T) \}$. Then, a valid $\sigma$-algebra for $\Omega$ would be $\mathcal{P}(\Omega)$, which is the power set of the sample space. If we were to extend all possible subsets, it would look like,
$$ \mathscr{B}_\Omega=\{ \varnothing; \{ (H,H) \}; \{ (H,T) \};\{ (T,H) \};\{ (T,T) \};\{ (H,H), (H,T) \};\dots;\{(H,T),(T,H),(T,T)\};\Omega\} $$
From the definition, we have satisfied the first property that there is both $\varnothing$ and $\Omega$ present (there exist alternate definitions where it is either $\varnothing$ or $\Omega$ themselves, both work here). As for the second property, without explicitly proving it, it is clear to see that any event $B$ has its complement $X\backslash B$ present in $\mathscr{B}_\Omega$; $\{(H, H) \}\in\mathscr{B}_\Omega\implies \{ (H,T) \},\{ (T,H) \},\{ (T,T) \}\in\mathscr{B}_\Omega$. Finally, the third property is satisfied since the union of any subset in $\mathscr{B}_\Omega$ also belongs in it.

Intuition for each, (1) it must contain the whole space itself, meaning the entire space is measurable, (2) $\mathscr{B}$ is closed under complements, which means that if we have a measurable set $A$, we should also be able to measure everything except $A$, and (3) $\mathscr{B}$ is closed under finite unions, which means that if two or more sets are measurable individually, their combination should also be measurable. With both of these, we can also build countable intersections, $A\cap B = (A^c \cup B^c)^c$, so closure under complements and countable unions guarantee closure under intersections via De Morgan's laws. Closure meaning that if you apply that operation to members of the set (or system), the result is still a member of the set.

These countable guarantees matter because without them, we could start with measurable sets, take countable unions, and end up with a set that isn't measurable -- breaking consistency of measure. So a $\sigma$-algebra is the smallest natural system of sets that (1) contains some initial collection, whether they be open sets or intervals, and (2) is closed under operations that arise in limits and infinite processes, them being countable unions, intersections, and complements.

For our purposes $\sigma$-algebras will serve as the domains of measures. By a measurable space we mean a pair $(X,\mathscr{B})$ where $X$ is a set and $\mathscr{B}$ is a $\sigma$-algebra of subsets of $X$.

**The trivial $\sigma$-algebra**  For any set $X$ the collection $\{ \varnothing, X \}$ is a $\sigma$-algebra, called the **trivial $\sigma$-algebra** on $X$. It is trivial in the sense that it doesn't contain any interesting subsets of $X$.

**The full $\sigma$-algebra**  For any set $X$ the collection $\mathcal{P}(X)$ is a $\sigma$-algebra called the **full $\sigma$-algebra** on $X$. 

**The $\sigma$-algebra generated by a collection**  For any set $X$ and any collection $\mathcal{F}\subset \mathcal{P}(X)$ there is a $\sigma$-algebra containing $\mathcal{F}$ that we can think of as the $\sigma$-algebra generated by $\mathcal{F}$. Defined to be,
$$ \sigma(\mathcal{F})=\{ A\subset \mathbb{R}:A \text{ in every sigma-algebra that contains } \mathcal{F} \} $$
and is in fact a $\sigma$-algebra.

We will work often with $\sigma$-algebras generated by this or that collection of sets. It is useful to think of $\sigma(\mathcal{F})$ as the *smallest* $\sigma$-algebra that contains $\mathcal{F}$. In particular, it is always true that if a $\sigma$-algebra $\mathscr{C}$ contains $\mathcal{F}$ then it automatically contains $\sigma(\mathcal{F})$. 

---

## Appendix
### The Intuition Behind It
This builds a framework for measuring the "size" of sets, not just finite sets or intervals, but possibly infinite ones. For this, we need a collection of sets we can assign measure to, and rules for how measures behave, especially when combining sets.

We can think of a $\sigma$-algebra as a well-behaved collection of subsets of a set $E$. The $\sigma$-algebra $\mathcal{E}$ tells us which subsets of $E$ are "measurable". It must satisfy that,
1. The empty set $\varnothing$ is measurable.
2. If you can measure $A$, you can measure what is *not* $A$.
3. If you can measure $A_1,A_2,A_3,\dots,$ then you can measure the union $A_1\cup A_2\cup \dots$.
We need these so that if we want to do operations like unions and complements, which are necessary for probability and integration, we stay within the world of measurable sets. As an example, on $E=\mathbb{R}$, the Borel $\sigma$-algebra is the collection of sets you can build from open intervals using countable unions, intersections, and complements.

---

### References
- [MATH 41021/61021 Measure Theory and Ergodic Theory, Donald Robertson](https://personalpages.manchester.ac.uk/staff/donald.robertson/teaching/23-24/41021/notes/length.html)
- [¿Qué es un sigma-algebra?, Topos Uranos](https://www.youtube.com/watch?v=cBnDYSiuyfE)
- 
