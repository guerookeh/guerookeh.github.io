---
title: Gradient Instability in General Multi-Layer Perceptrons
showDate: true
showReadingTime: true
draft: true
tags:
  - "#machine-learning"
---
**A derivation of activation-dependent layer-wise upper bounds on backpropagated gradient norms in a fully connected MLPs and their implications for gradient flow and stability.**
***
A recurring topic in machine learning is the [vanishing gradient problem](https://en.wikipedia.org/wiki/Vanishing_gradient_problem): *"greatly diverging gradient magnitudes between earlier and later layers encountered when training neural networks with backpropagation."* 

This problem, more specifically, is concerned on the contraction of the backpropagated signal induced by repeated multiplication of the layer Jacobians.

It is worth mentioning that when we talk about backward signal, we refer to the gradient of the loss with respect to a given intermediate state within the network, which in turn gets propagated backward during backpropagation to derive more backward signals with respect to other states. This tangibly comes in the form of activation gradients and pre-activation gradients. On the other hand, the forward signal refers to the pre-activations and activations that get propagated forward in the network. 

It is obvious that in the forward propagation of an MLP, the overall loss of the input under the current parameters is dependent on the earlier layers parameters as well. This dependence implies that a backward signal exists in principle; but the issue is that the backpropagated gradient can become too small and/or noisy to be useful for optimization.

***

The [vanishing and exploding gradients problem](https://en.wikipedia.org/wiki/Vanishing_gradient_problem) can be understood as the contraction or amplification of the backpropagated learning signal due to repeated multiplication by layer Jacobians, respectively. More generally, 







