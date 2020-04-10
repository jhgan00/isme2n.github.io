---
layout: post
title: "[ML/Stats] Why Standardized Residual instead of Pearson Residual?"
categories: [doc]
tags: [stat]
comments: true
---


$$
X_i \sim N(\mu, \sigma^2)
$$

$$
\begin{aligned}
Var(X_i - \overline{X}) &= E[(X_i - \overline {X})^2] - [E(X_i - \overline{X})]^2 \\ \\
&= [E(X_i^2) - E(X_i)^2] + [E(\overline{X}^2) - E(\overline{X})^2] -2[E(X_i\overline{X}) -E(X_i)E(\overline{X})] \\ \\
&= Var(X_i) + Var(\overline{X}) -2[E(X_i\overline{X}) -E(X_i)E(\overline{X})] \\ \\
\end{aligned}
$$

$$
\begin{aligned}
E(X_i\overline{X}) &= E(X_i\frac{1}{n}\sum_jX_j) \\
&= \frac{1}{n}\sum_jE(X_iX_j) \\
&= \frac{1}{n}\sum^n_{j=1,j\neq{i}}E(X_i)E(X_j) + \frac{1}{n}E(X_i^2)
\end{aligned}
$$