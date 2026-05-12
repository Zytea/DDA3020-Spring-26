# HW1 Written Problems Solution

## 1.1. Weighted Linear Regression

### (a)

Define

$$
\tilde{x}_i=\begin{bmatrix}x_i\\1\end{bmatrix}\in\mathbb{R}^{d+1}, \\
\quad \tilde{X}= \begin{bmatrix} \tilde{x}_1^T\\ \vdots\\ \tilde{x}_N^T \end{bmatrix}\in\mathbb{R}^{N\times(d+1)}, \\
\quad Y= \begin{bmatrix} y_1^T\\ \vdots\\ y_N^T \end{bmatrix}\in\mathbb{R}^{N\times m}, \\
\quad A=\mathrm{diag}(\alpha_1,\dots,\alpha_N).
$$

$$
\tilde{W}=\begin{bmatrix}W & b\end{bmatrix}\in\mathbb{R}^{m\times(d+1)}.
$$

Then

$$
J(W,b)=\sum_{i=1}^N \alpha_i\|y_i-(W x_i+b)\|^2 =\mathrm{tr}\Big((Y-\tilde{X}\tilde{W}^T)^T A (Y-\tilde{X}\tilde{W}^T)\Big).
$$

Derivative:
$$
 \frac{\partial J}{\partial \tilde{W}} = -2 A (Y-\tilde{X}\tilde{W}^T)^T \tilde{X} = -2\bigl(Y^T A \tilde{X}-\tilde{W} \tilde{X}^T A \tilde{X}\bigr).
$$
Setting it to zero:
$$
\tilde{X}^T A \tilde{X}\,\tilde{W}^T=\tilde{X}^T A Y.
$$

Hence the closed-form solution is

$$
\tilde{W}^{*T}=(\tilde{X}^T A \tilde{X})^{-1}\tilde{X}^T A Y.
$$

i.e.,

$$
\tilde{W}^* = Y^T A \tilde{X}\,(\tilde{X}^T A \tilde{X})^{-1},
$$

and we get $W^*$ (the first $d$ columns of $\tilde{W}^*$) and $b^*$ (the last column of $\tilde{W}^*$).

### (b)

Let $r_i = y_i-(W x_i+b)$. 

Then
$$
J(W,b)=\sum_{i=1}^N \alpha_i\, r_i^T r_i.
$$

Gradients:

$$
\frac{\partial J}{\partial W} = -2\sum_{i=1}^N \alpha_i\, r_i\, x_i^T, \\
\qquad \frac{\partial J}{\partial b} = -2\sum_{i=1}^N \alpha_i\, r_i.
$$

Gradient descent updates (learning rate $\eta$):

$$
W^{(t+1)} = W^{(t)} - \eta \frac{\partial J}{\partial W} = W^{(t)} + 2\eta\sum_{i=1}^N \alpha_i\big(y_i-(W^{(t)}x_i+b^{(t)})\big)x_i^T,
$$

$$
b^{(t+1)} = b^{(t)} - \eta \frac{\partial J}{\partial b} = b^{(t)} + 2\eta\sum_{i=1}^N \alpha_i\big(y_i-(W^{(t)}x_i+b^{(t)})\big).
$$

## 1.2. Mathematical Foundations: Optimization & Information Theory

### (a)

Objective:

$$
f_0(x_1,x_2)=(x_1-2)^2+(x_2-1)^2.
$$

Gradient:
$$
\nabla f_0 = \begin{bmatrix} 2(x_1-2)\\ 2(x_2-1) \end{bmatrix}.
$$


Hessian:
$$
\nabla^2 f_0= \begin{bmatrix} 2 & 0\\ 0 & 2 \end{bmatrix},
$$

which is positive definite, so $f_0$ is convex. 

The constraints
$$
x_1+x_2\le 1,\quad x_1\ge 0,\quad x_2\ge 0
$$

are affine (linear) constraints, hence they define a convex feasible set. 

**Therefore the problem is a convex optimization problem.**

Then we shall deal with KKT conditions:

Write constraints in the form $g_i(x)\le 0$:
$$
g_1(x)=x_1+x_2-1\le 0,\quad g_2(x)=-x_1\le 0,\quad g_3(x)=-x_2\le 0.
$$

Lagrangian:

$$
L(x,\mu)= (x_1-2)^2+(x_2-1)^2 +\mu_1(x_1+x_2-1)-\mu_2 x_1-\mu_3 x_2, \quad \mu_1,\mu_2,\mu_3\ge 0.
$$

**KKT conditions**

1.  **Stationarity**

$$
\nabla_x L = 0 \Rightarrow \begin{cases} 2(x_1-2)+\mu_1-\mu_2=0,\\ 2(x_2-1)+\mu_1-\mu_3=0. \end{cases}
$$

2.  **Primal feasibility**
    

$$
x_1+x_2\le 1,\quad x_1\ge 0,\quad x_2\ge 0.
$$

3.  **Dual feasibility**
    

$$
\mu_1,\mu_2,\mu_3\ge 0.
$$

4.  **Complementary slackness**
    

$$
\mu_1(x_1+x_2-1)=0,\quad \mu_2 x_1=0,\quad \mu_3 x_2=0.
$$

### (b)

From stationarity, we have:
$$
x_1=2-\frac{\mu_1}{2},\quad x_2=1-\frac{\mu_1}{2}.
$$

Given that $\mu_1(x_1+x_2-1)=0$, we have 2 cases here:

#### Case 1:  $\mu_1=0$.

Then $x_1=2$, $x_2=1$, which violates the primal feasibility condition $x_1+x_2\le 1$. 

So $\mu_1=0$ is not possible.

#### Case 2: $\mu_1>0$, $x_1+x_2=1.$

At the optimum we will have $x_1>0$ , so $\mu_2=0$ by $\mu_2 x_1=0$. 

Also, the closest point on the line $x_1+x_2=1$ in the first quadrant occurs at $x_2=0$, so $x_2=0$ (given $\mu_3 x_2=0$ ).

Using stationarity condition:

$$
2(1-2)+\mu_1-\mu_2=0 \Rightarrow -2+\mu_1=0 \Rightarrow \mu_1=2,
$$

$$
2(0-1)+\mu_1-\mu_3=0 \Rightarrow -2+2-\mu_3=0 \Rightarrow \mu_3=0.
$$

All other KKT conditions are also satisfied, so

$$
x^*=(1,0).
$$

### (c)

#### (1)

$$
D_{\mathrm{KL}}(P\|Q)=\sum_x P(x)\log\frac{P(x)}{Q(x)}, \\
 D_{\mathrm{KL}}(Q\|P)=\sum_x Q(x)\log\frac{Q(x)}{P(x)}.
$$

Difference:

$$
D_{\mathrm{KL}}(P\|Q)-D_{\mathrm{KL}}(Q\|P) \\ =\sum_x P(x)\log\frac{P(x)}{Q(x)}+\sum_x Q(x)\log\frac{P(x)}{Q(x)}\\  =\sum_x (P(x)+Q(x))\log\frac{P(x)}{Q(x)}.
$$

In general, $\sum_x (P(x)+Q(x))\log\frac{P(x)}{Q(x)}\neq 0$ (To support this argument, the next section gives a simple counterexample, i.e., $P(1)=\frac13,\; P(2)=\frac23;\qquad Q(1)=\frac12,\; Q(2)=\frac12. \qquad D_{\mathrm{KL}}(P\|Q)=\frac13\log\frac{2}{3}+\frac23\log\frac{4}{3}\approx 0.05663, $$\qquad D_{\mathrm{KL}}(Q\|P)=\frac12\log\frac{3}{2}+\frac12\log\frac{3}{4} =\frac12\log\frac{9}{8}\approx 0.05889.$), hence

$$
D_{\mathrm{KL}}(P\|Q)\ne D_{\mathrm{KL}}(Q\|P)
$$

#### (2)

Let the sample space be $\{1,2\}$, with

$$
P(1)=\frac13,\; P(2)=\frac23;\qquad Q(1)=\frac12,\; Q(2)=\frac12.
$$

Then

$$
D_{\mathrm{KL}}(P\|Q)=\frac13\log\frac{2}{3}+\frac23\log\frac{4}{3}\approx 0.05663,
$$

$$
D_{\mathrm{KL}}(Q\|P)=\frac12\log\frac{3}{2}+\frac12\log\frac{3}{4} =\frac12\log\frac{9}{8}\approx 0.05889.
$$

So $D_{\mathrm{KL}}(P\|Q)\neq D_{\mathrm{KL}}(Q\|P)$.

## 1.3. Logistic Regression Properties

### (a)

$$
\sigma(z)=\frac{1}{1+e^{-z}}.
$$

Differentiate:

$$
\sigma'(z)=\frac{e^{-z}}{(1+e^{-z})^2}.
$$

Also,

$$
\sigma(z)\bigl(1-\sigma(z)\bigr)=\frac{1}{1+e^{-z}}\left(1-\frac{1}{1+e^{-z}}\right) =\frac{1}{1+e^{-z}}\cdot\frac{e^{-z}}{1+e^{-z}} =\frac{e^{-z}}{(1+e^{-z})^2}.
$$

Hence,

$$
\sigma'(z)=\sigma(z)\bigl(1-\sigma(z)\bigr).
$$

### (b)

For one sample,

$$
J_i(w)=-\Big(y_i\log h_i+(1-y_i)\log(1-h_i)\Big),\quad h_i=\sigma(w^T x_i).
$$

Using chain rule:

$$
\frac{\partial J_i}{\partial w}=(h_i-y_i)x_i.
$$

Summing it up:

$$
\nabla_w J(w)=\sum_{i=1}^n (h_i-y_i)x_i.
$$

Let $X\in\mathbb{R}^{n\times d}$ be the data matrix (rows are $x_i^T$), $h\in\mathbb{R}^n$ and $y\in\mathbb{R}^n$. 

Then
$$
\nabla_w J(w)=X^T(h-y).
$$

### (c)

Let $S=\mathrm{diag}(h_i(1-h_i))$ (so $S\succeq 0$). 

Hessian:
$$
H=\nabla_w^2 J(w)=X^T S X.
$$

For any vector $v$,

$$
v^T H v = v^T X^T S X v = (Xv)^T S (Xv)=\sum_{i=1}^n s_i\,(x_i^T v)^2\ge 0,
$$

This is because each $s_i=h_i(1-h_i)\ge 0$. 

Thus $H$ is PSD, and $J(w)$ is convex.

### (d)

With L2 regularization:

$$
J_{\mathrm{reg}}(w)=J(w)+\frac{\lambda}{2}w^T w.
$$

Gradient:

$$
\nabla_w J_{\mathrm{reg}}(w)=X^T(h-y)+\lambda w.
$$

Gradient descent update:

$$
w^{(t+1)}=w^{(t)}-\eta\Big(X^T(h^{(t)}-y)+\lambda w^{(t)}\Big).
$$

### (e)

For softmax regression,

$$
P_{ik}=\frac{e^{w_k^T x_i}}{\sum_{j=1}^K e^{w_j^T x_i}}, \\
\qquad L=-\sum_{i=1}^n\sum_{k=1}^K y_{ik}\log P_{ik} \\
=-\sum_{i=1}^n\sum_{k=1}^K y_{ik}\bigl(w_k^T x_i-\log\sum_{j=1}^K e^{w_j^T x_i}\bigr)
$$

Since
$$
\nabla_{w_k} (-\sum_{i=1}^n\sum_{j=1}^K y_{ij} w_j^T x_i)=-\sum_{i=1}^n y_{ik} x_i,
$$
and
$$
\nabla_{w_k} \Bigl(\sum_{i=1}^n\sum_{j=1}^K y_{ij}\log\sum_{l=1}^K e^{w_l^T x_i}\Bigr) \\ 
=\nabla_{w_k} \Bigl(\sum_{i=1}^n \log\sum_{l=1}^K e^{w_l^T x_i}\Bigr) \\ 
=\sum_{i=1}^n \frac{e^{w_k^T x_i}}{\sum_{j=1}^K e^{w_j^T x_i}} x_i \\
=\sum_{i=1}^n P_{ik} x_i,
$$

Summing the two equations, and we have the gradient:
$$
\nabla_{w_k} L=\sum_{i=1}^n (P_{ik}-y_{ik})x_i.
$$

In matrix form (letting $p_k=[P_{1k},\dots,P_{nk}]^T$ and $y_k=[y_{1k},\dots,y_{nk}]^T$):

$$
\nabla_{w_k} L = X^T(p_k-y_k).
$$
