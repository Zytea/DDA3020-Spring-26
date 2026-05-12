# DDA3020 Machine Learning Final Exam 2022.12 - Solutions

## 1 Single-choice questions

### 1.1

**Answer: C**

Optimization is used as a tool in machine learning and AI, but it is not listed here as a sub-area of AI in the same way as Computer Vision, Machine Learning, and Natural Language Processing.

### 1.2

**Answer: B**

The binomial distribution is discrete. Gaussian, Laplace, and Beta distributions are continuous.

### 1.3

**Answer: C**

Following the course-intended single-choice reading, grouping body dimensions into clothing sizes is the clustering/unsupervised-learning example.

### 1.4

**Answer: B**

K-Nearest Neighbors is non-parametric.

### 1.5

**Answer: D**

The noise term is intrinsic to the data-generating process and cannot be reduced by choosing a different model.

### 1.6

**Answer: B**

Adding regularization terms penalizes overly complex parameter values and can reduce overfitting.

### 1.7

**Answer: C**

A neural network without nonlinear activation functions is equivalent to a linear model.

### 1.8

**Answer: D**

Setting a small maximum depth restricts the tree and makes it shallow.

### 1.9

**Answer: D**

The initial centroids are $50$ and $75$. Assigning by Euclidean distance gives

$$
G_A=\{50,60\},
\qquad
G_B=\{90,66,71,82,72,75,68,92\}.
$$

The new centroid of Group A is

$$
\frac{50+60}{2}=55.
$$

### 1.10

**Answer: B**

PCA is a linear projection method, not a nonlinear projection method.

### 1.11

**Answer: D**

GMM is usually more computationally expensive than K-means because it estimates soft responsibilities and Gaussian parameters.

### 1.12

**Answer: B**

From the figure, the counts are

$$
TP=5,
\quad
FN=1,
\quad
FP=2,
\quad
TN=2.
$$

Thus recall is

$$
\frac{TP}{TP+FN}=\frac{5}{6},
$$

not $4/6$. Therefore B is the incorrect statement.

### 1.13

**Answer: C**

Following the course's simplified linear-system terminology, when $m>d$, the system is over-determined and usually has no exact solution.

### 1.14

**Answer: D**

Both logistic regression and softmax regression are usually optimized iteratively and have no closed-form solution.

### 1.15

**Answer: C**

The predictions have mean

$$
\bar h=8.7.
$$

The MSE is

$$
\frac{1+4+4+9+16+25+1+0+4+9}{10}=7.3.
$$

The squared bias is

$$
(8.7-8.5)^2=0.04.
$$

The empirical variance is

$$
\frac{1}{10}\sum_{i=1}^{10}(h_i-8.7)^2=7.21\approx7.2,
$$

not $7.1$.

## 2 Calculations and Derivations

## 2.1 Discrete random variables

We have

$$
P(Y=2)=0.3,
\qquad
P(Y=3)=0.7,
\qquad
X\mid Y=y\sim \mathrm{Binomial}\left(2,\frac1y\right).
$$

### (1) Distribution of $X$

For $Y=2$:

$$
P(X=0\mid Y=2)=\frac14,
\quad
P(X=1\mid Y=2)=\frac12,
\quad
P(X=2\mid Y=2)=\frac14.
$$

For $Y=3$:

$$
P(X=0\mid Y=3)=\frac49,
\quad
P(X=1\mid Y=3)=\frac49,
\quad
P(X=2\mid Y=3)=\frac19.
$$

Thus

$$
\begin{aligned}
P(X=0)&=0.3\cdot\frac14+0.7\cdot\frac49=\frac{139}{360},\\
P(X=1)&=0.3\cdot\frac12+0.7\cdot\frac49=\frac{83}{180},\\
P(X=2)&=0.3\cdot\frac14+0.7\cdot\frac19=\frac{11}{72}.
\end{aligned}
$$

### (2) Conditional distribution $P(Y\mid X=1)$

$$
\begin{aligned}
P(Y=2\mid X=1)
&=\frac{P(X=1\mid Y=2)P(Y=2)}{P(X=1)}\\
&=\frac{\frac12\cdot0.3}{\frac{83}{180}}=\frac{27}{83},\\
P(Y=3\mid X=1)
&=\frac{P(X=1\mid Y=3)P(Y=3)}{P(X=1)}\\
&=\frac{\frac49\cdot0.7}{\frac{83}{180}}=\frac{56}{83}.
\end{aligned}
$$

### (3) $E(X)$ and $E(Y\mid X=1)$

$$
\begin{aligned}
E[X]
&=E[E[X\mid Y]]\\
&=0.3\cdot2\cdot\frac12+0.7\cdot2\cdot\frac13\\
&=\frac{23}{30}.
\end{aligned}
$$

Also,

$$
E[Y\mid X=1]
=2\cdot\frac{27}{83}+3\cdot\frac{56}{83}
=\frac{222}{83}.
$$

## 2.2 CNN model

The model is

$$
f_{CNN}(X)=\operatorname{Softmax}(FC_1(Conv_2(MP_1(Relu_1(Conv_1(X)))))).
$$

The input is $28\times28\times3$. The first convolution has $10$ filters of size $6\times6\times3$, stride $2$, padding $1$. The second convolution has $50$ filters of size $5\times5\times10$, stride $1$, padding $0$.

### (1) Feature-map sizes

For $Conv_1$:

$$
\frac{28+2\cdot1-6}{2}+1=13.
$$

Therefore,

$$
Relu_1: 13\times13\times10.
$$

For $MP_1$, using the common floor convention for pooling,

$$
\left\lfloor\frac{13-2}{2}\right\rfloor+1=6,
$$

so

$$
MP_1: 6\times6\times10.
$$

For $Conv_2$:

$$
\frac{6-5}{1}+1=2,
$$

so

$$
Conv_2: 2\times2\times50.
$$

### (2) Number of parameters

$$
\begin{aligned}
\#Conv_1&=10(6\cdot6\cdot3+1)=1090,\\
\#Conv_2&=50(5\cdot5\cdot10+1)=12550,\\
\#FC_1&=(2\cdot2\cdot50+1)\cdot10=2010.
\end{aligned}
$$

Total number of trainable parameters:

$$
1090+12550+2010=15650.
$$

### (3) Computational graph

```text
X -> Conv1 -> z1 -> Relu1 -> z2 -> MP1 -> z3 -> Conv2 -> z4 -> FC1 -> z5 -> Softmax -> z6 = y -> L(y,t)
```

Equivalently,

$$
\begin{aligned}
z_1&=Conv_1(X;w_1,b_1),\\
z_2&=Relu_1(z_1),\\
z_3&=MP_1(z_2),\\
z_4&=Conv_2(z_3;w_2,b_2),\\
z_5&=FC_1(z_4;w_3,b_3),\\
z_6&=Softmax(z_5),\\
y&=z_6.
\end{aligned}
$$

### (4) Backpropagation formulae

Forward pass:

$$
\begin{aligned}
z_1&=Conv_1(X;w_1,b_1),\\
z_2&=Relu(z_1),\\
z_3&=MP(z_2),\\
z_4&=Conv_2(z_3;w_2,b_2),\\
z_5&=W_3\operatorname{vec}(z_4)+b_3,\\
y=z_6&=Softmax(z_5),\\
\ell&=L(y,t).
\end{aligned}
$$

Backward pass:

$$
\begin{aligned}
\delta_6&=\frac{\partial L}{\partial z_6},\\
\delta_5&=\delta_6\frac{\partial z_6}{\partial z_5},\\
\frac{\partial L}{\partial W_3}&=\delta_5\operatorname{vec}(z_4)^\top,
\qquad
\frac{\partial L}{\partial b_3}=\delta_5,\\
\delta_4&=W_3^\top\delta_5,\\
\frac{\partial L}{\partial w_2}&=\delta_4\frac{\partial z_4}{\partial w_2},
\qquad
\frac{\partial L}{\partial b_2}=\delta_4\frac{\partial z_4}{\partial b_2},\\
\delta_3&=\delta_4\frac{\partial z_4}{\partial z_3},\\
\delta_2&=\delta_3\frac{\partial z_3}{\partial z_2},\\
\delta_1&=\delta_2\odot \mathbf 1[z_1>0],\\
\frac{\partial L}{\partial w_1}&=\delta_1\frac{\partial z_1}{\partial w_1},
\qquad
\frac{\partial L}{\partial b_1}=\delta_1\frac{\partial z_1}{\partial b_1}.
\end{aligned}
$$

For softmax with cross-entropy, we usually use

$$
\delta_5=y-t.
$$

## 2.3 SVM derivation

The hard-margin SVM problem is

$$
\min_{w,b}\frac12\|w\|^2
\quad
\text{s.t.}\quad
 y_i(w^\top x_i+b)\ge1,
\quad i=1,\ldots,n.
$$

### (1) Large-margin derivation

The distance from $x_i$ to the hyperplane $w^\top x+b=0$ is

$$
\frac{|w^\top x_i+b|}{\|w\|}.
$$

For separated data, choose the canonical scaling

$$
\min_i y_i(w^\top x_i+b)=1.
$$

Then the closest points are at distance $1/\|w\|$, and the margin width is $2/\|w\|$. Maximizing $2/\|w\|$ is equivalent to minimizing $\frac12\|w\|^2$, subject to

$$
y_i(w^\top x_i+b)\ge1.
$$

### (2) Hinge-loss derivation

The hinge loss is

$$
\ell_{hinge}(x_i,y_i;w,b)=\max\{0,1-y_i(w^\top x_i+b)\}.
$$

The regularized hinge-loss objective is

$$
C\sum_{i=1}^n\max\{0,1-y_i(w^\top x_i+b)\}+\frac12\|w\|^2.
$$

In the hard-margin case, requiring zero hinge loss means

$$
y_i(w^\top x_i+b)\ge1,
$$

so we obtain the hard-margin constrained problem.

### (3) Lagrangian, KKT conditions, and support vectors

The Lagrangian is

$$
L(w,b,\alpha)=\frac12\|w\|^2+
\sum_{i=1}^n\alpha_i\left(1-y_i(w^\top x_i+b)\right),
\qquad \alpha_i\ge0.
$$

Stationarity gives

$$
\begin{aligned}
\frac{\partial L}{\partial w}=0
&\Longrightarrow
w=\sum_{i=1}^n\alpha_i y_i x_i,\\
\frac{\partial L}{\partial b}=0
&\Longrightarrow
\sum_{i=1}^n\alpha_i y_i=0.
\end{aligned}
$$

Together with primal feasibility, dual feasibility, and complementary slackness,

$$
\begin{aligned}
y_i(w^\top x_i+b)&\ge1,\\
\alpha_i&\ge0,\\
\alpha_i\left(1-y_i(w^\top x_i+b)\right)&=0.
\end{aligned}
$$

The dual problem is

$$
\begin{aligned}
\max_\alpha\quad
&\sum_{i=1}^n\alpha_i-rac12\sum_{i=1}^n\sum_{j=1}^n
\alpha_i\alpha_j y_i y_j x_i^\top x_j\\
\text{s.t.}\quad
&\sum_{i=1}^n\alpha_i y_i=0,
\qquad
\alpha_i\ge0.
\end{aligned}
$$

A sample is a support vector when

$$
\alpha_i>0.
$$

By complementary slackness, such a point satisfies

$$
y_i(w^\top x_i+b)=1.
$$

## 2.4 Gaussian mixture model and EM

Let

$$
p(x)=\sum_{k=1}^K\pi_k\mathcal N(x\mid\mu_k,\Sigma_k),
\qquad
\sum_{k=1}^K\pi_k=1.
$$

### (1) Likelihood decomposition

For any

$$
q(z)=\prod_{n=1}^N q_n(z^{(n)}),
$$

we have

$$
\begin{aligned}
\ln p(D;\Theta)
&=\sum_{n=1}^N\ln\sum_{z^{(n)}}p(x^{(n)},z^{(n)};\Theta)\\
&=\sum_{n=1}^N\ln\sum_{z^{(n)}}q_n(z^{(n)})
\frac{p(x^{(n)},z^{(n)};\Theta)}{q_n(z^{(n)})}\\
&\ge
\sum_{n=1}^N\sum_{z^{(n)}}q_n(z^{(n)})
\ln\frac{p(x^{(n)},z^{(n)};\Theta)}{q_n(z^{(n)})}\\
&=L(q;\Theta).
\end{aligned}
$$

The gap is

$$
\ln p(D;\Theta)-L(q;\Theta)
=\sum_{n=1}^N
\mathrm{KL}\left(q_n(z^{(n)})\,\|\,p(z^{(n)}\mid x^{(n)};\Theta)\right)
\ge0.
$$

### (2) E-step

Set

$$
q_n(z^{(n)}=k)=\gamma_{nk}
=\frac{\pi_k\mathcal N(x^{(n)}\mid\mu_k,\Sigma_k)}
{\sum_{j=1}^K\pi_j\mathcal N(x^{(n)}\mid\mu_j,\Sigma_j)}.
$$

### (3) M-step

Let

$$
N_k=\sum_{n=1}^N\gamma_{nk}.
$$

Then

$$
\begin{aligned}
\mu_k^{new}&=\frac{1}{N_k}\sum_{n=1}^N\gamma_{nk}x^{(n)},\\
\Sigma_k^{new}&=\frac{1}{N_k}\sum_{n=1}^N\gamma_{nk}
(x^{(n)}-\mu_k^{new})(x^{(n)}-\mu_k^{new})^\top,\\
\pi_k^{new}&=\frac{N_k}{N}.
\end{aligned}
$$

## 2.5 Decision tree with entropy

There are $4$ positive and $4$ negative samples, so

$$
H(D)=1.
$$

### (1) Best root attribute

For **Color**:

- Red: $3+$, $0-$, entropy $0$.
- Blue: $1+$, $2-$, entropy $0.9183$.
- Green: $0+$, $2-$, entropy $0$.

Thus

$$
H(D\mid Color)=\frac38\cdot0+\frac38\cdot0.9183+\frac28\cdot0=0.3444,
$$

and

$$
Gain(Color)=1-0.3444=0.6556.
$$

For **Shape**:

- Square: $3+$, $1-$, entropy $0.8113$.
- Circle: $1+$, $2-$, entropy $0.9183$.
- Triangle: $0+$, $1-$, entropy $0$.

So

$$
H(D\mid Shape)=\frac48(0.8113)+\frac38(0.9183)+\frac18(0)=0.75,
$$

and

$$
Gain(Shape)=0.25.
$$

For **Size**:

- Big: $3+$, $2-$, entropy $0.9710$.
- Small: $1+$, $2-$, entropy $0.9183$.

So

$$
H(D\mid Size)=\frac58(0.9710)+\frac38(0.9183)=0.9512,
$$

and

$$
Gain(Size)=0.0488.
$$

The best root is **Color**.

### (2) Complete tree

```text
Color?
├── Red:    +   {1, 3, 4}
├── Green:  -   {5, 8}
└── Blue:   Shape?
    ├── Square:   +   {2}
    ├── Triangle: -   {6}
    └── Circle:   -   {7}
```

For the Blue branch, splitting by Shape gives pure leaves.

### (3) Predictions

For test sample **(Red, Triangle, Big)**:

```text
Color = Red -> class +
```

So the predicted label is **+**.

For test sample **(Green, Square, Small)**:

```text
Color = Green -> class -
```

So the predicted label is **-**.

## 2.6 PCA equivalence proof

Let the centered data be

$$
\bar x^{(n)}=x^{(n)}-\mu,
$$

and let $U\in\mathbb R^{D\times K}$ satisfy

$$
U^\top U=I.
$$

The low-dimensional representation and reconstruction are

$$
z_n=U^\top(x^{(n)}-\mu),
\qquad
\tilde x^{(n)}=Uz_n+\mu.
$$

Hence

$$
\tilde x^{(n)}-\mu=UU^\top(x^{(n)}-\mu).
$$

Because $UU^\top(x^{(n)}-\mu)$ is the orthogonal projection of $x^{(n)}-\mu$ onto the subspace spanned by $U$, we have the Pythagorean decomposition

$$
\begin{aligned}
\|x^{(n)}-\mu\|^2
&=\|\tilde x^{(n)}-\mu\|^2
+\|x^{(n)}-\tilde x^{(n)}\|^2.
\end{aligned}
$$

Averaging over $n$,

$$
\frac1N\sum_{n=1}^N\|x^{(n)}-\mu\|^2
=
\frac1N\sum_{n=1}^N\|\tilde x^{(n)}-\mu\|^2
+
\frac1N\sum_{n=1}^N\|x^{(n)}-\tilde x^{(n)}\|^2.
$$

The left-hand side is independent of $U$. Therefore, maximizing projected variance

$$
\frac1N\sum_{n=1}^N\|\tilde x^{(n)}-\mu\|^2
$$

is equivalent to minimizing reconstruction error

$$
\frac1N\sum_{n=1}^N\|x^{(n)}-\tilde x^{(n)}\|^2.
$$

To obtain $U$, form the empirical covariance matrix

$$
\Sigma=\frac1N\sum_{n=1}^N(x^{(n)}-\mu)(x^{(n)}-\mu)^\top.
$$

Then solve the eigenvalue problem

$$
\Sigma q_i=\lambda_i q_i,
\qquad
\lambda_1\ge\lambda_2\ge\cdots\ge\lambda_D.
$$

The PCA projection matrix is formed by the top $K$ eigenvectors:

$$
U=[q_1,q_2,\ldots,q_K].
$$
