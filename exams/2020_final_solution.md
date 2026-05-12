# DDA2020 Machine Learning Final Exam - Solutions

## 1 Multiple-choice questions

### 1.1

**Answer: A, B, C, D**

Computer Vision, Robotics, Natural Language Processing, and Machine Learning are treated as sub-areas of AI. Optimization is a mathematical tool used in AI, but not usually listed as a sub-area here.

### 1.2

**Answer: A, C**

The model has high training accuracy but poor test accuracy, so it is overfitting. Regularization and more training data may improve test performance.

### 1.3

**Answer: A, B, D**

A feedforward network passes information from input to output. Backpropagation computes gradients from the output side backward and updates the weights. A CNN is not a fully connected network.

### 1.4

**Answer: B, C**

Information is $-\log p$, so a less probable event has larger information. KL divergence is non-negative and non-symmetric. A binomial distribution is discrete, not continuous.

### 1.5

**Answer: A, D, E**

For small feature dimension, the closed-form solution is reasonable. Polynomial regression can give a nonlinear decision boundary in the original input space. Ridge regression can reduce overfitting.

### 1.6

**Answer: A, B, D**

Interpreting “standard SVM” as the hard-margin linear SVM, it cannot directly handle non-linearly separable data. Slack variables allow violations, and the margin is determined by the closest points to the decision boundary. The statement about kernel SVM “perfectly” fitting all nonlinear data is too strong.

### 1.7

**Answer: B, D**

PCA is linear, so A is false. Reconstructed points are in the original space, so C is false. Dimensionality reduction can be supervised or unsupervised, and PCA chooses top eigenvectors.

### 1.8

**Answer: C, D**

A small maximum depth grows a shallow tree. Increasing the number of trees in bagging often improves or stabilizes training performance. Decision trees can handle categorical attributes, and a large minimum leaf size makes the tree shallower, not deeper.

### 1.9

**Answer: A, D**

Clothing-size grouping is a clustering problem. Image compression can also be treated as unsupervised learning, for example by clustering pixel colors with K-means.

### 1.10

**Answer: A**

Initial centroids are $66$ and $75$. Assigning by Euclidean distance gives

$$
G_A=\{50,66,68,60\},\qquad
G_B=\{90,71,82,72,75,99\}.
$$

So Group A has $4$ students. The new centroid of Group A is

$$
\frac{50+66+68+60}{4}=61,
$$

not $60$.

### 1.11

**Answer: A, D**

ROC plots TPR against FPR. Also, as the classification threshold increases, more positives are rejected, so FNR increases. In general $TPR+FPR\ne 1$, and accuracy is not generally $(TPR+TNR)/2$ unless classes are balanced.

### 1.12

**Answer: B**

As model complexity increases, bias usually decreases and variance usually increases on unseen/test data.

### 1.13

**Answer: B, D**

EM monotonically improves, or at least does not decrease, the log-likelihood. For a concave function, Jensen's inequality gives

$$
f(E[X])\ge E[f(X)].
$$

The GMM-to-K-means equivalence needs a limiting covariance assumption, not simply $\Sigma=I$. Latent variables need not always be discrete.

### 1.14

**Answer: C, D**

Following the course's simplified linear-system terminology, $m>d$ is over-determined and usually has no exact solution, while $m<d$ is under-determined and usually has infinitely many solutions.

### 1.15

**Answer: B, C**

The task is to predict grades, and the accuracy of predicted grades can be used as the performance measure. Ungraded assignments do not provide supervised labels.

## 2 Calculations and Derivations

## 2.1 Discrete random variables

We have

$$
Y\in\{2,3\},\qquad P(Y=2)=0.4,\qquad P(Y=3)=0.6,
$$

and

$$
X\mid Y=y\sim \mathrm{Binomial}\left(2,\frac1y\right).
$$

### (1) Distribution of $X$

For $Y=2$,

$$
P(X=0\mid Y=2)=\frac14,
\quad
P(X=1\mid Y=2)=\frac12,
\quad
P(X=2\mid Y=2)=\frac14.
$$

For $Y=3$,

$$
P(X=0\mid Y=3)=\frac49,
\quad
P(X=1\mid Y=3)=\frac49,
\quad
P(X=2\mid Y=3)=\frac19.
$$

Therefore,

$$
\begin{aligned}
P(X=0)&=0.4\cdot\frac14+0.6\cdot\frac49=\frac{11}{30},\\
P(X=1)&=0.4\cdot\frac12+0.6\cdot\frac49=\frac{7}{15},\\
P(X=2)&=0.4\cdot\frac14+0.6\cdot\frac19=\frac16.
\end{aligned}
$$

### (2) Conditional distribution $P(Y\mid X=1)$

$$
\begin{aligned}
P(Y=2\mid X=1)
&=\frac{P(X=1\mid Y=2)P(Y=2)}{P(X=1)}\\
&=\frac{\frac12\cdot 0.4}{\frac{7}{15}}=\frac37,\\
P(Y=3\mid X=1)
&=\frac{P(X=1\mid Y=3)P(Y=3)}{P(X=1)}\\
&=\frac{\frac49\cdot 0.6}{\frac{7}{15}}=\frac47.
\end{aligned}
$$

### (3) $E(X)$ and $E(Y\mid X=1)$

$$
\begin{aligned}
E[X]
&=E[E[X\mid Y]]\\
&=0.4\cdot 2\cdot\frac12+0.6\cdot 2\cdot\frac13\\
&=\frac45.
\end{aligned}
$$

Also,

$$
E[Y\mid X=1]
=2\cdot\frac37+3\cdot\frac47
=\frac{18}{7}.
$$

## 2.2 CNN model

The model is

$$
f_{CNN}(X)=\operatorname{Softmax}(FC_1(Conv_2(MP_1(Relu_1(Conv_1(X)))))).
$$

### (1) Feature-map sizes

For $Conv_1$:

$$
\frac{36+2\cdot1-8}{2}+1=16.
$$

So

$$
Conv_1: 16\times 16\times 10,
\qquad
Relu_1: 16\times 16\times 10.
$$

After $MP_1$ with $2\times2$ pooling and stride $2$,

$$
MP_1: 8\times 8\times 10.
$$

For $Conv_2$:

$$
\frac{8-5}{1}+1=4.
$$

So

$$
Conv_2: 4\times 4\times 100.
$$

### (2) Number of parameters

$$
\begin{aligned}
\#Conv_1&=10(8\cdot 8\cdot 3+1)=1930,\\
\#Conv_2&=100(5\cdot 5\cdot 10+1)=25100,\\
\#FC_1&=(4\cdot4\cdot100+1)\cdot 10=16010.
\end{aligned}
$$

Thus the total number of trainable parameters is

$$
1930+25100+16010=43040.
$$

### (3) Computational graph

Let

$$
\begin{aligned}
z_1&=Conv_1(X;w_1,b_1),\\
z_2&=Relu_1(z_1),\\
z_3&=MP_1(z_2),\\
z_4&=Conv_2(z_3;w_2,b_2),\\
z_5&=FC_1(z_4;w_3,b_3),\\
z_6&=Softmax(z_5),\\
y&=z_6,\\
\ell&=L(y,t).
\end{aligned}
$$

The graph is

```text
X -> Conv1 -> z1 -> Relu1 -> z2 -> MP1 -> z3 -> Conv2 -> z4 -> FC1 -> z5 -> Softmax -> z6 = y -> L(y,t)
```

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

Backward pass follows the chain rule:

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

For softmax with cross-entropy, the common simplification is

$$
\delta_5=y-t.
$$

## 2.3 SVM derivation

The hard-margin SVM primal problem is

$$
\min_{w,b}\frac12\|w\|^2
\quad
\text{s.t.}\quad
 y_i(w^\top x_i+b)\ge 1,
\quad i=1,\ldots,n.
$$

### (1) Large-margin perspective

For a hyperplane

$$
f_{w,b}(x)=w^\top x+b=0,
$$

the distance from $x_i$ to the hyperplane is

$$
\frac{|w^\top x_i+b|}{\|w\|}.
$$

For correctly separated data, impose

$$
y_i(w^\top x_i+b)>0.
$$

Because $(w,b)$ can be rescaled without changing the hyperplane, choose the canonical scale

$$
\min_i y_i(w^\top x_i+b)=1.
$$

Then the closest points have distance $1/\|w\|$ from the boundary, and the total margin width is

$$
\frac{2}{\|w\|}.
$$

Maximizing the margin is equivalent to minimizing $\|w\|$, or equivalently

$$
\min_{w,b}\frac12\|w\|^2
\quad
\text{s.t.}\quad
 y_i(w^\top x_i+b)\ge 1.
$$

### (2) Hinge-loss perspective

The hinge loss is

$$
\ell_{hinge}(x_i,y_i;w,b)=\max\{0,1-y_i(w^\top x_i+b)\}.
$$

The soft-margin objective can be written as

$$
C\sum_{i=1}^n\max\{0,1-y_i(w^\top x_i+b)\}+\frac12\|w\|^2.
$$

In the hard-margin separable case, requiring zero hinge loss for every point gives

$$
1-y_i(w^\top x_i+b)\le 0
\quad\Longleftrightarrow\quad
 y_i(w^\top x_i+b)\ge 1.
$$

Thus the zero-hinge-loss constrained problem reduces to the hard-margin SVM primal objective above.

### (3) Lagrangian, KKT conditions, and solution

Write the constraints as

$$
1-y_i(w^\top x_i+b)\le 0.
$$

The Lagrangian is

$$
L(w,b,\alpha)
=\frac12\|w\|^2+
\sum_{i=1}^n\alpha_i\left(1-y_i(w^\top x_i+b)\right),
\qquad \alpha_i\ge 0.
$$

KKT stationarity gives

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

The other KKT conditions are

$$
\begin{aligned}
&y_i(w^\top x_i+b)\ge 1,\\
&\alpha_i\ge 0,\\
&\alpha_i\left(1-y_i(w^\top x_i+b)\right)=0.
\end{aligned}
$$

Substituting the stationarity conditions into the Lagrangian gives the dual problem

$$
\begin{aligned}
\max_\alpha\quad
&\sum_{i=1}^n\alpha_i-rac12\sum_{i=1}^n\sum_{j=1}^n
\alpha_i\alpha_j y_i y_j x_i^\top x_j\\
\text{s.t.}\quad
&\sum_{i=1}^n\alpha_i y_i=0,
\qquad
\alpha_i\ge 0.
\end{aligned}
$$

A training point is a **support vector** when

$$
\alpha_i>0.
$$

By complementary slackness, this implies

$$
y_i(w^\top x_i+b)=1,
$$

so the point lies exactly on the margin and affects the decision boundary.

## 2.4 Classification metrics and AUC

From the figure, using the stated decision boundary, the counts are

$$
TP=5,
\qquad
FN=1,
\qquad
FP=2,
\qquad
TN=2.
$$

### (1) Confusion matrix

Using rows as true labels and columns as predicted labels:

| | Predicted + | Predicted - |
|---|---:|---:|
| True + | $TP=5$ | $FN=1$ |
| True - | $FP=2$ | $TN=2$ |

### (2) Precision, recall, accuracy

$$
\begin{aligned}
\mathrm{Precision}&=\frac{TP}{TP+FP}=\frac{5}{7},\\
\mathrm{Recall}&=\frac{TP}{TP+FN}=\frac{5}{6},\\
\mathrm{Accuracy}&=\frac{TP+TN}{TP+TN+FP+FN}=\frac{7}{10}.
\end{aligned}
$$

### (3) FNR and FPR

$$
\begin{aligned}
\mathrm{FNR}&=\frac{FN}{TP+FN}=\frac16,\\
\mathrm{FPR}&=\frac{FP}{FP+TN}=\frac12.
\end{aligned}
$$

### (4) AUC

Positive scores:

$$
0.2,0.5,0.7,0.7,0.8,0.9.
$$

Negative scores:

$$
0.1,0.3,0.5,0.6.
$$

Use

$$
AUC=\frac{1}{m_+m_-}\sum_{i=1}^{m_+}\sum_{j=1}^{m_-}u(s_i^+-s_j^-),
$$

where

$$
u(e)=
\begin{cases}
1,&e>0,\\
0.5,&e=0,\\
0,&e<0.
\end{cases}
$$

The total pair score is $19.5$ out of $6\times4=24$ pairs. Hence

$$
AUC=\frac{19.5}{24}=0.8125=\frac{13}{16}.
$$

## 2.5 Gaussian mixture model and EM

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
&\ge\sum_{n=1}^N\sum_{z^{(n)}}q_n(z^{(n)})
\ln\frac{p(x^{(n)},z^{(n)};\Theta)}{q_n(z^{(n)})}\\
&=L(q;\Theta).
\end{aligned}
$$

The gap is

$$
\ln p(D;\Theta)-L(q;\Theta)
=\sum_{n=1}^N
\mathrm{KL}\left(q_n(z^{(n)})\,\|\,p(z^{(n)}\mid x^{(n)};\Theta)\right)
\ge 0.
$$

### (2) E-step

Set

$$
q_n(z^{(n)}=k)=\gamma_{nk}
=p(z^{(n)}=k\mid x^{(n)};\Theta),
$$

where

$$
\gamma_{nk}
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

## 2.6 Decision tree with entropy

The data have $4$ positive and $4$ negative samples, so

$$
H(D)=-\frac12\log_2\frac12-\frac12\log_2\frac12=1.
$$

### (1) Best root attribute

For **Color**:

- Red: $3+$, $0-$, entropy $0$.
- Blue: $1+$, $2-$, entropy

$$
-\frac13\log_2\frac13-\frac23\log_2\frac23\approx0.9183.
$$

- Green: $0+$, $2-$, entropy $0$.

Thus

$$
H(D\mid Color)=\frac38\cdot0+\frac38\cdot0.9183+\frac28\cdot0\approx0.3444.
$$

So

$$
Gain(Color)=1-0.3444=0.6556.
$$

For **Shape**:

- Square: $3+$, $1-$, entropy $0.8113$.
- Circle: $1+$, $3-$, entropy $0.8113$.

So

$$
H(D\mid Shape)=0.8113,
\qquad
Gain(Shape)=0.1887.
$$

For **Size**:

- Big: $4+$, $2-$, entropy $0.9183$.
- Small: $0+$, $2-$, entropy $0$.

So

$$
H(D\mid Size)=\frac68\cdot0.9183=0.6887,
\qquad
Gain(Size)=0.3113.
$$

The best root is **Color**.

### (2) Complete tree

```text
Color?
├── Red:    +   {1, 3, 4}
├── Green:  -   {5, 8}
└── Blue:   Shape?
    ├── Square: +   {2}
    └── Circle: -   {6, 7}
```

For the Blue branch, splitting by Shape gives pure leaves.

### (3) Classification error of each node

Use

$$
Error(node)=1-\max_c p(c\mid node).
$$

| Node | Samples | Class counts | Error |
|---|---:|---:|---:|
| Root | 8 | $4+$, $4-$ | $1-4/8=1/2$ |
| Color = Red | 3 | $3+$, $0-$ | $0$ |
| Color = Green | 2 | $0+$, $2-$ | $0$ |
| Color = Blue | 3 | $1+$, $2-$ | $1-2/3=1/3$ |
| Blue, Shape = Square | 1 | $1+$, $0-$ | $0$ |
| Blue, Shape = Circle | 2 | $0+$, $2-$ | $0$ |
