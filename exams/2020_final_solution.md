# DDA2020 Machine Learning Final Exam - Solutions

## 1 Multiple-choice questions

### 1.1

**Answer: A, B, C, D**

- ✅ A. Computer Vision is a standard sub-area/application area of AI.
- ✅ B. Robotics is treated as a sub-area of AI.
- ✅ C. Natural Language Processing is a standard sub-area of AI.
- ✅ D. Machine Learning is a core sub-area of AI.
- ❌ E. Optimization is mainly a mathematical tool used to train AI/ML models, not listed in the PPT as a sub-area of AI.

PPT statement: AI includes areas such as machine learning, computer vision, natural language processing, and robotics; optimization is a tool used inside learning algorithms.

### 1.2

**Answer: A, C**

- ✅ A. Regularized logistic regression can reduce overfitting by penalizing overly complex parameters.
- ❌ B. A polynomial hypothesis usually increases model complexity; for a model already overfitting, this is not the intended fix.
- ✅ C. More training data can reduce variance and improve generalization.
- ❌ D. Adding more testing data only changes evaluation precision; it does not train a better model.

PPT statement: When training accuracy is high but test accuracy is poor, the model is overfitting. Typical remedies include regularization and collecting more training data.

### 1.3

**Answer: A, B, D**

- ✅ A. Backpropagation uses a forward pass to compute activations/loss and a backward pass to compute gradients for updating weights.
- ✅ B. In a feedforward neural network, information flows from input layers toward output layers.
- ❌ C. A CNN is not simply a fully connected network; it contains convolutional layers with local connectivity and weight sharing.
- ✅ D. For a multilayer perceptron, the loss is computed at the output layer and gradients are propagated backward by the chain rule.

PPT statement: Neural network training consists of forward propagation, loss computation, and backpropagation of gradients; CNNs use convolutional layers rather than only fully connected layers.

### 1.4

**Answer: B**

- ❌ A. Tossing a coin 4 times has $2^4=16$ ordered outcomes, not 4 possible events under the usual sample-space interpretation.
- ✅ B. Information is $-\log p$; since $0.01<0.1$, event B carries more information than event A.
- ❌ C. The word "positive" is not strictly correct: KL divergence is non-negative and can be $0$ when the two distributions are identical. It is generally non-symmetric.
- ❌ D. A binomial distribution is discrete; a Gaussian distribution is continuous.

PPT statement: Self-information decreases as probability increases; the correct KL statement is $D_{KL}(P\|Q)\ge0$ with equality possible when $P=Q$, and usually $D_{KL}(P\|Q)\ne D_{KL}(Q\|P)$; binomial is discrete and Gaussian is continuous.

### 1.5

**Answer: A, D, E**

- ✅ A. When $n$ is large but feature dimension $d$ is small, the closed-form normal-equation solution can be practical.
- ❌ B. Linear regression can be used as a component in classification-style decision rules, so the statement "only" is too strong in this course context.
- ❌ C. Ridge regression is still linear in the original feature space unless nonlinear features are introduced.
- ✅ D. Polynomial linear regression is linear in transformed features but nonlinear in the original input space.
- ✅ E. Ridge regression adds an $L_2$ penalty and can reduce overfitting.
- ❌ F. Polynomial features increase model complexity and are not the intended fix for an already overfitting linear regression model.

PPT statement: Closed-form linear regression is efficient for small $d$; ridge regression controls overfitting with regularization; polynomial feature mapping can produce nonlinear decision boundaries in the original space.

### 1.6

**Answer: A, C, D**

- ✅ A. Standard hard-margin linear SVM cannot handle non-linearly separable data directly.
- ❌ B. Slack variables allow margin violations for non-separable or overlapping samples, but they do not create a nonlinear decision boundary.
- ✅ C. For data that are non-linearly separable in the original space but separable after a feature mapping, kernel SVM can fit the training data by using a nonlinear decision boundary.
- ✅ D. The margin is defined as the closest distance from positive/negative training points to the decision boundary.

PPT statement: Slack variables address non-separable or overlapped data by allowing $y_i(w^\top x_i+b)\ge1-\xi_i$; for non-linearly separable data, SVM uses kernels by replacing inner products with $k(x_i,x_j)=\phi(x_i)^\top\phi(x_j)$.

### 1.7

**Answer: B, D**

- ❌ A. PCA is an orthogonal linear projection method; nonlinear PCA/kernel PCA are different variants.
- ✅ B. Dimensionality reduction can be done by unsupervised or supervised methods.
- ❌ C. The low-dimensional representation has dimension $k$, but the reconstruction is mapped back to the original $N$-dimensional space.
- ✅ D. PCA chooses the $k$ eigenvectors with the largest eigenvalues of the empirical covariance matrix.

PPT statement: PCA projects data linearly onto the subspace spanned by the top eigenvectors of the covariance matrix; reconstruction remains in the original feature space.

### 1.8

**Answer: C, D**

- ❌ A. Decision trees can handle categorical attributes as well as numerical attributes.
- ❌ B. A large minimum leaf size prevents further splitting and usually makes the tree shallower, not deeper.
- ✅ C. A small maximum depth restricts the tree and makes it shallow.
- ✅ D. Increasing the number of trees in bagging often improves or stabilizes performance by reducing variance.

PPT statement: Tree depth and minimum leaf size control tree complexity; bagging trains multiple trees and aggregates them to reduce variance.

### 1.9

**Answer: A, D**

- ✅ A. The option does not literally say "clustering", but it gives only unlabeled body-dimension data and asks us to determine clothing sizes. Under the course definition, this is naturally treated as unsupervised grouping/vector quantization: learn size groups from the structure of the measurements. If target size labels were provided, it could instead be supervised classification or regression, but the option does not provide such labels.
- ❌ B. Stock-market prediction is a supervised prediction/regression task if labels or future targets are used.
- ❌ C. Predicting disease from labeled medical records is supervised classification.
- ✅ D. Image compression can be treated as unsupervised learning, for example by clustering colors or learning low-dimensional representations.

PPT statement: Unsupervised learning uses unlabeled examples to reveal structure or transform data for a practical problem. Clustering assigns unlabeled examples into groups; supervised regression/classification would require paired labels or target values.

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

- ✅ A. Group A has 4 students after the first assignment.
- ❌ B. Group B has 6 students, not 5.
- ❌ C. K-means can converge to different results under different initial centroids.
- ❌ D. The new centroid of Group A is 61, not 60.

PPT statement: K-means alternates between nearest-centroid assignment and centroid recomputation; the result depends on initialization.

### 1.11

**Answer: A, D**

- ✅ A. ROC plots TPR on the y-axis against FPR on the x-axis.
- ❌ B. The identity is $TPR+FNR=1$, not $TPR+FPR=1$.
- ❌ C. Accuracy is generally $\frac{TP+TN}{TP+TN+FP+FN}$; it equals $\frac{TPR+TNR}{2}$ only when positive and negative classes are balanced.
- ✅ D. As the threshold increases, fewer samples are predicted positive, so FNR increases.

PPT statement: $TPR=TP/(TP+FN)$, $FNR=FN/(TP+FN)$, $FPR=FP/(FP+TN)$, and ROC is TPR versus FPR.

### 1.12

**Answer: B**

- ❌ A. The bias-variance tradeoff is discussed for generalization/test behavior, not as this exact training-set statement.
- ✅ B. As model complexity increases, bias usually decreases and variance usually increases on unseen/test data.
- ❌ C. Bias and variance do not both generally drop as complexity increases.
- ❌ D. This reverses the usual tradeoff.

PPT statement: Increasing model complexity typically decreases bias but increases variance; the test error is governed by their tradeoff.

### 1.13

**Answer: B, D**

- ❌ A. With $\Sigma=I$, EM for GMM becomes a soft version of K-means in the PPT, not the standard hard-assignment K-means algorithm.
- ✅ B. In the PPT's EM theorem, "improve" means the log-likelihood does not decrease after an EM iteration.
- ❌ C. Latent variables need not always be discrete; latent-variable models can have continuous latent variables.
- ✅ D. For a concave function, Jensen's inequality gives

$$
f(E[X])\ge E[f(X)].
$$

PPT statement: Fixing $\Sigma=I$ gives a soft version of K-means; EM alternates E-step and M-step to non-decrease the likelihood; Jensen's inequality for concave $f$ is $f(E[X])\ge E[f(X)]$.

### 1.14

**Answer: C, D**

- ❌ A. When $m=d$, a unique solution is guaranteed only if $X$ is invertible.
- ❌ B. When $m>d$, the system is over-determined, not under-determined.
- ✅ C. In the PPT's simplified terminology, $m>d$ is over-determined and has no exact solution.
- ✅ D. In the PPT's simplified terminology, $m<d$ is under-determined and has infinitely many solutions.
- ❌ E. When $m<d$, the system is under-determined, not over-determined.

PPT statement: Even-determined means $m=d$ and needs invertibility for a unique solution; over-determined means $m>d$; under-determined means $m<d$ and is described as having infinitely many solutions.

Strict math caveat: the existence/number of exact solutions also depends on rank and consistency. The selected answer follows the PPT's simplified exam convention.

### 1.15

**Answer: B, C**

- ❌ A. Ungraded assignments do not provide labels, so they are not supervised-learning experience for training the grader.
- ✅ B. The task is to predict the grades of student assignments.
- ✅ C. The accuracy of predicted grades can be used as the performance measure.
- ❌ D. The accuracy of students' answers is not the model's performance measure; the model is evaluated by how accurately it predicts grades.

PPT statement: In Tom Mitchell's definition, supervised learning needs experience $E$ with labels, a task $T$, and a performance measure $P$.

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

Use

$$
W_{out}=\left\lfloor\frac{W+2P-F}{S}\right\rfloor+1.
$$

For $Conv_1$:

$$
\begin{aligned}
W_1
&=\left\lfloor\frac{36+2\cdot1-8}{2}\right\rfloor+1\\
&=16.
\end{aligned}
$$

So

$$
Conv_1: 16\times 16\times 10,
\qquad
Relu_1: 16\times 16\times 10.
$$

After $MP_1$ with $2\times2$ pooling and stride $2$,

$$
\begin{aligned}
W_{pool}
&=\left\lfloor\frac{16-2}{2}\right\rfloor+1\\
&=8.
\end{aligned}
$$

$$
MP_1: 8\times 8\times 10.
$$

For $Conv_2$:

$$
\begin{aligned}
W_2
&=\left\lfloor\frac{8+2\cdot0-5}{1}\right\rfloor+1\\
&=4.
\end{aligned}
$$

So

$$
Conv_2: 4\times 4\times 100.
$$

### (2) Number of parameters

$$
\begin{aligned}
\#Conv_1&=10(8\cdot 8\cdot 3+1)=10(192+1)=1930,\\
\#Conv_2&=100(5\cdot 5\cdot 10+1)=100(250+1)=25100,\\
\#FC_1&=(4\cdot4\cdot100+1)\cdot 10=(1600+1)\cdot10=16010.
\end{aligned}
$$

Thus the total number of trainable parameters is

$$
\begin{aligned}
\#Total
&=1930+25100+16010\\
&=43040.
\end{aligned}
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

This subquestion asks for the symbolic solution of $w,b$. Since no concrete training data are given, the solution is expressed through the optimal dual variables $\alpha^*$. The dual problem below is the intermediate problem used to obtain $\alpha^*$; the requested $w,b$ are recovered after that.

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

Substituting the stationarity conditions into the Lagrangian gives the dual objective. Since
$w=\sum_i\alpha_i y_i x_i$ and $\sum_i\alpha_i y_i=0$,

$$
\begin{aligned}
L(w,b,\alpha)
&=\frac12\|w\|^2+\sum_{i=1}^n\alpha_i
-\sum_{i=1}^n\alpha_i y_i w^\top x_i
-b\sum_{i=1}^n\alpha_i y_i\\
&=\frac12\|w\|^2+\sum_{i=1}^n\alpha_i-w^\top w\\
&=\sum_{i=1}^n\alpha_i-\frac12\|w\|^2\\
&=\sum_{i=1}^n\alpha_i-\frac12\sum_{i=1}^n\sum_{j=1}^n
\alpha_i\alpha_j y_i y_j x_i^\top x_j.
\end{aligned}
$$

Therefore the dual problem is

$$
\begin{aligned}
\max_\alpha\quad
&\sum_{i=1}^n\alpha_i-\frac12\sum_{i=1}^n\sum_{j=1}^n
\alpha_i\alpha_j y_i y_j x_i^\top x_j\\
\text{s.t.}\quad
&\sum_{i=1}^n\alpha_i y_i=0,
\qquad
\alpha_i\ge 0.
\end{aligned}
$$

Let $\alpha^*$ be the optimizer of this dual problem. Then the requested weight vector is

$$
w^*=\sum_{i=1}^n\alpha_i^* y_i x_i.
$$

For any support vector $x_j$ with $\alpha_j^*>0$, complementary slackness gives

$$
y_j({w^*}^\top x_j+b^*)=1.
$$

Since $y_j\in\{-1,+1\}$, this gives

$$
b^*=y_j-{w^*}^\top x_j.
$$

If there is more than one support vector, $b$ can be computed from each support vector and averaged.

A training point is a **support vector** when

$$
\alpha_i^*>0.
$$

By complementary slackness, this implies

$$
y_i({w^*}^\top x_i+b^*)=1,
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

The pairwise contributions are

$$
\begin{aligned}
s^+=0.2 &: 1,\\
s^+=0.5 &: 1+1+0.5+0=2.5,\\
s^+=0.7 &: 4,\\
s^+=0.7 &: 4,\\
s^+=0.8 &: 4,\\
s^+=0.9 &: 4.
\end{aligned}
$$

Thus the total pair score is

$$
1+2.5+4+4+4+4=19.5.
$$

There are $6\times4=24$ pairs. Hence

$$
\begin{aligned}
AUC
&=\frac{19.5}{24}\\
&=0.8125=\frac{13}{16}.
\end{aligned}
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

The M-step maximizes the expected complete-data log-likelihood

$$
Q(\Theta)=\sum_{n=1}^N\sum_{k=1}^K\gamma_{nk}
\left[\log\pi_k+\log\mathcal N(x^{(n)}\mid\mu_k,\Sigma_k)\right].
$$

For one component $k$, the terms depending on $\mu_k$ and $\Sigma_k$ are

$$
\sum_{n=1}^N\gamma_{nk}
\left[
-\frac12\log|\Sigma_k|
-\frac12(x^{(n)}-\mu_k)^\top\Sigma_k^{-1}(x^{(n)}-\mu_k)
\right].
$$

Taking the derivative with respect to $\mu_k$ gives

$$
\begin{aligned}
\frac{\partial Q}{\partial\mu_k}
&=\Sigma_k^{-1}\sum_{n=1}^N\gamma_{nk}(x^{(n)}-\mu_k)=0\\
&\Longrightarrow
\mu_k^{new}=\frac{1}{N_k}\sum_{n=1}^N\gamma_{nk}x^{(n)}.
\end{aligned}
$$

To derive the covariance update, substitute $\mu_k^{new}$ and define the weighted scatter matrix

$$
S_k=\sum_{n=1}^N\gamma_{nk}
(x^{(n)}-\mu_k^{new})(x^{(n)}-\mu_k^{new})^\top.
$$

Use the precision matrix $\Lambda_k=\Sigma_k^{-1}$. Since
$\log|\Sigma_k|=-\log|\Lambda_k|$, the covariance-dependent part of $Q$ becomes

$$
Q_k(\Lambda_k)
=\frac{N_k}{2}\log|\Lambda_k|-\frac12\mathrm{Tr}(\Lambda_kS_k)+\text{const}.
$$

Taking the matrix derivative gives

$$
\begin{aligned}
\frac{\partial Q_k}{\partial\Lambda_k}
&=\frac{N_k}{2}\Lambda_k^{-T}-\frac12S_k^T=0.
\end{aligned}
$$

Because $\Lambda_k$ and $S_k$ are symmetric,

$$
N_k\Lambda_k^{-1}=S_k.
$$

Thus

$$
\Sigma_k^{new}=\Lambda_k^{-1}
=\frac{S_k}{N_k}
=\frac{1}{N_k}\sum_{n=1}^N\gamma_{nk}
(x^{(n)}-\mu_k^{new})(x^{(n)}-\mu_k^{new})^\top.
$$

For the mixing weights, maximize

$$
\sum_{k=1}^K N_k\log\pi_k+\lambda\left(\sum_{k=1}^K\pi_k-1\right).
$$

Then

$$
\frac{N_k}{\pi_k}+\lambda=0
\quad\Longrightarrow\quad
\pi_k^{new}=\frac{N_k}{N}.
$$

Therefore

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

- Square: $3+$, $1-$,

$$
H(Square)=-\frac34\log_2\frac34-\frac14\log_2\frac14\approx0.8113.
$$

- Circle: $1+$, $3-$,

$$
H(Circle)=-\frac14\log_2\frac14-\frac34\log_2\frac34\approx0.8113.
$$

So

$$
H(D\mid Shape)=0.8113,
\qquad
Gain(Shape)=0.1887.
$$

For **Size**:

- Big: $4+$, $2-$,

$$
H(Big)=-\frac46\log_2\frac46-\frac26\log_2\frac26\approx0.9183.
$$

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
