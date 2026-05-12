# DDA3020 Machine Learning Final Exam 2022.12 - Solutions

## 1 Single-choice questions

### 1.1

**Answer: C**

- ✅ A. Computer Vision is a sub-area/application area of AI.
- ✅ B. Machine Learning is a core sub-area of AI.
- ❌ C. Optimization is not listed in the PPT as a sub-area of AI; it is a mathematical tool used to train models.
- ✅ D. Natural Language Processing is a sub-area of AI.

PPT statement: AI includes areas such as machine learning, computer vision, natural language processing, and robotics; optimization is a tool used inside learning algorithms.

### 1.2

**Answer: B**

- ✅ A. Gaussian distribution is continuous.
- ❌ B. Binomial distribution is discrete, so it is not a continuous distribution.
- ✅ C. Laplace distribution is continuous.
- ✅ D. Beta distribution is continuous.

PPT statement: Binomial is a discrete distribution; Gaussian, Laplace, and Beta are continuous distributions.

### 1.3

**Answer: C (official/intended single-choice)**

- ⚠️ A. Workspace materials explicitly support learned data/image compression as unsupervised learning: L14 lists data compression as an unsupervised-learning practical problem, and the review notes say vector quantization can be used for image compression. Direct regression would require paired targets $(x_i,y_i)$, such as desired low-resolution images. In the official single-choice reading, this option is apparently interpreted narrowly as ordinary high-to-low resolution conversion, so it is not the selected answer; it should not be described as strictly false under the broader course materials.
- ❌ B. Predicting the stock market is a prediction task, usually supervised if target values are available.
- ✅ C. The option does not literally say "clustering", but it gives only unlabeled body-dimension data and asks us to determine clothing sizes. Under the course definition, this is naturally treated as unsupervised grouping/vector quantization. If target size labels were provided, it could instead be supervised classification or regression, but the option does not provide such labels.
- ❌ D. Predicting disease from labeled medical records is supervised classification.

PPT statement: Unsupervised learning uses unlabeled examples to reveal data structure or transform data for practical problems, including clustering, dimensionality reduction, autoencoders on unlabeled data, and data compression. The official single-choice answer is C; A is defensible only when interpreted as learned compression rather than deterministic resizing.

### 1.4

**Answer: B**

- ❌ A. Linear regression with nonlinear basis functions still learns a finite parameter vector, so it is parametric.
- ✅ B. K-Nearest Neighbors stores training samples and has no fixed-size learned parameter vector, so it is non-parametric.
- ❌ C. Linear SVM learns parameters such as $w,b$.
- ❌ D. A neural network learns many parameters/weights.

PPT statement: KNN is a non-parametric method; linear models, SVMs, and neural networks are parametric models.

### 1.5

**Answer: D**

- ✅ A. In the usual bias-variance tradeoff, increasing bias is associated with decreasing variance.
- ✅ B. A more complex model often has smaller bias.
- ✅ C. A more complex model often has larger variance.
- ❌ D. The noise term is irreducible error from the data-generating process; model choice does not reduce it.

PPT statement: Expected error decomposes into bias, variance, and irreducible noise; only bias and variance are controlled by model complexity.

### 1.6

**Answer: B**

- ❌ A. Decreasing the number of training samples usually increases overfitting risk.
- ✅ B. Adding regularization terms penalizes overly complex parameters and can reduce overfitting.
- ❌ C. A more complex hypothesis model usually increases overfitting risk.
- ❌ D. Switching to a second-order optimizer changes optimization, not the statistical overfitting control.

PPT statement: Overfitting is reduced by regularization, more training data, or simpler models; optimizers do not directly reduce model variance.

### 1.7

**Answer: C**

- ❌ A. A neural network with one nonlinear hidden layer can have universal approximation ability under standard conditions.
- ❌ B. The output dimension can be larger than the input dimension.
- ✅ C. Without nonlinear activations, a composition of affine/linear layers is still affine/linear.
- ❌ D. CNNs can contain fully connected layers after convolution/pooling layers.

PPT statement: Nonlinear activation functions are what make multilayer neural networks nonlinear; CNN architectures may include convolutional, pooling, and fully connected layers.

### 1.8

**Answer: D**

- ❌ A. Decision trees can handle categorical attributes.
- ❌ B. Decision trees are nonlinear, piecewise-constant models, not linear models.
- ❌ C. A large minimum leaf size restricts splitting and makes the tree shallower.
- ✅ D. A small maximum depth restricts the tree and makes it shallow.

PPT statement: Tree complexity is controlled by maximum depth and minimum leaf size; smaller depth produces a shallower tree.

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

- ❌ A. Group A has 2 students, not 4.
- ❌ B. Group B has 8 students, not 5.
- ❌ C. K-means can change when the initial centroids change.
- ✅ D. With the given assignments, the new centroid of Group A is 55.

PPT statement: K-means alternates between nearest-centroid assignment and centroid update; the solution depends on initialization.

### 1.10

**Answer: B**

- ✅ A. PCA is an unsupervised method.
- ❌ B. PCA is an orthogonal linear projection method, not a nonlinear projection method.
- ✅ C. PCA chooses the $k$ eigenvectors with the largest eigenvalues.
- ✅ D. PCA can reduce noise by keeping leading principal components and discarding low-variance components.

PPT statement: PCA is a linear dimensionality-reduction method based on the top eigenvectors of the empirical covariance matrix.

### 1.11

**Answer: D**

- ✅ A. GMM models component means and variances/covariances.
- ✅ B. GMM parameters can be estimated by the EM algorithm.
- ✅ C. GMM is a soft clustering method because it assigns posterior responsibilities.
- ❌ D. GMM is usually more computationally expensive than K-means, not lower, because it estimates responsibilities and Gaussian parameters.

PPT statement: K-means uses hard assignments and centroids; GMM-EM uses soft responsibilities plus mean, covariance, and mixing-weight updates.

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

- ✅ A. Precision is $\frac{TP}{TP+FP}=\frac{5}{7}$.
- ❌ B. Recall is $\frac{TP}{TP+FN}=\frac{5}{6}$, not $\frac46$.
- ✅ C. Accuracy is $\frac{TP+TN}{TP+TN+FP+FN}=\frac{7}{10}$.
- ✅ D. Higher AUC means the classifier ranks positives above negatives more often and is better overall.

PPT statement: Precision, recall, accuracy, and AUC are defined by the confusion matrix and ranking scores; AUC ranges from 0.5 for random ranking to 1 for perfect ranking.

### 1.13

**Answer: C**

- ❌ A. When $m=d$, a unique solution is guaranteed only if $X$ is invertible; it is not guaranteed when $X^\top X$ is not invertible.
- ❌ B. When $m>d$, the system is over-determined, not under-determined.
- ✅ C. In the PPT's simplified terminology, $m>d$ is over-determined and has no exact solution.
- ❌ D. When $m<d$, the system is under-determined and is described in the PPT as having infinitely many solutions, not no solution.

PPT statement: Even-determined needs invertibility for a unique solution; over-determined means $m>d$; under-determined means $m<d$ with infinitely many solutions under the course wording.

Strict math caveat: the existence/number of exact solutions also depends on rank and consistency. The selected answer follows the PPT's simplified exam convention.

### 1.14

**Answer: D**

- ❌ A. Logistic regression does not maximize the margin; margin maximization is the SVM objective.
- ❌ B. Softmax regression has a linear decision boundary in the feature space.
- ❌ C. The sigmoid derivative is $g'(a)=g(a)(1-g(a))$, not $g(a)(1+g(a))$.
- ✅ D. Both logistic regression and softmax regression are usually optimized iteratively and have no closed-form solution.

PPT statement: Logistic/softmax regression minimize cross-entropy by iterative optimization; SVM is the large-margin classifier; sigmoid derivative is $g(a)(1-g(a))$.

### 1.15

**Answer: C**

The predictions have mean

$$
\bar h=8.7.
$$

The MSE is

$$
\begin{aligned}
\mathrm{MSE}
&=\frac{1}{10}\sum_{i=1}^{10}(h_i-9)^2\\
&=\frac{1+4+4+9+16+25+1+0+4+9}{10}\\
&=7.3.
\end{aligned}
$$

The squared bias is

$$
(8.7-8.5)^2=0.04.
$$

The empirical variance is

$$
\begin{aligned}
\frac{1}{10}\sum_{i=1}^{10}(h_i-8.7)^2
&=\frac{0.49+2.89+5.29+7.29+13.69+28.09+1.69+0.09+5.29+7.29}{10}\\
&=7.21\approx7.2,
\end{aligned}
$$

not $7.1$.

- ✅ A. The empirical MSE is $7.3$.
- ✅ B. The squared bias is $(8.7-8.5)^2=0.04$.
- ❌ C. The empirical variance is $7.21\approx7.2$, not $7.1$.

PPT statement: For the bias-variance calculation, use empirical mean prediction $\bar h$, squared bias $(\bar h-t)^2$, and empirical variance $\frac1M\sum_m(h_m-\bar h)^2$.

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

Use

$$
W_{out}=\left\lfloor\frac{W+2P-F}{S}\right\rfloor+1.
$$

For $Conv_1$:

$$
\begin{aligned}
W_1
&=\left\lfloor\frac{28+2\cdot1-6}{2}\right\rfloor+1\\
&=13.
\end{aligned}
$$

Therefore,

$$
Relu_1: 13\times13\times10.
$$

For $MP_1$, using the common floor convention for pooling,

$$
\begin{aligned}
W_{pool}
&=\left\lfloor\frac{13-2}{2}\right\rfloor+1\\
&=6,
\end{aligned}
$$

so

$$
MP_1: 6\times6\times10.
$$

For $Conv_2$:

$$
\begin{aligned}
W_2
&=\left\lfloor\frac{6+2\cdot0-5}{1}\right\rfloor+1\\
&=2,
\end{aligned}
$$

so

$$
Conv_2: 2\times2\times50.
$$

### (2) Number of parameters

$$
\begin{aligned}
\#Conv_1&=10(6\cdot6\cdot3+1)=10(108+1)=1090,\\
\#Conv_2&=50(5\cdot5\cdot10+1)=50(250+1)=12550,\\
\#FC_1&=(2\cdot2\cdot50+1)\cdot10=(200+1)\cdot10=2010.
\end{aligned}
$$

Total number of trainable parameters:

$$
\begin{aligned}
\#Total
&=1090+12550+2010\\
&=15650.
\end{aligned}
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

This subquestion asks for the symbolic solution of $w,b$. Since no concrete training data are given, the solution is expressed through the optimal dual variables $\alpha^*$. The dual problem below is the intermediate problem used to obtain $\alpha^*$; the requested $w,b$ are recovered after that.

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

Substitute the stationarity conditions into the Lagrangian. Since
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
\alpha_i\ge0.
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

Since $y_j\in\{-1,+1\}$,

$$
b^*=y_j-{w^*}^\top x_j.
$$

If there are several support vectors, compute this value for each one and average it.

A sample is a support vector when

$$
\alpha_i^*>0.
$$

By complementary slackness, such a point satisfies

$$
y_i({w^*}^\top x_i+b^*)=1.
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

The M-step maximizes

$$
Q(\Theta)=\sum_{n=1}^N\sum_{k=1}^K\gamma_{nk}
\left[\log\pi_k+\log\mathcal N(x^{(n)}\mid\mu_k,\Sigma_k)\right].
$$

For one component $k$, the relevant Gaussian part is

$$
\sum_{n=1}^N\gamma_{nk}
\left[
-\frac12\log|\Sigma_k|
-\frac12(x^{(n)}-\mu_k)^\top\Sigma_k^{-1}(x^{(n)}-\mu_k)
\right].
$$

Differentiate with respect to $\mu_k$:

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
$\log|\Sigma_k|=-\log|\Lambda_k|$, the covariance-dependent part is

$$
Q_k(\Lambda_k)
=\frac{N_k}{2}\log|\Lambda_k|-\frac12\mathrm{Tr}(\Lambda_kS_k)+\text{const}.
$$

Taking the matrix derivative,

$$
\frac{\partial Q_k}{\partial\Lambda_k}
=\frac{N_k}{2}\Lambda_k^{-T}-\frac12S_k^T=0.
$$

Because $\Lambda_k$ and $S_k$ are symmetric,

$$
N_k\Lambda_k^{-1}=S_k.
$$

Therefore

$$
\Sigma_k^{new}=\Lambda_k^{-1}
=\frac{S_k}{N_k}
=\frac{1}{N_k}\sum_{n=1}^N\gamma_{nk}
(x^{(n)}-\mu_k^{new})(x^{(n)}-\mu_k^{new})^\top.
$$

For $\pi_k$, use a Lagrange multiplier:

$$
\sum_{k=1}^K N_k\log\pi_k+\lambda\left(\sum_{k=1}^K\pi_k-1\right).
$$

Thus

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

## 2.5 Decision tree with entropy

There are $4$ positive and $4$ negative samples, so

$$
H(D)=1.
$$

### (1) Best root attribute

For **Color**:

- Red: $3+$, $0-$, entropy $0$.
- Blue: $1+$, $2-$,

$$
H(Blue)=-\frac13\log_2\frac13-\frac23\log_2\frac23\approx0.9183.
$$

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

- Square: $3+$, $1-$,

$$
H(Square)=-\frac34\log_2\frac34-\frac14\log_2\frac14\approx0.8113.
$$

- Circle: $1+$, $2-$,

$$
H(Circle)=-\frac13\log_2\frac13-\frac23\log_2\frac23\approx0.9183.
$$

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

- Big: $3+$, $2-$,

$$
H(Big)=-\frac35\log_2\frac35-\frac25\log_2\frac25\approx0.9710.
$$

- Small: $1+$, $2-$,

$$
H(Small)=-\frac13\log_2\frac13-\frac23\log_2\frac23\approx0.9183.
$$

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

The mean of the reconstructed data is

$$
\begin{aligned}
\tilde\mu
&=\frac1N\sum_{n=1}^N\tilde x^{(n)}\\
&=\mu+UU^\top\left(\frac1N\sum_{n=1}^N(x^{(n)}-\mu)\right)\\
&=\mu.
\end{aligned}
$$

Since $\tilde\mu=\mu$, the maximum-variance objective in the paper is

$$
J_{\mathrm{var}}(U)=\frac1N\sum_{n=1}^N\|\tilde x^{(n)}-\mu\|^2.
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

The left-hand side is independent of $U$. Therefore, maximizing reconstructed-data variance

$$
\frac1N\sum_{n=1}^N\|\tilde x^{(n)}-\mu\|^2
$$

is equivalent to minimizing reconstruction error

$$
\frac1N\sum_{n=1}^N\|x^{(n)}-\tilde x^{(n)}\|^2.
$$

To obtain $U$, write the same variance objective in trace form:

$$
\begin{aligned}
J_{\mathrm{var}}(U)
&=\frac1N\sum_{n=1}^N\|\tilde x^{(n)}-\mu\|^2\\
&=\frac1N\sum_{n=1}^N
(x^{(n)}-\mu)^\top UU^\top(x^{(n)}-\mu)\\
&=\mathrm{Tr}(U^\top\Sigma U),
\end{aligned}
$$

where $\Sigma=\frac1N\sum_n(x^{(n)}-\mu)(x^{(n)}-\mu)^\top$. The reconstruction error is

$$
\begin{aligned}
J_{\mathrm{rec}}(U)
&=\frac1N\sum_{n=1}^N\|(I-UU^\top)(x^{(n)}-\mu)\|^2\\
&=\mathrm{Tr}(\Sigma)-\mathrm{Tr}(U^\top\Sigma U).
\end{aligned}
$$

Now form the empirical covariance matrix

$$
\Sigma=\frac1N\sum_{n=1}^N(x^{(n)}-\mu)(x^{(n)}-\mu)^\top.
$$

To see why eigenvectors are obtained, first consider one direction $u$ with $u^\top u=1$ and maximize $u^\top\Sigma u$. The Lagrangian is

$$
\mathcal L(u,\lambda)=u^\top\Sigma u-\lambda(u^\top u-1).
$$

Stationarity gives

$$
\begin{aligned}
\frac{\partial\mathcal L}{\partial u}
&=2\Sigma u-2\lambda u=0\\
&\Longrightarrow \Sigma u=\lambda u.
\end{aligned}
$$

Thus we solve the eigenvalue problem

$$
\Sigma q_i=\lambda_i q_i,
\qquad
\lambda_1\ge\lambda_2\ge\cdots\ge\lambda_D.
$$

The PCA projection matrix is formed by the top $K$ eigenvectors:

$$
U=[q_1,q_2,\ldots,q_K].
$$
