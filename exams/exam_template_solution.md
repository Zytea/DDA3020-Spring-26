# Exam Template - Solutions

## 1 Single-choice questions

### 1.1

**Answer: C (official/intended single-choice)**

- ⚠️ A. Workspace materials explicitly support learned data/image compression as unsupervised learning: L14 lists data compression as an unsupervised-learning practical problem, and the review notes say vector quantization can be used for image compression. Direct regression would require paired targets $(x_i,y_i)$, such as desired low-resolution images. In the official single-choice reading, this option is apparently interpreted narrowly as ordinary high-to-low resolution conversion, so it is not the selected answer; it should not be described as strictly false under the broader course materials.
- ❌ B. Predicting the stock market is a prediction task, usually supervised if target values are available.
- ✅ C. The option does not literally say "clustering", but it gives only unlabeled body-dimension data and asks us to determine clothing sizes. Under the course definition, this is naturally treated as unsupervised grouping/vector quantization. If target size labels were provided, it could instead be supervised classification or regression, but the option does not provide such labels.
- ❌ D. Predicting disease from labeled medical records is supervised classification.

PPT statement: Unsupervised learning uses unlabeled examples to reveal data structure or transform data for practical problems, including clustering, dimensionality reduction, autoencoders on unlabeled data, and data compression. The official single-choice answer is C; A is defensible only when interpreted as learned compression rather than deterministic resizing.

### 1.2

**Answer: B**

- ❌ A. Linear regression with nonlinear basis functions still learns a finite parameter vector.
- ✅ B. K-Nearest Neighbors is non-parametric because it stores and uses training samples directly instead of learning a fixed-size parameter vector.
- ❌ C. Linear SVM learns parameters such as $w,b$.
- ❌ D. A neural network learns weights and biases, so it is parametric.

PPT statement: KNN is non-parametric; linear models, SVMs, and neural networks are parametric models.

### 1.3

**Answer: C**

- ❌ A. A neural network with one nonlinear hidden layer can have universal approximation ability under standard conditions.
- ❌ B. The output size can be larger than the input size.
- ✅ C. Without nonlinear activation functions, a composition of affine/linear layers is still an affine/linear model.
- ❌ D. CNNs can contain fully connected layers.

PPT statement: Nonlinear activations make multilayer networks nonlinear; CNNs may include convolutional, pooling, and fully connected layers.

### 1.4

**Answer: C**

The predictions are

$$
8,7,11,6,5,14,10,9,11,6.
$$

Their average is

$$
\bar h=\frac{87}{10}=8.7.
$$

The MSE is

$$
\begin{aligned}
\mathrm{MSE}
&=\frac{1}{10}\sum_{i=1}^{10}(h_i-y)^2 \\
&=\frac{1+4+4+9+16+25+1+0+4+9}{10} \\
&=7.3.
\end{aligned}
$$

The squared bias is

$$
(\bar h-t)^2=(8.7-8.5)^2=0.04.
$$

The empirical variance is

$$
\begin{aligned}
\mathrm{Var}
&=\frac{1}{10}\sum_{i=1}^{10}(h_i-\bar h)^2 \\
&=\frac{0.49+2.89+5.29+7.29+13.69+28.09+1.69+0.09+5.29+7.29}{10}\\
&=7.21\approx 7.2,
\end{aligned}
$$

not $7.1$. Therefore C is the incorrect statement.

- ✅ A. The empirical MSE is $7.3$.
- ✅ B. The squared bias is $(8.7-8.5)^2=0.04$.
- ❌ C. The empirical variance is $7.21\approx7.2$, not $7.1$.

PPT statement: Use empirical mean prediction $\bar h$, squared bias $(\bar h-t)^2$, and empirical variance $\frac1M\sum_m(h_m-\bar h)^2$.

## 2 Calculations and Derivations

## 2.1 CNN convolution and pooling

### (1) Feature-map size

For one filter, the output spatial size is

$$
\left(\frac{w-a+2v}{u}+1\right)\times
\left(\frac{h-b+2v}{u}+1\right).
$$

With one filter, the output depth is $1$. With $k$ filters, the output volume is

$$
\left(\frac{w-a+2v}{u}+1\right)\times
\left(\frac{h-b+2v}{u}+1\right)\times k.
$$

### (2) Number of parameters

Each filter has $abc$ weights and one bias. With $k$ filters, the number of parameters is

$$
k(abc+1).
$$

### (3) Concrete convolution outputs

Use the usual CNN cross-correlation convention, i.e., do not flip the filter.

For

$$
I=
\begin{bmatrix}
1&0&1&0\\
1&0&1&0\\
1&1&1&1\\
0&0&1&0
\end{bmatrix},
$$

the valid $3\times 3$ convolution output has size $2\times 2$.

For $F_1$,

$$
F_1=
\begin{bmatrix}
-\frac12&-\frac12&-\frac12\\
0&1&0\\
1&0&1
\end{bmatrix},
$$

For example, the top-left entry is

$$
1\left(-\frac12\right)+0\left(-\frac12\right)+1\left(-\frac12\right)
+1(0)+0(1)+1(0)+1(1)+1(0)+1(1)=1.
$$

Computing the four valid windows gives

$$
\begin{aligned}
O_1
&=
\begin{bmatrix}
1&\frac52\\
1&\frac12
\end{bmatrix}.
\end{aligned}
$$

For $F_2$,

$$
F_2=
\begin{bmatrix}
1&0&-1\\
1&0&-1\\
1&0&-1
\end{bmatrix},
$$

For $F_2$, computing the four valid windows gives

$$
O_2=
\begin{bmatrix}
0&0\\
-1&0
\end{bmatrix}.
$$

If we apply $2\times 2$ max-pooling to each output map, then

$$
\operatorname{MaxPool}(O_1)=\frac52,
\qquad
\operatorname{MaxPool}(O_2)=0.
$$

## 2.2 Gaussian mixture model and EM

Let

$$
p(x)=\sum_{k=1}^K \pi_k\mathcal N(x\mid \mu_k,\Sigma_k),
\qquad
\sum_{k=1}^K\pi_k=1,
\qquad
\pi_k\ge 0.
$$

### (1) Log-likelihood

For $D=\{x^{(1)},\ldots,x^{(N)}\}$,

$$
\ln p(D;\Theta)
=\sum_{n=1}^N\ln p(x^{(n)};\Theta)
=\sum_{n=1}^N\ln\left(\sum_{k=1}^K\pi_k\mathcal N(x^{(n)}\mid\mu_k,\Sigma_k)\right).
$$

### (2) Likelihood decomposition

For any auxiliary distribution

$$
q(z)=\prod_{n=1}^N q_n(z^{(n)}),
$$

we have

$$
\begin{aligned}
\ln p(D;\Theta)
&=\sum_{n=1}^N\ln p(x^{(n)};\Theta)\\
&=\sum_{n=1}^N\ln\sum_{z^{(n)}}p(x^{(n)},z^{(n)};\Theta)\\
&=\sum_{n=1}^N\ln\sum_{z^{(n)}}q_n(z^{(n)})
\frac{p(x^{(n)},z^{(n)};\Theta)}{q_n(z^{(n)})}\\
&\ge
\sum_{n=1}^N\sum_{z^{(n)}}q_n(z^{(n)})
\ln\frac{p(x^{(n)},z^{(n)};\Theta)}{q_n(z^{(n)})}\\
&=L(q;\Theta),
\end{aligned}
$$

where Jensen's inequality is used because $\ln(\cdot)$ is concave.

The gap is

$$
\begin{aligned}
\ln p(D;\Theta)-L(q;\Theta)
&=\sum_{n=1}^N
\mathrm{KL}\left(q_n(z^{(n)})\,\|\,p(z^{(n)}\mid x^{(n)};\Theta)\right)\\
&\ge 0.
\end{aligned}
$$

The gap is zero iff

$$
q_n(z^{(n)})=p(z^{(n)}\mid x^{(n)};\Theta).
$$

### (3) E-step

Given $\Theta=\{\pi,\mu,\Sigma\}$, set

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

### (4) M-step

Let

$$
N_k=\sum_{n=1}^N\gamma_{nk}.
$$

The expected complete-data log-likelihood is

$$
Q(\Theta)=\sum_{n=1}^N\sum_{k=1}^K\gamma_{nk}
\left[\log\pi_k+\log\mathcal N(x^{(n)}\mid\mu_k,\Sigma_k)\right].
$$

For one component $k$, keep the terms involving $\mu_k$ and $\Sigma_k$:

$$
\sum_{n=1}^N\gamma_{nk}
\left[
-\frac12\log|\Sigma_k|
-\frac12(x^{(n)}-\mu_k)^\top\Sigma_k^{-1}(x^{(n)}-\mu_k)
\right].
$$

The derivative with respect to $\mu_k$ is

$$
\begin{aligned}
\frac{\partial Q}{\partial\mu_k}
&=\Sigma_k^{-1}\sum_{n=1}^N\gamma_{nk}(x^{(n)}-\mu_k)=0\\
&\Longrightarrow
\mu_k^{new}=\frac{1}{N_k}\sum_{n=1}^N\gamma_{nk}x^{(n)}.
\end{aligned}
$$

To derive the covariance update, substitute $\mu_k^{new}$ and define

$$
S_k=\sum_{n=1}^N\gamma_{nk}
(x^{(n)}-\mu_k^{new})(x^{(n)}-\mu_k^{new})^\top.
$$

Let $\Lambda_k=\Sigma_k^{-1}$. Since $\log|\Sigma_k|=-\log|\Lambda_k|$, the covariance-dependent part is

$$
Q_k(\Lambda_k)
=\frac{N_k}{2}\log|\Lambda_k|-\frac12\mathrm{Tr}(\Lambda_kS_k)+\text{const}.
$$

Then

$$
\frac{\partial Q_k}{\partial\Lambda_k}
=\frac{N_k}{2}\Lambda_k^{-T}-\frac12S_k^T=0.
$$

Since $\Lambda_k$ and $S_k$ are symmetric,

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

For the mixing coefficients, maximize

$$
\sum_{k=1}^K N_k\log\pi_k+\lambda\left(\sum_{k=1}^K\pi_k-1\right).
$$

This gives

$$
\frac{N_k}{\pi_k}+\lambda=0
\quad\Longrightarrow\quad
\pi_k^{new}=\frac{N_k}{N}.
$$

Therefore,

$$
\mu_k^{new}=\frac{1}{N_k}\sum_{n=1}^N\gamma_{nk}x^{(n)},
$$

$$
\Sigma_k^{new}
=\frac{1}{N_k}\sum_{n=1}^N\gamma_{nk}
(x^{(n)}-\mu_k^{new})(x^{(n)}-\mu_k^{new})^\top,
$$

and

$$
\pi_k^{new}=\frac{N_k}{N}.
$$

### (5) Monotonicity of EM

At iteration $t$, the E-step chooses

$$
q^{t+1}(z)=p(z\mid D;\Theta^t),
$$

so the lower bound is tight:

$$
L(q^{t+1};\Theta^t)=\ln p(D;\Theta^t).
$$

The M-step chooses

$$
\Theta^{t+1}=\arg\max_\Theta L(q^{t+1};\Theta),
$$

so

$$
L(q^{t+1};\Theta^{t+1})\ge L(q^{t+1};\Theta^t).
$$

Since the log-likelihood is always at least its lower bound,

$$
\begin{aligned}
\ln p(D;\Theta^{t+1})
&\ge L(q^{t+1};\Theta^{t+1})\\
&\ge L(q^{t+1};\Theta^t)\\
&=\ln p(D;\Theta^t).
\end{aligned}
$$

Therefore,

$$
\ln p(D;\Theta^{t+1})\ge \ln p(D;\Theta^t).
$$
