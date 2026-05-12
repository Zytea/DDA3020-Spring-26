# DDA3020 考试向数学短板补齐手册

> 目标：把“传统线性代数课只讲矩阵、秩、特征值、SVD”的基础，补到 DDA3020 作业和期末所需的水平。重点不是再背一遍模型，而是补齐考试里最容易卡住的 **矩阵求导、概率到损失函数的转换、约束优化/KKT、隐变量/EM、投影/PCA、维度与参数量计算、backprop 链式法则**。
>
> 使用建议：先读第 0–4 章，它们是所有推导题的公共工具；再读第 5–13 章的模型推导；最后用第 14–16 章的题型模板检查自己是否会写英文答案。



## 资料覆盖说明

本手册按你上传的材料进行整合：

- `review_1.md`：概率、信息论、线性代数、优化、KKT。
- `review_2.md`：线性回归、logistic regression、SVM、decision tree/random forest、neural networks。
- `review_3.md`：CNN、RNN/Transformer、bias-variance、performance evaluation、K-means、GMM/EM、PCA。
- `hw1.pdf` / `hw1_solution.pdf`：weighted linear regression、KKT、KL、logistic gradient/Hessian/softmax、gradient descent from scratch。
- `hw2.pdf` / `hw2_solution.pdf`：decision tree entropy、NN 线性激活判断、LSTM/Transformer 概念、CNN shape/parameter、sklearn/PyTorch 编程要求。
- `hw3_solution.pdf`：mixture of Bernoullis EM、K-means/PCA 手算、PCA/eigenfaces、K-means implementation。
- `DDA3020_L2&3` 到 `DDA3020_L17` 练习：各章公式、选择题、计算题、推导题。
- `2020.pdf`、`2022.pdf`、`exam-template.pdf`：期末题型风格，尤其是 MC 判断、CNN 参数量、SVM/EM/PCA/decision tree 推导。

---

## 0. 你现在缺的不是“线代”，而是这些考试技能

你截图里的线性代数课覆盖了矩阵乘法、线性系统、LU、向量空间、秩、正交、QR、二次型、特征值、谱分解、SVD。这些是基础，但 DDA3020 的作业和考试还要求你能把它们用于机器学习推导。

| 老线代课通常会讲 | DDA3020 额外要求你会 |
|---|---|
| 矩阵乘法、逆、秩 | 把逐样本求和写成矩阵形式，例如 $\sum_i \alpha_i\|y_i-Wx_i-b\|^2 \to \operatorname{tr}((Y-XB)^TA(Y-XB))$ |
| 正交、投影、SVD | PCA 的方差最大化/重构误差最小化、投影残差正交、特征值解释方差 |
| 二次型 | 证明 Hessian PSD、证明 logistic regression convex、推 normal equation |
| 线性系统 | over-determined/under-determined、left/right pseudo-inverse、least squares |
| 行列式、特征值 | Gaussian log-likelihood、$\log |\Sigma|$、GMM covariance update |
| 基础求导 | 矩阵求导、trace trick、chain rule、softmax/backprop/BPTT |
| 无约束优化很少讲 | 梯度下降、学习率、Lagrangian、KKT、SVM dual、PCA Lagrange |
| 概率可能很少讲 | entropy、cross-entropy、KL、MLE/MAP、Bayes、EM/Jensen |

**DDA3020 的考试/作业核心动作**可以概括成 8 个动词：

1. **augment**：把 bias 吸收到输入向量中。
2. **vectorize**：把 $\sum_i$ 写成 $X,Y,A$ 的矩阵式。
3. **differentiate**：对向量/矩阵求导。
4. **set to zero**：令梯度为 0 得闭式解。
5. **prove PSD**：用 $v^THv\ge 0$ 证明凸性。
6. **constrain**：写 Lagrangian 和 KKT。
7. **log-transform**：把 likelihood 乘积变成 log-likelihood 求和。
8. **bound/latent**：对 log-sum 引入隐变量，用 Jensen/ELBO 做 EM。

---

## 1. 统一符号：先解决维度灾难

很多题做不出来不是因为不会算，而是因为不知道 $X$ 到底是 $N\times d$ 还是 $d\times N$。考试中先固定一个约定，然后所有推导都围绕维度检查。

### 1.1 行样本 convention：最常见于 sklearn/作业

- $N$：样本数。
- $d$：输入特征维度。
- $m$：输出维度。
- 单个输入：$x_i\in\mathbb{R}^d$，默认列向量。
- 单个输出：$y_i\in\mathbb{R}^m$，默认列向量。
- 设计矩阵：
  $$
  X=\begin{bmatrix}x_1^T\\ \vdots\\ x_N^T\end{bmatrix}\in\mathbb{R}^{N\times d}.
  $$
- 多输出目标矩阵：
  $$
  Y=\begin{bmatrix}y_1^T\\ \vdots\\ y_N^T\end{bmatrix}\in\mathbb{R}^{N\times m}.
  $$
- 如果模型写成 $\hat y_i=W x_i+b$，其中 $W\in\mathbb{R}^{m\times d}$，$b\in\mathbb{R}^m$，那么矩阵预测写作
  $$
  \hat Y=XW^T+\mathbf 1 b^T.
  $$
- 如果把权重写成 $B=W^T\in\mathbb{R}^{d\times m}$，则
  $$
  \hat Y=XB.
  $$

### 1.2 bias absorption / augmentation

把 bias 变成权重的一部分：

$$
\tilde x_i=\begin{bmatrix}x_i\\1\end{bmatrix}\in\mathbb{R}^{d+1},\qquad
\tilde W=\begin{bmatrix}W & b\end{bmatrix}\in\mathbb{R}^{m\times(d+1)}.
$$

于是

$$
\tilde W\tilde x_i=Wx_i+b.
$$

行样本设计矩阵变成

$$
\tilde X=\begin{bmatrix}\tilde x_1^T\\\vdots\\\tilde x_N^T\end{bmatrix}\in\mathbb{R}^{N\times(d+1)}.
$$

如果用 $B=\tilde W^T\in\mathbb{R}^{(d+1)\times m}$，预测为

$$
\hat Y=\tilde X B.
$$

**考试检查点**：

- $\tilde X B$ 的维度是 $N\times m$，必须和 $Y$ 一样。
- 如果你最后求出 $B^*$，原题要的是 $\tilde W^*$，则 $\tilde W^*=(B^*)^T$。

---

## 2. 矩阵求导核心工具箱

### 2.1 三种常用求导风格

考试中你只需要熟练两种：

1. **直接背公式**：快，但容易转置错。
2. **differential + trace trick**：最稳，适合矩阵变量。

标量函数 $J$ 对矩阵 $W$ 的梯度定义为

$$
dJ=\operatorname{tr}\left((\nabla_WJ)^T dW\right).
$$

对向量 $w$，定义为

$$
dJ=(\nabla_wJ)^Tdw.
$$

### 2.2 Trace trick 必背公式

1. 标量可写成 trace：
   $$
   a=\operatorname{tr}(a).
   $$

2. trace 循环不变：
   $$
   \operatorname{tr}(ABC)=\operatorname{tr}(BCA)=\operatorname{tr}(CAB).
   $$

3. Frobenius norm：
   $$
   \|M\|_F^2=\operatorname{tr}(M^TM).
   $$

4. differential product rule：
   $$
   d(AB)=dA\,B+A\,dB.
   $$

5. 逆矩阵求导：
   $$
   d(X^{-1})=-X^{-1}(dX)X^{-1}.
   $$

6. log-determinant：
   $$
   d\log|X|=\operatorname{tr}(X^{-1}dX)
   $$
   若 $X$ 对称正定，则 $\nabla_X\log|X|=X^{-1}$。

### 2.3 向量求导必背表

设 $x\in\mathbb{R}^d$，$A$ 与 $x$ 无关：

| 函数 | 梯度 |
|---|---|
| $a^Tx$ | $a$ |
| $x^Ta$ | $a$ |
| $x^TAx$ | $(A+A^T)x$ |
| $x^TAx$ 且 $A=A^T$ | $2Ax$ |
| $\|x\|_2^2=x^Tx$ | $2x$ |
| $\|Ax-b\|_2^2$ | $2A^T(Ax-b)$ |
| $\frac12\|Ax-b\|_2^2$ | $A^T(Ax-b)$ |

### 2.4 矩阵求导必背表

| 函数 | 梯度 |
|---|---|
| $\|Y-XB\|_F^2$ w.r.t. $B$ | $2X^T(XB-Y)$ |
| $\frac12\|Y-XB\|_F^2$ w.r.t. $B$ | $X^T(XB-Y)$ |
| $\operatorname{tr}((Y-XB)^TA(Y-XB))$ w.r.t. $B$, $A=A^T$ | $2X^TA(XB-Y)$ |
| $\|Y-WX\|_F^2$ w.r.t. $W$ | $2(WXX^T-YX^T)$ |
| $\operatorname{tr}(W^TAW)$, $A=A^T$ | $2AW$ |
| $\operatorname{tr}(W^TAWB)$ | $AWB+A^TWB^T$ |

### 2.5 最重要的 differential 示例：least squares

令

$$
J(B)=\|Y-XB\|_F^2=\operatorname{tr}((Y-XB)^T(Y-XB)).
$$

记 residual

$$
E=XB-Y.
$$

则

$$
J=\operatorname{tr}(E^TE),\qquad dE=X\,dB.
$$

所以

$$
\begin{aligned}
dJ
&=\operatorname{tr}(dE^TE+E^TdE)\\
&=2\operatorname{tr}(E^TdE)\\
&=2\operatorname{tr}(E^TXdB)\\
&=2\operatorname{tr}((X^TE)^TdB).
\end{aligned}
$$

因此

$$
\nabla_BJ=2X^T(XB-Y).
$$

**你以后看到 squared loss，都可以把它还原成这个模板。**

---

## 3. 从逐样本求和到矩阵闭式解：HW1 Q1 的完整套路

HW1 第一题是 weighted multiple-output linear regression：

$$
J(W,b)=\sum_{i=1}^N \alpha_i\|y_i-(Wx_i+b)\|^2,
\qquad \alpha_i>0.
$$

其中

- $x_i\in\mathbb{R}^d$；
- $y_i\in\mathbb{R}^m$；
- $W\in\mathbb{R}^{m\times d}$；
- $b\in\mathbb{R}^m$。

### 3.1 Step 1：augment bias

$$
\tilde x_i=\begin{bmatrix}x_i\\1\end{bmatrix},\qquad
\tilde W=\begin{bmatrix}W&b\end{bmatrix}.
$$

模型变成

$$
\hat y_i=\tilde W\tilde x_i.
$$

### 3.2 Step 2：写成行样本矩阵

令

$$
\tilde X=\begin{bmatrix}\tilde x_1^T\\\vdots\\\tilde x_N^T\end{bmatrix}\in\mathbb{R}^{N\times(d+1)},
\quad
Y=\begin{bmatrix}y_1^T\\\vdots\\y_N^T\end{bmatrix}\in\mathbb{R}^{N\times m},
\quad
A=\operatorname{diag}(\alpha_1,\ldots,\alpha_N).
$$

令

$$
B=\tilde W^T\in\mathbb{R}^{(d+1)\times m}.
$$

则预测矩阵是 $\tilde XB$，目标函数是

$$
J(B)=\operatorname{tr}\left((Y-\tilde X B)^T A(Y-\tilde X B)\right).
$$

这一步是 weighted sum 的本质：

$$
\sum_i \alpha_i\|e_i\|^2
=\operatorname{tr}(E^TAE),
\quad E=Y-\tilde X B.
$$

### 3.3 Step 3：对 $B$ 求导并令零

由于 $A=A^T$，根据上面公式：

$$
\nabla_BJ=2\tilde X^TA(\tilde X B-Y).
$$

令梯度为 0：

$$
\tilde X^TA\tilde XB=\tilde X^TAY.
$$

若 $\tilde X^TA\tilde X$ 可逆：

$$
B^*=(\tilde X^TA\tilde X)^{-1}\tilde X^TAY.
$$

回到题目里的 $\tilde W$：

$$
\boxed{\tilde W^*=Y^TA\tilde X(\tilde X^TA\tilde X)^{-1}}.
$$

然后

$$
W^*=\text{前 }d\text{ 列},\qquad b^*=\text{最后一列}.
$$

### 3.4 Step 4：gradient descent 形式

回到原始形式。定义

$$
e_i=Wx_i+b-y_i\in\mathbb{R}^m.
$$

则

$$
J(W,b)=\sum_i\alpha_i e_i^Te_i.
$$

梯度为

$$
\boxed{\nabla_WJ=2\sum_{i=1}^N\alpha_i e_i x_i^T},
\qquad
\boxed{\nabla_bJ=2\sum_{i=1}^N\alpha_i e_i}.
$$

更新：

$$
\boxed{W^{(t+1)}=W^{(t)}-2\eta\sum_i\alpha_i(W^{(t)}x_i+b^{(t)}-y_i)x_i^T}
$$

$$
\boxed{b^{(t+1)}=b^{(t)}-2\eta\sum_i\alpha_i(W^{(t)}x_i+b^{(t)}-y_i)}.
$$

### 3.5 英文答题模板

> We augment each input vector as $\tilde x_i=[x_i^T,1]^T$ and define $\tilde W=[W,b]$. Then the model becomes $\tilde W\tilde x_i$. Let $\tilde X$ be the design matrix, $Y$ be the target matrix, and $A=\operatorname{diag}(\alpha_i)$. The objective can be written as $J(B)=\operatorname{tr}((Y-\tilde XB)^TA(Y-\tilde XB))$, where $B=\tilde W^T$. Taking derivative and setting it to zero gives $\tilde X^TA\tilde XB=\tilde X^TAY$, hence $B^*=(\tilde X^TA\tilde X)^{-1}\tilde X^TAY$ and $\tilde W^*=Y^TA\tilde X(\tilde X^TA\tilde X)^{-1}$.

---

## 4. 概率、信息论、MLE/MAP：把概率题变成优化题

### 4.1 基础概率公式

#### Bayes rule

离散：

$$
P(Y=y\mid X=x)=\frac{P(X=x\mid Y=y)P(Y=y)}{\sum_{y'}P(X=x\mid Y=y')P(Y=y')}.
$$

连续：

$$
p(x\mid y)=\frac{p(y\mid x)p(x)}{p(y)},
\qquad
p(y)=\int p(y\mid x')p(x')dx'.
$$

#### Expectation / variance

$$
\mathbb{E}[X]=\sum_x xP(X=x),
\quad
\operatorname{Var}(X)=\mathbb{E}[X^2]-(\mathbb{E}[X])^2.
$$

### 4.2 常见分布

#### Bernoulli

$$
P(y\mid \mu)=\mu^y(1-\mu)^{1-y},\quad y\in\{0,1\}.
$$

用于 logistic regression：

$$
y_i\mid x_i,w\sim\operatorname{Bernoulli}(h_i),
\quad h_i=\sigma(w^Tx_i).
$$

#### Categorical

$$
P(y=k)=\pi_k,
\quad \sum_k\pi_k=1.
$$

用于 softmax / mixture component。

#### Gaussian

$$
\mathcal{N}(x\mid\mu,\Sigma)
=\frac{1}{\sqrt{(2\pi)^d|\Sigma|}}\exp\left(-\frac12(x-\mu)^T\Sigma^{-1}(x-\mu)\right).
$$

用于 linear regression noise / GMM。

#### Laplace

$$
p(e)\propto \exp\left(-\frac{|e|}{b}\right).
$$

用于 robust regression 的 $L_1$ loss。

### 4.3 信息论：entropy / cross-entropy / KL

Entropy：

$$
H(P)=-\sum_x P(x)\log P(x).
$$

Cross-entropy：

$$
H(P,Q)=-\sum_xP(x)\log Q(x).
$$

KL divergence：

$$
D_{KL}(P\|Q)=\sum_xP(x)\log\frac{P(x)}{Q(x)}.
$$

三者关系：

$$
\boxed{H(P,Q)=H(P)+D_{KL}(P\|Q)}.
$$

所以最小化 cross-entropy 等价于让模型分布 $Q$ 接近真实分布 $P$。由于 $H(P)$ 不依赖模型参数，优化时只剩 $D_{KL}$。

### 4.4 KL 非对称性证明套路

定义：

$$
D_{KL}(P\|Q)=\sum_xP(x)\log\frac{P(x)}{Q(x)},
\quad
D_{KL}(Q\|P)=\sum_xQ(x)\log\frac{Q(x)}{P(x)}.
$$

因为权重分别是 $P(x)$ 和 $Q(x)$，一般不相等。反例：

$$
P=(0.5,0.5),\quad Q=(0.9,0.1).
$$

则

$$
D_{KL}(P\|Q)=0.5\log\frac{0.5}{0.9}+0.5\log\frac{0.5}{0.1},
$$

$$
D_{KL}(Q\|P)=0.9\log\frac{0.9}{0.5}+0.1\log\frac{0.1}{0.5}.
$$

数值一般不同。

### 4.5 MLE 标准步骤

给定 dataset $D=\{(x_i,y_i)\}_{i=1}^N$，假设独立同分布：

$$
\mathcal{L}(\theta;D)=\prod_{i=1}^Np(y_i\mid x_i;\theta).
$$

取 log：

$$
\log\mathcal{L}(\theta;D)=\sum_i\log p(y_i\mid x_i;\theta).
$$

MLE：

$$
\theta_{MLE}=\arg\max_\theta\log\mathcal{L}(\theta;D).
$$

考试中常把最大化 log-likelihood 转成最小化 negative log-likelihood：

$$
\theta_{MLE}=\arg\min_\theta -\sum_i\log p(y_i\mid x_i;\theta).
$$

### 4.6 MAP 标准步骤

加入 prior：

$$
\theta_{MAP}=\arg\max_\theta\left[\sum_i\log p(y_i\mid x_i;\theta)+\log p(\theta)\right].
$$

常见对应关系：

| likelihood | prior | objective |
|---|---|---|
| Gaussian noise | uniform prior | least squares |
| Gaussian noise | Gaussian prior | Ridge / $L_2$ |
| Gaussian noise | Laplace prior | Lasso / $L_1$ |
| Laplace noise | uniform prior | robust $L_1$ regression |
| Bernoulli label | Gaussian prior | L2 logistic regression |

**一句话**：likelihood 决定 loss，prior 决定 regularizer。

---

## 5. 凸优化、梯度下降、Lagrangian、KKT

### 5.1 Convex set

集合 $C$ convex：

$$
\forall x,y\in C,\ \forall \theta\in[0,1],\quad \theta x+(1-\theta)y\in C.
$$

线性等式/不等式定义的半空间、多面体一般都是 convex。

### 5.2 Convex function

定义：

$$
f(\theta x+(1-\theta)y)\le \theta f(x)+(1-\theta)f(y).
$$

一阶条件：

$$
f(y)\ge f(x)+\nabla f(x)^T(y-x).
$$

二阶条件：如果 $f$ 二阶可导，

$$
f\text{ convex}\iff \nabla^2 f(x)\succeq0.
$$

### 5.3 PSD 的证明模板

矩阵 $H$ PSD 的定义：

$$
\forall v,\quad v^THv\ge0.
$$

如果 $H=X^TRX$ 且 $R$ 是对角非负矩阵，则

$$
v^THv=v^TX^TRXv=(Xv)^TR(Xv)=\sum_i r_i(x_i^Tv)^2\ge0.
$$

这个模板会直接用于 logistic regression Hessian。

### 5.4 Gradient descent

更新：

$$
x^{(t+1)}=x^{(t)}-\eta\nabla f(x^{(t)}).
$$

典型题 $f(x_1,x_2)=x_1^2+5x_2^2$：

$$
\nabla f=(2x_1,10x_2)^T.
$$

更新为

$$
x_1^{(t+1)}=(1-2\eta)x_1^{(t)},
\quad
x_2^{(t+1)}=(1-10\eta)x_2^{(t)}.
$$

当 $|1-10\eta|>1$，$x_2$ 方向会发散。这就是大 learning rate 发散的数学原因。

### 5.5 Lagrangian 标准形式

标准最小化问题：

$$
\min_x f(x)
\quad
\text{s.t. } h_i(x)\le0,
\quad \ell_j(x)=0.
$$

Lagrangian：

$$
L(x,u,v)=f(x)+\sum_i u_ih_i(x)+\sum_jv_j\ell_j(x),
\quad u_i\ge0.
$$

### 5.6 KKT 条件

1. Stationarity：
   $$
   \nabla_xL(x,u,v)=0.
   $$

2. Primal feasibility：
   $$
   h_i(x)\le0,
   \quad \ell_j(x)=0.
   $$

3. Dual feasibility：
   $$
   u_i\ge0.
   $$

4. Complementary slackness：
   $$
   u_i h_i(x)=0.
   $$

解释：

- constraint inactive：$h_i(x)<0\Rightarrow u_i=0$。
- multiplier positive：$u_i>0\Rightarrow h_i(x)=0$。

### 5.7 HW1 约束优化题模板

题型：

$$
\min_{x_1,x_2}(x_1-2)^2+(x_2-1)^2
$$

s.t.

$$
x_1+x_2\le1,
\quad x_1\ge0,
\quad x_2\ge0.
$$

先转成 $\le0$：

$$
g_1=x_1+x_2-1\le0,
\quad g_2=-x_1\le0,
\quad g_3=-x_2\le0.
$$

Lagrangian：

$$
L=(x_1-2)^2+(x_2-1)^2+u_1(x_1+x_2-1)-u_2x_1-u_3x_2.
$$

Stationarity：

$$
2(x_1-2)+u_1-u_2=0,
$$

$$
2(x_2-1)+u_1-u_3=0.
$$

几何直觉：无约束最优点是 $(2,1)$，违反 $x_1+x_2\le1$，所以解在边界。最终解：

$$
\boxed{x^*=(1,0)}.
$$

英文模板：

> The Hessian of the objective is $2I\succ0$, so the objective is strictly convex. All constraints are affine, hence the feasible region is convex. Therefore the problem is a convex optimization problem. The Lagrangian is ... The KKT conditions are stationarity, primal feasibility, dual feasibility and complementary slackness. Solving the KKT conditions gives $x^*=(1,0)$.

---

## 6. Linear Regression 全套推导

### 6.1 Standard linear regression

模型：

$$
\hat y=Xw.
$$

Loss：

$$
J(w)=\|Xw-y\|^2=(Xw-y)^T(Xw-y).
$$

展开：

$$
J=w^TX^TXw-2y^TXw+y^Ty.
$$

求导：

$$
\nabla_wJ=2X^TXw-2X^Ty.
$$

令零：

$$
X^TXw=X^Ty.
$$

闭式解：

$$
\boxed{w^*=(X^TX)^{-1}X^Ty}.
$$

### 6.2 Gradient descent

若用 $\frac12\|Xw-y\|^2$：

$$
\nabla_wJ=X^T(Xw-y).
$$

更新：

$$
w\leftarrow w-\eta X^T(Xw-y).
$$

### 6.3 多输出 linear regression

$Y\in\mathbb{R}^{N\times m}$，$B\in\mathbb{R}^{d\times m}$：

$$
J(B)=\|XB-Y\|_F^2.
$$

梯度：

$$
\nabla_BJ=2X^T(XB-Y).
$$

闭式解：

$$
\boxed{B^*=(X^TX)^{-1}X^TY}.
$$

如果原题模型写 $Wx$，则 $W^*=(B^*)^T$。

### 6.4 Ridge regression

目标：

$$
J(w)=\|Xw-y\|^2+\lambda\|w\|^2.
$$

梯度：

$$
\nabla_wJ=2X^T(Xw-y)+2\lambda w.
$$

闭式解：

$$
\boxed{w^*=(X^TX+\lambda I)^{-1}X^Ty}.
$$

若不惩罚 bias，用 $\hat I$，第一项对角线为 0：

$$
\boxed{w^*=(X^TX+\lambda \hat I)^{-1}X^Ty}.
$$

### 6.5 MLE 视角

假设

$$
y_i=w^Tx_i+\epsilon_i,
\quad \epsilon_i\sim\mathcal{N}(0,\sigma^2).
$$

则

$$
p(y_i\mid x_i,w)=\frac1{\sqrt{2\pi}\sigma}
\exp\left(-\frac{(y_i-w^Tx_i)^2}{2\sigma^2}\right).
$$

Log-likelihood：

$$
\log L(w)=\sum_i\left[-\log(\sqrt{2\pi}\sigma)-\frac{(y_i-w^Tx_i)^2}{2\sigma^2}\right].
$$

最大化 log-likelihood 等价于最小化

$$
\sum_i(y_i-w^Tx_i)^2.
$$

### 6.6 MAP 视角：Ridge

若

$$
w\sim\mathcal{N}(0,\tau^2I),
$$

则

$$
\log p(w)=-\frac1{2\tau^2}\|w\|^2+\text{const}.
$$

MAP：

$$
\arg\max_w\left[-\frac1{2\sigma^2}\sum_i(y_i-w^Tx_i)^2-\frac1{2\tau^2}\|w\|^2\right].
$$

等价于

$$
\arg\min_w\left[\sum_i(y_i-w^Tx_i)^2+\lambda\|w\|^2\right],
\quad \lambda=\frac{\sigma^2}{\tau^2}.
$$

---

## 7. Logistic Regression 与 Softmax：HW1 最高频推导

### 7.1 Sigmoid derivative

$$
\sigma(z)=\frac1{1+e^{-z}}.
$$

求导：

$$
\begin{aligned}
\sigma'(z)
&=\frac{e^{-z}}{(1+e^{-z})^2}\\
&=\frac1{1+e^{-z}}\left(1-\frac1{1+e^{-z}}\right)\\
&=\sigma(z)(1-\sigma(z)).
\end{aligned}
$$

### 7.2 Binary logistic regression

$$
h_i=\sigma(w^Tx_i).
$$

Negative log-likelihood / cross-entropy：

$$
J(w)=-\sum_{i=1}^n\left[y_i\log h_i+(1-y_i)\log(1-h_i)\right].
$$

### 7.3 Gradient 推导

对单样本：

$$
\ell_i=-y_i\log h_i-(1-y_i)\log(1-h_i).
$$

先求

$$
\frac{\partial \ell_i}{\partial h_i}
=-\frac{y_i}{h_i}+\frac{1-y_i}{1-h_i}.
$$

又

$$
\frac{\partial h_i}{\partial z_i}=h_i(1-h_i),
\quad z_i=w^Tx_i,
\quad \frac{\partial z_i}{\partial w}=x_i.
$$

连乘：

$$
\frac{\partial \ell_i}{\partial w}
=\left(-\frac{y_i}{h_i}+\frac{1-y_i}{1-h_i}\right)h_i(1-h_i)x_i
=(h_i-y_i)x_i.
$$

因此

$$
\boxed{\nabla_wJ=\sum_i(h_i-y_i)x_i=X^T(h-y)}.
$$

### 7.4 Hessian 与 convexity

梯度：

$$
\nabla_wJ=X^T(h-y).
$$

因为

$$
\frac{\partial h_i}{\partial w}=h_i(1-h_i)x_i,
$$

Hessian：

$$
\boxed{H=X^TRX},
\quad
R=\operatorname{diag}(h_i(1-h_i)).
$$

由于 $0<h_i(1-h_i)\le\frac14$，$R$ 对角非负。对任意 $v$：

$$
v^THv=v^TX^TRXv=(Xv)^TR(Xv)=\sum_i h_i(1-h_i)(x_i^Tv)^2\ge0.
$$

所以 $H\succeq0$，$J(w)$ convex。

### 7.5 L2-regularized logistic regression

目标：

$$
J_{reg}(w)=J(w)+\frac{\lambda}{2}w^Tw.
$$

梯度：

$$
\nabla_wJ_{reg}=X^T(h-y)+\lambda w.
$$

更新：

$$
\boxed{w^{(t+1)}=w^{(t)}-\eta\left[X^T(h^{(t)}-y)+\lambda w^{(t)}\right]}.
$$

也可写作 weight decay：

$$
w^{(t+1)}=(1-\eta\lambda)w^{(t)}-\eta X^T(h^{(t)}-y).
$$

若 bias 不正则化，把 $w_0$ 对应项设为 0。

### 7.6 Softmax regression gradient

对第 $i$ 个样本，第 $k$ 类概率：

$$
P_{ik}=\frac{e^{w_k^Tx_i}}{\sum_j e^{w_j^Tx_i}}.
$$

Cross-entropy：

$$
L=-\sum_i\sum_k y_{ik}\log P_{ik}.
$$

结论：

$$
\boxed{\nabla_{w_k}L=\sum_i(P_{ik}-y_{ik})x_i}.
$$

矩阵形式：若 $X\in\mathbb{R}^{n\times d}$，$P,Y\in\mathbb{R}^{n\times K}$，$W\in\mathbb{R}^{d\times K}$，则

$$
\boxed{\nabla_WL=X^T(P-Y)}.
$$

### 7.7 Softmax derivative 证明模板

设 $z_{ij}=w_j^Tx_i$，

$$
P_{ij}=\frac{e^{z_{ij}}}{\sum_l e^{z_{il}}}.
$$

有

$$
\frac{\partial P_{ij}}{\partial z_{ik}}=P_{ij}(\mathbf{1}_{j=k}-P_{ik}).
$$

对 loss：

$$
\frac{\partial L_i}{\partial z_{ik}}=P_{ik}-y_{ik}.
$$

再乘

$$
\frac{\partial z_{ik}}{\partial w_k}=x_i.
$$

得到上面的梯度。

### 7.8 Logistic regression 常见考点

| 问法 | 标准答案 |
|---|---|
| sigmoid output 是什么 | $P(y=1\mid x;w)$ |
| decision boundary | $w^Tx+b=0$，因为 $\sigma(z)=0.5\iff z=0$ |
| 如何得到非线性边界 | feature transform $\phi(x)$，边界 $w^T\phi(x)=0$ |
| 为什么不用 MSE | sigmoid + MSE 通常非凸，cross-entropy 来自 Bernoulli MLE 且凸 |
| L2 regularization 影响 | 增大 bias、降低 variance，抑制 overfitting |
| Softmax vs one-vs-all | Softmax 联合学习归一化概率；OvA 训练多个独立二分类器 |

---

## 8. SVM：large margin、hinge loss、dual、kernel

### 8.1 Hyperplane distance

超平面：

$$
w^Tx+b=0.
$$

点 $x$ 到超平面的距离：

$$
\frac{|w^Tx+b|}{\|w\|}.
$$

### 8.2 Hard-margin SVM primal

要求分类正确且 functional margin 至少 1：

$$
y_i(w^Tx_i+b)\ge1.
$$

最大化 geometric margin $1/\|w\|$ 等价于：

$$
\boxed{
\min_{w,b}\frac12\|w\|^2
\quad\text{s.t. } y_i(w^Tx_i+b)\ge1,
\forall i.
}
$$

### 8.3 Hinge loss 视角

Hinge loss：

$$
\ell_i=\max(0,1-y_i(w^Tx_i+b)).
$$

Soft-margin objective：

$$
\min_{w,b}\frac12\|w\|^2+C\sum_i\max(0,1-y_i(w^Tx_i+b)).
$$

### 8.4 Hard-margin dual 推导

把 constraint 写成

$$
1-y_i(w^Tx_i+b)\le0.
$$

Lagrangian：

$$
L(w,b,\alpha)=\frac12w^Tw+\sum_i\alpha_i[1-y_i(w^Tx_i+b)],
\quad \alpha_i\ge0.
$$

Stationarity：

$$
\frac{\partial L}{\partial w}=w-\sum_i\alpha_i y_i x_i=0
\Rightarrow
\boxed{w=\sum_i\alpha_i y_i x_i}.
$$

$$
\frac{\partial L}{\partial b}=-\sum_i\alpha_i y_i=0
\Rightarrow
\boxed{\sum_i\alpha_i y_i=0}.
$$

代回得 dual：

$$
\boxed{
\max_\alpha \sum_i\alpha_i-\frac12\sum_{i,j}\alpha_i\alpha_jy_iy_jx_i^Tx_j
}
$$

s.t.

$$
\alpha_i\ge0,
\quad
\sum_i\alpha_iy_i=0.
$$

### 8.5 Support vectors

Complementary slackness：

$$
\alpha_i[1-y_i(w^Tx_i+b)]=0.
$$

- 若 $\alpha_i=0$：该点不是 support vector。
- 若 $\alpha_i>0$：必须有 $y_i(w^Tx_i+b)=1$，该点在 margin boundary 上，是 support vector。

### 8.6 Soft-margin SVM

Primal：

$$
\min_{w,b,\xi}\frac12\|w\|^2+C\sum_i\xi_i
$$

s.t.

$$
y_i(w^Tx_i+b)\ge1-\xi_i,
\quad \xi_i\ge0.
$$

Dual 与 hard-margin 基本一样，只是约束变成：

$$
\boxed{0\le\alpha_i\le C}.
$$

几何解释：

- $\alpha_i=0$：正确分类且在 margin 外。
- $0<\alpha_i<C$：在 margin 上。
- $\alpha_i=C$：违反 margin，甚至可能误分类。

### 8.7 Kernel trick

Dual 中数据只通过 inner product 出现：

$$
x_i^Tx_j.
$$

映射到特征空间：

$$
\phi(x_i)^T\phi(x_j)=k(x_i,x_j).
$$

常见 kernel：

- polynomial：$k(x,z)=(x^Tz+c)^p$。
- RBF：
  $$
  k(x,z)=\exp\left(-\frac{\|x-z\|^2}{2\sigma^2}\right).
  $$

---

## 9. Decision Tree / Random Forest：entropy 计算题

### 9.1 Entropy

节点 $S$ 中 $K$ 类比例为 $p_1,
\dots,p_K$：

$$
H(S)=-\sum_{k=1}^Kp_k\log_2p_k.
$$

二分类：

$$
H(S)=-p\log_2p-(1-p)\log_2(1-p).
$$

纯节点 entropy 为 0；二分类 50/50 entropy 最大，为 1。

### 9.2 Conditional entropy and information gain

如果特征 $A$ 把 $S$ 分成 $D_1,
\dots,D_V$：

$$
H(S\mid A)=\sum_{v=1}^V\frac{|D_v|}{|S|}H(D_v).
$$

Information gain：

$$
Gain(A)=H(S)-H(S\mid A).
$$

选 root：选 conditional entropy 最小，等价于 gain 最大。

### 9.3 构建树的考试步骤

1. 计算 root entropy。
2. 对每个候选 feature 计算 $H(S\mid A)$。
3. 选择最小 conditional entropy 的 feature。
4. 对非纯子节点重复。
5. 画树，叶节点写预测类别。
6. 对新样本沿树路径预测。

### 9.4 Random forest vs bagging

- Bagging：bootstrap 采样，训练多棵树，投票/平均。
- Random forest：在 bagging 基础上，每个 split 只看随机子集特征。
- 作用：降低树之间相关性，降低 variance，提高泛化。

常见判断：

- 单棵深树容易 overfit。
- 限制 max_depth、min_samples_leaf 可减少 overfitting。
- 随机森林一般比单棵树更稳定，但解释性不如单棵树。

---

## 10. Neural Network / Backprop：链式法则才是重点

### 10.1 单神经元

$$
z=w^Tx+b,
\quad y=\phi(z),
\quad L=\frac12(y-t)^2.
$$

链式法则：

$$
\frac{\partial L}{\partial w}
=\frac{\partial L}{\partial y}\frac{\partial y}{\partial z}\frac{\partial z}{\partial w}.
$$

其中

$$
\frac{\partial L}{\partial y}=y-t,
\quad
\frac{\partial z}{\partial w}=x,
\quad
\frac{\partial z}{\partial b}=1.
$$

若 $\phi=\sigma$：

$$
\frac{\partial y}{\partial z}=y(1-y).
$$

所以

$$
\frac{\partial L}{\partial w}=(y-t)y(1-y)x,
\quad
\frac{\partial L}{\partial b}=(y-t)y(1-y).
$$

### 10.2 多层网络的矩阵式 backprop

设一层为

$$
z^{(l)}=W^{(l)}a^{(l-1)}+b^{(l)},
\quad
a^{(l)}=\phi(z^{(l)}).
$$

定义

$$
\delta^{(l)}=\frac{\partial L}{\partial z^{(l)}}.
$$

输出层先由 loss 得到 $\delta^{(L)}$。隐藏层递推：

$$
\boxed{\delta^{(l)}=((W^{(l+1)})^T\delta^{(l+1)})\odot\phi'(z^{(l)})}.
$$

梯度：

$$
\boxed{\frac{\partial L}{\partial W^{(l)}}=\delta^{(l)}(a^{(l-1)})^T},
\quad
\boxed{\frac{\partial L}{\partial b^{(l)}}=\delta^{(l)}}.
$$

Batch 版本对样本求和/平均。

### 10.3 线性激活的深层网络为什么不更强

如果每层都是线性：

$$
f(x)=W_LW_{L-1}\cdots W_1x.
$$

令

$$
W'=W_LW_{L-1}\cdots W_1,
$$

则

$$
f(x)=W'x,
$$

仍然是线性模型。因此没有 nonlinear activation，就没有 universal approximation，也不能解决 XOR。

### 10.4 ReLU 与 vanishing gradient

ReLU：

$$
\operatorname{ReLU}(z)=\max(0,z),
\quad
\operatorname{ReLU}'(z)=\begin{cases}1,&z>0,\\0,&z<0.
\end{cases}
$$

相比 sigmoid/tanh，ReLU 在正区间梯度不小于 1，能缓解深层网络中的 vanishing gradient。

---

## 11. CNN：activation shape 与参数量

### 11.1 Convolution 输出尺寸

输入空间尺寸 $W_1\times H_1\times C$。

卷积核数量 $K$，filter size $F\times F\times C$，stride $S$，padding $P$。

输出：

$$
W_2=\frac{W_1-F+2P}{S}+1,
\quad
H_2=\frac{H_1-F+2P}{S}+1,
$$

输出体积：

$$
W_2\times H_2\times K.
$$

**必须是整数**。

### 11.2 Convolution 参数量

每个 filter 参数：

$$
F\cdot F\cdot C+1
$$

其中 $+1$ 是 bias。

总参数：

$$
\boxed{K(F^2C+1)}.
$$

### 11.3 Pooling 输出尺寸与参数

通常无 padding：

$$
W_2=\frac{W_1-F}{S}+1,
\quad
H_2=\frac{H_1-F}{S}+1.
$$

Depth 不变，参数量为 0。

### 11.4 Fully connected 参数量

若输入 flatten 后长度为 $N$，输出 $M$ 个神经元：

$$
\boxed{(N+1)M}
$$

其中 $+1$ 是 bias。

### 11.5 HW2 CNN worked example

网络：

$$
60\times60\times3
\to Conv5(12)
\to Maxpool2
\to Conv3(16)
\to FC8.
$$

1. Conv5(12)，$F=5,S=1,P=0,C=3,K=12$：
   $$
   W_2=60-5+1=56.
   $$
   输出：$56\times56\times12$。
   参数：
   $$
   (5\cdot5\cdot3+1)\cdot12=76\cdot12=912.
   $$

2. Maxpool2，$F=2,S=2$：
   $$
   56/2=28.
   $$
   输出：$28\times28\times12$。
   参数：0。

3. Conv3(16)，$F=3,S=1,P=1,C=12,K=16$：
   $$
   W_2=\frac{28-3+2}{1}+1=28.
   $$
   输出：$28\times28\times16$。
   参数：
   $$
   (3\cdot3\cdot12+1)\cdot16=109\cdot16=1744.
   $$

4. FC8：flatten 长度
   $$
   28\cdot28\cdot16=12544.
   $$
   参数：
   $$
   (12544+1)\cdot8=100360.
   $$

---

## 12. RNN / LSTM / Transformer：考试常问概念与公式

### 12.1 Vanilla RNN

递推：

$$
h_t=\tanh(W_{hh}h_{t-1}+W_{xh}x_t+b_h),
$$

输出：

$$
o_t=W_{hy}h_t+b_y,
\quad
\hat y_t=g(o_t).
$$

参数在所有 time steps 共享。

### 12.2 BPTT 的核心

总 loss：

$$
E(\theta)=\sum_{t=1}^TL_t.
$$

由于 $h_t$ 依赖 $h_{t-1}$，梯度包含 Jacobian 连乘：

$$
\frac{\partial h_t}{\partial h_k}
=\prod_{j=k+1}^t\frac{\partial h_j}{\partial h_{j-1}}.
$$

若简化为线性 RNN：

$$
h_t=W_{hh}h_{t-1}+W_{xh}x_t,
$$

则

$$
\frac{\partial h_T}{\partial h_1}=W_{hh}^{T-1}.
$$

若最大奇异值 $>1$，容易 exploding gradients；若 $<1$，容易 vanishing gradients。

### 12.3 LSTM

LSTM 用 cell state 和 gates 缓解 long-term dependency 问题。

常用公式：

$$
f_t=\sigma(W_f[h_{t-1},x_t]+b_f),
$$

$$
i_t=\sigma(W_i[h_{t-1},x_t]+b_i),
$$

$$
\tilde c_t=\tanh(W_c[h_{t-1},x_t]+b_c),
$$

$$
o_t=\sigma(W_o[h_{t-1},x_t]+b_o),
$$

$$
c_t=f_t\odot c_{t-1}+i_t\odot\tilde c_t,
$$

$$
h_t=o_t\odot\tanh(c_t).
$$

### 12.4 Transformer attention

输入 embedding matrix：

$$
X\in\mathbb{R}^{n\times d}.
$$

Projection：

$$
Q=XW_Q,
\quad K=XW_K,
\quad V=XW_V.
$$

Scaled dot-product attention：

$$
\boxed{Z=\operatorname{softmax}\left(\frac{QK^T}{\sqrt{d'}}\right)V}.
$$

Attention map：

$$
A=\operatorname{softmax}\left(\frac{QK^T}{\sqrt{d'}}\right)\in\mathbb{R}^{n\times n}.
$$

常考结论：

- Transformer 不按时间递推，所以更容易并行。
- 因为没有 recurrence，需要 positional encoding 表示顺序。
- Self-attention：$Q,K,V$ 来自同一序列。
- Cross-attention：$Q$ 来自 decoder，$K,V$ 来自 encoder。

---

## 13. Unsupervised Learning：K-means、GMM/EM、PCA

### 13.1 K-means objective

数据 $x_i\in\mathbb{R}^D$，cluster center $c_k$，assignment $r_{ik}\in\{0,1\}$，且 $\sum_k r_{ik}=1$。

目标：

$$
\boxed{J(c,r)=\sum_{i=1}^n\sum_{k=1}^Kr_{ik}\|x_i-c_k\|^2}.
$$

### 13.2 K-means assignment step

固定 centers：

$$
k^*=\arg\min_k\|x_i-c_k\|^2,
\quad r_{ik^*}=1.
$$

### 13.3 K-means refitting step 推导

固定 assignments，对 $c_k$ 求导：

$$
J_k=\sum_i r_{ik}\|x_i-c_k\|^2.
$$

$$
\frac{\partial J_k}{\partial c_k}=2\sum_i r_{ik}(c_k-x_i)=0.
$$

所以

$$
\boxed{c_k=\frac{\sum_i r_{ik}x_i}{\sum_i r_{ik}}}.
$$

即簇内均值。

### 13.4 GMM

高斯混合：

$$
p(x)=\sum_{k=1}^K\pi_k\mathcal{N}(x\mid\mu_k,\Sigma_k),
\quad \pi_k\ge0,
\quad \sum_k\pi_k=1.
$$

Log-likelihood：

$$
\log p(D\mid\theta)=\sum_{n=1}^N\log\left(\sum_{k=1}^K\pi_k\mathcal{N}(x^{(n)}\mid\mu_k,\Sigma_k)\right).
$$

困难点是 $\log\sum$。

### 13.5 Responsibilities / E-step

$$
\boxed{
\gamma_k^{(n)}=p(z^{(n)}=k\mid x^{(n)};\theta)
=\frac{\pi_k\mathcal{N}(x^{(n)}\mid\mu_k,\Sigma_k)}{\sum_j\pi_j\mathcal{N}(x^{(n)}\mid\mu_j,\Sigma_j)}
}
$$

解释：第 $k$ 个 component 对第 $n$ 个样本的 soft assignment。

### 13.6 GMM M-step

定义

$$
N_k=\sum_{n=1}^N\gamma_k^{(n)}.
$$

更新：

$$
\boxed{\mu_k=\frac1{N_k}\sum_n\gamma_k^{(n)}x^{(n)}}.
$$

$$
\boxed{\Sigma_k=\frac1{N_k}\sum_n\gamma_k^{(n)}(x^{(n)}-\mu_k)(x^{(n)}-\mu_k)^T}.
$$

$$
\boxed{\pi_k=\frac{N_k}{N}}.
$$

### 13.7 General EM / ELBO

对任意 $q(z)$：

$$
\log p(D;\theta)
=\mathcal{L}(q,\theta)+D_{KL}(q(z)\|p(z\mid D;\theta)).
$$

其中

$$
\mathcal{L}(q,\theta)=\sum_z q(z)\log\frac{p(D,z;\theta)}{q(z)}.
$$

因为 KL $\ge0$：

$$
\log p(D;\theta)\ge\mathcal{L}(q,\theta).
$$

E-step：固定 $\theta^{old}$，令

$$
q(z)=p(z\mid D;\theta^{old}),
$$

使 bound tight。

M-step：固定 $q$，最大化

$$
\theta^{new}=\arg\max_\theta \mathbb{E}_{q(z)}[\log p(D,z;\theta)].
$$

Monotonic ascent：

$$
\log p(D;\theta^{new})\ge\mathcal{L}(q,\theta^{new})\ge\mathcal{L}(q,\theta^{old})=
\log p(D;\theta^{old}).
$$

### 13.8 Mixture of Bernoullis EM

对二值向量 $x\in\{0,1\}^D$：

$$
P(x\mid p^{(k)})=\prod_{d=1}^D(p_d^{(k)})^{x_d}(1-p_d^{(k)})^{1-x_d}.
$$

Mixture：

$$
P(x^{(i)}\mid p,\pi)=\sum_{k=1}^K\pi_kP(x^{(i)}\mid p^{(k)}).
$$

E-step：

$$
\boxed{
\eta(z_k^{(i)})=
\frac{\pi_k\prod_{d=1}^D(p_d^{(k)})^{x_d^{(i)}}(1-p_d^{(k)})^{1-x_d^{(i)}}}
{\sum_j\pi_j\prod_{d=1}^D(p_d^{(j)})^{x_d^{(i)}}(1-p_d^{(j)})^{1-x_d^{(i)}}}
}
$$

M-step：

$$
\boxed{\tilde p_d^{(k)}=\frac{\sum_i\eta(z_k^{(i)})x_d^{(i)}}{N_k}},
\quad
N_k=\sum_i\eta(z_k^{(i)}),
$$

$$
\boxed{\tilde\pi_k=\frac{N_k}{N}}.
$$

### 13.9 PCA：projection 与 reconstruction

数据中心化：

$$
\mu=\frac1N\sum_nx^{(n)},
\quad \bar x^{(n)}=x^{(n)}-\mu.
$$

取 $U\in\mathbb{R}^{D\times K}$，列向量正交归一：

$$
U^TU=I.
$$

低维表示：

$$
z=U^T(x-\mu).
$$

重构：

$$
\tilde x=\mu+Uz.
$$

残差正交：

$$
U^T(x-\tilde x)=0.
$$

### 13.10 PCA covariance/eigenvalue 解

中心化后 covariance：

$$
\Sigma=\frac1N\sum_nx^{(n)}(x^{(n)})^T.
$$

第一主成分优化：

$$
\max_u u^T\Sigma u
\quad\text{s.t. } u^Tu=1.
$$

Lagrangian：

$$
L(u,
\lambda)=u^T\Sigma u-
\lambda(u^Tu-1).
$$

Stationarity：

$$
2\Sigma u-2\lambda u=0
\Rightarrow
\boxed{\Sigma u=\lambda u}.
$$

所以主成分是 covariance matrix 的 eigenvectors。取最大 eigenvalues 对应的 eigenvectors。

### 13.11 PCA 重构误差与方差解释率

若

$$
\lambda_1\ge\lambda_2\ge\cdots\ge\lambda_D,
$$

保留前 $K$ 个主成分的 variance explained ratio：

$$
\boxed{r_K=\frac{\sum_{i=1}^K\lambda_i}{\sum_{i=1}^D\lambda_i}}.
$$

平均重构误差：

$$
\boxed{\frac1N\sum_n\|x^{(n)}-\tilde x^{(n)}\|^2=\sum_{i=K+1}^D\lambda_i}.
$$

---

## 14. Bias-Variance、Evaluation、考试常识判断

### 14.1 Underfitting / overfitting

- Underfitting：training error 高，test error 也高；模型太简单，高 bias。
- Overfitting：training error 很低，test error 高；模型太复杂，高 variance。

模型复杂度升高：

- training error 通常下降。
- bias 通常下降。
- variance 通常上升。
- test error 可能先降后升。

### 14.2 Bias-variance decomposition

数据生成：

$$
y=t(x)+\epsilon,
\quad \epsilon\sim\mathcal{N}(0,\sigma^2).
$$

学习算法在数据集 $D$ 上学到 $h_D$，平均预测：

$$
\bar h(x)=\mathbb{E}_D[h_D(x)].
$$

平方损失期望误差分解：

$$
\mathbb{E}_{x,y,D}[(h_D(x)-y)^2]
=\underbrace{\mathbb{E}_{x,D}[(h_D(x)-\bar h(x))^2]}_{variance}
+\underbrace{\mathbb{E}_x[(\bar h(x)-t(x))^2]}_{bias^2}
+\underbrace{\mathbb{E}_{x,y}[(t(x)-y)^2]}_{noise}.
$$

### 14.3 Cross-validation

K-fold CV：

1. 把 training set 分成 $K$ folds。
2. 每次拿一个 fold 做 validation，其余训练。
3. 重复 $K$ 次。
4. 平均 validation performance，选择 hyperparameter。

不要用 test set 调参；test set 只用于最终评估。

### 14.4 Regression metrics

$$
MSE=\frac1n\sum_i(y_i-\hat y_i)^2.
$$

$$
MAE=\frac1n\sum_i|y_i-\hat y_i|.
$$

### 14.5 Confusion matrix metrics

| | predicted positive | predicted negative |
|---|---:|---:|
| actual positive | TP | FN |
| actual negative | FP | TN |

$$
Accuracy=\frac{TP+TN}{TP+TN+FP+FN}.
$$

$$
Precision=\frac{TP}{TP+FP}.
$$

$$
Recall=TPR=\frac{TP}{TP+FN}.
$$

$$
FNR=\frac{FN}{TP+FN},
\quad
TNR=\frac{TN}{TN+FP},
\quad
FPR=\frac{FP}{FP+TN}.
$$

恒等式：

$$
TPR+FNR=1,
\quad
TNR+FPR=1.
$$

ROC curve：y-axis 是 TPR，x-axis 是 FPR。

DET curve：通常画 FNR vs FPR。

AUC：正样本分数大于负样本分数的概率。

---

## 15. 期末/作业推导题的固定答题模板

### 15.1 “derive gradient” 模板

1. Define prediction or residual。
2. Write loss in compact form。
3. Use known derivative or differential。
4. Sum over samples or write matrix form。
5. Check dimension。

英文模板：

> Let $e_i=\hat y_i-y_i$. The objective can be written as $J=\sum_i e_i^Te_i$. Since $de_i=...$, we have $dJ=2\sum_i e_i^Tde_i$. Therefore the gradient is ... In matrix form, $\nabla J=...$.

### 15.2 “closed-form solution” 模板

1. 写 objective。
2. 求 gradient。
3. 令 gradient 为 0。
4. 得 normal equation。
5. 如果矩阵可逆，左乘逆。

英文模板：

> Taking derivative with respect to the parameter and setting it to zero gives the normal equation. Assuming the corresponding Gram matrix is invertible, the closed-form solution is ...

### 15.3 “prove convexity” 模板

1. Objective Hessian。
2. Hessian PSD。
3. Constraints are affine/convex。
4. Conclusion。

英文模板：

> The Hessian of the objective is ... For any vector $v$, $v^THv=...\ge0$, hence the objective is convex. The constraints are affine, so the feasible set is convex. Therefore this is a convex optimization problem.

### 15.4 “derive KKT” 模板

1. Rewrite constraints as $g_i(x)\le0$。
2. Write Lagrangian。
3. Stationarity。
4. Primal feasibility。
5. Dual feasibility。
6. Complementary slackness。
7. Case analysis active/inactive constraints。

英文模板：

> Let $g_i(x)\le0$ be the inequality constraints and $u_i\ge0$ be the multipliers. The Lagrangian is ... The KKT conditions are: stationarity ..., primal feasibility ..., dual feasibility ..., and complementary slackness ...

### 15.5 “derive MLE/MAP” 模板

1. State likelihood。
2. Product over samples due to i.i.d.。
3. Take log。
4. Drop constants independent of parameter。
5. Convert maximization to minimization。
6. MAP adds log-prior。

英文模板：

> Assuming the samples are i.i.d., the likelihood is the product of conditional probabilities. Taking logarithm converts the product into a sum. Removing constants independent of the parameter, maximizing the log-likelihood is equivalent to minimizing ... With the prior ..., MAP adds the term ..., which is equivalent to ... regularization.

### 15.6 “derive EM” 模板

1. Write incomplete log-likelihood $\log\sum_zp(x,z;\theta)$。
2. Introduce auxiliary distribution $q(z)$。
3. Use Jensen / KL decomposition。
4. E-step：set $q(z)=p(z\mid x;\theta^{old})$。
5. M-step：maximize expected complete-data log-likelihood。

英文模板：

> The incomplete-data log-likelihood contains a log-sum term, which is hard to optimize directly. We introduce an auxiliary distribution $q(z)$ and obtain the lower bound $\mathcal L(q,\theta)$. In the E-step, we choose $q(z)$ to be the posterior under old parameters so that the bound is tight. In the M-step, we maximize the expected complete-data log-likelihood with respect to the parameters.

### 15.7 “CNN shape/parameter” 模板

1. For each conv layer：write $W_2=(W_1-F+2P)/S+1$。
2. Depth equals number of filters。
3. Parameters：$(F^2C+1)K$。
4. Pooling has 0 parameters。
5. FC：flatten then $(N+1)M$。

### 15.8 “PCA derivation” 模板

1. Center data。
2. Covariance $\Sigma=\frac1N\sum_nx_nx_n^T$。
3. Objective $\max u^T\Sigma u$ s.t. $u^Tu=1$。
4. Lagrange gives $\Sigma u=\lambda u$。
5. Choose top eigenvalues。
6. Reconstruction error equals discarded eigenvalues。

---

## 16. 高频错误清单

1. **忘记 augment bias**：看到 $Wx+b$ 就先想 $\tilde W\tilde x$。
2. **$W$ 和 $W^T$ 混乱**：若 $Y=XB$，则 $B$ 是 $d\times m$；若题目写 $Wx$，$W$ 是 $m\times d$。
3. **正则化 bias**：通常不惩罚 $w_0$，除非题目明确说 regularize all weights。
4. **factor 2 丢失**：若 loss 有 $1/2$，梯度没有 2；若没有 $1/2$，梯度有 2。GD 中 factor 可被 learning rate 吸收，但推导要一致。
5. **Hessian PSD 只写结论**：考试要写 $v^THv\ge0$。
6. **KKT constraints 没转成 $\le0$**：乘子符号会错。
7. **SVM support vector 说成最远点**：support vectors 是离 boundary 最近/在 margin 上的点。
8. **Decision tree entropy 没按子节点样本数加权**。
9. **CNN filter depth 写错**：filter depth 必须等于输入 depth。
10. **Pooling 参数量写非零**：普通 max pooling 没有 learnable parameters。
11. **PCA reconstruction 维度写成低维**：$z$ 是低维，$\tilde x=\mu+Uz$ 回到原始 $D$ 维。
12. **GMM M-step 忘记 responsibility 权重**。
13. **EM 说保证全局最优**：EM 保证 log-likelihood 单调不下降，但可能到局部最优。
14. **K-means 说一定全局最优**：通常只保证收敛到局部最优，受初始化影响。
15. **ROC 坐标写反**：ROC 是 TPR vs FPR。

---

## 17. 最小复习路线：从“做不出 HW1 Q1”到“能写期末推导”

### Day 1：矩阵求导与 weighted least squares

必须手推：

- $\nabla_w\|Xw-y\|^2$。
- $\nabla_B\|XB-Y\|_F^2$。
- $\nabla_B\operatorname{tr}((Y-XB)^TA(Y-XB))$。
- HW1 Q1 weighted linear regression closed form + GD。

### Day 2：Logistic / Softmax

必须手推：

- $\sigma'(z)=\sigma(z)(1-\sigma(z))$。
- binary CE gradient $X^T(h-y)$。
- Hessian $X^TRX$ and PSD proof。
- softmax gradient $\sum_i(P_{ik}-y_{ik})x_i$。

### Day 3：Optimization / KKT / SVM

必须手推：

- HW1 KKT 题。
- hard-margin SVM primal to dual。
- soft-margin dual constraint $0\le\alpha_i\le C$。

### Day 4：Entropy / CNN / NN

必须会算：

- decision tree root entropy and conditional entropy。
- CNN output shape and parameter count。
- backprop 单神经元与多层 delta 公式。

### Day 5：Unsupervised

必须手推：

- K-means center update。
- GMM responsibilities and M-step。
- EM lower bound / KL decomposition。
- PCA Lagrange eigenvector derivation。

### Day 6：Evaluation / Bias-variance / Exam MC

必须掌握：

- confusion matrix metrics。
- ROC/AUC/EER。
- overfit/underfit remedy。
- K-fold CV。
- 2020/2022/template 中的判断题逻辑。

---

## 18. 一页速查表

### Linear / Ridge

$$
\nabla_w\|Xw-y\|^2=2X^T(Xw-y)
$$

$$
w^*=(X^TX)^{-1}X^Ty
$$

$$
w^*_{ridge}=(X^TX+\lambda I)^{-1}X^Ty
$$

### Logistic

$$
\sigma'(z)=\sigma(z)(1-\sigma(z))
$$

$$
\nabla_wJ=X^T(h-y)
$$

$$
H=X^TRX,
\quad R=\operatorname{diag}(h_i(1-h_i))
$$

### Softmax

$$
P_{ik}=\frac{e^{w_k^Tx_i}}{\sum_j e^{w_j^Tx_i}}
$$

$$
\nabla_{w_k}L=\sum_i(P_{ik}-y_{ik})x_i
$$

### SVM

$$
\min_{w,b}\frac12\|w\|^2
\quad s.t.\quad y_i(w^Tx_i+b)\ge1
$$

$$
w=\sum_i\alpha_iy_ix_i,
\quad \sum_i\alpha_iy_i=0
$$

$$
\max_\alpha\sum_i\alpha_i-\frac12\sum_{i,j}\alpha_i\alpha_jy_iy_jx_i^Tx_j
$$

### Decision tree

$$
H(S)=-\sum_kp_k\log_2p_k
$$

$$
H(S\mid A)=\sum_v\frac{|D_v|}{|S|}H(D_v)
$$

$$
Gain(A)=H(S)-H(S\mid A)
$$

### CNN

$$
W_2=\frac{W_1-F+2P}{S}+1
$$

$$
\#params=K(F^2C+1)
$$

### K-means

$$
J=\sum_i\sum_kr_{ik}\|x_i-c_k\|^2
$$

$$
c_k=\frac{\sum_ir_{ik}x_i}{\sum_ir_{ik}}
$$

### GMM/EM

$$
\gamma_k^{(n)}=\frac{\pi_k\mathcal N(x^{(n)}\mid\mu_k,\Sigma_k)}{\sum_j\pi_j\mathcal N(x^{(n)}\mid\mu_j,\Sigma_j)}
$$

$$
\mu_k=\frac1{N_k}\sum_n\gamma_k^{(n)}x^{(n)}
$$

$$
\Sigma_k=\frac1{N_k}\sum_n\gamma_k^{(n)}(x^{(n)}-\mu_k)(x^{(n)}-\mu_k)^T
$$

$$
\pi_k=\frac{N_k}{N}
$$

### PCA

$$
\Sigma=\frac1N\sum_nx^{(n)}(x^{(n)})^T
$$

$$
\Sigma u=\lambda u
$$

$$
r_K=\frac{\sum_{i=1}^K\lambda_i}{\sum_{i=1}^D\lambda_i}
$$

$$
\text{reconstruction error}=\sum_{i=K+1}^D\lambda_i
$$

---

## 19. 自测：你应该能不看答案完成这些

1. 推导 weighted linear regression 的闭式解，能同时写 $B^*$ 和 $\tilde W^*$。
2. 写出 $\nabla_W\sum_i\alpha_i\|Wx_i+b-y_i\|^2$。
3. 证明 logistic regression loss convex。
4. 推导 softmax 第 $k$ 类权重的梯度。
5. 写出一个 constrained convex problem 的 Lagrangian 和四条 KKT。
6. 从 SVM primal 推到 dual。
7. 手算 decision tree entropy 选择 root。
8. 手算 CNN 每层 shape 和参数量。
9. 从 RNN Jacobian 连乘解释 vanishing/exploding gradients。
10. 从 GMM posterior 推出 responsibilities。
11. 用 Lagrange multiplier 推出 PCA eigenvector equation。
12. 根据 confusion matrix 计算 precision/recall/FPR/FNR/AUC。

如果这些都能写出来，DDA3020 的数学推导短板基本补齐。
