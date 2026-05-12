import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUT_DIR = ""

class Result:
    def __init__(self, W, losses, train_acc, test_acc):
        self.W = W
        self.losses = losses
        self.train_acc = train_acc
        self.test_acc = test_acc

def softmax(logits):
    # logit \in R^{n x K}, output \in R^{n x K} 
    z = logits - logits.max(axis=1, keepdims=True)
    return np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)

def one_hot(y, K): 
    # y is (n,), output is (n, K)
    return np.eye(K)[y].astype(float)

def add_bias(X):
    return np.hstack([np.ones((X.shape[0], 1)), X])

def accuracy(W, X, y):
    pred = np.argmax(softmax(X @ W), axis=1)
    return float((pred == y).mean())

def cross_entropy_loss(W, Xb, Y, reg_type=None, lam=0.0):
    n = Xb.shape[0]
    P = softmax(Xb @ W)
    ce = -np.sum(Y * np.log(P + 1e-12)) / n

    reg = 0.0
    if reg_type == "l2":
        reg = (lam / (2*n)) * np.sum(W[1:, :] ** 2)  
    elif reg_type == "l1":
        reg = (lam / (2*n)) * np.sum(np.abs(W[1:, :]))
    return ce + reg

def train(
    Xb,
    Y,
    y_int,
    Xb_test,
    y_test_int,
    lr=0.1,
    epochs=10000,
    reg_type=None,
    lam=0.0,
    seed=1,
    loss_every=50,
):
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.01, size=(Xb.shape[1], Y.shape[1]))

    n = Xb.shape[0]
    losses = []
    for t in range(epochs):
        P = softmax(Xb @ W)
        grad = (Xb.T @ (P - Y)) / n
        if reg_type == "l2":
            grad[1:, :] += (lam / n) * W[1:, :]         
        elif reg_type == "l1":
            grad[1:, :] += (lam / (2*n)) * np.sign(W[1:, :])  

        W -= lr * grad

        if t % loss_every == 0:
            losses.append(cross_entropy_loss(W, Xb, Y, reg_type, lam))

    return Result(W, losses, accuracy(W, Xb, y_int), accuracy(W, Xb_test, y_test_int))

def plot_loss(losses, loss_every, title, path):
    plt.figure(figsize=(6, 4))
    plt.plot(np.arange(len(losses)) * loss_every, losses)
    plt.xlabel("iteration")
    plt.ylabel("loss")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

def plot_losses(res_dict, loss_every, title, path):
    plt.figure(figsize=(6, 4))
    for lam, res in res_dict.items():
        plt.plot(np.arange(len(res.losses)) * loss_every, res.losses, label=f"λ={lam}")
    plt.xlabel("iteration")
    plt.ylabel("loss")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

def sparsity(W, thresh=1e-3):
    return int(np.sum(np.abs(W[1:, :] ) < thresh)), int(W[1:, :] .size)

def main():
    # Load dataset
    csv_path = OUT_DIR + "Iris.csv"
    df = pd.read_csv(csv_path)

    X = df[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]].to_numpy(dtype=float)
    species = df["Species"].astype(str).to_numpy()
    classes = np.unique(species)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y = np.array([class_to_idx[s] for s in species], dtype=int)
    K = len(classes)

    # Train/test = 80/20
    seed = 1
    rng = np.random.default_rng(seed)
    idx = np.arange(len(X))
    rng.shuffle(idx)
    split = int(0.8 * len(X))
    train_idx, test_idx = idx[:split], idx[split:]
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Normalization
    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0, ddof=0)
    sigma[sigma == 0] = 1.0
    X_train = (X_train - mu) / sigma
    X_test = (X_test - mu) / sigma

    # Add bias
    Xtr = add_bias(X_train)
    Xte = add_bias(X_test)
    Ytr = one_hot(y_train, K)

    # Hyperparameters
    lr = 0.1
    epochs = 10000
    loss_every = 50

    # (a) No regularization
    res_noreg = train(
        Xtr,
        Ytr,
        y_train,
        Xte,
        y_test,
        lr=lr,
        epochs=epochs,
        reg_type=None,
        lam=0.0,
        seed = seed,
        loss_every=loss_every,
    )
    print(f"Hyperparameters: lr={lr}, epochs={epochs}")
    print()
    print("Part (a) No regularization")
    print(f"Train acc = {res_noreg.train_acc:.4f}, Test acc = {res_noreg.test_acc:.4f}")
    plot_loss(
        res_noreg.losses,
        loss_every,
        "Training loss (no regularization)",
        OUT_DIR + "q2_2_noreg.png",
    )
    print("Saved: q2_2_noreg.png")
    print()

    # (b) L2
    lams = [0.01, 0.1, 1.0]
    res_l2 = {}
    print("Part (b) L2")
    for lam in lams:
        res = train(
            Xtr,
            Ytr,
            y_train,
            Xte,
            y_test,
            lr=lr,
            epochs=epochs,
            reg_type="l2",
            lam=lam,
            seed = seed,
            loss_every=loss_every,
        )
        res_l2[lam] = res
        nz, total = sparsity(res.W)
        print(
            f"λ={lam}: Train acc={res.train_acc:.4f}, Test acc={res.test_acc:.4f}, near-zero weights={nz}/{total}"
        )

    # (c) L1
    res_l1 = {}
    print("Part (c) L1")
    for lam in lams:
        res = train(
            Xtr,
            Ytr,
            y_train,
            Xte,
            y_test,
            lr=lr,
            epochs=epochs,
            reg_type="l1",
            lam=lam,
            seed = seed,
            loss_every=loss_every,
        )
        res_l1[lam] = res
        nz, total = sparsity(res.W)
        print(
            f"λ={lam}: Train acc={res.train_acc:.4f}, Test acc={res.test_acc:.4f}, near-zero weights={nz}/{total}"
        )

if __name__ == "__main__":
    main()

'''
Hyperparameters: lr=0.1, epochs=10000

Part (a) No regularization
Train acc = 0.9750, Test acc = 1.0000
Saved: q2_2_noreg.png

Part (b) L2
λ=0.01: Train acc=0.9750, Test acc=1.0000, near-zero weights=0/12
λ=0.1: Train acc=0.9750, Test acc=1.0000, near-zero weights=0/12
λ=1.0: Train acc=0.9500, Test acc=0.9667, near-zero weights=0/12

Part (c) L1
λ=0.01: Train acc=0.9750, Test acc=1.0000, near-zero weights=0/12
λ=0.1: Train acc=0.9750, Test acc=1.0000, near-zero weights=0/12
λ=1.0: Train acc=0.9667, Test acc=1.0000, near-zero weights=4/12
'''