import numpy as np
import argparse
from typing import Callable, List, Tuple

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('train_input', type=str, help='path to training input .csv file')
parser.add_argument('validation_input', type=str, help='path to validation input .csv file')
parser.add_argument('train_out', type=str, help='path to store training predictions')
parser.add_argument('validation_out', type=str, help='path to store validation predictions')
parser.add_argument('metrics_out', type=str, help='path to store training and testing metrics')
parser.add_argument('num_epoch', type=int, help='number of training epochs')
parser.add_argument('hidden_units', type=int, help='number of hidden units')
parser.add_argument('init_flag', type=int, choices=[1, 2], help='1: random init, 2: zero init')
parser.add_argument('learning_rate', type=float, help='learning rate')

# Helper functions and classes
def args2data(args) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str, str, str, int, int, int, float]:
    out_tr, out_te, out_metrics = args.train_out, args.validation_out, args.metrics_out
    n_epochs, n_hid, init_flag, lr = args.num_epoch, args.hidden_units, args.init_flag, args.learning_rate
    X_tr = np.loadtxt(args.train_input, delimiter=',')
    y_tr = X_tr[:, 0].astype(int)
    X_tr = X_tr[:, 1:]  
    X_te = np.loadtxt(args.validation_input, delimiter=',')
    y_te = X_te[:, 0].astype(int)
    X_te = X_te[:, 1:]  
    return X_tr, y_tr, X_te, y_te, out_tr, out_te, out_metrics, n_epochs, n_hid, init_flag, lr

def shuffle(X, y, epoch):
    np.random.seed(epoch)
    ordering = np.random.permutation(len(y))
    return X[ordering], y[ordering]

def zero_init(shape):
    return np.zeros(shape=shape)

def random_init(shape):
    np.random.seed(shape[0] * shape[1]) 
    return np.random.uniform(-0.1, 0.1, shape)

class SoftMaxCrossEntropy:
    def _softmax(self, z: np.ndarray) -> np.ndarray:
        e_z = np.exp(z - np.max(z))
        return e_z / e_z.sum(axis=0)

    def forward(self, z: np.ndarray, y: int, num_classes: int) -> Tuple[np.ndarray, float]:
        y_one_hot = np.zeros(num_classes)
        y_one_hot[y] = 1
        y_hat = self._softmax(z)
        loss = -np.sum(y_one_hot * np.log(y_hat + 1e-9))
        return y_hat, loss

    def backward(self, y: int, y_hat: np.ndarray) -> np.ndarray:
        grad = y_hat.copy()
        grad[y] -= 1
        return grad

class Sigmoid:
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.output = 1 / (1 + np.exp(-x))
        return self.output

    def backward(self, dz: np.ndarray) -> np.ndarray:
        return dz * self.output * (1 - self.output)

class Linear:
    def __init__(self, input_size: int, output_size: int, weight_init_fn: Callable[[Tuple[int, int]], np.ndarray], learning_rate: float):
        self.lr = learning_rate
        self.weights = weight_init_fn((output_size, input_size))
        self.bias = np.zeros(output_size)
        self.dw = np.zeros_like(self.weights)
        self.db = np.zeros_like(self.bias)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        return np.dot(self.weights, x) + self.bias

    def backward(self, dz: np.ndarray) -> np.ndarray:
        self.dw = np.outer(dz, self.x)
        self.db = dz
        return np.dot(self.weights.T, dz)

    def step(self) -> None:
        self.weights -= self.lr * self.dw
        self.bias -= self.lr * self.db

class NN:
    def __init__(self, input_size: int, hidden_size: int, output_size: int, weight_init_fn: Callable[[Tuple[int, int]], np.ndarray], learning_rate: float):
        self.hidden_layer = Linear(input_size, hidden_size, weight_init_fn, learning_rate)
        self.sigmoid = Sigmoid()
        self.output_layer = Linear(hidden_size, output_size, weight_init_fn, learning_rate)
        self.softmax_ce = SoftMaxCrossEntropy()

    def forward(self, x: np.ndarray, y: int, num_classes: int) -> Tuple[np.ndarray, float]:
        h = self.sigmoid.forward(self.hidden_layer.forward(x))
        y_hat, loss = self.softmax_ce.forward(self.output_layer.forward(h), y, num_classes)
        return y_hat, loss

    def backward(self, y: int, y_hat: np.ndarray) -> None:
        dz_output = self.softmax_ce.backward(y, y_hat)
        dz_hidden = self.sigmoid.backward(self.output_layer.backward(dz_output))
        self.hidden_layer.backward(dz_hidden)

    def step(self):
        self.hidden_layer.step()
        self.output_layer.step()

    def compute_loss(self, X: np.ndarray, y: np.ndarray, num_classes: int) -> float:
        losses = [self.forward(X[i], y[i], num_classes)[1] for i in range(len(y))]
        return np.mean(losses)

    def train(self, X_tr: np.ndarray, y_tr: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, n_epochs: int, num_classes: int) -> Tuple[List[float], List[float]]:
        train_losses, test_losses = [], []
        for epoch in range(n_epochs):
            X_shuf, y_shuf = shuffle(X_tr, y_tr, epoch)
            for i in range(len(X_shuf)):
                y_hat, _ = self.forward(X_shuf[i], y_shuf[i], num_classes)
                self.backward(y_shuf[i], y_hat)
                self.step()
            train_losses.append(self.compute_loss(X_tr, y_tr, num_classes))
            test_losses.append(self.compute_loss(X_test, y_test, num_classes))
            print(f"Epoch {epoch+1}: train_loss={train_losses[-1]}, validation_loss={test_losses[-1]}")
        return train_losses, test_losses

    def test(self, X: np.ndarray, y: np.ndarray, num_classes: int) -> Tuple[np.ndarray, float]:
        predictions = [np.argmax(self.forward(X[i], y[i], num_classes)[0]) for i in range(len(X))]
        error_rate = np.mean(predictions != y)
        return np.array(predictions), error_rate

if __name__ == "__main__":
    args = parser.parse_args()
    labels = ["a", "e", "g", "i", "l", "n", "o", "r", "t", "u"]
    num_classes = len(labels)
    (X_tr, y_tr, X_test, y_test, out_tr, out_te, out_metrics, n_epochs, n_hid, init_flag, lr) = args2data(args)

    nn = NN(input_size=X_tr.shape[-1], hidden_size=n_hid, output_size=num_classes,
            weight_init_fn=zero_init if init_flag == 2 else random_init, learning_rate=lr)

    train_losses, test_losses = nn.train(X_tr, y_tr, X_test, y_test, n_epochs, num_classes)

    train_labels, train_error_rate = nn.test(X_tr, y_tr, num_classes)
    test_labels, test_error_rate = nn.test(X_test, y_test, num_classes)

    with open(out_tr, "w") as f:
        for label in train_labels:
            f.write(str(label) + "\n")
    with open(out_te, "w") as f:
        for label in test_labels:
            f.write(str(label) + "\n")
    with open(out_metrics, "w") as f:
        for i in range(len(train_losses)):
            f.write(f"epoch={i + 1} crossentropy(train): {train_losses[i]}\n")
            f.write(f"epoch={i + 1} crossentropy(validation): {test_losses[i]}\n")
        f.write(f"error(train): {train_error_rate}\n")
        f.write(f"error(validation): {test_error_rate}\n")
