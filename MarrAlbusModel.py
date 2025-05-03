import numpy as np
import matplotlib.pyplot as plt

class MarrAlbusModel:
    def __init__(self, d_in, d_granule, d_out, lr=1.5e-2, ep = .0000001):
        self.E = np.random.randn(d_in, d_granule) * np.sqrt(2.0/d_in)
        self.W = np.random.randn(d_granule, d_out) * 0.1
        self.b = np.zeros((1, d_out))
        self.lr = lr
        self.ep = ep
        self.prev_E = None

    def granule_activation(self, z):
        return 1/(1+np.exp(-z))

    def forward(self, X):
        self.z = X.dot(self.E)             # fixed expansion
        self.h = self.granule_activation(self.z)
        self.y = self.h.dot(self.W) + self.b
        return self.y

    def backward(self, X, Y):
        # compute gradient of MSE loss
        err = self.y - Y                   # (n, d_out)
        dW  = self.h.T.dot(err) / X.shape[0]
        db  = np.mean(err, axis=0, keepdims=True)
        # update trainable params
        self.W -= self.lr * dW
        self.b -= self.lr * db
        return np.mean(err**2)

    def train(self, X, Y, epochs=500):
        for ep in range(epochs):
            self.forward(X)
            loss = self.backward(X, Y)
            if self.prev_E is not None and abs(self.prev_E - loss)<self.ep:
                print('Epoch:'+str(ep))
                print(self.prev_E,loss)
                break
            self.prev_E = loss
            if ep % 1000 == 0:
                print(f"Epoch {ep} loss {loss:.4f}")

    def predict(self, X):
        return self.forward(X)
