import numpy as np
import matplotlib.pyplot as plt

class ffnn:
    def __init__(self, d_in, d_hidden, d_out, lr=1.5e-2, ep = .0000001):
        self.W1 = np.random.randn(d_in, d_hidden) * np.sqrt(2.0/d_in)
        self.b1 = np.zeros((1, d_hidden))
        self.W2 = np.random.randn(d_hidden, d_out) * 0.1
        self.b2 = np.zeros((1, d_out))
        self.lr = lr
        self.ep = ep
        self.prev_E = None

    def sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def sigmoid_deriv(self, y):
        return y*(1-y)

    def forward(self, X):
        # hidden layer with sigmoid
        self.z1 = X.dot(self.W1) + self.b1   # (n, d_hidden)
        self.h  = self.sigmoid(self.z1)      # (n, d_hidden)

        self.z2 = self.h.dot(self.W2) + self.b2  # (n, d_out)
        self.y  = self.z2                       # identity
        return self.y

    def backward(self, X, Y):

        err = self.y - Y                     # (n, d_out)
        d2  = err                            

 
        dW2 = self.h.T.dot(d2) / X.shape[0]  # (d_hidden, d_out)
        db2 = np.mean(d2, axis=0, keepdims=True)  # (1, d_out)

      
        d1  = d2.dot(self.W2.T) * self.sigmoid_deriv(self.h)  # (n, d_hidden)

   
        dW1 = X.T.dot(d1) / X.shape[0]      # (d_in, d_hidden)
        db1 = np.mean(d1, axis=0, keepdims=True)  # (1, d_hidden)


        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

        return np.mean(err**2)

    def train(self, X, Y, epochs=1000):
        for ep in range(epochs):
            self.forward(X)
            loss = self.backward(X, Y)
            if self.prev_E is not None and abs(self.prev_E - loss)<self.ep:
                print('Epoch:'+str(ep))
                print(self.prev_E, loss)
                break
            self.prev_E = loss
            if ep % 10000 == 0:
                print(f"Epoch {ep:<4}  Loss {loss:.5f}")

    def predict(self, X):
        return self.forward(X)
