import math
import numpy as np

class LogisticClassifier:
    def __init__(self, learning_rate=0.01, num_iterations=1000, lambda_=0.0, using_one_hot=False):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.lambda_ = lambda_
        self.w = None
        self.b = None
        self.J_history = []
        self.using_one_hot = using_one_hot

    def sigmoid(self, x):
        
        if self.using_one_hot:
            #z = float(x)
            z = x.astype('float')
        else:
            z = np.clip(x, -500, 500)
            
        return 1.0 / (1.0 + np.exp(-z))

    def compute_cost(self, X, y):
        m, n = X.shape
        z = np.dot(X, self.w) + self.b
        f_wb = self.sigmoid(z)
        
        if self.using_one_hot:
            cost = np.sum(~y * np.log(f_wb + 1e-10))
        else:
            cost = np.sum(-y * np.log(f_wb + 1e-10) - (1 - y) * np.log(1 - f_wb + 1e-10))
        
        cost = cost / m

        reg_cost = 0
        if self.lambda_ != 0:
            reg_cost = (self.lambda_ / (2 * m)) * np.dot(self.w, self.w)

        return cost + reg_cost

    def calculate_gradient(self, X, y):
        m, n = X.shape
        dw = np.zeros((n,))
        db = 0.

        for i in range(m):
            z = np.dot(X[i], self.w) + self.b
            a = self.sigmoid(z)
            dz = a - y[i]
            for j in range(n):
                dw[j] += X[i][j] * dz + self.lambda_ / m * self.w[j]
            db += dz

        return dw / m, db / m
    
    def fit(self, X, y):
        m, n = X.shape
        self.w = np.zeros((n,))
        self.b = 0.

        for i in range(self.num_iterations):
            dw, db = self.calculate_gradient(X, y)
            self.w = self.w - self.learning_rate * dw
            self.b = self.b - self.learning_rate * db

            if i < 100000:
                self.J_history.append(self.compute_cost(X, y))

            if i % math.ceil(self.num_iterations / 10) == 0:
                print(f"Iteration {i:4d}: Cost {self.J_history[-1]}")
        
        return (self.w, self.b, self.J_history)

    def predict(self, X):
        z = np.dot(X, self.w) + self.b
        return np.round(self.sigmoid(z))

