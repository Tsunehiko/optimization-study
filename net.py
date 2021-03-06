import sys, os
sys.path.append(os.pardir)

import numpy as np
import matplotlib.pyplot as plt

from util import numerical_gradient

class Net:
    def __init__(self):
        super().__init__()
        self.params = {}
        self.params["W"] = np.array([1.0,0.0])

    def predict(self, x):
        W = self.params["W"]
        sigm = lambda a: 1/(1+np.exp(-a))
        y = sigm(x.dot(W))
        return y

    def loss(self, x, t):
        y = self.predict(x)
        loss = np.sum((y-t)**2)
        return loss

    def grad(self, x, t):
        loss = lambda W: self.loss(x, t)
        grads = {}
        grads["W"] = numerical_gradient(loss, self.params["W"])
        return grads

    