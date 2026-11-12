import numpy as np


def active_parameters(model):
    parameters = list(model.parameters().values())
    active = [p for p in parameters if p.grad is not None]
    for p in active:
        if p.grad.shape != p.data.shape or not np.isfinite(p.grad).all():
            raise ValueError("Invalid parameter gradient")
    return active


class SGD:
    def __init__(self, lr=0.01, momentum=0.0):
        if not np.isfinite(lr) or lr <= 0 or not 0 <= momentum < 1:
            raise ValueError("Invalid SGD settings")
        self.lr, self.momentum, self.velocity = lr, momentum, {}

    def step(self, model):
        for p in active_parameters(model):
            v = self.momentum * self.velocity.get(p, np.zeros_like(p.data)) + p.grad
            self.velocity[p] = v
            p.data -= self.lr * v


class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        if not np.isfinite(lr) or lr <= 0 or not 0 <= beta1 < 1 or not 0 <= beta2 < 1 or not np.isfinite(eps) or eps <= 0:
            raise ValueError("Invalid Adam settings")
        self.lr, self.beta1, self.beta2, self.eps = lr, beta1, beta2, eps
        self.t, self.m, self.v = 0, {}, {}

    def step(self, model):
        active = active_parameters(model)
        if not active:
            return
        self.t += 1
        for p in active:
            m = self.beta1 * self.m.get(p, np.zeros_like(p.data)) + (1 - self.beta1) * p.grad
            v = self.beta2 * self.v.get(p, np.zeros_like(p.data)) + (1 - self.beta2) * p.grad ** 2
            self.m[p], self.v[p] = m, v
            p.data -= self.lr * (m / (1 - self.beta1 ** self.t)) / (np.sqrt(v / (1 - self.beta2 ** self.t)) + self.eps)
