"""NCHW layers using im2col and hand-written reverse-mode derivatives."""
import numpy as np
from .core import Module, Parameter, pair, floating


def im2col(x, kernel_size, stride=1, padding=0):
    x = floating(x, 4)
    kh, kw = pair(kernel_size, "kernel_size")
    sh, sw = pair(stride, "stride")
    ph, pw = pair(padding, "padding", 0)
    n, c, h, w = x.shape
    oh, ow = (h + 2 * ph - kh) // sh + 1, (w + 2 * pw - kw) // sw + 1
    if min(oh, ow) <= 0:
        raise ValueError("Kernel is larger than padded input")
    padded = np.pad(x, ((0, 0), (0, 0), (ph, ph), (pw, pw)))
    rows = np.repeat(np.arange(kh), kw)
    rows = np.tile(rows, c)[:, None] + sh * np.repeat(np.arange(oh), ow)
    cols = np.tile(np.tile(np.arange(kw), kh), c)[:, None] + sw * np.tile(np.arange(ow), oh)
    channels = np.repeat(np.arange(c), kh * kw)[:, None]
    matrix = padded[:, channels, rows, cols].transpose(1, 2, 0).reshape(c * kh * kw, -1)
    return matrix, (channels, rows, cols, (oh, ow))


class Linear(Module):
    def __init__(self, in_features, out_features, rng=None):
        if type(in_features) is not int or type(out_features) is not int or min(in_features, out_features) <= 0:
            raise ValueError("Feature counts must be positive integers")
        rng = rng if rng is not None else np.random.default_rng()
        self.weight = Parameter(rng.normal(size=(in_features, out_features)) * np.sqrt(2 / in_features))
        self.bias = Parameter(np.zeros(out_features))
        self.x = None

    def parameters(self):
        return {"weight": self.weight, "bias": self.bias}

    def forward(self, x, training=True):
        x = floating(x, 2)
        if x.shape[1] != self.weight.data.shape[0]:
            raise ValueError("Linear input features do not match")
        self.x = x.copy()
        return x @ self.weight.data + self.bias.data

    def backward(self, dout):
        if self.x is None:
            raise RuntimeError("forward must precede backward")
        dout = floating(dout, 2)
        if dout.shape != (len(self.x), len(self.bias.data)):
            raise ValueError("Linear upstream gradient shape does not match")
        self.weight.grad = self.x.T @ dout
        self.bias.grad = dout.sum(axis=0)
        return dout @ self.weight.data.T


class Conv2D(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, rng=None):
        if type(in_channels) is not int or type(out_channels) is not int or min(in_channels, out_channels) <= 0:
            raise ValueError("Channel counts must be positive integers")
        self.kernel_size = pair(kernel_size, "kernel_size")
        self.stride = pair(stride, "stride")
        self.padding = pair(padding, "padding", 0)
        rng = rng if rng is not None else np.random.default_rng()
        kh, kw = self.kernel_size
        self.weight = Parameter(rng.normal(size=(out_channels, in_channels, kh, kw)) * np.sqrt(2 / (in_channels * kh * kw)))
        self.bias = Parameter(np.zeros(out_channels))
        self.cache = None

    def parameters(self):
        return {"weight": self.weight, "bias": self.bias}

    def forward(self, x, training=True):
        x = floating(x, 4)
        if x.shape[1] != self.weight.data.shape[1]:
            raise ValueError("Conv2D input channels do not match")
        matrix, indices = im2col(x, self.kernel_size, self.stride, self.padding)
        oh, ow = indices[-1]
        self.cache = (x.shape, matrix, indices)
        out = self.weight.data.reshape(len(self.bias.data), -1) @ matrix + self.bias.data[:, None]
        return out.reshape(len(self.bias.data), oh, ow, len(x)).transpose(3, 0, 1, 2)

    def backward(self, dout):
        if self.cache is None:
            raise RuntimeError("forward must precede backward")
        shape, matrix, (channels, rows, cols, (oh, ow)) = self.cache
        dout = floating(dout, 4)
        if dout.shape != (shape[0], len(self.bias.data), oh, ow):
            raise ValueError("Conv2D upstream gradient shape does not match")
        upstream = dout.transpose(1, 2, 3, 0).reshape(len(self.bias.data), -1)
        self.weight.grad = (upstream @ matrix.T).reshape(self.weight.data.shape)
        self.bias.grad = upstream.sum(axis=1)
        dx_col = self.weight.data.reshape(len(self.bias.data), -1).T @ upstream
        ph, pw = self.padding
        n, c, h, w = shape
        padded = np.zeros((n, c, h + 2 * ph, w + 2 * pw), dtype=dout.dtype)
        values = dx_col.reshape(c * self.kernel_size[0] * self.kernel_size[1], oh * ow, n).transpose(2, 0, 1)
        np.add.at(padded, (slice(None), channels, rows, cols), values)
        return padded[:, :, ph:ph + h, pw:pw + w]


class BatchNorm2d(Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        if type(num_features) is not int or num_features <= 0 or not np.isfinite(eps) or eps <= 0 or not 0 <= momentum <= 1:
            raise ValueError("Invalid BatchNorm2d parameters")
        self.eps, self.momentum = eps, momentum
        shape = (1, num_features, 1, 1)
        self.gamma, self.beta = Parameter(np.ones(shape)), Parameter(np.zeros(shape))
        self.running_mean, self.running_var = np.zeros(shape), np.ones(shape)
        self.cache = None

    def parameters(self):
        return {"gamma": self.gamma, "beta": self.beta}

    def buffers(self):
        return {"running_mean": self.running_mean, "running_var": self.running_var}

    def forward(self, x, training=True):
        x = floating(x, 4)
        if x.shape[1] != self.gamma.data.shape[1]:
            raise ValueError("BatchNorm2d channels do not match")
        if training:
            mean, var = x.mean(axis=(0, 2, 3), keepdims=True), x.var(axis=(0, 2, 3), keepdims=True)
            self.running_mean[...] = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var[...] = (1 - self.momentum) * self.running_var + self.momentum * var
            normalized = (x - mean) / np.sqrt(var + self.eps)
            self.cache = normalized, np.sqrt(var + self.eps)
        else:
            normalized = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)
            self.cache = None
        return self.gamma.data * normalized + self.beta.data

    def backward(self, dout):
        if self.cache is None:
            raise RuntimeError("A training forward must precede BatchNorm backward")
        dout = floating(dout, 4)
        normalized, std = self.cache
        if dout.shape != normalized.shape:
            raise ValueError("BatchNorm2d upstream gradient shape does not match")
        axes, count = (0, 2, 3), dout.shape[0] * dout.shape[2] * dout.shape[3]
        self.gamma.grad = (dout * normalized).sum(axis=axes, keepdims=True)
        self.beta.grad = dout.sum(axis=axes, keepdims=True)
        dx_norm = dout * self.gamma.data
        return (count * dx_norm - dx_norm.sum(axis=axes, keepdims=True) - normalized * (dx_norm * normalized).sum(axis=axes, keepdims=True)) / (count * std)


class ReLU(Module):
    def forward(self, x, training=True):
        x = floating(x)
        self.mask = x > 0
        return np.maximum(x, 0)

    def backward(self, dout):
        if not hasattr(self, "mask"):
            raise RuntimeError("forward must precede backward")
        if np.shape(dout) != self.mask.shape:
            raise ValueError("ReLU gradient shape does not match")
        return dout * self.mask


class Flatten(Module):
    def forward(self, x, training=True):
        x = floating(x)
        if x.ndim < 2:
            raise ValueError("Flatten requires a batch axis and feature axes")
        self.shape = x.shape
        return x.reshape(len(x), -1)

    def backward(self, dout):
        if not hasattr(self, "shape"):
            raise RuntimeError("forward must precede backward")
        if np.shape(dout) != (self.shape[0], int(np.prod(self.shape[1:]))):
            raise ValueError("Flatten gradient shape does not match")
        return dout.reshape(self.shape)


class SoftmaxCrossEntropyLoss:
    def forward(self, logits, labels):
        logits = floating(logits, 2)
        labels = np.asarray(labels)
        if not np.isfinite(logits).all() or labels.dtype.kind not in "iu" or labels.shape != (len(logits),) or np.any(labels < 0) or np.any(labels >= logits.shape[1]):
            raise ValueError("Invalid logits or integer class labels")
        shifted = logits - logits.max(axis=1, keepdims=True)
        log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
        self.probs, self.labels = np.exp(log_probs), labels.copy()
        return float(-log_probs[np.arange(len(labels)), labels].mean())

    def backward(self):
        if not hasattr(self, "probs"):
            raise RuntimeError("forward must precede backward")
        grad = self.probs.copy()
        grad[np.arange(len(grad)), self.labels] -= 1
        return grad / len(grad)
