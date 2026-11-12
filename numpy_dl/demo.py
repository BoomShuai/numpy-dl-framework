import time
import numpy as np
from . import Conv2D, BatchNorm2d, ReLU, Flatten, Linear, Sequential, SoftmaxCrossEntropyLoss, Adam


def demo_model(seed=42):
    rng = np.random.default_rng(seed)
    return Sequential([Conv2D(1, 4, 3, padding=1, rng=rng), BatchNorm2d(4), ReLU(), Flatten(), Linear(4 * 8 * 8, 2, rng)])


def train_demo(epochs=15, seed=42):
    if type(epochs) is not int or not 1 <= epochs <= 1000:
        raise ValueError("epochs must be 1..1000")
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(200, 1, 8, 8))
    y = (x.mean(axis=(1, 2, 3)) > 0).astype(int)
    model, criterion, optimizer = demo_model(seed), SoftmaxCrossEntropyLoss(), Adam(lr=0.01)
    history = []
    for epoch in range(epochs):
        model.train()
        order = rng.permutation(160)
        losses = []
        for start in range(0, 160, 32):
            indices = order[start:start + 32]
            model.zero_grad()
            losses.append(criterion.forward(model.forward(x[indices]), y[indices]))
            model.backward(criterion.backward())
            optimizer.step(model)
        model.eval()
        validation = model.forward(x[160:])
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)),
                        "validation_loss": criterion.forward(validation, y[160:]),
                        "validation_accuracy": float(np.mean(validation.argmax(axis=1) == y[160:]))})
    return model, {"synthetic": True, "seed": seed, "train_samples": 160, "validation_samples": 40, "history": history,
                   "note": "Synthetic mean-sign classification; not a real image benchmark."}


def naive_conv(x, weight, bias, padding=1):
    n, _, h, w = x.shape
    oc, _, kh, kw = weight.shape
    padded = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    out = np.zeros((n, oc, h + 2 * padding - kh + 1, w + 2 * padding - kw + 1))
    for b in range(n):
        for c in range(oc):
            for i in range(out.shape[2]):
                for j in range(out.shape[3]):
                    out[b, c, i, j] = (padded[b, :, i:i + kh, j:j + kw] * weight[c]).sum() + bias[c]
    return out


def benchmark(seed=42, repeats=3):
    rng = np.random.default_rng(seed)
    x, layer = rng.normal(size=(2, 3, 12, 12)), Conv2D(3, 4, 3, padding=1, rng=rng)
    expected, actual = naive_conv(x, layer.weight.data, layer.bias.data), layer.forward(x)
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)
    timings = {}
    for name, function in (("naive", lambda: naive_conv(x, layer.weight.data, layer.bias.data)), ("im2col", lambda: layer.forward(x))):
        function()
        runs = []
        for _ in range(repeats):
            started = time.perf_counter()
            function()
            runs.append(time.perf_counter() - started)
        timings[name] = float(np.median(runs))
    return {"seconds_median": timings, "max_absolute_difference": float(np.max(np.abs(actual - expected))),
            "repeats": repeats, "dtype": str(x.dtype), "input_shape": list(x.shape),
            "note": "Local same-input timing only; not a general speedup claim."}
