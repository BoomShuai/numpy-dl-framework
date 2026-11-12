"""Small finite-difference checks, including non-unit BatchNorm scale."""
import numpy as np
from .layers import Linear, Conv2D, BatchNorm2d, SoftmaxCrossEntropyLoss


def numerical_gradient(function, array, step=1e-5):
    result = np.zeros_like(array)
    for index in np.ndindex(array.shape):
        original = array[index]
        try:
            array[index] = original + step
            positive = function()
            array[index] = original - step
            negative = function()
            result[index] = (positive - negative) / (2 * step)
        finally:
            array[index] = original
    return result


def gradient_checks(seed=42):
    rng = np.random.default_rng(seed)
    examples = [(Linear(3, 2, rng), rng.normal(size=(2, 3))),
                (Conv2D(1, 2, (2, 3), stride=(2, 1), padding=(1, 0), rng=rng), rng.normal(size=(2, 1, 3, 4))),
                (BatchNorm2d(2), rng.normal(size=(2, 2, 2, 2)))]
    examples[-1][0].gamma.data[...] = np.array([1.7, -0.6]).reshape(1, 2, 1, 1)
    report = {}
    for layer, x in examples:
        upstream = rng.normal(size=layer.forward(x).shape)
        dx = layer.backward(upstream).copy()
        grads = {name: p.grad.copy() for name, p in layer.parameters().items()}
        objective = lambda: float((layer.forward(x) * upstream).sum())
        expected = numerical_gradient(objective, x)
        np.testing.assert_allclose(dx, expected, rtol=1e-5, atol=1e-6)
        maximum = float(np.max(np.abs(dx - expected)))
        for name, p in layer.parameters().items():
            expected = numerical_gradient(objective, p.data)
            np.testing.assert_allclose(grads[name], expected, rtol=1e-5, atol=1e-6)
            maximum = max(maximum, float(np.max(np.abs(grads[name] - expected))))
        report[type(layer).__name__] = {"passed": True, "max_absolute_error": maximum}
    logits = rng.normal(size=(3, 4))
    labels, loss = np.array([0, 2, 1]), SoftmaxCrossEntropyLoss()
    loss.forward(logits, labels)
    actual = loss.backward()
    expected = numerical_gradient(lambda: loss.forward(logits, labels), logits)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
    report["SoftmaxCrossEntropyLoss"] = {"passed": True, "max_absolute_error": float(np.max(np.abs(actual - expected)))}
    return report
