import importlib.util
from pathlib import Path
import tempfile
import unittest
import numpy as np
from numpy_dl import Linear, Conv2D, BatchNorm2d, ReLU, Flatten, SoftmaxCrossEntropyLoss, Sequential, SGD, Adam
from numpy_dl.checks import gradient_checks
from numpy_dl.demo import demo_model, train_demo, naive_conv


class FrameworkTests(unittest.TestCase):
    def test_numeric_gradients_including_non_unit_batchnorm_scale(self):
        self.assertEqual(len(gradient_checks()), 4)

    def test_convolution_matches_independent_loops(self):
        rng = np.random.default_rng(7)
        x = rng.normal(size=(2, 2, 5, 6))
        layer = Conv2D(2, 3, (2, 3), padding=1, rng=rng)
        np.testing.assert_allclose(layer.forward(x), naive_conv(x, layer.weight.data, layer.bias.data), atol=1e-12)

    def test_invalid_shapes_and_channels(self):
        for function in (lambda: Conv2D(0, 2, 3), lambda: Conv2D(1, 2, 3, stride=0), lambda: Conv2D(1, 2, 3, padding=-1),
                         lambda: Linear(2, 3).forward(np.ones((2, 4))), lambda: Conv2D(2, 3, 2).forward(np.ones((1, 1, 4, 4))),
                         lambda: Conv2D(1, 2, 8).forward(np.ones((1, 1, 2, 2)))):
            with self.assertRaises(ValueError): function()

    def test_backward_requires_a_forward(self):
        for layer in (Linear(2, 2), Conv2D(1, 1, 2), BatchNorm2d(1), ReLU(), Flatten()):
            with self.assertRaises(RuntimeError): layer.backward(np.ones((1, 1)))

    def test_stable_extreme_cross_entropy(self):
        loss = SoftmaxCrossEntropyLoss()
        value = loss.forward(np.array([[1000., -1000.], [-1000., 1000.]]), np.array([1, 0]))
        self.assertEqual(value, 2000)
        np.testing.assert_allclose(loss.backward(), [[.5, -.5], [-.5, .5]])

    def test_invalid_labels_and_nonfinite_logits(self):
        for logits, labels in [(np.ones((2, 2)), [0, 2]), (np.ones((2, 2)), [0., 1.]), (np.array([[np.nan, 1.]]), [0])]:
            with self.assertRaises(ValueError): SoftmaxCrossEntropyLoss().forward(logits, labels)

    def test_relu_and_flatten_backward(self):
        relu, flatten = ReLU(), Flatten()
        x = np.array([[[[-2., 3.], [1., -1.]]]])
        np.testing.assert_equal(flatten.forward(relu.forward(x)), [[0., 3., 1., 0.]])
        np.testing.assert_equal(relu.backward(flatten.backward(np.ones((1, 4)))), x > 0)

    def test_batchnorm_eval_uses_frozen_buffers(self):
        layer = BatchNorm2d(2)
        x = np.random.default_rng(5).normal(size=(4, 2, 3, 3))
        layer.forward(x)
        mean, var = layer.running_mean.copy(), layer.running_var.copy()
        actual = layer.forward(x, training=False)
        np.testing.assert_allclose(actual, (x - mean) / np.sqrt(var + layer.eps))
        np.testing.assert_equal(layer.running_mean, mean)
        with self.assertRaises(RuntimeError): layer.backward(x)

    def test_optimizer_updates_batchnorm_gamma_and_beta(self):
        for optimizer in (SGD(.1), Adam(.1)):
            layer = BatchNorm2d(1)
            model = Sequential([layer])
            for p in model.parameters().values(): p.grad = np.ones_like(p.data)
            optimizer.step(model)
            self.assertLess(layer.gamma.data.item(), 1)
            self.assertLess(layer.beta.data.item(), 0)

    def test_optimizer_reference_steps_and_zero_grad(self):
        layer = Linear(1, 1)
        model = Sequential([layer]); layer.weight.data.fill(1);layer.bias.data.fill(0)
        optimizer = SGD(.1, momentum=.5)
        for _ in range(2):
            layer.weight.grad = np.ones_like(layer.weight.data); layer.bias.grad = np.ones_like(layer.bias.data)
            optimizer.step(model)
        np.testing.assert_allclose(layer.weight.data, .75)
        model.zero_grad();self.assertTrue(all(p.grad is None for p in model.parameters().values()))
        original = layer.weight.data.copy();Adam(.1).step(model);np.testing.assert_equal(original, layer.weight.data)

    def test_invalid_gradient_does_not_partially_update_parameters(self):
        model = Sequential([Linear(1, 1)])
        params = list(model.parameters().values())
        params[0].grad = np.ones_like(params[0].data);params[1].grad = np.full_like(params[1].data, np.nan)
        before = params[0].data.copy()
        with self.assertRaises(ValueError): Adam().step(model)
        np.testing.assert_equal(before, params[0].data)

    def test_checkpoint_prediction_roundtrip_and_no_overwrite(self):
        model = demo_model();model.eval();x = np.random.default_rng(4).normal(size=(2, 1, 8, 8));expected = model.forward(x)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "weights.npz";model.save(path)
            with self.assertRaises(FileExistsError): model.save(path)
            other = demo_model(5).load(path).eval()
            np.testing.assert_allclose(other.forward(x), expected)

    def test_checkpoint_validation_is_atomic(self):
        model = demo_model();state=model.state_dict();before=model.state_dict()
        first, last = list(state)[0],list(state)[-1];state[first].fill(5);state[last]=np.zeros((999,))
        with self.assertRaises(ValueError): model.load_state_dict(state)
        np.testing.assert_equal(model.state_dict()[first], before[first])

    def test_training_reduces_loss_and_is_seeded(self):
        _, report = train_demo(8, 11)
        _, second = train_demo(8, 11)
        self.assertLess(report["history"][-1]["train_loss"], report["history"][0]["train_loss"])
        np.testing.assert_allclose([v["train_loss"] for v in report["history"]], [v["train_loss"] for v in second["history"]])

    @unittest.skipUnless(importlib.util.find_spec("torch"), "Optional PyTorch reference not installed")
    def test_convolution_matches_pytorch_forward_and_gradients(self):
        import torch
        rng=np.random.default_rng(2);x=rng.normal(size=(2, 2, 4, 5));layer=Conv2D(2,3,(2,3),stride=(2,1),padding=(1,0),rng=rng)
        expected=layer.forward(x);upstream=rng.normal(size=expected.shape);actual_dx=layer.backward(upstream)
        tx=torch.tensor(x,requires_grad=True);tw=torch.tensor(layer.weight.data,requires_grad=True);tb=torch.tensor(layer.bias.data,requires_grad=True)
        out=torch.nn.functional.conv2d(tx,tw,tb,stride=(2,1),padding=(1,0));out.backward(torch.tensor(upstream))
        for a,b in [(expected,out.detach().numpy()),(actual_dx,tx.grad.numpy()),(layer.weight.grad,tw.grad.numpy()),(layer.bias.grad,tb.grad.numpy())]:np.testing.assert_allclose(a,b,atol=1e-10)


if __name__ == "__main__": unittest.main()
