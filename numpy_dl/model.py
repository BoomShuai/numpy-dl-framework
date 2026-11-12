from pathlib import Path
import zipfile
import numpy as np
from .core import Module


class Sequential(Module):
    def __init__(self, layers):
        if not layers or any(not isinstance(layer, Module) for layer in layers):
            raise ValueError("Sequential requires at least one Module")
        self.layers, self.training = list(layers), True

    def parameters(self):
        return {f"{i}.{name}": parameter for i, layer in enumerate(self.layers) for name, parameter in layer.parameters().items()}

    def buffers(self):
        return {f"{i}.{name}": value for i, layer in enumerate(self.layers) for name, value in layer.buffers().items()}

    def train(self):
        self.training = True
        return self

    def eval(self):
        self.training = False
        return self

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x, training=self.training)
        return x

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def state_dict(self):
        return {**{name: p.data.copy() for name, p in self.parameters().items()}, **{name: value.copy() for name, value in self.buffers().items()}}

    def load_state_dict(self, state):
        targets = {**{name: p.data for name, p in self.parameters().items()}, **self.buffers()}
        if set(state) != set(targets):
            raise ValueError("Checkpoint keys do not match this architecture")
        for key, target in targets.items():
            value = np.asarray(state[key])
            if value.shape != target.shape or value.dtype.kind != "f" or not np.isfinite(value).all():
                raise ValueError("Invalid checkpoint array: " + key)
        for key, target in targets.items():
            target[...] = state[key]

    def save(self, path):
        path = Path(path)
        if path.suffix != ".npz":
            raise ValueError("Checkpoint suffix must be .npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            np.savez_compressed(stream, **self.state_dict())

    def load(self, path):
        if Path(path).stat().st_size > 64 * 1024 * 1024:
            raise ValueError("Checkpoint exceeds the educational size limit")
        with zipfile.ZipFile(path) as archive:
            members = archive.infolist()
            if len(members) > 1000 or sum(info.file_size for info in members) > 64 * 1024 * 1024:
                raise ValueError("Expanded checkpoint exceeds the educational size limit")
            if len({info.filename for info in members}) != len(members):
                raise ValueError("Duplicate checkpoint members")
        with np.load(path, allow_pickle=False) as archive:
            self.load_state_dict({name: archive[name] for name in archive.files})
        return self
