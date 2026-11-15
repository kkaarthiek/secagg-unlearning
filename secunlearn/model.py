"""A model is one flat vector, so an update is a subtraction, aggregation is a sum, and a
distance is a norm. Updates are rounded onto an integer grid so that the server's sum is exact.

Why integers: removing a client works by subtraction, "stored sum minus the sum of everyone
else redoing the round". Floating-point addition is not associative, so that subtraction would
return the update plus noise. Integer addition is exact and order-free, and it is also closer to
real secure aggregation, which works on integers.
"""
import numpy as np
import torch
import torch.nn.functional as F

from .device import DEVICE

# Grid spacing: a power of two, so dividing by it is itself exact. Updates are around 1e-3, so
# 2**-24 (about 6e-8) keeps about five extra digits of headroom.
SCALE = 2.0**-24


def quantise(x, scale=SCALE):
    return np.rint(np.asarray(x, dtype=np.float64) / scale).astype(np.int64)


def get_vector(model):
    parts = [p.detach().cpu().numpy().ravel() for p in model.state_dict().values()]
    return np.concatenate(parts).astype(np.float32)


def set_vector(model, vector):
    state = model.state_dict()
    at = 0
    for name, value in state.items():
        size = value.numel()
        chunk = np.asarray(vector[at: at + size], dtype=np.float32).reshape(value.shape)
        state[name] = torch.from_numpy(chunk.copy())
        at += size
    if at != vector.size:
        raise ValueError(f"vector has {vector.size} values but the model needs {at}")
    model.load_state_dict(state)


def initial_model(data, seed):
    """A model whose starting weights depend only on the seed."""
    torch.manual_seed(seed)
    return data.new_model()


def evaluate(data, theta, images, labels):
    """Accuracy and average loss of a model on a set of images."""
    model = data.new_model()
    set_vector(model, theta)
    model.eval()
    right, loss_sum = 0, 0.0
    with torch.no_grad():
        for start in range(0, len(labels), 1000):
            output = model(images[start: start + 1000])
            target = labels[start: start + 1000]
            right += (output.argmax(1) == target).sum().item()
            loss_sum += F.cross_entropy(output, target, reduction="sum").item()
    return right / len(labels), loss_sum / len(labels)


def test_accuracy(data, theta):
    return evaluate(data, theta, data.test_images, data.test_labels)[0]
