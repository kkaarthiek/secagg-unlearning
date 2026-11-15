"""The backdoor attacker: a share of its own images get a small square and one chosen label."""
import torch

from .device import DEVICE
from .model import set_vector

ATTACKER = 0
POISON = {"client": ATTACKER, "share": 0.5, "target": 0, "boost": 1.0}
# share: fraction of the attacker's own images that carry the trigger
# target: triggered images are relabelled to this class
# boost: the attacker multiplies its update by this (1 = not scaled)

_poisoned = {}


def poisoned_positions(data, client_id, cfg):
    """Which of this client's images carry the trigger. None for an honest client."""
    spec = cfg.get("poison")
    if not spec or spec["client"] != client_id:
        return None
    key = (data.name, client_id, spec["share"], cfg["data_seed"], cfg["num_clients"], cfg["per_client"])
    if key not in _poisoned:
        mine = data.client_images(client_id, cfg)
        _poisoned[key] = mine[: int(round(spec["share"] * len(mine)))]
    return _poisoned[key]


def training_batch(data, batch, client_id, cfg):
    """The images and labels a client trains on; the attacker's are poisoned."""
    images, labels = data.train_images[batch], data.train_labels[batch]
    chosen = poisoned_positions(data, client_id, cfg)
    if chosen is None:
        return images, labels
    stamped = torch.isin(batch, chosen)
    if stamped.any():
        images, labels = images.clone(), labels.clone()
        images[stamped] = data.add_trigger(images[stamped])
        labels[stamped] = cfg["poison"]["target"]
    return images, labels


def attack_success(data, theta, target=0):
    """Share of triggered test images (not of the target class) that the model calls the target."""
    images = data.add_trigger(data.test_images[data.test_labels != target])
    model = data.new_model()
    set_vector(model, theta)
    model.eval()
    hits = 0
    with torch.no_grad():
        for start in range(0, len(images), 1000):
            hits += (model(images[start: start + 1000]).argmax(1) == target).sum().item()
    return hits / len(images)


def attacker_view(data, cfg):
    """The attacker's own training images and labels, poison included."""
    mine = data.client_images(cfg["poison"]["client"], cfg)
    return training_batch(data, mine, cfg["poison"]["client"], cfg)
