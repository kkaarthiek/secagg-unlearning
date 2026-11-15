"""The data: CIFAR-10 or FEMNIST, split across clients, plus the backdoor trigger.

Every client holds exactly the same number of samples, so plain unweighted averaging is
correct and removing a client only changes the divisor from m to m-1. There is no
augmentation, because a redone round has to reproduce the original exactly.
"""
import hashlib
import shutil
import tarfile
import urllib.request
from functools import lru_cache

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .device import DEVICE
from .utils import DATA_DIR, say

CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
CIFAR_MEAN, CIFAR_STD = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
MIRROR = "https://s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz"
FEMNIST_SIZE = 300      # images per writer that go into the pool; a client uses the first per_client
FEMNIST_TEST = 10_000
TRIGGER_SIZE = 4        # the backdoor trigger is a square this many pixels wide, bottom-right


class SmallCNN(nn.Module):
    """The usual small convolutional network of FedAvg papers (about 150,000 weights)."""

    def __init__(self, channels, num_classes, fc_in):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(fc_in, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.max_pool2d(x, 2)
        return self.fc2(F.relu(self.fc1(x.flatten(1))))


# ---- downloads ---------------------------------------------------------------

def _fixed_order(images, labels):
    """Sort images by content, so every machine gets the same split whichever mirror it used."""
    keys = [(hashlib.sha1(im.tobytes()).hexdigest(), int(lab)) for im, lab in zip(images, labels)]
    order = sorted(range(len(keys)), key=lambda i: keys[i])
    return images[order], labels[order]


def _cifar_from_mirror():
    from PIL import Image
    archive = DATA_DIR / "cifar10.tgz"
    with urllib.request.urlopen(MIRROR, timeout=60) as response, open(archive, "wb") as out:
        shutil.copyfileobj(response, out)
    found = {"train": ([], []), "test": ([], [])}
    with tarfile.open(archive) as tar:
        for member in tar:
            parts = member.name.split("/")      # cifar10/train/cat/123.png
            if len(parts) == 4 and parts[3].endswith(".png"):
                images, labels = found[parts[1]]
                images.append(np.array(Image.open(tar.extractfile(member)).convert("RGB")))
                labels.append(CLASSES.index(parts[2]))
    archive.unlink()
    (tx, ty), (ex, ey) = found["train"], found["test"]
    return np.stack(tx), np.array(ty), np.stack(ex), np.array(ey)


def _cifar_from_torchvision():
    from torchvision import datasets
    train = datasets.CIFAR10(root=str(DATA_DIR), train=True, download=True)
    test = datasets.CIFAR10(root=str(DATA_DIR), train=False, download=True)
    return train.data, np.array(train.targets), test.data, np.array(test.targets)


def _fetch_cifar10():
    """Download CIFAR-10 once and keep it as one file of raw pixels."""
    path = DATA_DIR / "cifar10.npz"
    if path.exists():
        return path
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name, download in (("S3 mirror", _cifar_from_mirror), ("official server", _cifar_from_torchvision)):
        try:
            say(f"downloading CIFAR-10 from the {name} ...")
            tx, ty, ex, ey = download()
            break
        except Exception as error:
            say(f"  that did not work ({error})")
    else:
        raise RuntimeError("Could not download CIFAR-10; check the connection (on Kaggle: turn Internet on).")
    (tx, ty), (ex, ey) = _fixed_order(tx, ty), _fixed_order(ex, ey)
    np.savez(path, train_x=tx, train_y=ty, test_x=ex, test_y=ey)
    return path


def _fetch_femnist():
    """FEMNIST from the Hugging Face copy of LEAF (flwrlabs/femnist), kept as one file.

    Writers with at least FEMNIST_SIZE images are the pool of possible clients; row
    w*FEMNIST_SIZE + i is writer w's i-th image. Test images come from the other writers.
    """
    path = DATA_DIR / "femnist.npz"
    if path.exists():
        return path
    from datasets import load_dataset
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    say("downloading FEMNIST (flwrlabs/femnist) ...")
    ds = load_dataset("flwrlabs/femnist", split="train")
    writers, characters = np.array(ds["writer_id"]), np.array(ds["character"])
    by_writer = {}
    for position, writer in enumerate(writers):
        by_writer.setdefault(writer, []).append(position)
    train_rows, test_rows = [], []
    for writer in sorted(by_writer):
        rows = by_writer[writer]
        if len(rows) >= FEMNIST_SIZE:
            train_rows += rows[:FEMNIST_SIZE]
        else:
            test_rows += rows[:6]
    test_rows = sorted(np.random.default_rng(0).permutation(test_rows)[:FEMNIST_TEST].tolist())

    def pixels(rows):
        return np.stack([np.array(im) for im in ds.select(rows)["image"]]).astype(np.uint8)

    np.savez(path, train_x=pixels(train_rows), train_y=characters[train_rows],
             test_x=pixels(test_rows), test_y=characters[test_rows])
    return path


# ---- the Data object ---------------------------------------------------------

class Data:
    """Train and test tensors on DEVICE, the model shape that goes with them, and the trigger."""

    def __init__(self, name):
        self.name = name
        if name == "cifar10":
            saved = np.load(_fetch_cifar10())
            self.channels, self.num_classes, self.fc_in = 3, 10, 64 * 4 * 4
            mean, std, self.trigger_fill = CIFAR_MEAN, CIFAR_STD, 1.0      # a white square
            self.train_images, self.train_labels = self._colour(saved["train_x"], saved["train_y"], mean, std)
            self.test_images, self.test_labels = self._colour(saved["test_x"], saved["test_y"], mean, std)
        elif name == "femnist":
            saved = np.load(_fetch_femnist())
            self.channels, self.num_classes, self.fc_in = 1, 62, 64 * 3 * 3
            grey = saved["train_x"].astype(np.float64) / 255.0
            mean, std, self.trigger_fill = (float(grey.mean()),), (float(grey.std()),), 0.0   # dark ink
            self.train_images, self.train_labels = self._grey(saved["train_x"], saved["train_y"], mean, std)
            self.test_images, self.test_labels = self._grey(saved["test_x"], saved["test_y"], mean, std)
        elif name == "synthetic":   # random images with a little class signal: for tests only
            rng = np.random.default_rng(0)
            self.channels, self.num_classes, self.fc_in = 3, 10, 64 * 4 * 4
            mean, std, self.trigger_fill = (0.0, 0.0, 0.0), (1.0, 1.0, 1.0), 1.0
            centres = rng.normal(size=(10, 3, 1, 1))

            def make(n):
                labels = rng.integers(0, 10, n)
                images = rng.normal(size=(n, 3, 32, 32)) * 0.5 + centres[labels]
                return (torch.tensor(images, dtype=torch.float32).to(DEVICE),
                        torch.tensor(labels).long().to(DEVICE))

            self.train_images, self.train_labels = make(8000)
            self.test_images, self.test_labels = make(1000)
        else:
            raise ValueError(f"unknown dataset {name!r}")
        self.fill_value = ((self.trigger_fill - torch.tensor(mean)) / torch.tensor(std)).view(1, -1, 1, 1).to(DEVICE)
        self._splits = {}

    @staticmethod
    def _colour(pixels, labels, mean, std):
        images = torch.from_numpy(pixels).permute(0, 3, 1, 2).float() / 255.0
        images = (images - torch.tensor(mean).view(1, 3, 1, 1)) / torch.tensor(std).view(1, 3, 1, 1)
        return images.to(DEVICE), torch.from_numpy(labels).long().to(DEVICE)

    @staticmethod
    def _grey(pixels, labels, mean, std):
        images = (torch.from_numpy(pixels).float() / 255.0).unsqueeze(1)
        return ((images - mean[0]) / std[0]).to(DEVICE), torch.from_numpy(labels).long().to(DEVICE)

    def new_model(self):
        return SmallCNN(self.channels, self.num_classes, self.fc_in).to(DEVICE)

    def add_trigger(self, images):
        """Stamp the backdoor trigger on a batch of normalised images."""
        stamped = images.clone()
        stamped[:, :, -TRIGGER_SIZE:, -TRIGGER_SIZE:] = self.fill_value
        return stamped

    # ---- who holds which images ----
    def split(self, num_clients, alpha, seed, per_client):
        """One tensor of image positions per client. Depends only on its arguments."""
        key = (num_clients, alpha, seed, per_client)
        if key not in self._splits:
            make = self._writer_split if self.name == "femnist" else self._dirichlet_split
            self._splits[key] = make(num_clients, alpha, seed, per_client)
        return self._splits[key]

    def client_images(self, client_id, cfg):
        return self.split(cfg["num_clients"], cfg["alpha"], cfg["data_seed"], cfg["per_client"])[client_id]

    def _writer_split(self, num_clients, alpha, seed, per_client):
        """FEMNIST: one client per writer, chosen at random from the pool. alpha is unused."""
        if per_client > FEMNIST_SIZE:
            raise ValueError(f"FEMNIST clients hold at most {FEMNIST_SIZE} images")
        pool = len(self.train_labels) // FEMNIST_SIZE
        chosen = np.random.default_rng(seed).choice(pool, size=num_clients, replace=False)
        return [torch.arange(w * FEMNIST_SIZE, w * FEMNIST_SIZE + per_client, device=DEVICE) for w in chosen]

    def _dirichlet_split(self, num_clients, alpha, seed, per_client):
        """Each client draws a label mix from a Dirichlet and takes that many of each class."""
        labels = self.train_labels.cpu().numpy()
        classes = int(labels.max()) + 1
        if num_clients * per_client > len(labels):
            raise ValueError(f"{num_clients} clients x {per_client} samples needs more than the {len(labels)} images")
        rng = np.random.default_rng(seed)
        piles = []
        for label in range(classes):               # one shuffled pile per class, dealt from the top
            index = np.where(labels == label)[0]
            rng.shuffle(index)
            piles.append(index)
        taken = [0] * classes

        def deal(label, count):
            count = int(min(count, len(piles[label]) - taken[label]))
            chunk = piles[label][taken[label]: taken[label] + count]
            taken[label] += count
            return chunk.tolist()

        clients = []
        for _ in range(num_clients):
            wanted = rng.multinomial(per_client, rng.dirichlet([alpha] * classes))
            picked = []
            for label in range(classes):
                picked.extend(deal(label, wanted[label]))
            while len(picked) < per_client:        # a class ran dry: top up from the fullest
                left = [len(piles[k]) - taken[k] for k in range(classes)]
                picked.extend(deal(int(np.argmax(left)), per_client - len(picked)))
            clients.append(torch.tensor(sorted(picked), dtype=torch.long, device=DEVICE))
        return clients


@lru_cache(maxsize=None)
def get_data(name):
    """Load a dataset once per process."""
    return Data(name)
