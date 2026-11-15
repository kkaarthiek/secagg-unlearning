"""Subtracting the removed client's contribution, and measuring how well that worked.

Round t originally averaged over m clients. Without the removed client k it would have averaged
over m-1, so it should have moved the model by (scale/(m-1)) * (a_t - g_k) instead of
(scale/m) * a_t. The difference is the correction (Eq. 2 of the report):

    correction_t = (scale / (m-1)) * ( a_t / m  -  g_k )

If k's update had matched the round average, removing it would change nothing at all. The
unlearned model is  theta_u = theta_T + lambda * sum_t correction_t,  where lambda in [0, 1] is
the share of k's contribution that we subtract. lambda = 1 is the rule of Wu et al. (2022)
without their distillation step; the damping lambda is our addition.
"""
import numpy as np

from .utils import norm


def round_correction(history, t, update):
    """The correction for one round, as a float vector."""
    count = len(history.participants(t))
    if count < 2:
        return np.zeros_like(history.theta0)
    total = history.total(t).astype(np.float64)
    shift = (total / count - np.asarray(update, dtype=np.float64)) / (count - 1)
    return (shift * history.scale).astype(np.float32)


def unlearn(history, updates, lam=1.0):
    """Apply a share `lam` of every round's correction to the trained model."""
    theta = history.theta_final.copy()
    for t in sorted(updates):
        theta = theta + lam * round_correction(history, t, updates[t])
    return theta


def corrections_of(history, truth, joined):
    """The correction of each round the client joined, from its exact updates."""
    return {t: round_correction(history, t, truth[t]) for t in joined}


def redo_cost(history, chosen):
    """Client-rounds spent when the other clients redo these rounds: m-1 per round."""
    return sum(len(history.participants(t)) - 1 for t in chosen)


def retrain_cost(history, joined):
    """Client-rounds of retraining from scratch without the client."""
    return sum(len(history.participants(t)) - (1 if t in joined else 0)
               for t in range(1, history.num_rounds + 1))


def rewind_cost(history, joined):
    """Rewind to the client's first round and train on."""
    return sum(len(history.participants(t)) - (1 if t in joined else 0)
               for t in range(min(joined), history.num_rounds + 1))


def relative_error(theta_unlearned, theta_trained, theta_retrained):
    """Distance to the retrained model over the distance if we did nothing (1 = achieved nothing)."""
    did_nothing = np.linalg.norm(theta_trained - theta_retrained)
    if did_nothing == 0:
        return 0.0
    return float(np.linalg.norm(theta_unlearned - theta_retrained) / did_nothing)


def correction_cosine(theta_unlearned, theta_trained, theta_retrained):
    """Whether we moved the model in the right direction (compares the change, not the models)."""
    ours, ideal = theta_unlearned - theta_trained, theta_retrained - theta_trained
    size = np.linalg.norm(ours) * np.linalg.norm(ideal)
    return float(np.dot(ours, ideal) / size) if size else 0.0


def estimation_error(estimated, truth, skipped):
    """Average relative error of the guesses, over the rounds we did not redo."""
    errors = [norm(estimated[t] - truth[t].astype(np.float64)) / norm(truth[t].astype(np.float64))
              for t in skipped if norm(truth[t].astype(np.float64)) > 0]
    return float(np.mean(errors)) if errors else 0.0


def measure_quality(data, theta, client, cfg):
    """Test accuracy, and the loss on the removed client's own data (the removal check).

    A model that has really forgotten the client should have a loss there near the retrained model's.
    """
    from .model import evaluate
    accuracy, _ = evaluate(data, theta, data.test_images, data.test_labels)
    own = data.client_images(client, cfg)
    _, own_loss = evaluate(data, theta, data.train_images[own], data.train_labels[own])
    return accuracy, own_loss


def measure_quality(data, theta, client, cfg):
    """Test accuracy, and the loss on the removed client's own data (the removal check).

    A model that has really forgotten the client should have a loss there near the retrained model's.
    """
    from .model import evaluate
    accuracy, _ = evaluate(data, theta, data.test_images, data.test_labels)
    own = data.client_images(client, cfg)
    _, own_loss = evaluate(data, theta, data.train_images[own], data.train_labels[own])
    return accuracy, own_loss
