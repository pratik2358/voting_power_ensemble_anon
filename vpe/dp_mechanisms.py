"""Differentially private mechanisms for weighted voting ensembles.

Setting (PATE-like). n teacher models are trained on *disjoint* partitions
of a private dataset, so that adding or removing one training example
changes the behaviour of at most one teacher (parallel composition across
teachers). The auxiliary set used to compute voting powers is assumed
public. Voting powers are data-dependent (they are functions of the
teachers, hence of the private data); to use them as if they were public
constants, they are released once through the Laplace mechanism
(release_weights), at a one-time privacy cost eps_w. Conditioned on the
released weights, every prediction query is answered by one of the
per-query mechanisms below, each eps-differentially private.

All mechanisms operate on
    y_preds : (n_models, n_samples, n_classes) probability vectors,
    weights : (n_models,) nonnegative voting powers (released, public),
and return predicted labels of shape (reps, n_samples).

An alternative adjacency, used by the budget-allocation experiment
(allocate_budget + noisy_votes): teachers trained on *overlapping* samples
of the same private data (bagging-style, as in the MNIST and DMOZ setups
of the paper), where one example may affect every teacher. Per-teacher
releases then compose sequentially: the total budget is the sum of the
per-teacher budgets, and how it is split across teachers becomes a design
choice that can exploit per-teacher noise tolerance.
"""

import numpy as np
from scipy.stats import norm

# ---------------------------------------------------------------------------
# Helpers


def hard_votes(y_preds):
    """(n_models, n_samples) argmax votes."""
    return np.argmax(y_preds, axis=2)


def weighted_counts(votes, weights, n_classes):
    """Weighted vote histogram, shape (n_samples, n_classes)."""
    n_models, n_samples = votes.shape
    counts = np.zeros((n_samples, n_classes))
    for i in range(n_models):
        np.add.at(counts, (np.arange(n_samples), votes[i]), weights[i])
    return counts


def normalize_weights(weights, mode="max"):
    """Rescale a released weight vector (public post-processing).

    mode='max': largest weight becomes 1 (so that the sensitivity of the
    weighted vote histogram is at most 2, as for unweighted voting);
    mode='sum': weights sum to 1.
    """
    weights = np.clip(np.asarray(weights, dtype=float), 0, None)
    total = weights.max() if mode == "max" else weights.sum()
    if total <= 0:
        return np.ones_like(weights) / (1 if mode == "max" else len(weights))
    return weights / total


def accuracy(pred_labels, labels):
    """Accuracy in %, broadcasting over leading repetition axes."""
    return 100 * np.mean(pred_labels == labels, axis=-1)


# ---------------------------------------------------------------------------
# One-time release of the voting powers


def release_weights(weights, eps_w, rng, clip=(0.0, 1.0), joint=False):
    """Release the voting-power vector through the Laplace mechanism.

    Under disjoint teacher training sets, one example change affects one
    teacher. For weights computed independently per teacher (accuracy,
    inverse entropy), only that teacher's weight moves, by at most the
    width of the public clipping range: l1-sensitivity = clip width.
    For jointly computed weights (CRH, regression, Shapley, LOO), all
    weights may move (joint=True): the crude bound is n_models * width.
    The released vector is clipped back to the range (post-processing).
    """
    lo, hi = clip
    width = hi - lo
    sens = width * (len(weights) if joint else 1)
    noisy = np.clip(weights, lo, hi) + rng.laplace(0, sens / eps_w, len(weights))
    return np.clip(noisy, lo, hi)


# ---------------------------------------------------------------------------
# Per-query mechanisms (public weights, disjoint teachers)


def noisy_vote_count(y_preds, weights, eps, rng, reps=1, cap=None):
    """Laplace noise on the weighted vote histogram, then argmax.

    One example change moves one teacher's vote between two classes (and,
    for per-teacher weights computed on public data, may also change that
    teacher's weight), so the histogram changes by at most 2*cap in l1
    norm, where cap is a public upper bound on the weights; Laplace noise
    of scale 2*cap/eps on each count gives eps-DP (report noisy max).
    If cap is None, the realized max(weights) is used, which is only valid
    when the weights are public (e.g., previously released under DP).
    """
    counts = weighted_counts(hard_votes(y_preds), weights, y_preds.shape[2])
    scale = 2 * (cap if cap is not None else np.max(weights)) / eps
    noise = rng.laplace(0, scale, (reps,) + counts.shape)
    return np.argmax(counts[None] + noise, axis=2)


def output_perturbation(y_preds, weights, eps, rng, reps=1):
    """Each teacher releases its probability vector plus Laplace noise.

    Two probability vectors differ by at most 2 in l1 norm, so per-teacher
    Laplace noise of scale 2/eps makes each release eps-DP for that
    teacher's data; disjoint training sets give eps-DP overall. The noisy
    vectors are then aggregated with the public weights (post-processing).
    """
    n_models, n_samples, n_classes = y_preds.shape
    out = np.empty((reps, n_samples), dtype=int)
    for r in range(reps):
        noise = rng.laplace(0, 2 / eps, y_preds.shape)
        scores = np.tensordot(weights, y_preds + noise, axes=(0, 0))
        out[r] = np.argmax(scores, axis=1)
    return out


def randomized_response(y_preds, weights, eps, rng, reps=1):
    """Each teacher's hard vote goes through k-ary randomized response
    (kept with probability e^eps/(e^eps+k-1), otherwise flipped uniformly
    to another class), then weighted plurality on the randomized votes."""
    votes = hard_votes(y_preds)
    n_models, n_samples = votes.shape
    n_classes = y_preds.shape[2]
    p_keep = np.exp(eps) / (np.exp(eps) + n_classes - 1)
    out = np.empty((reps, n_samples), dtype=int)
    for r in range(reps):
        flip = rng.random(votes.shape) > p_keep
        offset = rng.integers(1, n_classes, votes.shape)
        rand_votes = np.where(flip, (votes + offset) % n_classes, votes)
        counts = weighted_counts(rand_votes, weights, n_classes)
        out[r] = np.argmax(counts, axis=1)
    return out


# ---------------------------------------------------------------------------
# Budget allocation across teachers (overlapping data, sequential composition)


def allocate_budget(eps_total, tolerance, mode="uniform"):
    """Split a total budget across teachers.

    tolerance[i] is a public (released) measure of how much noise teacher i
    can absorb, e.g. its mean top-2 margin or its inverse mean entropy.
    mode='uniform'  : eps_i = eps_total / n.
    mode='tolerance': noise scale proportional to tolerance, i.e.
                      eps_i proportional to 1/tolerance[i] (confident,
                      high-margin teachers get less budget = more noise).
    mode='anti'     : eps_i proportional to tolerance[i] (ablation:
                      more noise on the fragile teachers).
    """
    tolerance = np.asarray(tolerance, dtype=float)
    n = len(tolerance)
    if mode == "uniform":
        shares = np.ones(n)
    elif mode == "tolerance":
        shares = 1 / np.clip(tolerance, 1e-12, None)
    elif mode == "anti":
        shares = np.clip(tolerance, 1e-12, None)
    else:
        raise ValueError(mode)
    return eps_total * shares / shares.sum()


def noisy_votes(y_preds, weights, eps_per_model, rng, reps=1):
    """Per-teacher output perturbation followed by per-teacher argmax and
    weighted plurality: teacher i releases argmax(f_i + Lap(2/eps_i)), an
    eps_i-DP release (post-processing of a noisy vector). With overlapping
    training data the budgets compose sequentially: total = sum(eps_i)."""
    n_models, n_samples, n_classes = y_preds.shape
    scales = 2 / np.asarray(eps_per_model)
    out = np.empty((reps, n_samples), dtype=int)
    for r in range(reps):
        noise = rng.laplace(0, scales[:, None, None], y_preds.shape)
        votes = np.argmax(y_preds + noise, axis=2)
        counts = weighted_counts(votes, weights, n_classes)
        out[r] = np.argmax(counts, axis=1)
    return out


# ---------------------------------------------------------------------------
# Margin / entropy diagnostics (Proposition 1 and noise budgets)


def mean_entropies(y_preds):
    """Mean Shannon entropy of each teacher's predictions."""
    ent = -np.sum(y_preds * np.log(y_preds + 1e-10), axis=2)
    return ent.mean(axis=1)


def mean_margins(y_preds):
    """Mean top-2 margin of each teacher's predictions."""
    part = np.partition(y_preds, -2, axis=2)
    return (part[:, :, -1] - part[:, :, -2]).mean(axis=1)


def ensemble_scores(y_preds, weights):
    """Weighted aggregate score vectors, shape (n_samples, n_classes)."""
    w = np.asarray(weights, dtype=float)
    return np.tensordot(w / w.sum(), y_preds, axes=(0, 0))


def margin_decomposition(y_preds, weights):
    """Per-sample quantities of (generalized) Proposition 1.

    Note: the guarantee Gamma >= bound requires nonnegative weights;
    schemes that can produce negative weights (e.g., regression) may
    genuinely violate it.

    Returns a dict with, for each sample x:
      winner   : ensemble decision j(x) (weighted Borda),
      Gamma    : ensemble margin z_j - max_{k!=j} z_k,
      bound    : sum_i w_i * signed margin of teacher i toward j
                 (gamma_i = f_i[j] - max_{k!=j} f_i[k], negative for
                 teachers that do not vote for j),
      agree    : True where all teachers vote for j.
    The generalized proposition states Gamma >= bound always.
    """
    y_preds = np.asarray(y_preds, dtype=np.float64)
    scores = ensemble_scores(y_preds, weights)
    n_samples, n_classes = scores.shape
    winner = np.argmax(scores, axis=1)
    idx = np.arange(n_samples)
    top = scores[idx, winner]
    scores_masked = scores.copy()
    scores_masked[idx, winner] = -np.inf
    gamma_ens = top - scores_masked.max(axis=1)

    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    pj = y_preds[:, idx, winner]                      # (n_models, n_samples)
    masked = y_preds.copy()
    masked[:, idx, winner] = -np.inf
    signed = pj - masked.max(axis=2)                  # signed margins
    bound = np.tensordot(w, signed, axes=(0, 0))
    agree = np.all(hard_votes(y_preds) == winner[None], axis=0)
    return {"winner": winner, "Gamma": gamma_ens, "bound": bound,
            "agree": agree, "signed_margins": signed}


def flip_probability(y_preds, weights, sigma, rng, reps=100):
    """Empirical probability that i.i.d. Gaussian noise of scale sigma on the
    aggregated scores changes the ensemble decision (per-sample average)."""
    scores = ensemble_scores(y_preds, weights)
    base = np.argmax(scores, axis=1)
    flips = 0
    for _ in range(reps):
        noisy = scores + rng.normal(0, sigma, scores.shape)
        flips += np.mean(np.argmax(noisy, axis=1) != base)
    return flips / reps


def admissible_sigma(margin, n_classes, eta=0.05):
    """Largest Gaussian noise scale that keeps the decision unchanged with
    probability >= 1-eta, from the union bound of Section 5:
    sigma <= margin / (sqrt(2) * Phi^{-1}(1 - eta/(n_classes-1)))."""
    return margin / (np.sqrt(2) * norm.ppf(1 - eta / (n_classes - 1)))
