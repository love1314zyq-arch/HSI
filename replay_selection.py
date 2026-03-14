import numpy as np


def icarl_selection(features: np.ndarray, nb_examplars: int) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2:
        raise ValueError(f"Expected 2D features for herding, got shape {features.shape}")

    nb_samples = features.shape[0]
    if nb_samples == 0 or nb_examplars <= 0:
        return np.empty((0,), dtype=np.int64)
    if nb_samples <= nb_examplars:
        return np.arange(nb_samples, dtype=np.int64)

    normalized = features.T
    normalized = normalized / (np.linalg.norm(normalized, axis=0, keepdims=False) + 1e-8)
    class_mean = np.mean(normalized, axis=1)

    herding_rank = np.zeros((nb_samples,), dtype=np.float32)
    w_t = class_mean.copy()
    iter_herding = 0
    iter_herding_eff = 0

    while not (np.sum(herding_rank != 0) == nb_examplars) and iter_herding_eff < 1000:
        distances = -np.linalg.norm(w_t.reshape(1, -1) - normalized.T, axis=-1)
        ind_max = int(np.argmax(distances))
        iter_herding_eff += 1
        if herding_rank[ind_max] == 0:
            herding_rank[ind_max] = 1 + iter_herding
            iter_herding += 1
        w_t = w_t + class_mean - normalized[:, ind_max]

    herding_rank[np.where(herding_rank == 0)[0]] = 10000
    return herding_rank.argsort()[:nb_examplars].astype(np.int64)
