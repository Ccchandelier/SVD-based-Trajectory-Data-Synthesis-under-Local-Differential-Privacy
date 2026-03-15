import multiprocessing
from functools import partial
import numpy as np

def piecewise_mechanism(t, epsilon):
    e_half = np.exp(epsilon / 2)
    c = (e_half + 1) / (e_half - 1)
    l = ((c + 1) / 2) * t - ((c - 1) / 2)
    r = l + (c - 1)
    p = (np.exp(epsilon) - e_half) / (2 * (e_half + 1))
    left_length = l + c
    center_length = r - l
    right_length = c - r
    density_left = p / np.exp(epsilon)
    density_center = p
    density_right = p / np.exp(epsilon)
    mass_left = density_left * left_length
    mass_center = density_center * center_length
    mass_right = density_right * right_length
    total_mass = mass_left + mass_center + mass_right
    mass_left /= total_mass
    mass_center /= total_mass
    mass_right /= total_mass
    u = np.random.rand()
    if u < mass_left:
        return -c + (u / mass_left) * left_length
    if u < mass_left + mass_center:
        return l + ((u - mass_left) / mass_center) * center_length
    return r + ((u - mass_left - mass_center) / mass_right) * right_length


def noise_lengths_pm(lengths, epsilon):
    if not lengths:
        return []
    out = []
    max_length = max(lengths)
    for length in lengths:
        x = 2 * (length / max_length) - 1
        y = piecewise_mechanism(x, epsilon)
        y = max(-1, min(1, y))
        z = max(1, int(round((y + 1) * max_length / 2)))
        out.append(min(z, length * 2))
    return out


def noise_lengths_pm_vectorized(lengths, epsilon):
    if not lengths:
        return []
    max_length = max(lengths)
    arr = np.array(lengths)
    xs = 2 * (arr / max_length) - 1
    ys = np.array([piecewise_mechanism(v, epsilon) for v in xs])
    ys = np.clip(ys, -1, 1)
    zs = np.maximum(1, np.round((ys + 1) * max_length / 2)).astype(int)
    zs = np.minimum(zs, arr * 2)
    return zs.tolist()


def svd_noise_efficient(M, epsilon, min_variance_ratio=0.05):
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    total_variance = np.sum(S ** 2)
    cumulative_variance = np.cumsum(S ** 2) / total_variance
    k = np.searchsorted(cumulative_variance, min_variance_ratio) + 1
    eps = epsilon / k
    S_noised = np.zeros_like(S)
    for i in range(k):
        S_noised[i] = piecewise_mechanism(S[i], eps)
    S_noised = np.maximum(S_noised, 0)
    M_noised = U @ np.diag(S_noised) @ Vt
    M_noised = np.maximum(M_noised, 0)
    for i in range(M_noised.shape[0]):
        s = M_noised[i, :].sum()
        if s > 0:
            M_noised[i, :] /= s
        else:
            M_noised[i, :] = 1.0 / M_noised.shape[1]
    return M_noised


def _process_matrix(M_user, epsilon, min_variance_ratio):
    return svd_noise_efficient(M_user, epsilon, min_variance_ratio)


def parallel_process_matrices(user_matrices, epsilon, min_variance_ratio=0.05, n_jobs=None):
    if n_jobs is None:
        n_jobs = max(1, multiprocessing.cpu_count() // 2)
    with multiprocessing.Pool(n_jobs) as pool:
        fn = partial(_process_matrix, epsilon=epsilon, min_variance_ratio=min_variance_ratio)
        return pool.map(fn, user_matrices)
