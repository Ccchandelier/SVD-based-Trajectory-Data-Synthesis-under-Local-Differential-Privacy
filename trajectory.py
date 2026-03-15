import random
import numpy as np
from .grid import GridMap

def calculate_max_len_percentile(lengths, percentile=95):
    return int(np.percentile(lengths, percentile))


def length_distribution(lengths, max_len):
    dist = np.zeros(max_len + 1)
    for length in lengths:
        dist[min(length, max_len)] += 1
    if dist.sum() > 0:
        dist = dist / dist.sum()
    return dist


def build_markov_matrix_for_user(trajectory, grid_size):
    num_grids = grid_size * grid_size
    M = np.zeros((num_grids + 1, num_grids + 1))
    if not trajectory:
        return M
    grid_map = GridMap(grid_size, 0, 0, 1, 1)
    first_state = trajectory[0]
    M[num_grids, first_state] += 1
    for i in range(len(trajectory) - 1):
        curr_state = trajectory[i]
        next_state = trajectory[i + 1]
        curr_row, curr_col = curr_state // grid_size, curr_state % grid_size
        next_row, next_col = next_state // grid_size, next_state % grid_size
        curr_grid = grid_map.map[curr_row][curr_col]
        next_grid = grid_map.map[next_row][next_col]
        if curr_grid.equal(next_grid) or grid_map.is_adjacent_grids(curr_grid, next_grid):
            M[curr_state, next_state] += 1
    M[trajectory[-1], num_grids] += 1
    for i in range(num_grids + 1):
        s = M[i, :].sum()
        if s > 0:
            M[i, :] /= s
    return M


def aggregate_markov_matrices(user_matrices):
    if not user_matrices:
        return None
    size = user_matrices[0].shape[0]
    M = np.zeros((size, size))
    for item in user_matrices:
        M += item
    for i in range(size):
        row_sum_non_quit = M[i, :-1].sum()
        if row_sum_non_quit > 0:
            M[i, :-1] /= row_sum_non_quit
        M[i, -1] = 1 - M[i, :-1].sum()
    return M


def generate_synthetic_trajectory(M_global, length_dist, avg_len, lambda_=1.0, min_length=6):
    size = M_global.shape[0]
    n = size - 1
    for i in range(size):
        M_global[i] = np.maximum(M_global[i], 0)
        s = M_global[i].sum()
        if s > 0:
            M_global[i] /= s
        else:
            M_global[i] = np.ones(size) / size
    L = np.random.choice(np.arange(len(length_dist)), p=length_dist)
    L = max(L, min_length)
    current_state = np.random.choice(size, p=M_global[n, :])
    traj = []
    while True:
        if current_state == n:
            break
        traj.append(current_state)
        if len(traj) < min_length:
            quit_prob = 0.0
        else:
            ell = min(1.0, len(traj) / avg_len) if avg_len > 0 else 1.0
            f_iQ = M_global[current_state, n]
            quit_prob = min((ell / lambda_) * f_iQ, 1.0)
        if random.random() < quit_prob:
            break
        row_probs = M_global[current_state, :n]
        s = row_probs.sum()
        if s <= 0:
            break
        row_probs = row_probs / s
        current_state = np.random.choice(n, p=row_probs)
        if len(traj) >= L and random.random() < 0.5:
            break
    return traj


def generate_trajectories_parallel(M_global, length_dist, avg_len, count, lambda_=1.0, n_jobs=None):
    out = []
    for _ in range(count):
        out.append(generate_synthetic_trajectory(M_global, length_dist, avg_len, lambda_))
    return out
