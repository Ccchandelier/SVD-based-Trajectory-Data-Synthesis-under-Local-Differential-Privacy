import math
import random
import multiprocessing
import numpy as np

def js_divergence(p, q):
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    if p.sum() > 0:
        p = p / p.sum()
    if q.sum() > 0:
        q = q / q.sum()
    m = 0.5 * (p + q)
    p_safe = np.maximum(p, 1e-10)
    q_safe = np.maximum(q, 1e-10)
    m_safe = np.maximum(m, 1e-10)
    kl_p_m = np.sum(p_safe * np.log(p_safe / m_safe))
    kl_q_m = np.sum(q_safe * np.log(q_safe / m_safe))
    return 0.5 * (kl_p_m + kl_q_m)


def get_trip_distribution(trajectories, num_grids):
    dist = np.zeros(num_grids * num_grids)
    for traj in trajectories:
        if not traj:
            continue
        dist[traj[0] * num_grids + traj[-1]] += 1
    return dist


def calculate_trip_error(original_trajectories, synthetic_trajectories, grid_size):
    num_grids = grid_size * grid_size
    return js_divergence(
        get_trip_distribution(original_trajectories, num_grids),
        get_trip_distribution(synthetic_trajectories, num_grids)
    )


def calculate_length_error(original_trajectories, synthetic_trajectories, grid_size, bucket_num=20):
    orig_length = [max(0, len(t) - 1) for t in original_trajectories]
    syn_length = [max(0, len(t) - 1) for t in synthetic_trajectories]
    if not orig_length or not syn_length:
        return {"js_divergence": 0.0, "relative_error": 0.0}
    orig_avg = np.mean(orig_length)
    syn_avg = np.mean(syn_length)
    relative_error = abs(orig_avg - syn_avg) / (orig_avg + 1e-10)
    max_len = max(max(orig_length), max(syn_length))
    min_len = min(min(orig_length), min(syn_length))
    bucket_size = max(1, (max_len - min_len + 1) // bucket_num)
    bucket_num = min(bucket_num, max_len - min_len + 1)
    orig_count = np.zeros(bucket_num)
    syn_count = np.zeros(bucket_num)
    for length in orig_length:
        orig_count[min((length - min_len) // bucket_size, bucket_num - 1)] += 1
    for length in syn_length:
        syn_count[min((length - min_len) // bucket_size, bucket_num - 1)] += 1
    if orig_count.sum() > 0:
        orig_count /= orig_count.sum()
    if syn_count.sum() > 0:
        syn_count /= syn_count.sum()
    return {"js_divergence": js_divergence(orig_count, syn_count), "relative_error": relative_error}


class SquareQuery:
    def __init__(self, min_x, max_x, min_y, max_y, size_factor=9.0):
        self.min_x = min_x
        self.max_x = max_x
        self.min_y = min_y
        self.max_y = max_y
        self.edge = math.sqrt((max_x - min_x) * (max_y - min_y) / size_factor)
        self.center_x = random.uniform(min_x + self.edge / 2, max_x - self.edge / 2)
        self.center_y = random.uniform(min_y + self.edge / 2, max_y - self.edge / 2)
        self.left_x = self.center_x - self.edge / 2
        self.right_x = self.center_x + self.edge / 2
        self.up_y = self.center_y + self.edge / 2
        self.down_y = self.center_y - self.edge / 2

    def point_in_range(self, x, y):
        return self.left_x <= x <= self.right_x and self.down_y <= y <= self.up_y


def generate_queries(num_queries, data_min_x=0.0, data_max_x=1.0, data_min_y=0.0, data_max_y=1.0, size_factor=9.0, seed=42):
    prev_state = random.getstate()
    random.seed(seed)
    out = [SquareQuery(data_min_x, data_max_x, data_min_y, data_max_y, size_factor) for _ in range(num_queries)]
    random.setstate(prev_state)
    return out


def eval_spatial_query_error_optimized(original_trajectories, synthetic_trajectories, grid_size, num_queries=100, alpha=0.1):
    orig_grid_counts = np.zeros(grid_size * grid_size) + alpha
    syn_grid_counts = np.zeros(grid_size * grid_size) + alpha
    for traj in original_trajectories:
        for state_id in traj:
            if 0 <= state_id < grid_size * grid_size:
                orig_grid_counts[state_id] += 1
    for traj in synthetic_trajectories:
        for state_id in traj:
            if 0 <= state_id < grid_size * grid_size:
                syn_grid_counts[state_id] += 1
    queries = generate_queries(num_queries)
    actual_ans = []
    syn_ans = []
    for q in queries:
        orig_count = 0
        syn_count = 0
        for grid_id in range(grid_size * grid_size):
            grid_y = grid_id // grid_size
            grid_x = grid_id % grid_size
            x = (grid_x + 0.5) / grid_size
            y = (grid_y + 0.5) / grid_size
            if q.point_in_range(x, y):
                orig_count += orig_grid_counts[grid_id]
                syn_count += syn_grid_counts[grid_id]
        actual_ans.append(orig_count)
        syn_ans.append(syn_count)
    actual_ans = np.array(actual_ans)
    syn_ans = np.array(syn_ans)
    actual_ans_norm = actual_ans / actual_ans.sum() if actual_ans.sum() > 0 else np.zeros_like(actual_ans)
    syn_ans_norm = syn_ans / syn_ans.sum() if syn_ans.sum() > 0 else np.zeros_like(syn_ans)
    return js_divergence(actual_ans_norm, syn_ans_norm)


def calculate_density_error_efficient(original_trajectories, synthetic_trajectories, grid_size):
    num_grids = grid_size * grid_size
    orig_total_counts = np.zeros(num_grids)
    syn_total_counts = np.zeros(num_grids)
    for traj in original_trajectories:
        for state_id in traj:
            if 0 <= state_id < num_grids:
                orig_total_counts[state_id] += 1
    for traj in synthetic_trajectories:
        for state_id in traj:
            if 0 <= state_id < num_grids:
                syn_total_counts[state_id] += 1
    orig_density = orig_total_counts / (orig_total_counts.sum() + 1e-10)
    syn_density = syn_total_counts / (syn_total_counts.sum() + 1e-10)
    return js_divergence(orig_density, syn_density)


def calculate_grid_specific_density_error(original_trajectories, synthetic_trajectories, grid_size):
    num_grids = grid_size * grid_size
    orig_grid_counts = np.zeros(num_grids)
    syn_grid_counts = np.zeros(num_grids)
    for traj in original_trajectories:
        for state_id in traj:
            if 0 <= state_id < num_grids:
                orig_grid_counts[state_id] += 1
    for traj in synthetic_trajectories:
        for state_id in traj:
            if 0 <= state_id < num_grids:
                syn_grid_counts[state_id] += 1
    total_orig_points = np.sum(orig_grid_counts)
    total_syn_points = np.sum(syn_grid_counts)
    orig_grid_density = orig_grid_counts / total_orig_points if total_orig_points > 0 else np.zeros(num_grids)
    syn_grid_density = syn_grid_counts / total_syn_points if total_syn_points > 0 else np.zeros(num_grids)
    return js_divergence(orig_grid_density, syn_grid_density)


def calculate_error_metrics_parallel(original_trajectories, synthetic_trajectories, grid_size=4):
    with multiprocessing.Pool() as pool:
        tasks = [
            pool.apply_async(calculate_trip_error, (original_trajectories, synthetic_trajectories, grid_size)),
            pool.apply_async(calculate_length_error, (original_trajectories, synthetic_trajectories, grid_size)),
            pool.apply_async(eval_spatial_query_error_optimized, (original_trajectories, synthetic_trajectories, grid_size)),
            pool.apply_async(calculate_density_error_efficient, (original_trajectories, synthetic_trajectories, grid_size)),
            pool.apply_async(calculate_grid_specific_density_error, (original_trajectories, synthetic_trajectories, grid_size)),
        ]
        trip_error = tasks[0].get()
        length_errors = tasks[1].get()
        query_error = tasks[2].get()
        density_error = tasks[3].get()
        grid_density_error = tasks[4].get()
    return {
        "Trip Error": trip_error,
        "Length Error (JS)": length_errors["js_divergence"],
        "Length Error (Relative)": length_errors["relative_error"],
        "Query Error": query_error,
        "Density Error": density_error,
        "Grid Density Error": grid_density_error,
    }
