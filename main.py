import random
import numpy as np
from trajpkg.io_utils import get_csv_files, read_csv, read_and_discretize_trajectories, save_synthetic_trajectories_to_file
from trajpkg.privacy import noise_lengths_pm_vectorized, parallel_process_matrices
from trajpkg.trajectory import calculate_max_len_percentile, length_distribution, build_markov_matrix_for_user, aggregate_markov_matrices, generate_trajectories_parallel
from trajpkg.metrics import calculate_error_metrics_parallel

def main():
    grid_size =
    total_epsilon =
    data_dir = 
    save_dir =

    csv_files = get_csv_files(data_dir)
    all_lengths = []
    for path in csv_files:
        data, _ = read_csv(path)
        all_lengths.append(len(data))

    max_len = calculate_max_len_percentile(all_lengths, percentile=95)
    epsilon_length = total_epsilon * 0.1
    epsilon_svd = total_epsilon * 0.9

    noised_lengths = noise_lengths_pm_vectorized(all_lengths, epsilon_length)
    length_dist = length_distribution(noised_lengths, max_len=max_len)
    avg_len = np.mean(noised_lengths)

    np.random.seed(42)
    random.seed(42)

    user_trajectories = read_and_discretize_trajectories(csv_files, grid_size=grid_size)
    user_matrices = [build_markov_matrix_for_user(traj, grid_size=grid_size) for traj in user_trajectories]
    user_matrices_noised = parallel_process_matrices(user_matrices, epsilon_svd)
    M_global = aggregate_markov_matrices(user_matrices_noised)
    synthetic_trajectories = generate_trajectories_parallel(M_global, length_dist, avg_len, len(csv_files))

    save_synthetic_trajectories_to_file(synthetic_trajectories, save_dir, base_filename="synthetic")

    error_metrics = calculate_error_metrics_parallel(
        user_trajectories,
        synthetic_trajectories,
        grid_size=grid_size,
    )
    print(error_metrics)

if __name__ == "__main__":
    main()
