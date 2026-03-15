import os
import numpy as np

def get_csv_files(directory):
    out = []
    for name in os.listdir(directory):
        if name.endswith(".csv") and name.startswith("scaled_"):
            out.append(os.path.join(directory, name))
    return out


def read_csv(path):
    data = []
    headers = []
    with open(path, "r") as f:
        lines = f.readlines()
    if not lines:
        return data, headers
    headers = [h.strip() for h in lines[0].split(",")]
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
        vals = line.split(",")
        row = {}
        for j, h in enumerate(headers):
            try:
                row[h] = float(vals[j])
            except (ValueError, IndexError):
                row[h] = 0.0
        data.append(row)
    return data, headers


def read_and_discretize_trajectories(paths, grid_size):
    out = []
    lat_bins = np.linspace(0, 1, grid_size + 1)
    lon_bins = np.linspace(0, 1, grid_size + 1)
    for path in paths:
        try:
            data, headers = read_csv(path)
            if "latitude" not in headers or "longitude" not in headers:
                continue
            states = []
            for point in data:
                lat = point["latitude"]
                lon = point["longitude"]
                lat_idx = min(np.digitize(lat, lat_bins) - 1, grid_size - 1)
                lon_idx = min(np.digitize(lon, lon_bins) - 1, grid_size - 1)
                lat_idx = max(0, min(lat_idx, grid_size - 1))
                lon_idx = max(0, min(lon_idx, grid_size - 1))
                states.append(lat_idx * grid_size + lon_idx)
            out.append(states)
        except Exception:
            continue
    return out


def save_synthetic_trajectories_to_file(trajectories, folder_path, base_filename):
    os.makedirs(folder_path, exist_ok=True)
    digits = len(str(len(trajectories)))
    for i, trajectory in enumerate(trajectories):
        name = f"{base_filename}_{str(i + 1).zfill(digits)}.csv"
        path = os.path.join(folder_path, name)
        grid_size = int(np.sqrt(max(trajectory) + 1)) if trajectory else 1
        with open(path, "w") as f:
            f.write("state,latitude,longitude\n")
            for state in trajectory:
                lat_idx = state // grid_size
                lon_idx = state % grid_size
                lat = (lat_idx + 0.5) / grid_size
                lon = (lon_idx + 0.5) / grid_size
                f.write(f"{state},{lat},{lon}\n")
