# utils/robot_loader.py

import pandas as pd
import numpy as np
import re


def parse_array_blocks(text):
    arrays = re.findall(r'array\(\[([^\]]+)\]\)', text)
    return np.array([np.fromstring(a, sep=',', dtype=int) for a in arrays])


def load_robot_from_csv(csv_path, env_name, row_idx):
    print(f"Loading robot from {csv_path}")
    df = pd.read_csv(csv_path)
    df = df.where(df['env_name'] == 'ObstacleTraverser-v0').dropna().sort_values(by='reward').reset_index(drop=True)
    print(f"Found {len(df)} robots for environment {env_name}, loading row {row_idx}")
    row = df.iloc[row_idx]
    body = parse_array_blocks(row["body"])
    connections = parse_array_blocks(row["connections"])
    return body, connections
