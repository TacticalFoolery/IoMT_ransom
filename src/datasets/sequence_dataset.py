import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import List, Union


class GroupedSequenceDataset(Dataset):
    """
    Parameters:
    
    df : pd.DataFrame
        The input dataframe.
    feature_columns : List[str]
        Names of feature columns to include in each timestep.
    label_column : str
        Name of the label column (binary 0/1).
    group_column : str
        Column to group by (e.g., 'device_id' or 'src_ip').
    time_column : str
        Timestamp column (must be sortable).
    seq_len : int
        Length of sliding window.
    label_mode : str
        "any" => sequence labeled 1 if any flow in window is malicious.
        "last" => sequence labeled as the last element's label.
    min_group_size : int
        Minimum group length to produce any sequences (default==seq_len).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_columns: List[str],
        label_column: str = "label",
        group_column: str = "device_id",
        time_column: str = "timestamp",
        seq_len: int = 10,
        label_mode: str = "any",
        min_group_size: int = None,
    ):
        assert label_mode in ("any", "last", "max")
        self.seq_len = seq_len
        self.feature_columns = feature_columns
        self.label_column = label_column
        self.samples = []
        self.labels = []

        if min_group_size is None:
            min_group_size = seq_len

        grouped = df.groupby(group_column)

        for _, group in grouped:
            # sort by time (assumes time column is parseable/sort-friendly)
            group = group.sort_values(time_column).reset_index(drop=True)

            if len(group) < min_group_size:
                continue

            X = group[feature_columns].values
            y = group[label_column].values

            for i in range(len(group) - seq_len + 1):
                x_window = X[i : i + seq_len]
                y_window = y[i : i + seq_len]

                if label_mode == "last":
                    seq_label = int(y_window[-1])
                elif label_mode == "max":
                    seq_label = int(np.max(y_window))
                else:  # "any"
                    seq_label = int(np.any(y_window == 1))

                self.samples.append(x_window)
                self.labels.append(seq_label)

        if len(self.samples) == 0:
            raise ValueError("No sequences created. Check seq_len, grouping, and time ordering.")

        label_dtype = torch.long if label_mode == "max" else torch.float32
        self.samples = torch.tensor(np.array(self.samples), dtype=torch.float32)
        self.labels = torch.tensor(np.array(self.labels), dtype=label_dtype)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]


class ArraySequenceDataset(Dataset):
    """
    Parameters:

    features : np.ndarray, shape (N, F)
        Per-row features (e.g., latent vectors or latent+recon).
    labels : np.ndarray, shape (N,)
        Per-row labels (0/1).
    group_ids : np.ndarray or List, shape (N,)
        Group identifier per-row (device_id, src_ip hash, etc). Must match row order of features/labels.
    seq_len : int
        Sliding window length.
    label_mode : str
        "any" or "last"
    """

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        group_ids: Union[np.ndarray, List],
        seq_len: int = 10,
        label_mode: str = "any",
    ):
        assert features.shape[0] == labels.shape[0] == len(group_ids)
        assert label_mode in ("any", "last", "max")

        self.seq_len = seq_len
        self.samples = []
        self.labels = []

        # iterate group-by-group preserving original order
        unique_groups, group_indices = np.unique(group_ids, return_inverse=True)
        # collect indices per group in original order
        group_to_rowidx = {g: [] for g in range(len(unique_groups))}
        for row_idx, grp_idx in enumerate(group_indices):
            group_to_rowidx[grp_idx].append(row_idx)

        for grp_idx, rows in group_to_rowidx.items():
            if len(rows) < seq_len:
                continue

            # rows are in original order; build sliding windows on these indices
            for i in range(len(rows) - seq_len + 1):
                window_idx = rows[i : i + seq_len]
                x_window = features[window_idx]           # shape (seq_len, F)
                y_window = labels[window_idx]            # shape (seq_len,)

                if label_mode == "last":
                    seq_label = int(y_window[-1])
                elif label_mode == "max":
                    seq_label = int(np.max(y_window))
                else:  # "any"
                    seq_label = int(np.any(y_window == 1))

                self.samples.append(x_window)
                self.labels.append(seq_label)

        if len(self.samples) == 0:
            raise ValueError("No sequences created from arrays. Check seq_len and group_ids.")

        label_dtype = torch.long if label_mode == "max" else torch.float32
        self.samples = torch.tensor(np.array(self.samples), dtype=torch.float32)
        self.labels = torch.tensor(np.array(self.labels), dtype=label_dtype)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]