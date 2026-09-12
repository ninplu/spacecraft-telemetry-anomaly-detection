import numpy as np 
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.dataset import load_channel

# ==========================
# Mission 1 constants
# ==========================

MONOTONIC_CHANNELS = set(range(4, 12))

MISSION1_VAL_START = "2006-10-01"
MISSION1_TEST_START = "2007-01-01"

# ==========================
# Channel cleaning

def clean_channel(data): 
    """
    Sort telemetry chronoligically and remove duplicate timestamps.

    No intrepolation is done here. 
    """

    data = data.sort_index()

    data = data[~data.index.duplicated(keep="last")]

    return data

def difference_channel(data, channel_number):
     """
    ESA Mission 1 channels 4-11 are monotonic channels.

    The official ESA preprocessing takes their first
    difference before resampling.
    """

     if channel_number not in MONOTONIC_CHANNELS:
         return data 

     data = data.copy()
     column = data.columns[0]

     values = data[column].to_numpy()

     data[column] = np.diff(
         values, 
         append=values[-1]
     )

     return data 


def resample_channel(channel, interval='30s'):
    """
    Resample telemetry using zero-order hold.

    This follows the ESA Mission 1 preprocessing approach:
    values are forward-filled to a regular 30-second grid.

    No averaging and no linear/time interpolation.
    """

    channel = channel.sort_index()

    rule = pd.Timedelta(interval)

    first_timestamp = (pd.Timestamp(channel.index[0]).floor(rule))

    last_timestamp = (pd.Timestamp(channel.index[-1]).ceil(rule))

    resamples_index = pd.date_range(
        start=first_timestamp,
        end=last_timestamp,
        freq=rule
    )

    resampled = channel.reindex(
        resamples_index,
        method="ffill"
    )

    resampled.iloc[0] = channel.iloc[0]

    return resampled 


def build_dataset(interval='30s'):
    """
    Build Mission 1 telemetry dataset.

    Pipeline:
        load
        -> clean
        -> difference channels 4-11
        -> 30-second zero-order-hold resampling
        -> combine all 76 channels
        -> fill channel start/end gaps
    """

    all_channels = []

    for ch in range(1, 77):

        channel_data = load_channel(ch)
        channel_data = clean_channel(channel_data)
        channel_data = difference_channel(channel_data, ch)
        channel_data = resample_channel(channel_data, interval=interval)

        channel_data.columns = [f"channel_{ch}"]
        all_channels.append(channel_data)

    dataset = pd.concat(all_channels, axis=1)
    dataset = dataset.sort_index()

    dataset = (dataset.ffill().bfill())
    dataset = dataset.astype(np.float32)

    return dataset 

def create_label_series(dataset, labels, interval='30s'):
    """
    Create a global binary anomaly label for every
    resampled timestamp.

    0 = normal
    1 = anomaly

    Short anomalies that fall between two 30-second
    timestamps are explicitly retained.
    """

    label_series = pd.Series(
        0, 
        index = dataset.index,
        dtype = np.uint8
    )

    rule = pd.Timedelta(interval)

    for _, row in labels.iterrows():
        anomaly_start = row["StartTime"]
        anomaly_end = row["EndTime"]

        if dataset.index.tz is not None: 
            if anomaly_start.tzinfo is None:
                anomaly_start = (anomaly_start.tz_localize(dataset.index.tz))

            else: 
                anomaly_start = (anomaly_start.tz_convert(dataset.index.tz))

            if anomaly_end.tzinfo is None:
                anomaly_end = (anomaly_end.tz_localize(dataset.index.tz))

            else:
                anomaly_end = (anomaly_end.tz_convert(dataset.index.tz))


        mask = ((label_series.index >= anomaly_start) & (label_series.index <= anomaly_end))


        # Normal case: mark every grid point that lies inside the anomaly interval.
        if mask.any(): 
            label_series.loc[mask] = 1

        # Very short anomaly that falls entirely between two resampled timestamps.
        else: 
            nearest_timestamp = (anomaly_start.ceil(rule))

            if (nearest_timestamp >= label_series.index[0]) and (nearest_timestamp <= label_series.index[-1]):
                label_series.loc[nearest_timestamp] = 1

    return label_series

def _align_timestamp_timezone(timestamp, index):
    """
    Make a split timestamp compatible with the
    timezone of the dataset index.
    """

    timestamp = pd.Timestamp(timestamp)

    if index.tz is not None:

        if timestamp.tzinfo is None: 
            timestamp = timestamp.tz_localize(index.tz)

        else: 
            timestamp = timestamp.tz_convert(index.tz)

    elif timestamp.tzinfo is not None: 
        timestamp = timestamp.tz_localize(None)

    return timestamp 


def dataset_split(dataset, labels, val_start=MISSION1_VAL_START, test_start=MISSION1_TEST_START):
    """
    Chronological Mission 1 split.

    Train:
        before 2006-10-01

    Validation:
        2006-10-01 through 2007-01-01

    Test:
        strictly after 2007-01-01
    """

    val_start = _align_timestamp_timezone(val_start, dataset.index)
    test_start = _align_timestamp_timezone(test_start, dataset.index)

    train_mask = (dataset.index < val_start)
    val_mask = ((dataset.index >= val_start) & (dataset.index <= test_start))
    test_mask = (dataset.index > test_start)

    train_data = dataset.loc[train_mask]
    train_labels = labels.loc[train_mask]

    val_data = dataset.loc[val_mask]
    val_labels = labels.loc[val_mask]

    test_data = dataset.loc[test_mask]
    test_labels = labels.loc[test_mask]

    return (
        train_data, 
        train_labels,
        val_data,
        val_labels,
        test_data,
        test_labels
    )

def scale_splits(train_data, train_labels, val_data, test_data, normal_only=True):
    """
    Fit StandardScaler using TRAINING DATA ONLY.

    For anomaly detection we optionally fit the scaler
    using only normal training timestamps.
    """

    scaler = StandardScaler()

    if normal_only:

        scaler_fit_data = train_data.loc[train_labels == 0]

    else: 
        scaler_fit_data = train_data

    scaler.fit(scaler_fit_data.values)

    train_scaled = pd.DataFrame(
        scaler.transform(train_data.values),
        index=train_data.index,
        columns=train_data.columns,
        dtype=np.float32
    )

    val_scaled = pd.DataFrame(
        scaler.transform(val_data.values),
        index=val_data.index,
        columns=val_data.columns,
        dtype=np.float32
    )

    test_scaled = pd.DataFrame(
        scaler.transform(test_data.values),
        index=test_data.index,
        columns=test_data.columns, 
        dtype=np.float32
    )

    return (
        train_scaled,
        val_scaled,
        test_scaled,
        scaler
    )

def create_windows(data, window_size=256, stride=128):
    """
    Convert telemetry into overlapping windows.

    Expected input shape:
        (timestamps, 76)

    Output:
        (num_windows, 256, 76)
    """

    if isinstance(data, pd.DataFrame):
        data = data.values

    windows = []

    for start in range(0, len(data) - window_size + 1, stride):
        end = start + window_size
        windows.append(data[start:end])

    return np.asarray(windows, dtype=np.float32)

def create_window_labels(labels, window_size=256, stride=128, anomaly_threshold=0.1):
    """
    Convert timestamp-level labels into binary
    window labels.

    NOTE:
    anomaly_threshold=0.1 is a modelling choice,
    not an ESA benchmark rule.
    """

    labels = np.asarray(labels, dtype=np.float32)

    window_labels = []

    for start in range(0, len(labels) - window_size + 1, stride):
        end = start + window_size
        fraction = labels[start:end].mean()
        label = int(fraction >= anomaly_threshold)

        window_labels.append(label)

    return np.asarray(window_labels, dtype=np.uint8)

def create_windows_to_disk(data, output_path, window_size=256, stride=128):
    """
    Create overlapping windows and write them directly
    to a .npy file using a memory map.

    This avoids storing all windows in RAM at once.

    Args:
        data:
            Array-like telemetry with shape
            (timestamps, channels).

        output_path:
            Path where the .npy file will be saved.

        window_size:
            Number of timestamps per window.

        stride:
            Number of timestamps to move between windows.

    Returns:
        Number of windows created.
    """

    num_windows = ((len(data) - window_size) // stride) + 1

    num_channels = data.shape[1]

    windows = np.lib.format.open_memmap(
        output_path, 
        mode='w+',
        dtype=np.float32,
        shape=(num_windows, window_size, num_channels)
    )

    for i, start in enumerate(range(0, len(data) - window_size + 1, stride)):
        end = start + window_size
        windows[i] = data[start:end]

    windows.flush()  # Ensure data is written to disk

    del windows  # Free memory

    return num_windows 

def create_per_channel_label_matrix(index, label_with_types, target_channels):

    zeros_matrix = np.zeros((len(index), len(target_channels)), dtype=np.uint8)

    for i, channel in enumerate(target_channels):
        target_channel_labels = label_with_types[label_with_types["Channel"] == channel]

        target_channel_labels = target_channel_labels[target_channel_labels["Category"].isin(["Anomaly", "Rare Event"])]

        for _, row in target_channel_labels.iterrows():

            anomaly_start = row["StartTime"]
            anomaly_end = row["EndTime"]

            mask = ((index >= anomaly_start) & (index <= anomaly_end))
            zeros_matrix[mask, i] = 1


    return pd.DataFrame(
        zeros_matrix,
        index=index,
        columns=target_channels
    )

def create_per_channel_window_labels(per_channel_labels, window_size=256, stride=128, anomaly_threshold=0.1):
    """
    Convert timestamp-level per-channel labels into binary
    window labels.
    
    NOTE:
    anomaly_threshold=0.1 is a modelling choice,
    not an ESA benchmark rule.
    """


    per_channel_labels = np.asarray(per_channel_labels)

    window_labels = []

    for start in range(0, len(per_channel_labels) - window_size + 1, stride):

        end = start + window_size
        window_slice = per_channel_labels[start:end]

        fraction_per_channel = window_slice.mean(axis=0)
        label_per_channel = (fraction_per_channel >= anomaly_threshold).astype(np.uint8)

        window_labels.append(label_per_channel)

    return np.asarray(window_labels, dtype=np.uint8)