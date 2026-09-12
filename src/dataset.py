from pathlib import Path
import pickle
import zipfile
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PATH_TO_DATASET = PROJECT_ROOT / "ESA-Mission1"


def load_channel(channel_number):
    zip_path = PATH_TO_DATASET / "channels" / f"channel_{channel_number}.zip"

    with zipfile.ZipFile(zip_path) as z:
        filename = z.namelist()[0]

        with z.open(filename) as f:
            data = pickle.load(f)

    if data.index.tz is None:
        data.index = data.index.tz_localize("UTC")
    else:
        data.index = data.index.tz_convert("UTC")

    return data


def load_labels():
    labels = pd.read_csv(
        PATH_TO_DATASET / "labels.csv",
        index_col=0
    )

    labels["StartTime"] = pd.to_datetime(
        labels["StartTime"],
        utc=True
    )

    labels["EndTime"] = pd.to_datetime(
        labels["EndTime"],
        utc=True
    )

    return labels