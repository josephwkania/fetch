"""
Shared fixtures.

The environment has to be set before tensorflow is imported, so it is set here
at import time rather than in a fixture.
"""

import os

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import importlib.util
import pathlib
import sys

import h5py
import numpy as np
import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def write_candidate(path, index, ft_dim=(256, 256), dt_dim=(256, 256), nan=False):
    """
    Write one candidate h5 in the shape your_candmaker produces.

    Both arrays carry a spike whose row encodes `index`. DataGenerator detrends
    and normalises, neither of which moves the maximum, so a batch can still be
    traced back to the file it came from.
    """
    ft = np.zeros(ft_dim, dtype=np.float32)
    dt = np.zeros(dt_dim, dtype=np.float32)
    ft[index % ft_dim[0], 1] = 5.0
    dt[index % dt_dim[0], 2] = 7.0
    if nan:
        ft[10, 10] = np.nan
        dt[10, 10] = np.nan
    with h5py.File(path, "w") as f:
        f.create_dataset("data_freq_time", data=ft)
        f.create_dataset("data_dm_time", data=dt)
        attrs = {
            "fch1": 1500.0, "foff": -0.5, "nchans": ft_dim[0], "dm": 100.0,
            "cand_id": f"cand_{index}", "tsamp": 256e-6, "dm_opt": 100.0,
            "snr": 10.0, "snr_opt": 10.0, "width": 1,
        }
        for key, value in attrs.items():
            f.attrs[key] = value
    return str(path)


@pytest.fixture
def candidates(tmp_path):
    """Make `n` candidate h5s in a fresh directory and return their paths."""

    def make(n, **kwargs):
        return [
            write_candidate(tmp_path / f"cand_{i:04d}.h5", i, **kwargs)
            for i in range(n)
        ]

    return make


@pytest.fixture(scope="session")
def predict():
    """bin/predict.py as a module. It is a script, so it is loaded by path."""
    spec = importlib.util.spec_from_file_location(
        "fetch_predict", REPO / "bin" / "predict.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
