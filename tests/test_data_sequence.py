"""
DataGenerator's contract: every candidate read once, in the order given, with
the right shapes and no NaNs left in the data.
"""

import numpy as np
import pytest

from fetch.data_sequence import DataGenerator


def generator(paths, batch_size=4, labels=None):
    return DataGenerator(
        list_IDs=paths,
        labels=[0] * len(paths) if labels is None else labels,
        shuffle=False,
        noise=False,
        batch_size=batch_size,
    )


def spike_rows(array):
    """The marker row written by write_candidate, one per candidate."""
    return [
        int(np.unravel_index(np.argmax(candidate), candidate.shape)[0])
        for candidate in array
    ]


def test_length_rounds_up(candidates):
    assert len(generator(candidates(10), batch_size=4)) == 3


def test_every_candidate_once_and_in_order(candidates):
    gen = generator(candidates(10), batch_size=4)
    seen = []
    for index in range(len(gen)):
        seen += spike_rows(gen[index][0]["data_dm_time"])
    assert seen == list(range(10))


def test_last_batch_holds_the_remainder(candidates):
    gen = generator(candidates(10), batch_size=4)
    assert gen[2][0]["data_freq_time"].shape[0] == 2


def test_keys_and_shapes(candidates):
    data, labels = generator(candidates(4), batch_size=4)[0]
    assert set(data) == {"data_freq_time", "data_dm_time"}
    assert data["data_freq_time"].shape == (4, 256, 256, 1)
    assert data["data_dm_time"].shape == (4, 256, 256, 1)
    assert labels.shape == (4, 2)


def test_labels_come_back_categorical(candidates):
    _, labels = generator(candidates(4), batch_size=4, labels=[0, 1, 1, 0])[0]
    assert labels.argmax(axis=1).tolist() == [0, 1, 1, 0]


def test_nans_do_not_survive(candidates):
    data, _ = generator(candidates(4, nan=True), batch_size=4)[0]
    assert np.isfinite(data["data_freq_time"]).all()
    assert np.isfinite(data["data_dm_time"]).all()


def test_shuffle_off_keeps_file_order(candidates):
    gen = generator(candidates(6), batch_size=2)
    assert gen.indexes.tolist() == list(range(6))


def test_batches_are_normalised(candidates):
    data, _ = generator(candidates(4), batch_size=4)[0]
    for candidate in data["data_dm_time"]:
        assert abs(np.median(candidate)) < 1e-3
        assert 0.5 < np.std(candidate) < 2.0


def test_staged_in_float32(candidates):
    """
    Staging in float64 doubles what every reader hands over for no extra
    precision, since the h5 data is float32. Expected to fail before the
    float32 staging change.
    """
    data, _ = generator(candidates(4), batch_size=4)[0]
    assert data["data_freq_time"].dtype == np.float32
    assert data["data_dm_time"].dtype == np.float32
