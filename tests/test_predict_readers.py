"""
The reading half of predict.py: whatever reads ahead must hand every batch over
exactly once, in order, and must not leave threads behind.

Both helpers are absent from some commits and have changed signature between
others, so each test asks for what it needs and skips when it is not there.
"""

import inspect
import threading
import time

import numpy as np
import pytest
from tensorflow.keras.utils import Sequence


class SlowSequence(Sequence):
    """
    A Sequence with uneven read times, so a reader that returns batches in
    completion order rather than index order is caught.
    """

    def __init__(self, n_batches, batch_size=2):
        self.n_batches = n_batches
        self.batch_size = batch_size

    def __len__(self):
        return self.n_batches

    def __getitem__(self, index):
        time.sleep(0.02 * ((self.n_batches - index) % 5))
        marker = np.full((self.batch_size, 4, 4, 1), float(index), dtype=np.float64)
        labels = np.zeros((self.batch_size, 2))
        return {"data_freq_time": marker, "data_dm_time": marker.copy()}, labels


def markers(array):
    return array[:, 0, 0, 0].astype(int).tolist()


def call_batches(predict, sequence, workers):
    if not hasattr(predict, "batches"):
        pytest.skip("no batches(); this commit reads inside predict_generator")
    if "use_multiprocessing" in inspect.signature(predict.batches).parameters:
        return predict.batches(sequence, workers, False)
    return predict.batches(sequence, workers)


def call_chunks(predict, sequence, batches_per_chunk, workers):
    if not hasattr(predict, "chunks"):
        pytest.skip("no chunks(); this commit predicts a batch at a time")
    if "use_multiprocessing" in inspect.signature(predict.chunks).parameters:
        return predict.chunks(sequence, batches_per_chunk, workers, False)
    return predict.chunks(sequence, batches_per_chunk, workers)


@pytest.mark.parametrize("workers", [1, 2, 4])
def test_batches_yields_every_batch_in_order(predict, workers):
    sequence = SlowSequence(12)
    got = [markers(data["data_freq_time"])[0] for data, _ in
           call_batches(predict, sequence, workers)]
    assert got == list(range(12))


def test_chunks_cover_every_candidate_in_order(predict):
    sequence = SlowSequence(10, batch_size=2)
    got = []
    for chunk in call_chunks(predict, sequence, 4, 4):
        got += markers(chunk["data_freq_time"])
    assert got == [index for index in range(10) for _ in range(2)]


def test_chunks_hold_whole_batches_and_a_remainder(predict):
    sequence = SlowSequence(10, batch_size=2)
    sizes = [len(chunk["data_freq_time"]) for chunk in
             call_chunks(predict, sequence, 4, 4)]
    assert sizes == [8, 8, 4]


@pytest.mark.parametrize("workers", [1, 2])
def test_chunks_with_few_workers(predict, workers):
    sequence = SlowSequence(5, batch_size=2)
    sizes = [len(chunk["data_freq_time"]) for chunk in
             call_chunks(predict, sequence, 2, workers)]
    assert sizes == [4, 4, 2]


def test_one_chunk_when_it_holds_everything(predict):
    sequence = SlowSequence(3, batch_size=2)
    chunks = list(call_chunks(predict, sequence, 100, 2))
    assert len(chunks) == 1
    assert len(chunks[0]["data_freq_time"]) == 6


def test_chunks_are_float32(predict):
    """The Sequence above hands over float64; the model wants float32."""
    sequence = SlowSequence(4, batch_size=2)
    for chunk in call_chunks(predict, sequence, 2, 2):
        assert chunk["data_freq_time"].dtype == np.float32
        assert chunk["data_dm_time"].dtype == np.float32


def wait_for_threads(count, deadline=60.0):
    end = time.time() + deadline
    while threading.active_count() > count and time.time() < end:
        time.sleep(0.2)
    return threading.active_count()


def test_readers_stop_when_the_work_is_done(predict):
    before = threading.active_count()
    sequence = SlowSequence(8, batch_size=2)
    list(call_chunks(predict, sequence, 2, 4))
    assert wait_for_threads(before) == before


def test_readers_stop_when_the_generator_is_abandoned(predict):
    """
    The failure that hung a run for six hours was at shutdown, after the work
    had finished, so a reader left running is worth a test of its own.
    """
    before = threading.active_count()
    sequence = SlowSequence(40, batch_size=2)
    chunks = call_chunks(predict, sequence, 2, 4)
    next(chunks)
    chunks.close()
    assert wait_for_threads(before) == before
