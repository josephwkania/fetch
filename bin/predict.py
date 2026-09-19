#!/usr/bin/env python3

import argparse
import glob
import logging
import os
import string

# tensorflow.keras resolves on import, so this is set before it; see
# fetch/__init__.py.
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from tensorflow.keras.utils import Progbar

from fetch.data_sequence import DataGenerator
from fetch.utils import get_model

logger = logging.getLogger(__name__)

MODEL_INDICES = list(string.ascii_lowercase)[:11]


def batches(sequence, workers, max_pending=32):
    """
    Yield a Sequence's batches in order, read ahead on `workers` threads. h5py
    drops the GIL, so threads read in parallel without pickling every batch
    back to the parent the way a process pool does, and one thread is enough to
    hide the reading behind the GPU. `max_pending` bounds the read-ahead.
    """
    pool = ThreadPoolExecutor(max_workers=max(1, workers))
    try:
        pending = deque()
        nxt = 0
        while nxt < min(max_pending, len(sequence)):
            pending.append(pool.submit(sequence.__getitem__, nxt))
            nxt += 1
        while pending:
            item = pending.popleft().result()
            if nxt < len(sequence):
                pending.append(pool.submit(sequence.__getitem__, nxt))
                nxt += 1
            yield item
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


def chunks(sequence, batches_per_chunk, workers):
    """
    Yield the Sequence's batches regrouped into chunks of `batches_per_chunk`,
    each one dict of stacked arrays. predict() costs a host-device round trip
    per call, so a model is shown a whole chunk rather than a batch at a time.
    The readers are given a chunk's worth of queue, which is what lets them
    fill the next chunk while the current one is on the GPU.
    """
    # Closing this generator does not finalise a suspended one it is looping
    # over, so batches() is closed here or its readers outlive the caller.
    source = batches(sequence, workers, max(8, batches_per_chunk))
    try:
        group = []
        for data, _ in source:
            # float32 on arrival: DataGenerator stages in float64, and holding
            # a chunk of that costs twice the memory for no extra precision.
            group.append(
                {k: v.astype(np.float32, copy=False) for k, v in data.items()}
            )
            if len(group) == batches_per_chunk:
                yield {k: np.concatenate([b[k] for b in group]) for k in group[0]}
                group = []
        if group:
            yield {k: np.concatenate([b[k] for b in group]) for k in group[0]}
    finally:
        source.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fast Extragalactic Transient Candiate Hunter (FETCH)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-v", "--verbose", help="Be verbose", action="store_true")
    parser.add_argument(
        "-g",
        "--gpu_id",
        help="GPU ID (use -1 for CPU)",
        type=int,
        required=False,
        default=0,
    )
    parser.add_argument(
        "-n",
        "--nproc",
        help="Number of reader threads",
        default=4,
        type=int,
    )
    parser.add_argument(
        "-c",
        "--data_dir",
        help="Directory with candidate h5s.",
        required=True,
        type=str,
        action='append'
    )
    parser.add_argument(
        "-b", "--batch_size", help="Batch size for training data", default=8, type=int
    )
    parser.add_argument(
        "-k",
        "--chunk_size",
        help="Candidates held in memory per predict() call. Costs about 1 MB "
        "per candidate; rounded down to whole batches.",
        default=256,
        type=int,
    )
    parser.add_argument(
        "-m",
        "--model",
        help="Index of the model to use. Several may be given; they share one "
        "pass over the candidates and each writes its own results_<model>.csv.",
        required=True,
        nargs="+",
    )
    parser.add_argument(
        "-p", "--probability", help="Detection threshold", default=0.5, type=float
    )
    args = parser.parse_args()

    logging_format = (
        "%(asctime)s - %(funcName)s -%(name)s - %(levelname)s - %(message)s"
    )

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format=logging_format)
    else:
        logging.basicConfig(level=logging.INFO, format=logging_format)

    for model_idx in args.model:
        if model_idx not in MODEL_INDICES:
            raise ValueError(
                f"Model {model_idx} unknown: models only range from "
                f"{MODEL_INDICES[0]} -- {MODEL_INDICES[-1]}."
            )
    if len(set(args.model)) != len(args.model):
        raise ValueError(f"Model given more than once: {args.model}")

    if args.gpu_id >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpu_id}"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    logging.info(f"Reading candidates on {args.nproc} threads")

    batches_per_chunk = max(1, args.chunk_size // args.batch_size)

    models = [(model_idx, get_model(model_idx)) for model_idx in args.model]

    for data_dir in args.data_dir:

        cands_to_eval = glob.glob(f"{data_dir}/*h5")

        if len(cands_to_eval) == 0:
            logger.warning(f"No candidates to evaluate in directory: {data_dir}")
            continue

        logging.debug(f"Read {len(cands_to_eval)} candidates")

        # Get the data generator, make sure noise and shuffle are off.
        cand_datagen = DataGenerator(
            list_IDs=cands_to_eval,
            labels=[0] * len(cands_to_eval),
            shuffle=False,
            noise=False,
            batch_size=args.batch_size,
        )

        # get's get predicting. Read once, show each chunk to every model. No
        # shuffle, so the predictions concatenate in the order of cands_to_eval.
        chunk_probs = {model_idx: [] for model_idx, _ in models}
        progbar = Progbar(target=len(cand_datagen))
        done = 0
        for chunk in chunks(cand_datagen, batches_per_chunk, args.nproc):
            for model_idx, model in models:
                chunk_probs[model_idx].append(
                    model.predict(chunk, batch_size=args.batch_size, verbose=0)
                )
            done += int(np.ceil(len(chunk["data_freq_time"]) / args.batch_size))
            progbar.update(done)

        # Save results
        for model_idx, _ in models:
            probs = np.concatenate(chunk_probs[model_idx])
            results_dict = {}
            results_dict["candidate"] = cands_to_eval
            results_dict["probability"] = probs[:, 1]
            results_dict["label"] = np.round(probs[:, 1] >= args.probability)
            results_file = data_dir + f"/results_{model_idx}.csv"
            pd.DataFrame(results_dict).to_csv(results_file)
