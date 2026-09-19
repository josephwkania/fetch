"""
predict.py end to end on a handful of candidates. Slow, and it needs the model
weights already cached, so it skips rather than downloading them.
"""

import os
import pathlib
import subprocess
import sys

import pandas as pd
import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
WEIGHTS = pathlib.Path.home() / ".keras" / "models"
MODEL_A = "a_ft_DenseNet121_2_dt_Xception_13_256.h5"
MODEL_K = "k_ft_DenseNet121_2_dt_InceptionV3_18_64.h5"

needs_weights = pytest.mark.skipif(
    not (WEIGHTS / MODEL_A).exists(), reason=f"{MODEL_A} is not cached"
)


def run_predict(data_dir, args):
    # TF_USE_LEGACY_KERAS is deliberately not set: predict.py has to pick its
    # own Keras, or an install from requirements.txt cannot load the models.
    env = dict(os.environ, PYTHONPATH=str(REPO), TF_CPP_MIN_LOG_LEVEL="3")
    env.pop("TF_USE_LEGACY_KERAS", None)
    return subprocess.run(
        [sys.executable, str(REPO / "bin" / "predict.py"),
         "--data_dir", str(data_dir), "-g", "-1"] + args,
        capture_output=True, text=True, env=env, timeout=1800,
    )


@needs_weights
def test_one_row_per_candidate(candidates, tmp_path):
    paths = candidates(6)
    result = run_predict(tmp_path, ["-m", "a", "-b", "4"])
    assert result.returncode == 0, result.stderr[-2000:]

    frame = pd.read_csv(tmp_path / "results_a.csv")
    assert len(frame) == len(paths)
    assert sorted(frame.candidate) == sorted(paths)
    assert frame.probability.between(0, 1).all()
    assert set(frame.label.unique()) <= {0.0, 1.0}
    assert (frame.label == (frame.probability >= 0.5)).all()


@needs_weights
def test_probability_belongs_to_its_own_candidate(candidates, tmp_path):
    """
    Predictions are concatenated in read order and pasted next to the file
    list, so anything that reorders the reading silently mislabels every
    candidate. Two runs over the same directory must agree row for row.
    """
    candidates(6)
    first = run_predict(tmp_path, ["-m", "a", "-b", "4"])
    assert first.returncode == 0, first.stderr[-2000:]
    one = pd.read_csv(tmp_path / "results_a.csv")

    second = run_predict(tmp_path, ["-m", "a", "-b", "4"])
    assert second.returncode == 0, second.stderr[-2000:]
    two = pd.read_csv(tmp_path / "results_a.csv")

    assert list(one.candidate) == list(two.candidate)
    assert one.probability.round(6).tolist() == two.probability.round(6).tolist()


@needs_weights
@pytest.mark.skipif(not (WEIGHTS / MODEL_K).exists(), reason=f"{MODEL_K} is not cached")
def test_several_models_share_one_pass(candidates, tmp_path):
    paths = candidates(4)
    result = run_predict(tmp_path, ["-m", "a", "k", "-b", "4"])
    if result.returncode != 0 and "unrecognized arguments" in result.stderr:
        pytest.skip("-m takes a single model at this commit")
    assert result.returncode == 0, result.stderr[-2000:]

    first = pd.read_csv(tmp_path / "results_a.csv")
    second = pd.read_csv(tmp_path / "results_k.csv")
    assert list(first.candidate) == list(second.candidate)
    assert len(first) == len(paths)


@needs_weights
def test_an_unknown_model_is_refused(candidates, tmp_path):
    candidates(1)
    result = run_predict(tmp_path, ["-m", "z"])
    assert result.returncode != 0
    assert not (tmp_path / "results_z.csv").exists()


def test_an_empty_directory_is_not_an_error(tmp_path):
    result = run_predict(tmp_path, ["-m", "a"])
    assert result.returncode == 0, result.stderr[-2000:]
    assert not list(tmp_path.glob("results_*.csv"))
