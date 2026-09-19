# What the predict.py rewrite changed

Measured on a Tesla T4 (sm_75, 70 W, a 1590 MHz boost it cannot hold), tf-keras
2.19 on TensorFlow 2.19, Python 3.12, against 6,400 candidate h5s. Those are
stand-ins, not sky: the production run's own cutouts went with an ephemeral
disk, so they were generated as Gaussian noise with a crude dispersed pulse
injected, written the way `your_candmaker.py` writes them -- two 256x256 float32
planes, gzip-9, the same attributes. They average 228 kB against the run's
measured 295 kB, so timings taken on them are representative and if anything a
little optimistic. What they are *not* is representative of the classifier's
output distribution, which matters below. Two versions are compared throughout:

- **upstream** — `5b85488`, with its own `bin/predict.py` and
  `fetch/data_sequence.py`, one model per process.
- **tip** — reading on a bounded window of reader threads, showing each model a
  whole chunk per `predict()` call, all models in one pass.

A third version appears where it is informative: the first attempt at sharing
one read across models, which called `predict_on_batch` per batch behind Keras's
`OrderedEnqueuer`. It was slower than upstream for a single model, which is what
prompted the rewrite.

## Runtime

Inference leg only, 6,400 candidates, FETCH's default `-b 8`, `-n 4`, every
repeat shown.

| path | model a | model c |
| --- | --- | --- |
| upstream | 89.18, 88.42 | 91.16, 91.04 |
| first shared-read attempt | 97.05, 97.10 | 110.86, 110.35 |
| tip | 69.03, 68.14, 67.03 | 73.80, 74.35 |
| upstream at `-n 1` | 62.71, 62.31 | 65.40, 67.22 |

The last row is the floor. With one worker Keras prefetches through `tf.data`
and the reading disappears behind the card entirely, so it is close to what the
arithmetic alone costs. The tip comes within 9% of it at `-n 4`, where upstream
sits 42% above it: reading through four reader *processes* costs more than it
saves, because every batch is pickled back to the parent.

Seven models over the same 6,400 candidates:

| | inference | per candidate |
| --- | --- | --- |
| upstream, seven separate runs | 630 s (87+87+88+94+92+91+91) | 98.4 ms |
| tip, one pass | 502.76, 501.19 s | 78.4 ms |

1.26x on the inference leg, before counting the six extra process startups and
model loads upstream needs. At 78.4 ms a candidate the seven models run at
2,956 GFLOP/s, 54% of this card's measured sustained 3x3 convolution ceiling of
5.48 TFLOP/s. The stage as originally run managed 761 GFLOP/s, 14%.

`-b 32` is now slower than `-b 8` (70.56 s against 66.58 for model a). The
earlier gain from a larger batch was the per-call dispatch cost being spread
over more candidates, and the rewrite removes that cost rather than amortising
it.

## The probabilities move, slightly

Every model, all 6,400 candidates, tip against upstream:

| model | max abs diff | closest p to 0.5 | margin | labels changed |
| --- | --- | --- | --- | --- |
| a | 1.05e-06 | 7.81e-02 | 74,400x | 0 |
| b | 1.56e-07 | 4.82e-01 | 3,090,000x | 0 |
| c | 7.20e-07 | 3.00e-01 | 416,000x | 0 |
| d | 2.63e-07 | 4.49e-01 | 1,710,000x | 0 |
| e | 6.90e-08 | 4.69e-01 | 6,800,000x | 0 |
| f | 1.58e-06 | 3.00e-01 | 182,000x | 0 |
| k | 3.20e-06 | 3.28e-04 | 102x | 0 |

No candidate changes label, for any model, on this set.

**The margins in that table are not the real ones.** On synthetic noise the
classifier is confident and almost nothing lands near the threshold: model a
flags 0 of 6,400 here, where in production it flagged 5,000 of 30,527. The
production run's own `results_*.csv` are the right place to look, and they are
far tighter:

| model | rows >= 0.5 | closest p to 0.5 | measured shift | ratio |
| --- | --- | --- | --- | --- |
| a | 5,000 | 3.22e-05 | 1.05e-06 | 31x |
| b | 11,284 | 5.80e-05 | 1.56e-07 | 372x |
| c | 9,929 | 7.73e-05 | 7.20e-07 | 107x |
| d | 6,472 | **1.04e-06** | 2.63e-07 | **4.0x** |
| e | 6,764 | 1.90e-04 | 6.90e-08 | 2,746x |
| f | 6,169 | 6.71e-05 | 1.58e-06 | 42x |
| k | 11,360 | 1.51e-05 | 3.20e-06 | **4.7x** |

Real candidates do sit within a part in a million of the threshold: model d has
one at 1.04e-06 and one candidate within 3.2e-06, model k has five within
1e-04. So the worst-case headroom is about 4x, not the hundreds of thousands the
synthetic set suggests.

That ratio also mixes two measurements. The shifts were measured on synthetic
inputs; the margins come from real ones. Kernel-level rounding depends on the
activations, so the shifts would have to be re-measured on real cutouts before
the ratio means anything exact. The filterbanks to make those cutouts are
available -- three 8.1 GB J1705 observations, though not the three the run
itself used -- so this is a gap that can be closed rather than an unknowable.

## Why they move

Two causes, and it took a control run to separate them.

**The inputs are identical, so nothing mathematical differs.** `DataGenerator`
does all its arithmetic in float32 and then writes into a staging buffer that is
float64 upstream and float32 at the tip. Widening float32 to float64 and
narrowing back is exact, so the tensor the model receives is the same either
way: over 400 candidates both trees hash to `e9ab8e68...`. The batching is
identical too — `shuffle=False`, the same file order, `batch_size=8` in both,
and 6,400 divides exactly into chunks of 256 and batches of 8, so no batch is
partial. At inference BatchNorm uses its stored moving statistics, so candidates
in a batch do not influence each other regardless.

**Most of the difference is between graphs, not between runs.** Upstream traces
one `tf.function` stepping a `tf.data` pipeline built by `from_generator` with
unknown cardinality; the tip calls `predict()` on an in-memory array, which
gives `from_tensor_slices` with fully known shapes. Grappler's layout pass and
cuDNN's algorithm choice see different graphs and settle on different kernels,
which sum in different orders and round differently. That choice is then fixed
for the graph, which is why models a, c and k reproduce bit for bit across three
separate runs of the tip while still differing from upstream by up to 3.2e-06.

**A little of it is genuine run-to-run nondeterminism.** Three runs of the tip
over the same candidates:

| model | run 1 vs 2 | run 1 vs 3 | run 2 vs 3 | vs upstream |
| --- | --- | --- | --- | --- |
| a | 0 | 0 | 0 | 1.05e-06 |
| b | 7.30e-08 | 7.30e-08 | 0 | 1.56e-07 |
| c | 0 | 0 | 0 | 7.20e-07 |
| d | 2.50e-08 | 5.70e-09 | 2.50e-08 | 2.63e-07 |
| e | 7.60e-08 | 7.60e-08 | 0 | 6.90e-08 |
| f | 3.80e-07 | 3.80e-07 | 0 | 1.58e-06 |
| k | 0 | 0 | 0 | 3.20e-06 |

Runs 2 and 3 were consecutive in one script and agree exactly except for model
d; run 1 was a separate process after seven other runs, and differs for b, d, e
and f. cuDNN's autotuner chooses on timing and available workspace, so a
different GPU memory state can yield a different kernel. For model e the spread
between two runs of the same binary (7.6e-08) is larger than the difference from
upstream (6.9e-08): there the version makes no difference distinguishable from
noise.

So for a, c and k the shift is systematic and reproducible; for b, d, e and f it
is partly systematic and partly which process happened to run. Neither cause is
close to large enough to move a label.

## What this does not cover

- **Half precision.** Everything here is fp32 on the CUDA cores. The T4's tensor
  cores measured 24.2 TFLOP/s against the 5.48 TFLOP/s fp32 convolution ceiling,
  and nothing in FETCH sets a mixed-precision policy. fp16 would perturb
  probabilities at around 1e-03, three orders above anything measured here, so
  it needs this same label comparison before it could be trusted.
- **Scale.** The production stage classified 30,527 candidates; 6,400 is the
  largest stand-in set built here.
- **Real cutouts.** Every number above was taken on generated candidates. The
  timings carry over; the probability margins do not, as the table above shows.
- **The runtime the pipeline actually used.** That image carried Keras 2.14 and
  went with the ephemeral disk. Every comparison here is tf-keras 2.19, and the
  mechanisms involved are version-specific.
