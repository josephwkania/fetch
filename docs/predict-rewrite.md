# What the predict.py rewrite changed

Measured on a Tesla T4 (sm_75, 70 W, a 1590 MHz boost it cannot hold).

The headline results are in "On real candidates" below: 18,618 cutouts of two
pulsars, made by the pipeline itself and classified in its own container on
Keras 2.14. The sections before that are the earlier work on generated
stand-ins, kept because the mechanism is easier to see there and because the
runtime comparison needed a fixed set -- but where the two disagree, the real
candidates are the answer.

The stand-ins: 6,400 h5s, tf-keras 2.19 on TensorFlow 2.19, Python 3.12. Not
sky: the production run's own cutouts went with an ephemeral
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

## On real candidates

The numbers above were taken on generated stand-ins. The pipeline has since
been run end to end on two pulsars, and everything below is from those: 8,484
cutouts of PSR J0358 and 10,134 of PSR J1705, made by `your_candmaker.py` and
classified in the pipeline's own container, which carries Keras 2.14.

### Timing

Seven models over one pass against seven separate runs, same files, same four
cores:

| | upstream, 7 runs | rewrite, 1 pass | |
| --- | --- | --- | --- |
| J0358, Keras 2.14 | 1126 s | 728 s | 1.55x |
| J1705, Keras 2.14 | 1361 s | 857 s | 1.59x |
| J0358, tf-keras 2.19 | 898 s | 710 s | 1.26x |
| J1705, tf-keras 2.19 | 1128 s | 844 s | 1.34x |

Per candidate the rewrite costs 83-86 ms in both runtimes, while upstream costs
133 ms on 2.14 and 106-111 ms on 2.19. The rewrite is runtime-independent
because it is no longer paying framework overhead per model, which is why the
gain is larger on the runtime the pipeline actually runs.

Upstream's per-model times vary by only a few per cent -- 153 to 167 s on
J0358 -- while the models differ by a factor of four in arithmetic. A stage
whose cost is blind to the work it does is not compute-bound; it is paying to
read the same 8,484 cutouts seven times.

### The probabilities still move, and the real margins are tight

Rewrite against upstream, same runtime, every model, all 18,618 candidates:
the largest disagreement is 6.7e-06 and **no candidate changes label**, across
260,652 comparisons.

But the headroom is nothing like what the generated set suggested. Taking the
closest candidate to the 0.5 threshold in each model's real results, against
that model's measured shift:

| | closest p to 0.5 | shift | ratio |
| --- | --- | --- | --- |
| J1705, model k | 3.70e-05 | 5.95e-06 | **6.2x** |
| J1705, model f | 5.03e-05 | 4.70e-06 | 10.7x |
| J0358, model f | 1.06e-04 | 6.72e-06 | 15.8x |
| J0358, model a | 7.02e-04 | 4.00e-07 | 1755x |

On generated noise the same calculation gave 74,400x to 6,800,000x, because
the classifier is confident there and almost nothing lands near the threshold.
On real sky the worst case is six. Nothing flipped here, and no candidate sits
within one shift of the threshold in either run -- but the production J1705
results contain a candidate 1.04e-06 from it, which is inside one shift. "No
labels changed" is a measurement of these 18,618 candidates, not a guarantee
for the next set.

## The first measurements, on generated stand-ins

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

## The pod is slower than the same container, and it is one variable

The classification stage takes far longer as a Kubernetes Job than the same
image does by hand on the same four cores: 369 s against 160 s for one model
over the 8,484 J0358 cutouts. The cause is not the container, the storage or
the card. `limits: {cpu: "4"}` becomes a CFS quota -- 400 ms of CPU per 100 ms
period -- while `--cpuset-cpus 0-3` is four whole cores, and `predict.py`
oversubscribes badly: each of its reader processes starts as many OpenMP
threads as the host has CPUs, because a cgroup limit does not change what the
kernel reports. Those threads burn the quota early in each period and then all
stall until it refills.

Measured, same model and cutouts, all under `--cpus=4`:

| | seconds |
| --- | --- |
| as the stage runs it (`-n 4`) | 354 |
| `-n 2` | 360 |
| `-n 1` | 246 |
| `-n 4`, `OMP_NUM_THREADS=1` | **91** |
| `-n 4`, OpenMP and both TensorFlow pools capped | 86 |
| `-n 1`, all capped | 99 |
| for reference, `--cpuset-cpus=0-3`, `-n 4` | 160 |

The reader count is not the variable: `-n 2` is no better than `-n 4`.
`OMP_NUM_THREADS=1` alone accounts for almost all of it, taking the stage from
354 s to 91 s -- faster than four dedicated cores, because the oversubscription
was costing more than the fourth core was worth. This is a ConfigMap change,
and it compounds with the rewrite rather than overlapping with it.

## What this does not cover

- **Half precision.** Everything here is fp32 on the CUDA cores. The T4's tensor
  cores measured 24.2 TFLOP/s against the 5.48 TFLOP/s fp32 convolution ceiling,
  and nothing in FETCH sets a mixed-precision policy. fp16 would perturb
  probabilities at around 1e-03, three orders above anything measured here, so
  it needs this same label comparison before it could be trusted.
- **Scale.** The production stage classified 30,527 candidates; 6,400 is the
  largest stand-in set built here.
- **The threshold.** Nothing flipped across 260,652 real comparisons, but the
  worst margin is a factor of six and the production results hold a candidate
  inside one shift. A change that perturbs probabilities further -- half
  precision, a different card, a newer cuDNN -- needs this comparison run
  again rather than inheriting its conclusion.
- **The runtime the pipeline actually used.** That image carried Keras 2.14 and
  went with the ephemeral disk. Every comparison here is tf-keras 2.19, and the
  mechanisms involved are version-specific.
