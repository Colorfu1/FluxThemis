# Official π0.5 LIBERO Checkpoint Evaluation and FluxVLA FT Comparison

Results compiled on 2026-10-16; all three evaluations completed on 2026-10-15.

**The official checkpoint results reported here use the FT ZMQ RNG protocol:
92.80% success with a prediction/execution horizon of 10/5, and 95.80% with
10/10.** Both results are retained; specify the execution horizon when citing them.

Here, "official" refers to the `pi05_libero` weights released by Physical
Intelligence. It does not mean that every OpenPI evaluation default was used.

## 1. Results and FT Comparison

Each evaluation used eight A800 80GB GPUs, with three model replicas and three
simulator workers per GPU. Spatial, Object, Goal and Long each contain 10 tasks,
using fixed initial states 0–49 per task: 500 episodes per suite and 2,000 in
total. The respective action-step limits are 220, 280, 300 and 520. Episodes end
on success; reaching the limit without success counts as failure. The overall
score is the arithmetic mean of the four suite success rates. Pass@10 is not
used, and LIBERO-90 is excluded.

| Suite | Official 10/5 | Official 10/10 | FT ZMQ 10/10 (this run) | Official 10/10 − FT (percentage points) |
| --- | ---: | ---: | ---: | ---: |
| Spatial | 97.8% (489/500) | 99.2% (496/500) | 99.0% (495/500) | +0.2 |
| Object | 95.8% (479/500) | 98.4% (492/500) | 99.6% (498/500) | −1.2 |
| Goal | 95.0% (475/500) | 96.4% (482/500) | 98.0% (490/500) | −1.6 |
| Long | 82.6% (413/500) | 89.2% (446/500) | 94.2% (471/500) | −5.0 |
| **Mean success rate** | **92.80%** | **95.80%** | **97.70%** | **−1.90** |
| **Successful episodes** | **1,856/2,000** | **1,916/2,000** | **1,954/2,000** | **38 fewer** |

Official 10/10 completed 60 more episodes successfully than 10/5, including 33
more in Long. It completed 38 fewer successfully than FT, including 25 fewer in
Long. Each configuration had only one full run; these results do not establish
statistical significance.

The FT comparison uses **97.70%** from rerunning the four `full_finetune_bs64`
checkpoints, rather than the 97.95% in the
[public FluxVLA performance table](https://github.com/FluxVLA/FluxVLA#performance).
The public reference values remain 98.6 / 99.6 / 98.0 / 95.6±1.0. The historical
checkpoint selection, number of repeated runs and meaning of `±` in that table
have not been fully confirmed.

## 2. Models and Evaluation Protocol

The official model comes from `gs://openpi-assets/checkpoints/pi05_libero` and
was already fine-tuned for LIBERO by Physical Intelligence. No additional
training was performed for these evaluations. See
[OpenPI Fine-Tuned Models](https://github.com/Physical-Intelligence/openpi#fine-tuned-models)
for the release details. The same official weights serve all four suites; FT
uses four separately fine-tuned checkpoints. Their sources and pinned revisions
are listed under [FT model preparation](libero.md#pi05-ft-model-preparation).

| Setting | Official 10/5 | Official 10/10 | FT ZMQ (this run) |
| --- | --- | --- | --- |
| Predicted / executed actions | 10 / 5 | 10 / 10 | 10 / 10 |
| Environment seed / settling steps | 0 / 10 | 0 / 10 | 0 / 10 |
| Model seed assignment | `7 + global episode index within the suite` | Same | Same |
| Model RNG | Reset per request, `sequence_noise=False` | Same | Same |
| Parameter storage precision | FP32, retaining the official precision handling | Same | BF16 |
| Model and simulator scheduling | fixed | fixed | dynamic |
| Preprocessing / normalization | Official image, prompt, state-quantile and action transforms | Same | Image and mean/std transforms from the FT config |
| Two-view video | 4 fps | 2 fps | 2 fps |

For a full suite, the seed is `7 + task_index × 50 + episode_index`, with indexing
restarting for each suite. This protocol resets the RNG to the episode's seed
on every request, then draws the first initial noise sample. With the current
deterministic preprocessing, replanning within an episode uses the same initial
noise tensor. Observations still differ, so actions can change. This differs
from the continuous RNG stream in FT's native `scripts/eval.py` and from OpenPI
JAX's default RNG behavior.

The request fields `policy_seed`, `environment_seed` and `environment_case_index`
identify different aspects of an episode. LIBERO selects each task's fixed
initial state by case index, while the actual `simulator_seed=0`. Neither the
policy seed nor the recorded environment seed should be interpreted as the
simulator seed passed to LIBERO.

Relative to the official configs at the same revision, 10/5 overrides only
`themis.server.runner.sequence_noise=False`. The 10/10 run additionally sets
`eval.eval_chunk_size` and `themis.evaluation.execute_horizon` to 10 and adjusts
video capture to 2 fps to match the replanning frequency. Both retain
`keep_params_fp32=True`, official preprocessing and fixed scheduling.
`evaluation_protocol.json` and `official_reference_configs/` are archives
produced by external job tooling at the time. They are not distributed with this
repository or generated automatically by the current launcher.

Although all three runs use the same model RNG protocol, FT and the official
model still differ in weights, preprocessing, parameter precision and
scheduling. The score difference cannot be attributed solely to additional
fine-tuning. Matching episode identities and fixed worker assignments also does
not guarantee bitwise determinism on GPUs.

## 3. Runtime and GPU Statistics

| Runtime and throughput | Official 10/5 | Official 10/10 | FT ZMQ (this run) |
| --- | ---: | ---: | ---: |
| Total launcher runtime | 41 min 01 sec | 29 min 19 sec | 40 min 54 sec |
| Total loading and preparation (approx.) | 8 min 51 sec | 8 min 58 sec | 11 min 53 sec |
| Total rollout time (approx.) | 31 min 43 sec | 19 min 57 sec | 28 min 26 sec |
| Rollout throughput (episodes/hour) | 3,784 | 6,013 | 4,219 |

These timings come from the evaluation launcher logs and exclude platform
queuing, image pulls and dependency/hash checks performed before launch.
Rollout time is the sum of the suites' wall-clock times, not the sum of parallel
worker runtimes. Stage durations are approximate because they were derived from
log timestamps with one-second resolution and rounded throughput figures.

Compared with official 10/5, official 10/10 reduced total launcher runtime by
approximately 28.5% and increased rollout throughput by approximately 59%.
Peak memory usage per GPU was approximately 59,427 MiB for both official runs
and 25,287 MiB for FT. GPU measurements cover the whole device. Some `nvidia-smi`
requests timed out, leaving sampling gaps in all three summaries; these are not
complete performance profiles.

## 4. Result Artifacts, Versions and Validation Scope

All three runs used Themis `ee4257c3c8b2a17926d37683981f6e47be626453` and VLA
`f77e1ce1ee5c388ccf5cd906b1573c89f97b437b`. Startup logs recorded robosuite 1.5.1,
MuJoCo 3.2.6 and LIBERO 0.1.0, with Torch 2.8.0+cu128 for inference and Torch
2.8.0+cpu for simulation. These source SHAs identify the versions actually run;
they do not validate subsequent workspace changes.

At startup, both official runs verified the same expected SHA256 hashes and
wrote `checkpoint_hashes.json`:

| File | SHA256 |
| --- | --- |
| `model.safetensors` | `598451c937e8839b5a15fa54c35a233dc1cefd633c560d25d9fcbe7b6cf96a1a` |
| `dataset_statistics.json` | `72ff25e64ccdddea45dc816489682c776e5b4da22b7c75cdc6dd9ff3df72cb8c` |
| `tokenizer/tokenizer.model` | `8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6` |

All three cloud jobs reached `Succeeded`. Their complete console logs contain
`run_end status=finished episodes=500/500` for all four suites. The success
counts in this report were uniquely reconstructed from the cumulative success
rates at suite boundaries; subtracting successive cumulative counts yielded
each suite's result. **The episode files, task coverage and videos for these
three runs have not been downloaded again and audited individually.**

This report retains evaluation scores, model sources, the protocol and the
validation scope. Detailed statistical attachments have been archived outside
the repository. Raw logs, per-episode records and videos remain in the external
evaluation artifacts.

## 5. Reproduction Parameters and Config Sources

The official model evaluations used
`configs/pi05/libero_official/{SUITE}.py` from the FluxVLA commit listed above.
First prepare config copies with absolute paths to the checkpoint, statistics
and tokenizer. These configs and their parameter precision handling belong to
that VLA revision.

Run each of the four suites separately. Using a prepared `libero_spatial.py` as
an example, first start the model service in the model environment with the
10/5 configuration:

```bash
bash scripts/distributed_model.sh --fluxvla-root /path/to/FluxVLA \
  --config /prepared/libero_spatial.py \
  --ckpt-path /models/pi05_libero/model.safetensors \
  --gpus 8 --workers-per-gpu 3 --scheduling fixed \
  --cfg-options themis.server.runner.sequence_noise=False \
    eval.eval_chunk_size=5 eval.observation_video.fps=4
```

Then connect to that service from the simulator environment:

```bash
bash scripts/run_evaluation.sh --config /prepared/libero_spatial.py \
  --server-endpoint tcp://MODEL_HOST:15555 \
  --gpus 8 --workers-per-gpu 3 --scheduling fixed --full \
  --cfg-options themis.evaluation.execute_horizon=5
```

Use the corresponding prepared config for each remaining suite, starting the
model and simulator independently. For 10/10, set both the model's execution
horizon and the simulator's `execute_horizon` to 10, and set model video capture
to 2 fps. The prediction horizon remains 10.

See [FT model preparation](libero.md#pi05-ft-model-preparation) for model paths
and preparation of the four FT configs, and the
[ZMQ evaluation launcher guide](../guides/evaluation.md) for general launch
options. These commands reflect the evaluation protocol in this report. The
cleaned-up launcher has not yet been rerun for a full cloud evaluation; the
recorded scores do not replace validation of the new version.
