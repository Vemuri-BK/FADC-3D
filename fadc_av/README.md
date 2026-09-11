# FADC-AV

Development branch: `feature/fadc-AV`.

This package builds the verified discrete 3D FADC adaptation step by step.
Python implementation and tests live in Git; Kaggle will check out a pinned
commit and execute GPU verification and training scripts.

## Implemented: frequency decomposition

`frequency.py` provides immutable `FrequencyConfig` and
`FrequencyDecomposition3D`. Inputs use `(B,C,D,H,W)` layout. Symmetric,
unshifted FFT masks select `abs(f_axis) < 1/(2*k)` on every spatial axis.
The default cutoff denominators are `(2,4,8)`.

The module returns ordered high-frequency bands, the final low-frequency
residual, and a tuple describing which band slots have nonempty frequency
support. Duplicate masks produce explicit zero bands; the number of slots
stays fixed. DC remains in the low residual. Float16/bfloat16 inputs produce
float32 bands for FFT safety; float64 inputs retain float64 precision.

```python
from fadc_av import FrequencyDecomposition3D

parts = FrequencyDecomposition3D()(features)
reconstructed = sum(parts.high) + parts.low
```

Run from the repository root in the `fadc3d` environment:

```text
python -B -m unittest discover -s fadc_av/tests -v
```

Tests cover production-stage shapes, DC preservation with nonidentity gains,
conjugate symmetry, nested masks, sinusoids, duplicate bands, precision,
invalid inputs, and numerical gradient checking. GPU AMP testing remains a
later Kaggle integration step.

## Implemented: trainable frequency selection

`selection.py` adds `FrequencySelection3D` and immutable `SelectionConfig`.
One grouped spatial convolution per active high band predicts gains in [0,2]
via `2*sigmoid(logits)`. Zero initialization gives unit gains. By default
the low residual is unchanged; optional low-frequency attention is explicit.
Inactive band heads are skipped while their registered parameters remain
stable across input shapes. Ordinary forward returns only the selected features.

```python
from fadc_av import FrequencySelection3D, SelectionConfig

selector = FrequencySelection3D(in_channels=32, config=SelectionConfig())
selected = selector(features)
diagnostics = selector.forward_with_gates(features)
```

Diagnostic gains are returned, not cached on the model; they retain gradients
unless the caller detaches them or runs under `torch.no_grad()`. Inactive high
gains are `None`. Outputs retain the decomposition's at-least-float32 policy.
Recreate the same configuration before loading a state_dict; experiment-level
configuration serialization will be added with the checkpoint pipeline.

Selection tests cover nonidentity group gains, spatial variation, constant
preservation, optional low attention, inactive heads, parameter stability,
CPU bf16 autocast backward, and strict nonidentity weight reload.

## Implemented: shared-kernel dilation and voxelwise selection

`dilation.py` supplies `SharedKernelConv3D`, `VoxelwiseDilationSelector`, and
the integrated `AdaptiveDilatedConv3D`. Its default flow is:

```text
input -> FrequencySelection3D -> one shared kernel at dilations 1/2/3
                             -> voxelwise softmax weighted sum
```

```python
from fadc_av import AdaptiveDilatedConv3D, DilationConfig

block = AdaptiveDilatedConv3D(32, 64, DilationConfig(dilations=(1, 2, 3)))
block.set_temperature(1.5)
output = block(features)
diagnostics = block.forward_with_attention(features)
```

There is one learnable base kernel. Optional AdaKern adapts it per sample;
the default uses the static base kernel directly.
Branches preserve spatial size using stride one and padding equal to dilation.
The attention head is a small spatial convolution, ReLU, and zero-initialized
output head. Initial probabilities are uniform; this produces the average of
the dilation branches, not a dilation-one identity. `selection=None` disables
frequency selection for an ablation. Normalization and residual connections
will be added in matched network blocks rather than hidden inside this operator.

Temperature is a registered buffer and survives state_dict save/reload.
Extra state records an operator version and configuration, rejecting semantic
mismatches such as loading dilation-(1,2) weights into dilation-(1,3). Rebuild
with the same architecture configuration before loading. Runtime temperature
is restored independently of its constructor initialization value.

Weighted branches are accumulated without an additional branch stack. Training
still retains autograd intermediates; this is not a claim of constant-memory
backpropagation. Returned probabilities and expected dilation retain gradients
unless detached by the caller. Expected dilation is a diagnostic summary, not
an equivalent continuous sampling operator.

Tests compare nonuniform mixing and gradients against ordinary convolution
with explicitly zero-inserted kernels. They also cover odd/tiny/noncontiguous
inputs, gradients through frequency and attention heads, CPU bf16 backward,
non-default-temperature reload, and configuration mismatch rejection. A CUDA
FP16/GradScaler optimizer-step test is included and skips automatically without
CUDA.

## Implemented: optional AdaKern

`adakern.py` predicts separate input-channel and output-filter gains for the
spatial mean and residual components of the base kernel. A pooled Linear/ReLU
descriptor feeds four zero-initialized heads. Gains are `2*sigmoid(logits)`;
unit initial gains reconstruct the base kernel within floating-point tolerance.
No batch-dependent normalization or kernel-position attention is added.

```python
from fadc_av import AdaKernConfig, DilationConfig, AdaptiveDilatedConv3D

config = DilationConfig(adaptive_kernel=AdaKernConfig(hidden_channels=16))
block = AdaptiveDilatedConv3D(32, 64, config)
output = block(features)
```

The adapted tensor `(B,O,I,3,3,3)` is computed once per forward and reused
at every dilation. A grouped-convolution packing implements independent
per-sample kernels; no learned branch-specific kernels are introduced.
Set `adaptive_kernel=None` for the static-kernel ablation.

Operator state version is now 2 and includes the AdaKern configuration.
Version-1 prototype state dictionaries are intentionally rejected; no automatic
migration is provided. New states restore both adaptation weights and runtime
attention temperature. Tests include an independent scalar kernel reference,
separate-sample convolution outputs and gradients, reuse of the same adapted
tensor across branches, static/adaptive equivalence at initialization,
nonidentity checkpoint reload, and CPU mixed-precision backward.

Module-only local result before network integration: **36 tests passed, 2 CUDA tests skipped**. Both static
and adaptive CUDA FP16 optimizer-step checks remain for Kaggle verification.

## First image-only experiment

`model.py` now assembles the complete two-channel 3D segmentation network by
reusing `models/unet_3d.py` and replacing only `enc3.conv.block[3]`. Existing
BatchNorm, ReLU, pooling, decoder and skip connections are preserved. No new
residual blocks or dropout are added. Four frequency components, dilation
choices `(1,2,3)` and AdaKern are explicitly enabled. Temperature is fixed at
1.0 for this first experiment; uniform initial mixing is a branch average.

`experiment.json` specifies seed 42, width 32, patch 128x128x64, physical batch
2, 100 epochs, AdamW 1e-4, weight decay 1e-5, five warmup epochs, and validation
every 10 epochs on all 306 validation cases using full-volume sliding-window
inference (epochs 10, 20, ..., 100). The unchanged batch Dice+CE definition uses float32 loss
reductions. Validation uses argmax, overlap zero, constant blending and no TTA.
GPU windows are processed one at a time and assembled on CPU. Empty-ground-
truth cases are excluded from mean foreground Dice, matching MONAI's default;
per-case false-positive counts remain available.

`run.py` scans all cache arrays for shape, finite values, binary masks and
training patch compatibility; enforces 1200/306 cases and disjoint patient IDs;
and records the enumerated split. Its fingerprint covers patient paths and
file sizes, **not cache contents or preprocessing provenance**. Keep the Kaggle
dataset version fixed. This default split is validation, not a held-out test.

Augmentations reuse the existing cached dataset transforms, with MONAI worker
seeding and per-epoch loader seeds. Checkpoints save optimizer, scheduler,
scaler, RNG state, runtime attention temperature and configuration. Resume is
at epoch boundaries; an interrupted partial epoch is repeated. Preserve both
`last.pt` and `best.pt` together when moving to a new Kaggle session. Exact
cross-GPU/library reproducibility is not guaranteed. Synthetic CPU testing
compares resumed and uninterrupted model tensors exactly.

Historical baselines remain reference results. Corrected augmentation seeding
and explicit float32 loss reductions mean the final strict comparison should
also run `variant: baseline` through this same runner; this is supported by
changing the JSON variant and using a separate output directory.

The notebook `kaggle_fadc_av_enc3_full_s42.ipynb` is generated by
`python -m fadc_av.make_notebook`. Before Kaggle execution, publish these files
on `feature/fadc-AV`, enter that commit's full SHA in `EXPECTED_COMMIT`, attach
the two-channel cache and set its root path. It checks out that exact commit,
installs the locally tested MONAI 1.5.2 while retaining Kaggle PyTorch, runs
tests, then runs production-size real-data preflight before training. Preflight
uses two optimizer steps, one whole-volume validation and a weight reload.
It is an execution check, not evidence of segmentation accuracy or convergence.
GPU out-of-memory errors stop the run; batch size is never silently reduced.

Preflight displays per-case scan progress and optimizer-step progress. Training
and final evaluation pass `--preflight-report` to reuse successful checks:
the root, configuration, patient inventory, file sizes and modification times
must match. These metadata checks avoid decompressing all volumes again; they
are not cryptographic content checks. Old reports without file metadata require
one new preflight. Training displays epoch/total, batch progress, live/mean loss
and best Dice, and announces each new best checkpoint. A failed new preflight
invalidates the previous success report.

AMP gradient overflow now skips the unsafe optimizer update and lets GradScaler
lower its scale before the next batch. Nonfinite unscaled gradients without a
scaler, nonfinite loss, and eight consecutive overflows still stop execution.
Affected parameter names and scale changes are printed; cumulative skipped
updates and the current scale are logged. This handles transient overflow but
does not establish the source of persistent numerical instability. The model,
loss and optimizer configuration are unchanged; existing last.pt checkpoints
remain loadable. Resume restarts the incomplete epoch, not the interrupted batch.

Each epoch writes `train_log.json` and `train_log.csv` for later charts, including
training/validation/combined time in minutes (excluding checkpoint writes), loss,
Dice-loss and CE-loss components, learning rate used and next learning rate,
AMP skips, and pooled training-patch Dice/IoU/sensitivity/precision. Training
metrics use argmax predictions before updates on augmented sampled patches,
including batches whose AMP update was skipped; they are not full-volume scores.
Validation Dice/IoU/sensitivity are per-case means recorded only on scheduled
validation epochs; missing values remain null/blank. Older resumed epochs keep
their original fields: new metrics cannot be reconstructed retrospectively.

Local suite after integration: 40 passed, two CUDA checks skipped (42 total).
Local integration uses small synthetic volumes, not real patient data. Full-
size CUDA memory, AMP, and real-data behavior remain for Kaggle verification.
Best/last checkpoints, per-case metrics, logs, split inventory and package
versions are saved under `/kaggle/working/outputs`. Save a notebook version
with outputs to retain them. No GPU training has been launched locally.

Pretraining and metadata/language conditioning remain separate experiments.
The new operator does not import legacy FADC code.

## All-encoder placement experiment

`experiment_all_encoders.json` changes only `variant` to `fadc_all_encoders`.
Both convolutions in enc1 through enc4 use full FADC (eight adaptive operators).
Bottleneck and decoder remain plain; no residual connections or dropout are
introduced. Four bands, dilations 1/2/3, AdaKern, seed 42, width 32, batch 2,
100 epochs and validation every 10 epochs on all 306 cases match the enc3 run.

Generate the separate launcher with `python -m fadc_av.make_notebook
--variant all_encoders --commit FULL_SHA` (on one command line). The artifact is
`kaggle_fadc_av_all_encoders_full_s42.ipynb`, with output directory
`/kaggle/working/outputs/fadc_av_all_encoders_full_s42`. On another Kaggle account,
attach the same cache dataset version and update CACHE_ROOT for that account.
Start from scratch: an enc3 checkpoint cannot resume this architecture.
Production GPU memory/throughput must pass preflight; eight adaptive operators
can be substantially more expensive. Do not reduce batch or width silently,
because doing so changes the placement comparison. Local tests check all eight
locations, volumetric output shape, finite gradients and checkpoint reload.
