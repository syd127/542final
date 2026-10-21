# RhythmFlow

Music-conditioned human motion in-betweening with conditional flow matching,
trained on AIST++. Given a start pose, an end pose, a music track and a vector
of kinematic style dials, the model generates the motion in between.

EECS 542 final project. This file records what was built, what the numbers say,
and what did **not** work, so none of it has to be rediscovered.

---

## 1. Headline findings

**Endpoint conditioning works.** 4.4 cm mean endpoint error on held-out data at
the best sampling setting. The test split shares neither a sequence nor a music
track with training, so this is generalisation to new choreography *and* new
audio simultaneously.

**The kinematic style dials work.** All five are monotonic (Spearman 0.90–1.00),
calibrated to within 0.02–0.11 percentile units, and off-diagonal leakage is
2–4x below the diagonal gain. On the energy/travel axis the model is *more*
disentangled than the data: real motion couples them at rho = +0.59, the model
propagates only +0.13.

**Flow matching is efficient, more so than the literature claim.** Quality peaks
at **2–5 ODE steps** and degrades with more. Related work cites ~10 steps for
flow against ~50 for diffusion; this model is best at 2–5.

**Music conditioning does not produce beat synchronisation.** Dropping the audio
entirely leaves beat alignment unchanged (BAS 0.815 without audio vs 0.810 with).
This is the project's main negative result and it is load-bearing — see §6.

**A methodological finding, probably the most novel part.** Generated motion
carries 20–300x more high-frequency power than the ground truth's own noise
floor. Any style metric built on unfiltered finite differences therefore measures
the generator's jitter rather than the choreography, and will report that style
control has failed when it has not. Band-limiting the descriptor recovers the
control the naive metric hides. See §5.

---

## 2. Results

### Sampling-quality sweep (40 held-out clips, 20 test sequences x 2 windows)

| method | endpoint (cm) | BAS | beats/s | HFR % | foot skate |
|---|---|---|---|---|---|
| ground truth | 0.00 | 0.746 | 3.25 | 0.013 | 0.337 |
| SLERP interpolation | 0.00 | 0.075 | 0.37 | 0.000 | 0.055 |
| flow, 1 step | 8.87 | 0.854 | 5.71 | 2.158 | 1.522 |
| flow, 2 steps | **4.36** | 0.775 | 4.71 | 0.516 | 0.755 |
| flow, 5 steps | 4.60 | 0.761 | 4.22 | **0.179** | **0.684** |
| flow, 10 steps | 5.28 | 0.773 | 4.43 | 0.336 | 0.806 |
| flow, 25 steps | 5.77 | 0.794 | 4.62 | 0.453 | 0.925 |
| flow, 50 steps | 5.98 | 0.810 | 4.91 | 0.699 | 1.032 |
| flow, 50 steps, **no audio** | 5.07 | 0.815 | 5.15 | 0.657 | 0.979 |

`HFR` = fraction of joint-position power above 8 Hz, i.e. jitter.
`beats/s` = density of motion beats (velocity minima).
Reproduce with `scripts/evaluate.sbatch`.

**How to read this table, carefully:**

- **SLERP is the baseline that matters.** It gets the endpoints for free (it *is*
  the endpoints) and has no jitter, but BAS 0.075 against the model's 0.76. The
  model produces something interpolation fundamentally cannot.
- **BAS above ground truth is an artifact, not a win.** The model scores higher
  than real dance, but the `beats/s` column shows why: GT has 3.25 velocity
  minima per second, the model 4.2–5.7. More beats means more chances to land
  near some audio beat. Per-beat alignment is *worse* than real dance. Do not
  report BAS without beat density beside it.
- **More ODE steps is worse**, which is surprising. The sweep ran at fixed
  guidance 2.0, and optimal guidance depends on step count, so accumulated
  classifier-free-guidance overshoot is a plausible alternative explanation to
  anything intrinsic. A guidance x NFE grid would settle it and is cheap. Until
  then, state the efficiency result with that caveat.
- **Jitter and foot skate are real defects.** HFR is 14x the ground-truth floor
  even at the best step count; foot skate is 2–4.5x GT. Both are visible in the
  sample videos.

### Style dial controllability (`scripts/controllability.py`)

| dial | Spearman | gain | achieved range | calib. MAE |
|---|---|---|---|---|
| `energy` | 0.90 | 0.40 | 0.31 → 0.63 | 0.050 |
| `travel` | 0.90 | 0.52 | 0.38 → 0.84 | 0.081 |
| `level` | 1.00 | 0.65 | 0.41 → 0.93 | 0.023 |
| `reach` | 1.00 | 0.72 | 0.27 → 0.84 | 0.052 |
| `updown` | 0.95 | 0.38 | 0.36 → 0.65 | 0.112 |

Gain < 1 means the model undershoots: requesting the full range moves the output
38–72% of it. That is ordinary compression for classifier-free guidance and can
be increased with stronger style guidance at sampling time.

Leakage matrix `M_ij` (sweep dial i, measure dial j) has all off-diagonals
|M| <= 0.18 against diagonals of 0.38–0.72. Mean |excess over the data prior| is
0.184. **Leakage must be judged against the data prior, not against zero** — the
dials are genuinely correlated in real motion (max rho 0.48 between `energy` and
`travel`), and a generative model following that correlation is behaving
correctly. Only excess leakage indicates entanglement.

**Caveat:** the dial sweeps use one base clip and one seed per level, so Spearman
is computed on five points. Directions are clear; the numbers are not tight.
Scaling to ~10 clips x 8 seeds is one GPU job.

### Music swap (same endpoints, same dials, tied noise)

Five unseen tracks shift the achieved dials by 0.08–0.19 percentile units while
endpoint error holds at 4.3 cm. So audio does influence movement character. But
note the effect is **smaller than the style dials' own authority** and comparable
to residual leakage — and §6 shows it does not yield beat sync.

### Single dial vs five dials

The 5-dial model (`runs/style5`) and the earlier 1-scalar model (`runs/main`)
are indistinguishable on quality: endpoint 4.4–5.7 vs 4.3–5.2 cm, v-loss
0.19–0.28 vs 0.21–0.29. **Richer conditioning cost nothing.**

---

## 3. Data pipeline

### The official AIST++ motion archive is broken

`motions.zip` from the AIST++ GitHub release contains **411 of 1408 sequences**.
The download page advertises 306 MB; the hosted asset is 78.7 MB. The download
is byte-complete and passes `unzip -t`, so the upload itself is incomplete, and
`ignore_list.txt` (46 entries) does not explain the gap. The Google Cloud bucket
is private and no mirror of the original exists on OpenDataLab or hyper.ai.

**Workaround:** the HuggingFace mirror `haiphamcse/aistpp_v3` holds the same SMPL
annotations cut into 150-frame windows at stride 75, covering all 1408 sequences.
Those windows tile exactly, so `scripts/build_motion_store.py` stitches them back
into full sequences and **verifies every one of the 411 authentic sequences
bit-for-bit — zero mismatches.** The mirror is unmodified AIST++, already at
30 fps.

Cost: stitching loses a tail of up to 149 frames per sequence, **13.5% of
frames**, measured against the 411 ground-truth files. The alternative was using
only those 411 (1.27 h of motion) instead of 4.64 h.

### Other data facts worth knowing

- **Audio alignment is zero-offset.** Verified by cross-correlating the mirror's
  per-window audio against the full tracks: every window lands exactly where its
  frame index predicts, correlation 0.999. Motion frame *k* ↔ music frame *k*.
  (`scripts/check_audio_align.py`)
- **49 of 151 feature channels are frozen.** AIST++ is fit from COCO-17 keypoints,
  which have no toe or hand landmarks, so SMPL joints 7, 8, 10, 11, 20, 21, 22, 23
  (ankles, feet, wrists, hands) sit at identity rotation. Their std is ~1e-4;
  normalising by it would amplify pure fitting noise ~5000x and hand a third of
  the loss budget to unpredictable garbage. `compute_stats.py --std-floor 1e-2`
  gives them std 1, so they cost nothing and still decode to identity.
  **Consequence for the paper: feet cannot flex in this data**, so foot-contact
  supervision is coarser than EDGE's. That is a dataset limitation, not a choice.
- **The floor is not at y=0 and moves between sequences** (mean 0.761 m, range
  0.561–0.921). It must be estimated per sequence from a low percentile of foot
  height. `motion.estimate_floor()`.
- **`keypoints3d` contains NaNs** (0.03% of frames, 14/120 sequences sampled).
  Harmless but must be masked wherever keypoints are used; it silently poisoned
  the first FK validation run.
- **The test split is unusually clean**: zero sequence overlap *and* zero music
  overlap with training (10 held-out tracks, one per genre).

### SMPL body model

License-gated, registration required at <https://smpl.is.tue.mpg.de/>. Version
**1.1.0** includes the neutral model (earlier versions shipped male/female only,
with neutral distributed separately through SMPLify). AIST++ is fit at zero
betas.

Joint-level forward kinematics needs only two things from the 247 MB pickle: the
**24x3 rest-joint positions** (stored directly as `J`, no need to compute
`J_regressor @ v_template`) and the **kinematic tree**. Everything else is for
meshes. `scripts/smpl_fk.py` reads it with a restricted unpickler that allows
only numpy/scipy reconstruction and stubs out chumpy, which avoids installing
chumpy at all (it is unmaintained and breaks on numpy >= 2).

FK validated against `keypoints3d`, which is independently triangulated:
**median 3.4 cm, zero NaNs over 120 sequences.** That 3.4 cm is the *expected*
disagreement between COCO's surface landmarks and SMPL's internal joints, not
error — a wrong kinematic tree or rotation order would show tens of centimetres.

The model file must never be committed (`*.pkl` is gitignored).

---

## 4. Method

**Representation**, 151-d per frame at 30 fps, matching EDGE so numbers stay
comparable: 4 foot contacts + 3 root translation (metres, floor-aligned) +
24 joints x 6D rotation.

**Clips** are windows of 15–150 frames (0.5–5 s), padded to 150 with a validity
mask. Frames 0 and N-1 are the conditioned keyframes (one frame each end).
Windows are canonicalised to the first frame: root translated to the xz origin,
heading rotated to face +z. Variable gap length is a training detail, not a
contribution — CondMDI already trains with random keyframe masks.

**Conditioning**, three signals:
1. *Keyframes* — values plus a binary observation mask concatenated to the input
   (CondMDI's mechanism, moved into a flow-matching model).
2. *Music* — 35-d librosa stack (envelope, 20 MFCC, 12 chroma, onset, beat) at
   exactly 30 feature frames per second, by setting the sample rate to
   `FPS * HOP_LENGTH` = 15360 Hz. Per-frame, via cross-attention.
3. *Style* — five kinematic dials, per-dial Fourier embeddings, summed into every
   token. **Per-dial classifier-free dropout** so a single dial can be swept
   while the others sit neutral, plus a forced whole-vector drop so the
   unconditional branch guidance extrapolates from is actually trained
   (independent drops would co-occur with probability `cfg_drop^5`).

**Objective.** Linear/OT path `x_t = (1-t) x_0 + t x_1`, regressing `v = x_1 - x_0`.
Auxiliary kinematic losses are applied to the clean estimate recovered from the
predicted velocity, `x1_hat = x_t + (1-t) v`: FK joint position, velocity, foot
contact and a skate penalty.

**Model.** 8-layer transformer, d_model 512, self-attention over frames plus
cross-attention to audio, ~35 M parameters.

---

## 5. The style dials, and why they are band-limited

See `scripts/kinematics.py`. All computed on joint positions passed through a
**zero-phase 4th-order Butterworth low-pass at 6 Hz** before any differencing,
then normalised to [0,1] by quantile-normal mapping against training-set
quantile tables.

| dial | definition | reads as |
|---|---|---|
| `energy` | RMS body-relative, yaw-removed live-joint speed / body height | calm ↔ vigorous |
| `travel` | mean horizontal pelvis speed / body height | in place ↔ travelling |
| `level` | mean pelvis height / body height | crouched ↔ upright |
| `reach` | mean end-effector distance from pelvis / body height | enclosed ↔ open |
| `updown` | upper vs lower body share of kinetic energy, in [-1,1] | footwork ↔ arm-led |

Yaw is removed per frame in `energy` so that a pirouette does not register as
limb articulation and travelling does not inflate articulation energy.

Dial correlation in real motion (this is the **data prior** for leakage analysis):

```
           energy  travel   level   reach  updown
 energy     +1.00   +0.48   -0.41   -0.13   +0.12
 travel     +0.48   +1.00   +0.07   +0.09   -0.17
  level     -0.41   +0.07   +1.00   +0.25   -0.07
  reach     -0.13   +0.09   +0.25   +1.00   +0.15
 updown     +0.12   -0.17   -0.07   +0.15   +1.00
```

Max off-diagonal 0.48, so these are five distinct axes, not one statistic in
disguise.

### Why filtering is not optional

Measured with `scripts/spectral_analysis.py`:

| | >4 Hz | >6 Hz | >8 Hz | >10 Hz |
|---|---|---|---|---|
| ground truth (test) | 0.110% | 0.019% | 0.007% | 0.004% |
| training motion | 0.310% | 0.060% | 0.021% | 0.010% |
| **generated** | **2.203%** | **1.590%** | **1.275%** | **0.924%** |
| excess | 20x | 82x | 181x | **245x** |

Real dance motion is essentially bandlimited — 99.9% of its power is below 4 Hz.
The generator adds roughly 30 dB of broadband jitter from ~4 Hz upward. Since
velocity, acceleration and jerk weight the spectrum by w^2, w^4, w^6, an
unfiltered descriptor measures that jitter. Effect of the cutoff on the
generated/ground-truth acceleration ratio:

| cutoff | raw | 10 Hz | 8 Hz | 6 Hz | 5 Hz | 4 Hz |
|---|---|---|---|---|---|---|
| ratio | 3.98x | 2.20x | 1.68x | **1.35x** | 1.20x | 1.05x |

**6 Hz is chosen deliberately over 4 Hz.** 4 Hz drives the bias to 1.05x, but
ground truth keeps 99.98% of its power below 6 Hz, so filtering harder would
*hide* a genuine model defect behind the metric rather than measure it. With the
final dials the generated/GT ratios are 0.86 / 1.01 / 1.01 / 0.97 for
energy / travel / level / reach, against **3.98x** for the old scalar.

### What this fixed

The first style descriptor was a single mean-acceleration scalar. It reported
that the dial had a floor — requesting percentile 0.05 produced 7.29, and the
sweep inverted between 0.55 and 0.80. Re-measured band-limited, **the same
samples show a strictly monotonic response** (0.128 → 0.603 over the request
range). The apparent floor was the generator's own noise floor sitting at roughly
the 30th percentile of the old metric, which made every request below 0.3
unmeasurable. The control had been working the whole time; the metric was hiding it.

### Rejected descriptors, and why

- **SPARC** (spectral arc length) — standard smoothness measure, nearly immune to
  noise, but unusable on our window lengths: rank correlation between a 15-frame
  estimate and the full-clip value is ~0, because its frequency resolution
  collapses. Fine as an *evaluation* metric on clips >= 90 frames.
- **LDLJ** (log dimensionless jerk) — strongly noise-sensitive as the motor-control
  literature warns, and confounded with articulation energy (rho -0.58).
- **A tempo dial** (Hjorth mobility of velocity) — computable and informative, but
  carries ~10% upward bias from residual jitter and largely duplicates what the
  music conditioning already supplies. Including it would muddy the central
  question of whether the music drives the motion.

---

## 6. The music-conditioning problem

`flow, 50 steps` scores BAS 0.810. The **same model with audio dropped** scores
**0.815**, with better endpoint error. Beat alignment is unaffected by removing
the music entirely.

Taken with the BAS density artifact (§2), the honest reading is that the model
produces generically dance-like motion with many velocity minima, which scores
well on BAS without actually following the beat.

What the music *does* do: swapping tracks with noise tied shifts achieved dials
by 0.08–0.19 percentile units, so it influences movement character.

Caveat in fairness: dropping audio through the classifier-free-guidance null path
is not identical to a model trained without audio. The null branch *was* trained
(10% dropout), so the ablation is legitimate, but a from-scratch no-audio model
is the clean comparison and has not been run.

Plausible causes worth investigating: the 35-d librosa stack may be too weak a
representation (EDGE uses 4800-d Jukebox features); cross-attention to audio may
be too easy to ignore when keyframes already constrain the motion heavily; and
0.5–5 s windows may be too short for beat structure to matter.

---

## 7. Running it

```bash
module load python/3.11.5 && source .venv/bin/activate
```

| step | command | notes |
|---|---|---|
| build motion store | `scripts/build_motion_store.py` | stitches the mirror, verifies vs official |
| audio features | `scripts/extract_audio.py` | 35-d at 30 fps |
| per-sequence arrays | `sbatch scripts/prep.sbatch` | FK, floor alignment, contacts |
| stats + dial quantiles | inside `prep.sbatch` / `restyle_and_launch.sbatch` | prints the dial correlation matrix |
| train | `sbatch scripts/train.sbatch <run>` | self-chaining, see §8 |
| samples + HTML | `sbatch scripts/samples.sbatch` | self-contained page with videos |
| metrics | `sbatch scripts/evaluate.sbatch` | NFE sweep, SLERP, no-audio, controllability |
| status | `scripts/status.sh <run>` | queue, step count, loss history |

### Every file in `scripts/`

**Library modules** (imported, not run directly):

| file | what it holds |
|---|---|
| `kinematics.py` | the five style dials, the 6 Hz filter, quantile normalisation. **The only file to edit to change the dial set** |
| `motion.py` | 6D rotation conversions, differentiable FK, foot contacts, the 151-d layout |
| `smpl_fk.py` | reads the SMPL pickle with a restricted unpickler; rest joints + kinematic tree |
| `dataset.py` | clip sampling, per-clip canonicalisation, style labelling |
| `model.py` | the transformer velocity field and its three conditioning paths |

**Pipeline stages** (run in this order):

| file | stage |
|---|---|
| `build_motion_store.py` | stitch the HF mirror into full sequences, verify against the official 411 |
| `extract_audio.py` | 60 wav tracks → 35-d features at 30 fps |
| `build_dataset.py` | FK, floor alignment, contacts → per-sequence arrays |
| `compute_stats.py` | normalisation stats + per-dial quantile tables + dial correlation matrix |
| `train.py` | the training loop, resumable |
| `generate_samples.py` | held-out samples, music swap, per-dial sweeps |
| `render_html.py` | skeleton videos → one self-contained HTML page |
| `evaluate.py` | NFE sweep, SLERP baseline, no-audio ablation, all metrics |
| `controllability.py` | per-dial response, gain, leakage vs the data prior |

**Checks and one-off analyses** (safe to run any time, nothing depends on them):

| file | what it answers |
|---|---|
| `check_model.py` | do shapes, masking and gradients work before spending a GPU job |
| `check_audio_align.py` | is the audio zero-offset to the motion (yes, 0.999) |
| `spectral_analysis.py` | where motion energy and jitter sit in frequency — produces §5's tables |

**Job scripts**: `prep.sbatch`, `restyle_and_launch.sbatch`, `train.sbatch`,
`samples.sbatch`, `evaluate.sbatch`, plus `status.sh`. Note `.gitignore` excludes
`*.sbatch`, so these are **not in version control** — `train.sbatch` in particular
holds the crash-resilient chaining logic described in §8.

### Artifacts

| path | what it is |
|---|---|
| `results/rhythmflow_style5.html` | **current** qualitative page, 3.7 MB, self-contained — `scp` and open |
| `results/eval_style5.npz` | the metrics table from `evaluate.py` |
| `results/samples_style5.npz` | generated + ground-truth joints for the 36 sample clips |
| `results/baseline_1dial.html` | the earlier single-scalar model, kept for comparison |
| `runs/style5/last.pt` | **the model**, step 200000, EMA weights included |
| `runs/main/last.pt` | the 1-scalar baseline, also 200000 steps |

---

## 8. Cluster notes (Great Lakes)

These shaped the design and are easy to rediscover painfully.

- **1 GPU per user, 8 h max wall time** (`sacctmgr show assoc user=$USER`).
  Training must checkpoint and resume; it cannot be one long run.
- **Short jobs backfill dramatically better.** A 1 h request started immediately
  while 2 h and 8 h requests waited 2.5 h — and 8 h bought no worse position than
  2 h. Resource size matters too: 8 CPU / 64 GB / 2 h queued 3 hours behind
  4 CPU / 24 GB / 1 h.
- **`spgpu2` rejects the class account** (restricted to owned accounts). `spgpu`
  had 409 queued jobs; the V100 `gpu` partition was free. Test with
  `sbatch --test-only`.
- **No PyTorch module exists** — only `tensorflow/2.7.0`. Use a venv.
- **`train.sbatch` queues its successor with `afterany` *before* training starts**,
  so the chain survives crashes, OOM, node failure and timeouts. This earned its
  keep: in the 5-dial run **15 of 21 chunks ended in TIMEOUT** and it still
  reached 200 k steps. An `afterok` resubmit would have died at chunk 1.
  Two guards stop runaways: a job starting at the step target exits without a
  successor, and a job finding no progress since the previous job stops the chain.
- The timeouts come from the in-job budget timer starting *after* setup, so later
  chunks overran their wall limit. Loss was capped at <2000 steps per incident by
  periodic checkpointing. **Fix:** base the budget on Slurm's job start time, and
  stop writing optimiser state on every save.
- Do not run preprocessing on the login node; it crawled at load 22.

---

## 9. What is missing

- **No FID / diversity metrics.** Distributional realism is unmeasured; only
  endpoint accuracy, beat alignment, jitter and skate are.
- **No matched diffusion baseline.** The flow-vs-diffusion efficiency claim in the
  related-work draft is not tested. Scope the claim to "flow matching at varying
  NFE" or train a diffusion model with the identical backbone (~15 GPU-hours).
- **Controllability needs more samples** — currently 1 base clip, 1 seed per
  level, Spearman over 5 points.
- **NFE sweep is confounded with guidance**, fixed at 2.0. Needs a guidance x NFE grid.
- **No from-scratch no-audio model** for a clean audio ablation.
- **Jitter is unfixed.** The spectral analysis gives a principled target: penalise
  high-frequency power during training, or filter at output.
- **Skeleton rendering only**, no mesh, so subtle quality is hard to judge.

## 10. Honest novelty assessment

**Standard, do not claim:** attribute conditioning with classifier-free guidance;
LMA-inspired kinematic features; SPARC/LDLJ as smoothness measures; zero-phase
Butterworth filtering of mocap; genre-conditioned music-to-dance generation.

**Defensible, stated carefully:** continuous, interpretable, multi-dimensional
kinematic dials for *music-conditioned in-betweening* appear not to have been
reported (phrase as "to our knowledge", and search again before submitting); and
the finding that generator jitter sits ~30 dB above the ground-truth noise floor
and corrupts unfiltered energy-style metrics, with band-limiting recovering
control the naive metric hides. The second is the most likely to be genuinely new
and is a useful caution for anyone evaluating "energy" in generated motion.

**Do not overclaim:** that the dials are LMA Effort factors (they are proxies);
that the dials are disentangled (they are correlated in data up to rho 0.48, and
should be); that keyframe conditioning in flow matching is new (MotionLab 2025
does rectified flow with sparse keyframes — read it before writing the related
work); or that the model controls style independently of music.
