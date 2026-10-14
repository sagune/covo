# Pronunciation-grounded correction learning v1 — 2026-10-03

User authorized adjustment after reviewing original9B error analysis. This is a new immutable experiment; old code, checkpoints and outputs remain unchanged. No output protection or inference routing is introduced. No claim of novelty or guaranteed CER preservation.

## What changes

Keep the same MLP connector architecture, frozen SenseVoice caches, original correction prompt and evidence whitelist. Restart from old-best checkpoint30022 LoRA plus the same Afull3243 connector used to initialize prior B; do not continue from smoke or a losing adapted9B checkpoint.

Train objective per row:

`L = CE_true + 0.5 CE_text + 0.1 I_correct KL(p_text.detach || p_true) + 0.1 I_aux softplus(1 - 5 (gap_true - gap_zero.detach))`.

CE targets are the existing reference JSON, not an added explanation. KL is over causally aligned answer-token distributions only. Correct-input flag uses TRAIN reference equality with TRAIN top1. Text peer uses eval/no-grad for KL; it is not a frozen independent teacher. Paired likelihood gradient calls disable Dropout modules only while keeping the model in training mode for gradient checkpointing; peers use eval/no-grad. Ordinary CE retains inherited training dropout. Base and SenseVoice remain frozen; existing LoRA and MLP jointly update. Only the final answer-length-plus-one logits are materialized, preserving exact causal answer scoring while reducing memory use.

Negative: an actual visible TRAIN top1/first8 N-best alternative within1..4 normalized character edits of the train reference. Both strings must be all Han, and dictionary sentence-context toneless pinyin must differ. Same-phoneme spellings, including tone-only distinctions, are excluded from acoustic contrast. No dev/test examples or errors influence pair mining. No synthetic names, oracle choice at inference, or reference passed to the prompt.

Eligible negatives are ordered by edit distance then candidate order. Acoustic contrast applies only to the deterministic quarter selected by sha256(917|train_id) first8 hex modulo4==0, reducing extra passes without changing the17294 base examples. Both likelihoods are divided by a common maximum target-token length. The zero-prefix gap is stop-gradient: no direct incentive to worsen the zero branch. Dictionary classification is a proxy, not wave-level supervision.

## Locked workflow

- Existing17294 train /1334 contextualdev,21102refchars,599mentions; not official full dev or unseen test. Audit IDs and normalized-reference separation.
- Same seed917, order,2162 updates, accumulation8 with final6, connectorLR5e-5/LoRALR5e-6, warmup32, decay.01, clipping1. One epoch; final checkpoint only, no dev-based early stopping.
- Extra forwards mean this is not compute matched to previous B. Four smoke inputs cover ordinary CE, correct-input KL, contrast and longest lengths. Verify finite losses, exact frozen weights, LoRA+connector updates, saving/reloading optimizer/RNG/adapter/connector and generation.
- New dev runs text/true/zero/mismatch, ALL1334 each, raw greedy JSON output, max96, original normalization. No rules that copy top1 or restore hotwords. Invalid/empty output counts, none dropped.
- Reuse hash-verified old D/text and B/text/true/zero only as explicitly labeled prior baselines with identical data/prompt/initialization provenance; do not claim equal objective or compute.
- Report CER, designated recall, S/D/I, improved/worsened, correct-input corruption and paired95%CIs. True must outperform its text/zero and old baselines without recall loss to support useful coupling. Point nondegradation is not a statistical guarantee.
- Stop on GPU contention, hash/identity changes, nonfinite values, OOM, failed reload, missing outputs. Preserve old outputs. Do not silently shorten train/dev or change coefficients based on dev.
- Existing cache and source locks are verified before every GPU phase; persistent pipeline and shared GPU lock prevent duplicate runs. Checkpoints every256 plus final retain optimizer/RNG state.

This experiment tests whether changing learning objectives makes the existing representation useful. It does not separately identify each loss contribution or prove an architecture improvement; ablations and a fresh held-out evaluation would follow only after validation.
