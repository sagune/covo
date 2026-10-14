# Joint 9B LoRA adaptation diagnostic — 2026-09-23, old-best revision

User authorized the three-condition experiment. No new frontend generation or full-data training is authorized by this run.

- User's latest instruction overrides the625 draft: use historical AISHELL dev-CER-selected `qwen35_9b_aishell_hardneg_dropout_2epoch_20260815/checkpoint-30022`. This does not mean recall-optimal or universally best. Local evidence: outputs/bridge-ablation-9b-20260908/RECALL_DIAGNOSIS.md and the corresponding REPORT.md. Recheck remote identity before launch.
- A: newly train connector only with frozen30022. Do NOT reuse625 predictions/checkpoint as a matched control. Afull initializer,17294 rows,seed917,accumulation8,2162 updates,connector LR5e-5,warmup32,AdamW weight decay0.01,clipping1.0.
- B: same Afull +30022, jointly train MLP and existing LoRA. LoRA LR5e-6, actual30022 module targets/rank/dropout retained. Base9B and SenseVoice stay frozen.
- C: restart from30022, train the same LoRA on text only. Same examples, order, optimizer-update count and LoRA LR as B. This is not an equal-FLOP comparison.
- D: unchanged30022 pure-text baseline using identical correction prompt/evidence/dev; no training.
- Afull was pretrained against625. A/B equally inherit this cross-adapter connector transfer; it is not a connector already aligned to30022. Do not silently add pure-audio pretraining or claim a clean from-scratch architecture comparison.
- Final checkpoint only, one epoch. Save resumable immutable checkpoints every256 updates plus final, including adapter, connector, optimizer, scheduler and RNG. Smoke weights are not used in formal training.
- A/B: text/true/zero/mismatch; C/D: text. Ten conditions, each all1334 historical contextual dev rows,21102 reference chars,599 hotword mentions. Not official14326dev and not unseen test.
- Same prompts, evidence whitelist, greedy generation96tokens, scoring and imperfect-duration-matched permutation as A. No restoration/protection rules. Invalid/empty outputs counted, no dropped samples.
- Score raw corpus CER, designated hotword recall, input-relative improved/worsened/unchanged, exact matches, S/D/I and corruption of initially correct inputs. Bootstrap Btrue against Dtext/Atext/Atrue/Btext/Bzero/Bmismatch/Ctext,2000 paired utterance resamples.
- Interpretation: audio sensitivity alone is not success; joint training must not degrade original text CER, should improve recall, and should beat text-only LoRA and relevant audio controls. Report point estimates and intervals, no statistical noninferiority claim from point estimates alone.
- Stop on source/cache mismatch, GPU contention, nonfinite loss/gradients, OOM, identity/metric audit failure or process failure. Preserve all output. No automatic protocol changes or parameter search.
- Actual scorer is NFKC + remove whitespace/punctuation, no OpenCC. Historical frontend/context/Afull history and error-enriched train distribution are inherited limitations.

Pipeline: CPU prepare → A/B/C smoke+reload → D matched baseline → B train+full dev → C train+full dev → A train+full dev → audited comparison bundle.
