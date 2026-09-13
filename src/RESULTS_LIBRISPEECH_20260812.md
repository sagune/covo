# LibriSpeech Experiment Notes (2026-08-12)

## Dataset

- Corpus: LibriSpeech
- Available splits: train-clean-100 (28,539), dev-clean (2,703), dev-other (2,864), test-clean (2,620), test-other (2,939)
- Smoke SFT set: 16 rows sampled from dev-clean, with 10 ranked ASR-style hypotheses per row

## Model Selection

- Base model: Qwen/Qwen3.5-9B
- Local path: `/root/autodl-tmp/cbwhisper_covo_migration_20260609_tar_extracted/models/Qwen3.5-9B`
- Training method: BF16 LoRA SFT, matching the existing COVO training pipeline
- LoRA targets: `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`
- LoRA configuration: rank 16, alpha 32, dropout 0.05

## Smoke Test

- Steps: 1
- Sequence length: 1,024
- Batch size: 1
- Gradient accumulation: 1
- Gradient checkpointing: enabled
- Trainable parameters: 29,097,984 / 8,982,901,248 (0.3239%)
- Train loss: 5.3488
- Train runtime: 3.464 seconds
- Observed peak GPU memory: 19,385 MiB
- Exit code: 0
- Adapter reload: successful (`PeftModelForCausalLM`)

## Artifacts

- Smoke data: `/root/autodl-tmp/src/logs/librispeech_qwen35_9b_smoke_sft_20260812.jsonl`
- Training log: `/root/autodl-tmp/src/logs/train_qwen35_9b_librispeech_smoke_20260812.log`
- Memory log: `/root/autodl-tmp/src/logs/train_qwen35_9b_librispeech_smoke_memory_20260812.txt`
- Adapter: `/root/autodl-tmp/cbwhisper_covo_migration_20260609_tar_extracted/covo/outputs/qwen35_9b_librispeech_smoke_20260812`

## Next Stage

Generate English ASR N-best/acoustic evidence for the training split, construct hard-negative and balanced COVO SFT examples with the existing method, then launch the full LoRA run. No full LibriSpeech training was started in this smoke test.

## Storage Cleanup

Old adapters, summaries, configs, trainer states, and result files were preserved. Optimizer/scheduler/RNG resume payloads from old checkpoints were removed, so retained checkpoints support inference and comparison but not exact optimizer-state resume.
