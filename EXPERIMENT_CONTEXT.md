# Experiment Context

## Current Scope

The maintained route is now **CB-SenseVoice + COVO**:

```text
audio
  -> SenseVoice encoder states
  -> TCResNet KWS
  -> phrase cross-attention context adapter
  -> context-biased SenseVoice CTC beam
  -> CB-SenseVoice candidate reranking
  -> COVO conservative correction
  -> final transcript
```

The earlier CB-Whisper implementation is retired. Its dedicated entry points,
configs and diagnostic patches are not part of the maintained branch. Historical
results remain available in Git history and in the comparison tables in
`README.md`.

## Research Constraints

- The method must remain lightweight and suitable for a paper.
- Do not use confidence gates, reference-aware selection or test-set correction
  dictionaries as the main method.
- Keep SenseVoice, the KWS backbone and the Qwen base model frozen; train only
  lightweight context adapters or LoRA adapters.
- Record each full experiment with its dataset, metric normalization, checkpoint
  and comparison against the previous accepted result.
- The target for AISHELL-NE is at least 90% hotword recall while minimizing CER.
- Run all experiments in `/root/autodl-tmp/great`, not the base environment.

## Accepted Components

### SenseVoice

- Base model: `iic/SenseVoiceSmall`
- Hidden states: extracted directly from the SenseVoice encoder
- Decoder: CTC prefix beam search with acoustic phrase evidence

### KWS

- Backbone: TCResNet
- Main checkpoint:
  `src/outputs/aishell_sensevoice_kws/checkpoints/f1G/f1G-epoch=16-step=102085.ckpt`
- KWS retrieves context candidates; it is not changed during CB-SenseVoice or
  COVO experiments.

### Context Adapter

- Architecture: phrase cross-attention
- Main checkpoint:
  `src/outputs/sensevoice_context_adapter/phrase_crossattn_aishell_full_20260722.pt`
- Accepted phrase-confidence weight: `14.0`

### COVO

- Base model directory: `models/Qwen3.5-4B`
- COVO source: `covo/`
- CER-oriented adapter:
  `covo/outputs/qwen35_cbwhisper_preserve2_aishell_train_noop_1epoch_bf16_bs7`
- Joint recall/CER adapter:
  `covo/outputs/qwen35_cbwhisper_hotword_preserve_dpo_30steps_bf16`

The adapter directory names retain their historical `cbwhisper` labels to match
existing checkpoints; they are consumed by the CB-SenseVoice pipeline.

## Accepted Results

All AISHELL numbers use corpus CER after OpenCC traditional-to-simplified
conversion, NFKC normalization and removal of whitespace and punctuation.

### AISHELL-NE 808

| System | CER | Recall@400 | R1 Recall@226 |
|---|---:|---:|---:|
| SenseVoiceSmall | 10.3609% | 39.75% | 10.18% |
| Original CB-SenseVoice | 7.8231% | 78.25% | 62.39% |
| CB-SenseVoice phrase cross-attention | 4.8273% | 80.75% | 66.37% |
| CB-SenseVoice w14 | 3.7951% | 91.50% | 84.96% |
| w14 + preserve2 COVO | 3.0268% | 89.25% | 81.42% |
| **w14 + DPO-30 COVO** | **3.1354%** | **90.50%** | **83.63%** |

The DPO-30 system is the joint main result because it satisfies the 90% recall
target. Preserve2 is retained as the lowest-CER ablation.

### Full AISHELL-1 Test

| System | Corpus CER |
|---|---:|
| SenseVoiceSmall | 6.2922% |
| SenseVoice + preserve2 COVO | 3.7647% |
| Routed SenseVoice / CB-SenseVoice w14 | 5.4894% |
| **Routed + DPO-30 COVO** | **3.2062%** |
| Routed + preserve2 COVO | 3.1929% |

Routing is based only on whether an external context list exists: the 808
contextual utterances use CB-SenseVoice and the other 6,368 utterances use plain
SenseVoice.

## Reproduction

Train SenseVoice KWS:

```bash
cd src
/root/autodl-tmp/great/bin/python run_CLI.py fit \
  --config configs/train-sensevoice-kws.yaml
```

Run CB-SenseVoice:

```bash
cd src
/root/autodl-tmp/great/bin/python run_CLI.py test \
  --config configs/cb-sensevoice-aishell.yaml
```

Export and convert COVO evidence:

```bash
cd src
CBW_EVIDENCE_ONLY=1 \
CBW_EVIDENCE_OUT=logs/cb_sensevoice_evidence.jsonl \
/root/autodl-tmp/great/bin/python run_CLI.py test \
  --config configs/cb-sensevoice-aishell.yaml

/root/autodl-tmp/great/bin/python analysis/cbsensevoice_covo_bridge.py prepare \
  --input logs/cb_sensevoice_evidence.jsonl \
  --output ../covo/data/processed/cb_sensevoice_messages.jsonl \
  --include-pinyin \
  --protect-supported-hotwords
```

Run COVO inference:

```bash
cd /root/autodl-tmp
PYTHONPATH=covo/src /root/autodl-tmp/great/bin/python \
  covo/scripts/infer_lora_text.py --help
```

Model weights, datasets, generated evidence, predictions and runtime logs are
local artifacts and must not be committed.
