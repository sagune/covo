
# Adding User Feedback To Enhance CB-Whisper

This repository contains code that allows to reproduce all experiments performed in the paper "Adding User Feedback To Enhance CB-Whisper".

## Setup

Create a conda environment and activate it

```bash

  conda activate biasing-whisper
```

Install ffmpeg and the necessary requirements

```bash
  conda install 'ffmpeg<5'
  pip install -r requirements.txt
```
    
## Build Datasets

The following bash scripts will create compatible folder structures for the scripts present in this repository, as well as do all the pre-processing necessary to train and evaluate the KWS classifier.

### Aishell-KWS and Aishell hotwords subsets

To build the Aishell-KWS dataset, download the `data_aishel.tgz` file from [here](https://www.openslr.org/33/) and place it on the directory where the dataset will be built. 

In the project directory, do as follows

```bash
cd datasets/aishell/
```

Activate the conda environment

```bash
conda activate kws
```

And run the bash script

```bash
bash build.sh
```

You will be asked to provide the path to the `tgz` file. It will take several hours, so make sure you open some `tmux` session and let it run uninterruptedly.

### MLS-KWS

To build the MLS-KWS dataset, download the zip files from [here](https://www.openslr.org/94/) for the English, German, French, Spanish, Portuguese and Polish languages. Everything else is equivalent to what was done with the Aishell-KWS dataset. 

### ACL6060

To build the ACL6060 dataset, download the zip file from [here](https://aclanthology.org/2023.iwslt-1.2/). Everything else is equivalent to what was done with the Aishell-KWS dataset. 

## The KWS Classifier

The CNN classifier for KWS was inspired in the one originally proposed in [CB-Whisper](https://arxiv.org/abs/2309.09552). This repository contains additional features that allow to reproduce the experiments on KWS performed in the afforementioned paper:
* Training using features derived from either TTS-generated or natural-speech audios for the keywords, or a mixture of both;
* DANN and [DANNCE](https://arxiv.org/abs/2102.03924) implementation;
* Validation of checkpoints using more than one dataset (for domain generalization analysis).

### Training

In the project directory, do as follows

```bash
cd src/
```

And run the following command

```bash
python3 run_CLI.py fit --config configs/train.yaml
```

In the `train.yaml` file, you will be able to set different hyperparameters, the paths to the dataset folders, which datasets to validate the model every epoch, the logger, and other training details. Important settings that must be introduced are capitalized and between square brackets.

### Evaluation

The following can be used to evaluate the precision, recall and F1 scores of the KWS classifier on the test sets of the different datasets.

In the project directory, do as follows

```bash
cd src/
```

And run the following command

```bash
python3 kws.py test --config configs/kws-aishell.yaml
```

For ease of use, there is one config `yaml` file per dataset. Do not forget to set the paths to the dataset folders and the given checkpoint to evaluate. Important settings that must be introduced are capitalized and between square brackets.

## Evaluate CB-Whisper with PBAWhisper

The following can be used to evaluate the entity recall of the CB-Whisper model on the test sets of the different datasets, using the KWS classifiers developed with these scripts. These results were not reported in the paper "Adding User Feedback To Enhance CB-Whisper". This version of CB-Whisper uses a wrapped version of Huggingface's `WhisperForConditionalGeneration`, also known as PBAWhisper, that can perform longform transcription jointly with keyword spotting on the go.

In the project directory, do as follows

```bash
cd src/
```

And run the following command

```bash
python3 cb-whisper.py test --config configs/cb-whisper-aishell.yaml
```

For ease of use, there is one config `yaml` file per dataset. Do not forget to set the paths to the dataset folders and the given checkpoint to evaluate. Important settings that must be introduced are capitalized and between square brackets.

### Current CB-Whisper metric-improvement experiments

Current work focuses on improving CB-Whisper entity recall and hotword-specific recognition quality on Aishell hotword evaluation.

Run from the repository root:

```bash
cd src/
python3 cb-whisper.py test --config configs/cb-whisper-aishell.yaml
```

Useful debug/output environment variables:

```bash
cd src/
CBW_DEBUG_LOG=logs/runtime_probe.jsonl \
CBW_DEBUG_MAX_SAMPLES=100 \
CBW_DEBUG_CLEAR_ON_START=1 \
CBW_METRICS_OUT=logs/test_metrics.csv \
python3 cb-whisper.py test --config configs/cb-whisper-aishell.yaml
```

The current Aishell CB-Whisper config enables KWS-based prompting, short-form n-best rescoring, pinyin-based scoring/repair, consensus reranking, and oracle n-best diagnostics. Main output files:

* `logs/test_metrics.csv`: final Entity Recall, CER, Hotword Sentence CER, Hotword Only CER, and WER with confidence intervals.
* `logs/runtime_probe.jsonl`: sampled per-utterance debug events, keyword candidates, prompt state, n-best rerank traces, and normalization previews.
* `logs/oracle_nbest_detail_aishell.csv`: per-candidate n-best diagnostic table.
* `logs/oracle_nbest_summary_aishell.csv`: per-sample top1-vs-oracle diagnostic summary.

KWS checkpoint policy: keep the KWS model fixed unless the experiment is explicitly about KWS retraining or KWS ablation. The current best KWS checkpoint is:

```text
src/mlruns/641753688314575260/d9b9fafe87d64bc98400705d2c58525c/checkpoints/f1G-epoch=7-step=48040.ckpt
```

This KWS model already includes local improvements over the original CB-Whisper KWS setup. Current experiments should primarily modify CB-Whisper decoding, prompting, normalization, rescoring, repair, and evaluation logic, while keeping this checkpoint as a controlled variable.

Method constraint: CB-Whisper is intended to be a lightweight improvement over Whisper, so proposed changes must stay lightweight at inference time. Avoid methods that require a much longer recognition pass, many repeated ASR calls, heavy external models, or large post-hoc search. Because this work is for a paper, changes should also remain methodologically clean and explainable; avoid over-engineered case-specific patches that only tune around observed failures.

Experiment goal: push KWS/keyword-related recall toward at least 0.90 while keeping CER as low as possible. Entity Recall is the primary hotword-success metric for CB-Whisper evaluation; CER, Hotword Only CER, and WER are guardrail metrics and should not be sacrificed casually for recall gains.

Experiment bookkeeping rule: after each code change, commit to git; after each experiment, append the result here and compare against the previous run or best known run.

| Date | Config | Checkpoint | Key settings | Entity Recall | CER | Hotword Only CER | WER | Notes |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| 2026-04-27 | `configs/cb-whisper-aishell.yaml` | `src/mlruns/641753688314575260/d9b9fafe87d64bc98400705d2c58525c/checkpoints/f1G-epoch=7-step=48040.ckpt` | KWS prompt, n-best=8, pinyin rescore, surface/consensus repair, consensus rerank | 0.8475 | 0.0822 | 0.1087 | 0.5161 | Current baseline. Oracle n-best top1 recall is 0.8455, oracle recall is 0.8650, leaving about +0.0194 recall headroom in the candidate pool. |
| 2026-04-30 | `configs/cb-whisper-aishell.yaml` | `src/mlruns/641753688314575260/d9b9fafe87d64bc98400705d2c58525c/checkpoints/f1G-epoch=7-step=48040.ckpt` | Baseline rerun in `/root/autodl-tmp/great`; same settings as above | 0.8475 | 0.0822 | 0.1087 | 0.5161 | Reproduced baseline in about 35m16s. Point metrics match 2026-04-27; only bootstrap confidence intervals changed slightly. Oracle n-best summary unchanged: top1 recall 0.8455, oracle recall 0.8650. |
| 2026-04-30 | `configs/cb-whisper-aishell.yaml` | same fixed KWS checkpoint | Tried broader keyword use: `prompt_max_injected_keywords=6`, `rescore_max_keywords=16` | 0.8464 | 0.0878 | 0.1122 | 0.5099 | Reverted. Recall decreased by 0.0011 and CER worsened by 0.0057 versus baseline, so broader prompt/rescore keyword budgets are not a good direction in this form. |
| 2026-04-30 | `configs/cb-whisper-aishell.yaml` | same fixed KWS checkpoint | Tried rerank weight shift: `rescore_asr_weight=1.2`, `rescore_keyword_weight=2.2` | 0.8475 | 0.0822 | 0.1087 | 0.5161 | Reverted. Point metrics matched baseline and did not improve recall, so static rerank weight tuning alone is not a promising direction. |
| 2026-04-30 | `configs/cb-whisper-aishell.yaml` | same fixed KWS checkpoint | Lightweight decode-time hotword token bias: `hotword_bias_weight=0.35`, `unmatched_scale=0.20`, `max_steps=48` | 0.8486 | 0.0845 | 0.1077 | 0.5272 | Mixed result. Recall improved by 0.0011 and Hotword Only CER improved by 0.0010 versus baseline, but CER worsened by 0.0023 and WER by 0.0111. Error inspection suggests broad first-token bias can hallucinate or append hotwords, so the next structural variant should bias only hotword continuations/prefix recovery instead of starting unmatched hotwords. |
| 2026-04-30 | `configs/cb-whisper-aishell.yaml` | same fixed KWS checkpoint | Decoder-evidence-gated hotword bias: `weight=0.30`, `unmatched_scale=0.0`, `restart_topk=16`, `restart_gap=1.0`, `restart_score>=0.55` | 0.8497 | 0.0842 | 0.1126 | 0.5248 | Not accepted as-is. Recall improved by 0.0022 versus baseline, but CER worsened by 0.0020, Hotword Only CER by 0.0038, and WER by 0.0087. The gate reduced overall CER damage slightly versus the first bias run but made hotword-only CER worse, so this needs another conservative variant or rollback. |
| 2026-04-30 | `configs/cb-whisper-aishell.yaml` | same fixed KWS checkpoint | ASR-plausibility guarded n-best selection: `max_asr_drop=0.25`, `max_len_growth=6`, `max_len_ratio=1.35` | 0.7945 | 0.0882 | 0.1370 | 0.5545 | Reverted. The guard was too conservative and collapsed many beneficial hotword rescoring decisions back to the ASR-best candidate, sharply reducing recall by 0.0530 versus baseline and worsening all guardrail metrics. |
| 2026-04-30 | `configs/cb-whisper-aishell.yaml` | same fixed KWS checkpoint | Overlong candidate penalty: `per_char=0.15`, `slack=2`, `cap=1.2` | 0.8475 | 0.0822 | 0.1087 | 0.5161 | Reverted. Metrics matched baseline exactly, so length-growth penalty did not change final decisions and is not useful in this form. |
| 2026-05-01 | `configs/cb-whisper-aishell.yaml` | same fixed KWS checkpoint | Nested phonetic keyword pruning: remove short hotwords when a similar-score longer hotword contains or phonetic-prefixes them | 0.8420 | 0.0824 | 0.1108 | 0.5161 | Reverted. It reduced short-hotword false competition but also removed true short entities, dropping recall by 0.0055 versus baseline. Next variant should promote long nested hotwords without deleting short ones. |
| 2026-05-01 | `configs/cb-whisper-aishell.yaml` | same fixed KWS checkpoint | Nested long-hotword promotion: keep short hotwords but also promote similar-score longer containing/phonetic-prefix hotwords into prompt/rescore sets | 0.8486 | 0.0820 | 0.1081 | 0.5149 | Accepted for now. Recall improved by 0.0011, CER by 0.0001, Hotword Only CER by 0.0007, and WER by 0.0012 versus baseline. This follows contextual-biasing practice of keeping competing bias phrases available instead of pruning the shorter form. |
| 2026-05-01 | `configs/cb-whisper-aishell.yaml` | same fixed KWS checkpoint | Length-weighted exact hotword coverage on top of nested long-hotword promotion | 0.8486 | 0.0820 | 0.1081 | 0.5149 | Reverted. Metrics were identical to nested long-hotword promotion, so the extra scoring complexity did not change final decisions. |
| 2026-05-01 | `configs/cb-whisper-aishell.yaml` | same fixed KWS checkpoint | Prefix-only hotword bias on top of nested long-hotword promotion: `weight=0.18`, no unmatched first-token bias | 0.8475 | 0.0816 | 0.1077 | 0.5161 | Not accepted as-is. CER and Hotword Only CER improved versus baseline, but recall fell back to baseline and lost the nested-promotion recall gain. Continue with a weaker prefix bias before deciding whether to keep this route. |
| 2026-05-02 | `configs/cb-whisper-aishell.yaml` | same fixed KWS checkpoint | Prefix-only hotword bias on top of nested long-hotword promotion: `weight=0.08`, no unmatched first-token bias | 0.8475 | 0.0819 | 0.1084 | 0.5136 | Reverted. Guardrails improved versus baseline, but recall again fell to baseline and lost the nested-promotion recall gain. Prefix-only logits bias is not useful for the recall target in this form. |
| 2026-05-02 | `configs/cb-whisper-aishell.yaml` | same fixed KWS checkpoint | Candidate-supported exact surface repair on top of nested long-hotword promotion | 0.8486 | 0.0820 | 0.1081 | 0.5149 | Reverted. Metrics matched nested long-hotword promotion and debug showed zero candidate-exact repairs, so this extra repair path is dead code for the current candidate pools. |
| 2026-05-02 | `configs/cb-whisper-aishell.yaml` | same fixed KWS checkpoint | Broader nested long-hotword promotion: `score_ratio=0.90`, `max_extra=3` | 0.8486 | 0.0823 | 0.1081 | 0.5149 | Reverted. Recall and WER matched the accepted nested-promotion run, but CER worsened by 0.0003, so the narrower `score_ratio=0.95`, `max_extra=2` setting remains better. |
| 2026-05-02 | `configs/cb-whisper-aishell.yaml` | same fixed KWS checkpoint | Increase rescoring generation candidate cap from 16 to 24 while keeping `rescore_nbest=8` | 0.8552 | 0.0859 | 0.1028 | 0.5223 | Mixed result. Recall improved by 0.0066 and Hotword Only CER improved by 0.0052 versus the accepted nested-promotion run, and oracle recall rose to 0.8778. However CER worsened by 0.0039 and WER by 0.0074, with a longer recognition pass. Keep only as evidence that candidate-pool expansion helps recall; next variant needs a lightweight quality constraint before this route can be accepted. |

## License

See the [LICENSE.md](LICENSE.md) file for details.

## Citation

If you use any of the resources in this repository, please cite the following paper:

Citation will be added in the future.
