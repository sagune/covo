# CB-SenseVoice / COVO Research Workspace

This workspace contains the ongoing contextual-biasing ASR experiments derived
from CB-Whisper and COVO.  The current research direction is to migrate the
CB-Whisper hotword/KWS idea from Whisper encoder states to SenseVoice encoder
states, and then feed the resulting hotword-aware ASR evidence into a COVO-style
post-correction model.

The original CB-Whisper README is kept below for reproducibility.  This top
section documents the current working workflow.

## Current Goal

Build a paper-clean CB-SenseVoice pipeline:

1. Extract SenseVoice encoder hidden states for utterances and hotword audio.
2. Train the existing lightweight TCResNet KWS model on SenseVoice similarity
   matrices.
3. Use KWS hotword evidence to bias ASR candidate generation and COVO
   correction.
4. Compare against original CB-Whisper, Whisper large-v3, SenseVoice-only, and
   COVO-only baselines.

The main constraint is that the method should remain lightweight and
explainable.  Avoid heavy case-specific patches as main paper claims.

## Environment

Use the prepared conda environment:

```bash
conda activate /root/autodl-tmp/great
```

Most commands below are run from the repository root `/root/autodl-tmp` unless
noted otherwise.

## SenseVoice KWS Pipeline

The SenseVoice KWS data root is:

```text
datasets/aishell/data_aishell_sensevoice
```

It reuses the original AISHELL/CB-Whisper metadata and keyword audio assets, but
stores new SenseVoice hidden states under fresh `hs` and `keywords-hs`
directories.

Prepare the dataset skeleton:

```bash
python src/analysis/prepare_sensevoice_kws_dataset.py \
  --source datasets/aishell/data_aishell \
  --target datasets/aishell/data_aishell_sensevoice
```

Extract all AISHELL KWS and hotword hidden states:

```bash
src/scripts/run_sensevoice_kws_extraction_20260714.sh
```

This uses `iic/SenseVoiceSmall`, extracts `model.encoder(...)` outputs, strips
the four SenseVoice query tokens, L2-normalizes the frame states, and saves the
same quantized `.bin` format used by the existing KWS dataloader.  SenseVoice
hidden-state dimension is 512.

Train the SenseVoice KWS model:

```bash
cd src
python run_CLI.py fit --config configs/train-sensevoice-kws.yaml
```

The training config mirrors the strong large-v3 KWS recipe while changing the
feature source to SenseVoice hidden states.  The KWS model itself remains the
lightweight 1-channel TCResNet over keyword/utterance similarity matrices.

Current training artifacts are written to:

```text
src/outputs/aishell_sensevoice_kws/checkpoints/
src/outputs/mlruns/
src/logs/train_sensevoice_kws_20260714.log
```

The full run stopped normally after epoch 21. The best validation checkpoint is:

```text
src/outputs/aishell_sensevoice_kws/checkpoints/f1G/f1G-epoch=16-step=102085.ckpt
```

Its validation metrics are `f1_zh=0.88416`, precision `0.95420`, recall
`0.82380`, and selected threshold `0.954`.

## SenseVoice KWS and CB Validation (2026-07-15)

On the full 808-row AISHELL hotword test subset with TTS keyword states, the
SenseVoice KWS checkpoint reaches its best test F1 at threshold `0.787`:
precision `0.90948`, recall `0.90658`, and F1 `0.90803`. At the conservative
validation threshold `0.954`, precision is `0.95721` and recall is `0.83121`.
The sweep is stored in
`src/logs/kws_threshold_sweep_sensevoice_tts_20260715.csv`.

CB-Whisper evaluation now supports reusing precomputed KWS similarity matrices.
This lets the large-v3 ASR decoder consume KWS evidence produced from the
512-dimensional SenseVoice states without loading a mismatched Whisper KWS
encoder. The full 808-row result is Entity Recall `0.92155`, CER `0.07174`,
Hotword Only CER `0.05228`, and WER `0.46906`; metrics are in
`src/logs/test_metrics_sensevoice_kws_cb_full_20260715.csv`. This validates the
cross-encoder workflow, but it does not replace the previous true-v3 CB result
(`0.92707` recall, `0.07102` CER, `0.04531` Hotword Only CER). The likely next
step is improving SenseVoice KWS ranking/calibration rather than changing the
CB reranker.

The matching naked SenseVoice baseline was then decoded on the same 808 audio
files with `iic/SenseVoiceSmall`, Chinese decoding, and ITN enabled. Its direct
simplified/punctuation-normalized CER is `0.10376` (`1337/12885`, exact
`268/808`). Under the exact numeric/surface normalization used by the current
CB evaluator, CER is `0.08554` (`1105/12918`, exact `301/808`). Under the
broader evaluation that treats all Chinese/Arabic number forms as equivalent,
CER is `0.07756`. Therefore the SenseVoice-KWS + Whisper large-v3 CB result
(`0.07174`) is better than naked SenseVoice on this subset, although the margin
depends on number normalization. Predictions are in
`src/logs/aishell808_funasr_sensevoice_small_full_20260715.jsonl`.

## Important Files

```text
src/analysis/extract_sensevoice_hidden_states.py
src/analysis/prepare_sensevoice_kws_dataset.py
src/configs/train-sensevoice-kws.yaml
src/scripts/run_sensevoice_kws_extraction_20260714.sh
src/scripts/shutdown_after_sensevoice_kws_train_20260715.sh
EXPERIMENT_CONTEXT.md
```

`EXPERIMENT_CONTEXT.md` is the durable experiment ledger.  Record major
experiment settings, checkpoints, results, and failure reasons there.

## COVO / Post-Correction Direction

The strongest recent Shuili route used:

1. SenseVoice as a clean ASR anchor.
2. CB-Whisper-style hotword/candidate evidence.
3. A COVO-style correction prompt with n-best, pinyin, consensus/confusable
   evidence, and hotword signals.

The next clean paper direction is to replace the external SenseVoice anchor
with a full CB-SenseVoice workflow: SenseVoice hidden-state KWS, SenseVoice
candidate/evidence generation, then COVO correction.

## Working Rules

- Commit code changes after each coherent step.
- Keep generated logs, checkpoints, and datasets out of normal commits unless
  explicitly needed.
- For rejected routes, record why they failed before reverting or moving on.
- Prefer method-level changes that can be explained in a paper over brittle
  dataset-specific patches.

---

# Original CB-Whisper README

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

Log cleanup policy: keep the current accepted-run metrics/probe/oracle diagnostics in `logs/` and use this README table as the durable experiment ledger. Rejected or exploratory run logs can be deleted after their metrics and comparison have been recorded here.

Large-v3 adaptation: `configs/cb-whisper-aishell-v3.yaml` switches only the Whisper ASR generator/processor checkpoint to `openai/whisper-large-v3` and writes oracle diagnostics to separate `*_v3.csv` files. The fixed KWS checkpoint and KWS encoder path remain unchanged for controlled comparison against the large-v2 baseline.

Whisper large-v2 vs large-v3 architecture note: the main encoder/decoder hidden structure is the same (`d_model=1280`, 32 encoder layers, 32 decoder layers, 20 attention heads). The important compatibility differences are the acoustic front end and token/generation config: large-v2 uses 80 mel bins and vocab size 51865, while large-v3 uses 128 mel bins and vocab size 51866, with shifted task/no-timestamp/pad token ids. Current large-v3 failures are therefore unlikely to come from hidden-layer shape mismatch; they are more likely due to large-v3's different frontend/token conventions and weaker candidate pools under this CB-Whisper prompting path.

KWS checkpoint policy: keep the KWS model fixed unless the experiment is explicitly about KWS retraining or KWS ablation. The current best KWS checkpoint is:

```text
src/mlruns/641753688314575260/d9b9fafe87d64bc98400705d2c58525c/checkpoints/f1G-epoch=7-step=48040.ckpt
```

This KWS model already includes local improvements over the original CB-Whisper KWS setup. Current experiments should primarily modify CB-Whisper decoding, prompting, normalization, rescoring, repair, and evaluation logic, while keeping this checkpoint as a controlled variable.

Method constraint: CB-Whisper is intended to be a lightweight improvement over Whisper, so proposed changes must stay lightweight at inference time. Avoid methods that require a much longer recognition pass, many repeated ASR calls, heavy external models, or large post-hoc search. Because this work is for a paper, changes should also remain methodologically clean and explainable; avoid over-engineered case-specific patches that only tune around observed failures.

Experiment goal: push KWS/keyword-related recall toward at least 0.90 while keeping CER as low as possible. Entity Recall is the primary hotword-success metric for CB-Whisper evaluation; CER, Hotword Only CER, and WER are guardrail metrics and should not be sacrificed casually for recall gains.

Experiment bookkeeping rule: after each code change, commit to git; after each experiment, append the result here and compare against the previous run or best known run.

KWS top-k diagnostic on the Aishell test set with the fixed KWS checkpoint: 808 samples, 942 true hotword mentions. Global KWS coverage is high: micro Recall@1 0.8036, Recall@3 0.9522, Recall@6 0.9692, Recall@12 0.9862. Under the current CB-Whisper selection logic, the KWS candidate pool/rescore set covers 0.9607 of true hotwords, and the prompt set covers 0.9321. This means the current 0.85-range CB-Whisper entity recall is not primarily capped by KWS top-k retrieval; most true hotwords already reach the prompt/rescore interface. The main bottleneck remains Whisper candidate generation/decoding and how hotword evidence is realized in ASR hypotheses.

Custom Shuili dataset: `configs/cb-whisper-shuili.yaml` evaluates the self-built Shuili test set under `datasets/shuili`. It reuses the AISHELL-style hotword folder format, with utterance audio extracted from `datasets/shuili/wav.zip` into `datasets/shuili/test/S0001/*.wav`. The KWS checkpoint and CB-Whisper method settings are kept the same as the accepted large-v2 AISHELL run, so this is a dataset-transfer baseline rather than a new method. The first imported Shuili hidden-state files were incompatible with the current loader/KWS distribution: their payloads used `quantized=True` with float16 values and their vectors were strongly positive-biased. The project-local reproducible extraction path is now `python utils.py --extract_hs -a <audio_dir> -t <target_dir> -w openai/whisper-medium`, which matches the AISHELL hidden-state distribution.

Large-v3 hidden-state extraction: to retrain and test a KWS/CB-Whisper path that is genuinely large-v3-style, extract utterance and keyword hidden states with the same v3 profile, for example `python utils.py --extract_hs -a <audio_dir> -t <target_dir> -w openai/whisper-large-v3 --whisper_style large-v3`. This profile checks that the Whisper frontend is using 128 mel bins and writes `_hs_manifest.json` into the target folder with the checkpoint, feature size, encoder size, layer-fusion mode, normalization, and quantization metadata. If `--hs_fuse grouped_mean` is used without explicit groups, large-v3 defaults to upper encoder groups `20-23,24-27,28-32`; explicit `--hs_groups` still takes priority. Do not mix these v3 hidden states with the current medium-style KWS checkpoint/data: medium and large-v2 can coexist in the present system because KWS uses the medium encoder/database while ASR uses large-v2, but a v3 KWS experiment must keep utterance hidden states, keyword hidden states, and KWS training data in the same v3 representation space.

Large-v3 KWS training result: `configs/train-large-v3-kws.yaml` reproduces the previous best KWS training hyperparameters from run `641753688314575260/d9b9fafe87d64bc98400705d2c58525c`, but trains on the large-v3-style `datasets/aishell/data_aishell` hidden states with `whisper_ckpt: openai/whisper-large-v3`. The archive contained one corrupt zero-byte training hidden-state file, `kws/hs/BAC009S0068W0457.bin`; the corresponding row was removed from `kws/positives.tsv` before the successful run. Successful run id: `outputs/mlruns/810143305197259608/495a58ed117c49bcaf592f74bc9395a9`. Early stopping ended after epoch 16. Best checkpoint: `outputs/aishell_large_v3_kws_reproduce/checkpoints/f1G/f1G-epoch=11-step=72060.ckpt` with `f1_zh=0.8752`, `precision_zh=0.9347`, `recall_zh=0.8228`, and threshold `0.9820`. The final epoch-16 checkpoint is `outputs/aishell_large_v3_kws_reproduce/checkpoints/final/final-epoch=16-step=102085.ckpt` with `f1_zh=0.8711`, `precision_zh=0.9283`, `recall_zh=0.8205`, and `val/loss_zh=0.0252`. Compared with the previous large-v2-style KWS best checkpoint (`f1G-epoch=7-step=48040.ckpt`, `f1_zh=0.8651`, `precision_zh=0.9133`, `recall_zh=0.8218`), large-v3-style training improves validation F1 and precision slightly, while recall remains essentially capped around 0.82 under this validation protocol. Standalone Aishell test with `configs/kws-aishell-large-v3.yaml` gives threshold `0.982`, precision `0.9213`, recall about `0.7824`, and F1 `0.8462`, so the test-set operating point is precision-heavy and lower-recall than desired.

True large-v3 KWS retraining result: after replacing the stale 1024-dimensional hidden-state package with `datasets/aishell/data_aishell_largev3.7z`, all KWS and hotword hidden-state manifests report `openai/whisper-large-v3`, `feature_size=128`, and `d_model=1280`. The new run is `outputs/mlruns/810143305197259608/fede196a08b64159adadc9d329bc4176` with output root `outputs/aishell_large_v3_kws_true`. Best checkpoint: `outputs/aishell_large_v3_kws_true/checkpoints/f1G/f1G-epoch=11-step=72060.ckpt` with validation `f1_zh=0.8759`, `precision_zh=0.9334`, `recall_zh=0.8251`, and threshold `0.9740`. The final epoch-16 checkpoint gives `f1_zh=0.8659`, `precision_zh=0.9145`, `recall_zh=0.8222`, and threshold `0.9300`. Standalone Aishell KWS test with the true-v3 checkpoint gives threshold `0.974`, precision `0.9178`, recall about `0.8057`, and F1 `0.8581`, improving test recall and F1 over the previous v3-style checkpoint but still well below the 0.90 recall target.

True large-v3 online KWS diagnosis: the low-recall full true-v3 CB-Whisper run was traced to a hidden-state length mismatch inside the CB path, not to the prompt token order. Offline KWS validation uses hidden states trimmed to the real utterance length, but online CB-Whisper was scoring KWS over the full 30-second padded Whisper encoder output. This made unrelated hotwords receive near-1.0 scores and polluted the prompt/rescore keyword set. The online KWS path now trims encoder hidden states by the input attention mask before normalization and keyword scoring, keeping online inference consistent with the training/evaluation hidden-state distribution.

True large-v3 custom generation diagnosis: after the online KWS fix, the next bottleneck was the short-form custom generation path. Hotword prompt tokens are prepended to `decoder_input_ids`, but the standard `no_repeat_ngram_size=3` processor treats those prompt tokens as already-generated history. That can forbid the decoder from copying a 3-token hotword phrase from the prompt into the actual transcript. Fully disabling no-repeat confirmed the diagnosis on a 120-sample smoke test by pushing Entity Recall above 0.90, but it caused severe repetition and CER blow-up. The accepted fix keeps no-repeat active while ignoring only the prompt prefix when computing banned n-grams, so generated text is still protected from loops while prompt hotwords remain copyable.

Original Priberam KWS baseline on the current large-v3 data: `Enhance-CB-Whisper` was cloned from `https://github.com/Priberam/Enhance-CB-Whisper.git` and minimally adapted to the current quantized large-v3 hidden-state files by adding a compatible hidden-state loader and changing the original ResNet KWS input from 12 channels to 1 channel. The natural-keyword baseline config is `Enhance-CB-Whisper/src/configs/train-aishell-large-v3-resnet1.yaml`, with run id `outputs/mlruns/133323746360263826/00ecf843301d45ca97b2bc6b7410548d`. Training ended successfully after epoch 10. The best natural validation checkpoint is `Enhance-CB-Whisper/outputs/aishell_large_v3_resnet1_original_natural/checkpoints/f1G/f1G-epoch=5-step=18018.ckpt`, with `f1_1=0.7978`, `precision_1=0.7115`, and `recall_1=0.9080`. The final epoch-10 checkpoint has `f1_1=0.7413`, `precision_1=0.6091`, and `recall_1=0.9467`. This baseline is much weaker in F1 than the later TCResNet-style large-v3 KWS (`f1_zh=0.8759` validation, standalone test F1 `0.8581`) but confirms that the original ResNet can still achieve high recall on natural keyword audio after single-channel adaptation.

Original Priberam large-v3 endpoint baseline on AISHELL: using the original `Enhance-CB-Whisper` model path with only compatibility adaptations for current large-v3 single-channel hidden states and modern Transformers, standalone KWS test on natural AISHELL hotwords gives Precision `0.8023`, Recall `0.8747`, and F1 `0.8370` (`Enhance-CB-Whisper/src/logs/resnet1_natural_kws_test_full_stdout.log`). After aligning the original endpoint evaluator with the current code's metric口径, including the same surface normalization and bootstrap metrics, end-to-end original CB-Whisper with `openai/whisper-large-v3` and the same original ResNet checkpoint gives Entity Recall `0.7801`, CER `0.0907`, Hotword Sentence CER `0.0907`, Hotword Only CER `0.1129`, and WER `0.4975` (`Enhance-CB-Whisper/src/logs/cbwhisper_original_large_v3_resnet1_natural_metrics_aligned_stdout.log`). Compared with the current best true-v3 CB-Whisper result (`0.9238` Entity Recall, `0.0661` CER, `0.0467` Hotword Only CER, `0.4641` WER), the original-code baseline is far behind at the ASR endpoint even though its standalone KWS recall is already reasonably high.

Original Priberam large-v3 endpoint baseline on the self-built Shuili dataset: `datasets/shuili/shuil.7z` was extracted into `datasets/shuili/data_shuil_largev3` and `datasets/shuili/data_shuil_medium`. The large-v3 split has 1152 utterances, 124 hotwords, and manifests showing `openai/whisper-large-v3`, `feature_size=128`, `d_model=1280`, and `hs_fuse=last` for utterance and keyword hidden states. Using the original ResNet natural AISHELL checkpoint above as a transfer KWS model, the metric-aligned original CB-Whisper endpoint gives Entity Recall `0.8007`, CER `0.2395`, Hotword Sentence CER `0.2376`, Hotword Only CER `0.3262`, and WER `0.5833` (`Enhance-CB-Whisper/src/logs/cbwhisper_original_large_v3_resnet1_shuili_metrics_aligned_stdout.log`). Compared with the earlier current-code Shuili wide setting (`0.8666` Entity Recall, `0.0807` CER, `0.1447` Hotword Only CER, `0.4977` WER), this original-code transfer baseline has somewhat lower recall and much worse CER/hotword CER, so it is mainly useful as a weak original-code baseline rather than a competitive Shuili result.

Current-code true large-v3 Shuili run: `configs/cb-whisper-shuili-v3-kws.yaml` and `run_cbwhisper_shuili_v3_kws_test.py` evaluate `datasets/shuili/data_shuil_largev3` with the current prompt-aware no-repeat implementation and the true-v3 AISHELL KWS checkpoint `outputs/aishell_large_v3_kws_true/checkpoints/f1G/f1G-epoch=11-step=72060.ckpt`. Because Shuili KWS scores are less calibrated, this run keeps the broader Shuili candidate-pool settings (`kws_positive_threshold=0.05`, `kws_topk_per_group=50`, `kws_max_prompt_keywords=200`, `rescore_max_keywords=160`) while still limiting the injected prompt to 4 keywords. Result: Entity Recall `0.8715`, CER `0.1402`, Hotword Sentence CER `0.1396`, Hotword Only CER `0.3956`, and WER `0.7613` (`src/logs/experiment_shuili_v3_kws_stdout.log`; metrics in `src/logs/test_metrics.csv`). Compared with the original-code large-v3 Shuili baseline, recall improves by `+0.0708` and CER improves by `-0.0992`, but WER and Hotword Only CER are worse. Compared with the earlier current-code large-v2/medium Shuili wide setting (`0.8666` Entity Recall, `0.0807` CER, `0.1447` Hotword Only CER, `0.4977` WER), true-v3 slightly improves recall but substantially hurts error-rate metrics, so the Shuili true-v3 route is not currently the best Shuili setting.

Shuili large-v3 KWS-only comparison on the current package: the current true-v3 TCResNet KWS checkpoint above gives threshold `0.974`, Precision `0.9515`, Recall `0.4788`, and F1 `0.6370` on `datasets/shuili/data_shuil_largev3` (`src/logs/kws_shuili_large_v3_current_stdout.log`). A Shuili-specific threshold sweep (`analysis/kws_threshold_sweep.py`) finds a better natural-keyword F1 operating point at threshold `0.942`, with Precision `0.7868`, Recall `0.6233`, and F1 `0.6956` (`src/logs/kws_shuili_large_v3_current_thr0942_stdout.log`). For the TTS keyword set, the best F1 threshold is `0.838`, with Precision `0.8078`, Recall `0.7136`, and F1 `0.7578`; this is now captured by `configs/kws-shuili-large-v3-current-tts-bestf1.yaml` and verified in `src/logs/kws_shuili_large_v3_current_tts_bestf1_stdout.log`. If the natural-keyword threshold is forced down to `0.403`, Recall reaches `0.9006`, but Precision drops to `0.1709`; for TTS, forcing Recall to `0.9006` requires threshold `0.207` and gives Precision `0.3253`, so these high-recall points are useful diagnostically but not good standalone KWS operating points. The current KWS ranking is much stronger than either fixed-threshold result suggests: micro Recall@1 `0.4318`, Recall@3 `0.7678`, Recall@6 `0.8907`, Recall@12 `0.9539`, Recall@28 `0.9837`, Recall@50 `0.9982` for the natural keyword set used by the current Shuili endpoint; under the broad Shuili CB-Whisper selection settings, the CB/rescore pool covers `0.9684` of true hotwords, while the compact 4-keyword prompt covers `0.7859` (`src/logs/kws_topk_recall_shuili_large_v3_current_natural_stdout.log`). The original Priberam ResNet KWS checkpoint `Enhance-CB-Whisper/outputs/aishell_large_v3_resnet1_original_natural/checkpoints/f1G/f1G-epoch=5-step=18018.ckpt`, evaluated through the original `Enhance-CB-Whisper/src/kws.py test` path, gives TTS Precision `0.2711`, Recall `0.0551`, and F1 `0.0916` at the original 0.5 threshold. Threshold tuning only improves TTS to threshold `0.144`, Precision `0.1188`, Recall `0.1337`, and F1 `0.1258`, so TTS remains collapsed. On natural keywords, the original 0.5 threshold gives Precision `0.3684`, Recall `0.6585`, and F1 `0.4725`; the best-F1 threshold is `0.892`, giving Precision `0.7098`, Recall `0.5104`, and F1 `0.5938` (`src/logs/original_kws_threshold_sweep_shuili_large_v3_natural_stdout.log`). The original ResNet natural ranking is better than its TTS ranking but still below the current TCResNet KWS: micro Recall@1 `0.3930`, Recall@3 `0.6504`, Recall@6 `0.7778`, Recall@12 `0.8636`, Recall@28 `0.9530`, Recall@50 `0.9846`. This explains why the current true-v3 endpoint can still reach high Shuili entity recall after broad candidate pooling, while the original ResNet transfer model is a weaker KWS source on this current Shuili package.

Current-code Shuili medium-package rerun with the previous best KWS checkpoint: `run_cbwhisper_shuili_medium_test.py` evaluates `datasets/shuili/data_shuil_medium` using the earlier fixed KWS checkpoint `src/mlruns/641753688314575260/d9b9fafe87d64bc98400705d2c58525c/checkpoints/f1G-epoch=7-step=48040.ckpt`, online KWS encoder `openai/whisper-medium`, and ASR `openai/whisper-large-v2`, with the same broad Shuili KWS/rescore settings as the accepted Shuili route. Result: Entity Recall `0.7575`, CER `0.1674`, Hotword Sentence CER `0.1804`, Hotword Only CER `0.4491`, and WER `0.7839` (`src/logs/experiment_shuili_medium_stdout.log`; metrics in `src/logs/test_metrics.csv`). This is worse than both the current-code true-v3 Shuili run and the earlier wide Shuili setting, so this new medium package plus previous KWS checkpoint does not reproduce the earlier strong Shuili result.

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
| 2026-05-02 | `configs/cb-whisper-aishell.yaml` | same fixed KWS checkpoint | N-best 24 plus extra-candidate beam-rank prior: penalty starts after rank 16 with weight 0.08 | 0.8552 | 0.0859 | 0.1028 | 0.5223 | Reverted with the n-best expansion route. The rank prior did not change point metrics versus plain n-best 24, so the CER/WER damage is not explained by only late-ranked candidates. Candidate-pool expansion remains useful diagnostically, but not acceptable under the lightweight/low-CER constraint in this form. |
| 2026-05-03 | `configs/cb-whisper-aishell.yaml` | same fixed KWS checkpoint | Confidence-gated candidate budget: use n-best 24 only when KWS top score >= 0.90 and top-vs-runner-up gap >= 0.03 | 0.8464 | 0.0844 | 0.1108 | 0.5186 | Reverted. This was a paper-clean dynamic contextual-biasing idea, but the gate selected a harmful subset: recall dropped by 0.0022 and CER worsened by 0.0024 versus accepted nested promotion. Oracle recall was only 0.8673, close to the original candidate-pool limit, so this KWS-confidence gate did not preserve the useful part of n-best expansion. |
| 2026-05-03 | `configs/cb-whisper-aishell.yaml` | same fixed KWS checkpoint | Rescore context keyword filtering: keep keywords with score >= max(0.85, 0.95 * top score) before nested promotion | 0.8431 | 0.0827 | 0.1105 | 0.5161 | Reverted. This paper-clean phrase-filtering idea reduced distractor context, but it also removed useful competing hotwords for rescoring. Recall dropped by 0.0055 versus accepted nested promotion and Hotword Only CER worsened by 0.0024, so the rescore stage needs richer context than the prompt stage. |
| 2026-05-03 | `configs/cb-whisper-aishell.yaml` | same fixed KWS checkpoint | Softmax-calibrated rescore keyword priors with temperature 0.05 | 0.8486 | 0.0820 | 0.1081 | 0.5149 | Reverted. Metrics matched the accepted nested-promotion run exactly, so KWS-prior calibration did not change final candidate choices. This suggests current failures are not mainly caused by raw KWS score weighting among the existing rescore keywords. |
| 2026-05-03 | `configs/cb-whisper-aishell.yaml` | same fixed KWS checkpoint | Gated phonetic-only rescoring: allow exact-pinyin candidate evidence without exact keyword match when phonetic similarity is 1.0 and margin >= 0.2 | 0.8486 | 0.0820 | 0.1081 | 0.5149 | Reverted. Metrics matched accepted nested promotion exactly. This suggests the current n-best pools rarely contain high-margin phonetic-only alternatives that can beat exact contextual candidates under the existing ASR/context fusion. |
| 2026-05-03 | `configs/cb-whisper-aishell.yaml` | same fixed KWS checkpoint | Specificity-aware keyword ordering: apply a length bonus of 0.06 when ranking prompt/rescore context phrases | 0.8475 | 0.0851 | 0.1063 | 0.5111 | Reverted. Longer phrase ordering slightly improved Hotword Only CER and WER, but lost the nested-promotion recall gain and worsened CER by 0.0031 versus accepted nested promotion. Phrase specificity alone is not a stable ranking criterion for this setup. |
| 2026-05-03 | `configs/cb-whisper-aishell.yaml` | same fixed KWS checkpoint | Diverse beam search with 4 beam groups and diversity penalty 0.2 | N/A | N/A | N/A | N/A | Reverted before metrics. The installed Transformers generation path requires loading a remote `transformers-community/group-beam-search` custom generator for group beam search. That dependency is not acceptable for the lightweight, reproducible paper setting, so this route was abandoned without a full run. |
| 2026-05-03 | `configs/cb-whisper-aishell.yaml` | same fixed KWS checkpoint | Phonetic n-best consensus reranking: boost exact hotword candidates when other n-best candidates strongly support the same hotword phonetically | 0.8486 | 0.0820 | 0.1081 | 0.5149 | Reverted. Metrics matched accepted nested promotion exactly, indicating that current candidate pools do not contain enough cases where phonetic cross-candidate support changes the selected hypothesis. |
| 2026-05-03 | `configs/cb-whisper-aishell-v3.yaml` | same fixed KWS checkpoint | Direct Whisper large-v3 swap, keeping KWS encoder fixed and using `language: chinese` | 0.1204 | 3.7586 | 0.8452 | 0.9431 | Not accepted. The run produced many English transcripts on Aishell, so v3 is not correctly language-constrained by the current custom PBAWhisper generation path. Next v3 adaptation should explicitly force the `<\|zh\|>` language token or otherwise repair language/task token handling before comparing v3 fairly. |
| 2026-05-03 | `configs/cb-whisper-aishell-v3.yaml` | same fixed KWS checkpoint | 20-batch smoke test after explicitly forcing decoder prefix tokens `<\|zh\|>`, `<\|transcribe\|>`, and `<\|notimestamps\|>` in the custom PBAWhisper generation path | 0.5000 | 0.0767 | 0.2361 | 0.5500 | Smoke test only; not directly comparable with full-set metrics. The main failure mode from the direct v3 swap is fixed: predictions are now Chinese transcriptions instead of English translations. A full v3 fixed-baseline run is required before accepting or rejecting large-v3. |
| 2026-05-03 | `configs/cb-whisper-aishell-v3.yaml` | same fixed KWS checkpoint | Full large-v3 run after explicit decoder prefix forcing; oracle diagnostics written to `*_v3_fixed.csv` | 0.4796 | 0.1180 | 0.2994 | 0.7067 | Not accepted. Compared with the accepted large-v2 nested-promotion run, recall drops by 0.3691, CER worsens by 0.0360, Hotword Only CER worsens by 0.1914, and WER worsens by 0.1918. Oracle n-best top1 recall is 0.4867 and oracle recall is only 0.6665, so the large-v3 candidate pool itself is much weaker under the current custom generation/prompting path. |
| 2026-05-12 | `configs/cb-whisper-aishell-v3-kws.yaml` | `outputs/aishell_large_v3_kws_reproduce/checkpoints/f1G/f1G-epoch=11-step=72060.ckpt` | Full large-v3 ASR run with the newly trained large-v3-style KWS checkpoint; ASR remains `openai/whisper-large-v3`, while KWS encoder stays `openai/whisper-medium` because the provided hidden states are 1024-dimensional | 0.4796 | 0.1180 | 0.2994 | 0.7067 | Not accepted. Metrics are effectively identical to the previous large-v3 fixed-baseline run using the old KWS checkpoint. Oracle n-best top1 recall is 0.4867 and oracle recall is 0.6665, again showing that the large-v3 bottleneck is the ASR candidate pool/generation path rather than the KWS checkpoint. A failed first attempt with `encoder_ckpt=openai/whisper-large-v3` confirmed a 1280-vs-1024 hidden-state mismatch, so this data package is not truly end-to-end large-v3 KWS hidden-state compatible. |
| 2026-05-14 | `configs/cb-whisper-aishell-v3-kws.yaml` | `outputs/aishell_large_v3_kws_true/checkpoints/f1G/f1G-epoch=11-step=72060.ckpt` | Full true large-v3 path: ASR `openai/whisper-large-v3`, online KWS encoder `openai/whisper-large-v3`, and 1280-dimensional large-v3 hotword database | 0.5481 | 0.1132 | 0.2729 | 0.6683 | Not accepted, but this is a real improvement over the stale-hidden-state v3 KWS run: Entity Recall +0.0685, CER -0.0048, Hotword Only CER -0.0265, and WER -0.0384. It is still far below the accepted large-v2 CB-Whisper baseline. Oracle n-best top1 recall is 0.5563 and oracle recall is 0.6579, so true-v3 KWS helps select better keywords, but the large-v3 ASR candidate pool remains the dominant bottleneck. |
| 2026-05-14 | `configs/cb-whisper-aishell-v3-kws.yaml` | `outputs/aishell_large_v3_kws_true/checkpoints/f1G/f1G-epoch=11-step=72060.ckpt` | Full true large-v3 path after trimming online KWS encoder hidden states to the real utterance length using the attention mask | 0.7215 | 0.1196 | 0.2356 | 0.5842 | Diagnosis accepted, method not yet accepted. Compared with the previous true-v3 run, Entity Recall improves by +0.1735, Hotword Only CER improves by -0.0373, and WER improves by -0.0842, confirming that padded hidden frames were corrupting online KWS keyword selection. CER worsens by +0.0064, and the result is still below the accepted large-v2 baseline, so the next step should focus on making large-v3 use the now-correct hotword set without selecting lower-quality ASR candidates. Oracle n-best top1 recall is 0.7288 and oracle recall is 0.7351. |
| 2026-05-14 | `configs/cb-whisper-aishell-v3-kws.yaml` | `outputs/aishell_large_v3_kws_true/checkpoints/f1G/f1G-epoch=11-step=72060.ckpt` | Full true large-v3 path with prompt-aware no-repeat n-gram blocking: no-repeat is computed over generated text while ignoring the hotword prompt prefix | 0.9238 | 0.0661 | 0.0467 | 0.4641 | Accepted as the current best result. Compared with the online-KWS-trim run, Entity Recall improves by +0.2022, CER improves by -0.0535, Hotword Only CER improves by -0.1889, and WER improves by -0.1200. Compared with the accepted large-v2 baseline, recall improves by about +0.0763 and CER improves by about -0.0161. Oracle n-best top1 recall is 0.9289 and oracle recall is 0.9433, showing that the custom generation path now produces a substantially stronger candidate pool instead of merely relying on reranking. |
| 2026-05-28 | `configs/cb-whisper-aishell-v3-kws.yaml` with CLI overrides | `outputs/aishell_large_v3_kws_true/checkpoints/f1G/f1G-epoch=11-step=72060.ckpt` | Clean large-v3 fallback probe: `prompt=false`, phonetic rescore/repair/consensus disabled, and oracle n-best diagnostics disabled; metrics written to `logs/test_metrics_aishell_v3_clean.csv` | 0.4785 | 0.1145 | 0.2973 | 0.7005 | Rejected as a CER-reduction route. The no-prompt clean hypothesis is much worse than the current best true-v3 CB-Whisper run, so a simple clean-Whisper fallback cannot push AISHELL CER below 6%. Future CER reduction should use a more selective gate among contextual candidates or a weaker prompt/rescore variant, not unconditional clean fallback. |
| 2026-05-28 | offline n-best MBR on `logs/oracle_nbest_detail_aishell_v3_best_rerun.csv` | same true-v3 KWS checkpoint | Character-level MBR/consensus candidate selection over the existing n-best pool; no extra Whisper decoding | 0.9289 | 0.0645 | 0.0494 | N/A | Implemented as `src/analysis/nbest_consensus.py`. The best test-side MBR setting is `mbr_uniform_a0.2_kw0`, improving candidate-detail CER from 0.0666 to 0.0645 while keeping recall unchanged. Dev-selected MBR (`a=0.1`) reaches 0.0587 CER on dev but transfers only to 0.0660 on test, so MBR is useful as evidence that n-best consistency helps but is not yet a <6% test-CER solution. |
| 2026-05-28 | dev-trained offline n-best reranker | same true-v3 KWS checkpoint | Lightweight linear candidate selector trained on dev n-best features (`rank`, ASR score, exact/phonetic/consensus scores, support, length, top1 distance), then evaluated on test n-best | 0.9258 | 0.0643 | 0.0508 | N/A | Implemented as `src/analysis/nbest_reranker.py`. Training on `logs/oracle_nbest_detail_aishell_v3_dev_for_reranker.csv` and testing on `logs/oracle_nbest_detail_aishell_v3_best_rerun.csv` improves test candidate-detail CER from 0.0666 to 0.0643, but recall drops by 0.0031 and hotword-only CER worsens by 0.0014. A direct min-CER label variant was worse (`0.0650` CER), so the current feature-only linear reranker is not sufficient for the 6% CER target. |
| 2026-05-03 | `configs/cb-whisper-aishell.yaml` | same fixed KWS checkpoint | Chinese natural-language contextual prompt format: `以下关键词可能出现：kw1，kw2。` | 0.8320 | 0.0786 | 0.1185 | 0.5136 | Reverted/not accepted. This prompt form improves CER by 0.0034 versus accepted nested promotion, but recall drops by 0.0166 and Hotword Only CER worsens by 0.0105. Oracle recall also drops to 0.8486, so natural-language prompt wording weakens the hotword candidate pool despite better generic transcription. |
| 2026-05-03 | `configs/cb-whisper-aishell.yaml` | same fixed KWS checkpoint | Individual bracketed keyword prompt format: `(kw1) (kw2) (kw3)` | 0.8453 | 0.0864 | 0.1223 | 0.5173 | Not accepted. Compared with accepted nested promotion, recall drops by 0.0033 and CER worsens by 0.0044. Oracle recall is 0.8572, below the accepted 0.8663 oracle recall, so separating keywords into independent bracketed chunks does not improve candidate generation. |
| 2026-05-03 | `configs/cb-whisper-aishell.yaml` | same fixed KWS checkpoint | Prompt recency ordering: reverse selected prompt keywords so higher-KWS-score keywords appear closest to the decoder start | 0.8166 | 0.0898 | 0.1415 | 0.5408 | Reverted/not accepted. This candidate-generation-side idea substantially hurts recall and all guardrail metrics; oracle recall drops to 0.8317. The original descending KWS order is important for the current prompt format, even if the highest-score keyword is farther from the decoder start token. |
| 2026-05-05 | `configs/cb-whisper-shuili.yaml` | same fixed KWS checkpoint | Shuili custom dataset baseline after re-extracting utterance/keyword hidden states with `utils.py --extract_hs` and `openai/whisper-medium` | 0.7639 | 0.0998 | 0.2153 | 0.5872 | Corrected Shuili baseline. The bad imported hidden states caused KWS collapse: before re-extraction, KWS Recall@12 was only 0.0134 and prompt repeatedly selected the same unrelated keyword. Re-extraction restored normal hidden-state distribution and improved Entity Recall from 0.7556 to 0.7639, CER from 0.1000 to 0.0998, and WER from 0.5917 to 0.5872, but Hotword Only CER worsened from 0.1794 to 0.2153. KWS remains much weaker than AISHELL on this custom set: after correction, KWS Recall@12 is 0.4273 and Recall@28 is 0.6865, so the main remaining issue is KWS/data distribution mismatch rather than Whisper hidden-state file corruption alone. Oracle n-best summary: top1 recall 0.5497, oracle recall 0.6388, top1 hit rate 0.3881, oracle hit rate 0.5405. |
| 2026-05-05 | `configs/cb-whisper-shuili.yaml` | same fixed KWS checkpoint | New Shuili `hotword.7z` package, replacing the old hotword-side files while keeping the original utterance wavs | 0.7615 | 0.0999 | 0.2148 | 0.5849 | New package result. Hidden-state distribution is now normal out of the archive, so the earlier file-format issue is fixed. Metrics are close to the corrected Shuili baseline: Entity Recall -0.0024, CER +0.0001, Hotword Only CER -0.0005, WER -0.0023. KWS diagnosis remains the bottleneck: Recall@12 0.4281 and Recall@28 0.6823, far below AISHELL. The TTS keyword-audio domain is a plausible contributor, because KWS compares utterance hidden states against TTS keyword hidden states; a natural-keyword or KWS-adapted Shuili variant would test this directly. Oracle n-best summary: top1 recall 0.5484, oracle recall 0.6361, top1 hit rate 0.3872, oracle hit rate 0.5392. |
| 2026-05-06 | `configs/cb-whisper-shuili-narrow-baseline.yaml` | same fixed KWS checkpoint | Narrow Shuili candidate-pool baseline: `kws_positive_threshold=0.05`, `kws_topk_per_group=25`, `kws_max_prompt_keywords=100`, `rescore_max_keywords=50`; prompt still limited to 4 keywords | 0.8501 | 0.0841 | 0.1543 | 0.5069 | Accepted as a controlled weaker baseline. It is clearly worse than the current wider Shuili setting, but not collapsed, so it can show the value of broader KWS rank coverage and rescore context. Compared with the current setting, Entity Recall is lower by 0.0165, CER worse by 0.0034, Hotword Only CER worse by 0.0096, and WER worse by 0.0092. |
| 2026-05-05 | `configs/cb-whisper-shuili.yaml` | same fixed KWS checkpoint | Wider Shuili rescore pool: `kws_positive_threshold=0.05`, `kws_topk_per_group=50`, `kws_max_prompt_keywords=200`, `rescore_max_keywords=160`; prompt still limited to 4 keywords | 0.8666 | 0.0807 | 0.1447 | 0.4977 | Accepted as the current Shuili setting. Compared with the previous low-threshold setting, Entity Recall improves by 0.0165, CER by 0.0034, Hotword Only CER by 0.0096, while WER is roughly flat (+0.0008). Compared with the new-package baseline, Entity Recall improves by 0.1051 and CER by 0.0192. This supports the diagnosis that Shuili KWS score calibration is too strict and rank coverage is more useful than raw score thresholding; broadening the rescore pool helps without increasing the number of Whisper decoding passes. Oracle n-best summary: top1 recall 0.6312, oracle recall 0.6361, top1 hit rate 0.5297, oracle hit rate 0.5392. |
| 2026-05-05 | `configs/cb-whisper-shuili.yaml` | same fixed KWS checkpoint | Disable prompt while keeping wide Shuili KWS/rescore path | 0.7048 | 0.3005 | 0.2140 | 0.6950 | Rejected. Prompting is essential on Shuili; without prompt, generic ASR quality collapses and recall drops by 0.1617 versus the current wide-rescore setting. |
| 2026-05-05 | `configs/cb-whisper-shuili.yaml` | same fixed KWS checkpoint | Very wide Shuili rescore pool: `kws_positive_threshold=0.02`, `kws_topk_per_group=80`, `kws_max_prompt_keywords=320`, `rescore_max_keywords=320`; prompt still limited to 4 keywords | 0.8666 | 0.0811 | 0.1453 | 0.5023 | Rejected. Recall matches the current wide-rescore setting, but CER worsens by 0.0004, Hotword Only CER by 0.0005, and WER by 0.0046. This suggests the useful rank-coverage gain saturates around the `topk_per_group=50`, `rescore_max_keywords=160` setting; pushing toward the full keyword list mostly adds distractors. |
| 2026-05-05 | `configs/cb-whisper-shuili.yaml` | same fixed KWS checkpoint | Shuili natural-language prompt wording: `以下关键词可能出现：kw1，kw2。`; KWS/rescore settings unchanged from the current wide-rescore setting | 0.8442 | 0.0855 | 0.2029 | 0.5092 | Rejected. Compared with the current Shuili setting, Entity Recall drops by 0.0224, CER worsens by 0.0048, Hotword Only CER worsens by 0.0582, and WER worsens by 0.0115. The same natural-language prompt route that was weak on AISHELL is also weak on Shuili; the compact bracket prompt remains better for hotword realization. Oracle n-best summary: top1 recall 0.6168, oracle recall 0.6218, top1 hit rate 0.5083, oracle hit rate 0.5201. |
| 2026-05-05 | `configs/cb-whisper-shuili.yaml` | same fixed KWS checkpoint | Increase Shuili prompt budget from 4 to 8 injected keywords; KWS/rescore settings unchanged | 0.8536 | 0.0840 | 0.1817 | 0.5252 | Rejected. Compared with the current Shuili setting, Entity Recall drops by 0.0130, CER worsens by 0.0033, Hotword Only CER worsens by 0.0370, and WER worsens by 0.0275. This suggests Shuili does need prompting, but the prompt must stay compact; adding more KWS candidates mainly introduces distractors and hurts both hotword and generic ASR quality. Oracle n-best summary: top1 recall 0.6223, oracle recall 0.6277, top1 hit rate 0.5095, oracle hit rate 0.5238. |
| 2026-05-05 | `configs/cb-whisper-shuili.yaml` | same fixed KWS checkpoint | Specificity-first ordering inside the existing 4-keyword prompt; selected prompt set unchanged, only longer/full terms are placed earlier | 0.8666 | 0.0808 | 0.1665 | 0.4954 | Rejected and code option reverted. Entity Recall matches the current Shuili setting and WER improves slightly, but CER worsens by 0.0001 and Hotword Only CER worsens by 0.0217. Ordering alone does not fix Shuili's surface-form errors; the prompt likely needs better selection, not just reordering. Oracle n-best summary: top1 recall 0.6317, oracle recall 0.6382, top1 hit rate 0.5321, oracle hit rate 0.5463. |
| 2026-05-06 | original `Enhance-CB-Whisper` KWS on Shuili | original ResNet KWS trained on local `dataaishell/kws` TTS data | Test original-paper KWS transfer to Shuili with AISHELL-style loader and Chinese normalization | 0.1388 | N/A | N/A | N/A | KWS-only baseline: Precision 0.1378, Recall about 0.1388, F1 0.1383. This shows the original ResNet KWS trained on AISHELL/TTS transfers very poorly to Shuili. |
| 2026-05-06 | original `Enhance-CB-Whisper` CB-Whisper on Shuili | same original ResNet KWS checkpoint | Original CB-Whisper + original KWS, adapted only for local data format, current Transformers compatibility, and Chinese normalization | 0.0679 | N/A | N/A | N/A | Rejected as a baseline only. Compared with the current Shuili setting above (Entity Recall 0.8666, CER 0.0807), original CB-Whisper collapses mainly because the original KWS candidate source has almost no transfer recall on Shuili. The original evaluation script reports Entity Recall only, so CER/WER are not available from this run. |
| 2026-05-06 | original `Enhance-CB-Whisper` on AISHELL test | original ResNet KWS trained on local `dataaishell/kws` TTS data | Original CB-Whisper baseline after replacing outdated prompt internals with the current Transformers `prompt_ids` path and adding CER reporting | 0.7578 | 0.1121 | N/A | N/A | This is the usable original-code baseline. The earlier original-code AISHELL run was artificially low because prompt tokens were forced in a way that bypassed Whisper's current language/task-token handling. Current full log: `original/Enhance-CB-Whisper/src/logs/cbwhisper_aishell_original_kws_test_stdout.log`. |

## End-to-end CB-SenseVoice

The recognition side can now run without loading Whisper. The new
`configs/cb-sensevoice-aishell.yaml` workflow is:

```text
audio
  -> SenseVoice encoder hidden states
  -> SenseVoice-based KWS and hotword filtering
  -> neutral + hotword-biased SenseVoice CTC prefix beam search
  -> acoustic/exact/phonetic/consensus reranking
  -> final transcript and COVO evidence
```

The contextual CTC scorer rewards only token extensions that advance a
filtered hotword prefix, while the neutral beam preserves ordinary ASR
candidates. The reranker uses the original SenseVoice CTC acoustic score; no
Whisper checkpoint or Whisper-generated candidate is used on this path.

Full AISHELL hotword test (`808` rows, 2026-07-15):

| System | Entity Recall | CER | Hotword Only CER | WER |
| --- | ---: | ---: | ---: | ---: |
| Naked SenseVoice, current CB normalization | N/A | 0.08554 | N/A | N/A |
| SenseVoice KWS + Whisper-large-v3 CB | 0.92155 | 0.07174 | 0.05228 | 0.46906 |
| End-to-end CB-SenseVoice | 0.83204 | **0.06202** | 0.10979 | 0.40099 |

CB-SenseVoice improves CER by `0.02352` absolute over naked SenseVoice and by
`0.00972` over the mixed SenseVoice-KWS/Whisper path. Its remaining weakness is
hotword realization: candidate-oracle Entity Recall is only `0.85128`, versus
the standalone KWS recall ceiling of `0.90658`. Candidate-oracle CER is
`0.03686`, so the next research target is better contextual CTC candidate
generation rather than another hand-tuned reranking formula.

An additional 100-row probe that biased decoding with all detected KWS words
instead of only the filtered prompt words changed six candidate pools but left
CER and recall exactly unchanged. That expansion was rejected and the compact
filtered-hotword formulation was retained.

### CB-SenseVoice + COVO

CB-SenseVoice evidence is compatible with the existing COVO bridge. The
current no-gate workflow uses six SenseVoice candidates, their pinyin and
scores, and all available KWS/prompt hotword evidence with the conservative
`qwen35_cbwhisper_preserve2_aishell_train_noop_1epoch_bf16_bs7` adapter:

```text
CB-SenseVoice top1 + n-best + KWS evidence
  -> compact reliability-labeled COVO prompt
  -> Qwen3.5-4B + preserve2 LoRA
  -> corrected final transcript
```

Full AISHELL hotword test (`808` rows):

| System | Entity Recall | CB-normalized mean CER | Exact rows |
| --- | ---: | ---: | ---: |
| End-to-end CB-SenseVoice | 0.83204 | 0.06202 | 484 |
| CB-SenseVoice + COVO preserve2 | **0.83978** | **0.04379** | **541** |

COVO recovered `38/105` true hotwords that had entered the filtered prompt but
were absent from every SenseVoice candidate. Across all mentions it gained 48
hotwords and lost 42, so recall improves only modestly while CER improves
substantially. No post-filter, fallback gate, or reference-derived field is
used by the model prompt.

The bridge now preserves `keyword_mentions` as evaluation-only metadata during
simplified-Chinese conversion. These gold mentions are not rendered into the
COVO prompt.

Run from `src/`:

```bash
/root/autodl-tmp/great/bin/python analysis/cbwhisper_covo_bridge.py run \
  --input logs/cb_sensevoice_evidence_aishell.jsonl \
  --output logs/cb_sensevoice_covo_messages.jsonl \
  --prediction-output logs/cb_sensevoice_covo_predictions.jsonl \
  --adapter-path ../cbwhisper_covo_migration_20260609_tar_extracted/covo/outputs/qwen35_cbwhisper_preserve2_aishell_train_noop_1epoch_bf16_bs7 \
  --max-nbest 6 --max-pinyin 3 \
  --max-hotwords 8 --max-prompt-hotwords 6 \
  --max-candidates-with-scores 6 \
  --hotword-source all --include-pinyin \
  --protect-supported-hotwords \
  --batch-size 7 --disable-thinking --evaluate
```

Run the full workflow from `src/`:

```bash
CBW_METRICS_OUT=logs/test_metrics_cb_sensevoice.csv \
CBW_EVIDENCE_OUT=logs/cb_sensevoice_evidence_aishell.jsonl \
/root/autodl-tmp/great/bin/python run_CLI.py test \
  --config configs/cb-sensevoice-aishell.yaml
```

## CB-Whisper + covo workflow

The current code can export CB-Whisper evidence for a downstream covo/Qwen text-rewrite corrector without changing the normal CB-Whisper metrics path. Set `CBW_EVIDENCE_OUT` during a CB-Whisper test run:

```bash
cd /root/autodl-tmp/src
TRANSFORMERS_VERBOSITY=error \
CBW_EVIDENCE_OUT=logs/cbwhisper_covo_evidence_aishell.jsonl \
CBW_METRICS_OUT=logs/test_metrics.csv \
/root/autodl-tmp/great/bin/python cb-whisper.py test --config configs/cb-whisper-aishell-v3-kws.yaml
```

Each evidence record stores the CB-Whisper final output as `input.asr_top1`, the reranked candidate pool as `input.nbest`, KWS candidates as `input.hotwords`, injected prompt words as `input.prompt_hotwords`, pinyin strings, and candidate scores under `input.cbwhisper`. The bridge injects prompt-side CB-Whisper hotwords into `input.covo_hotwords` by default and writes them into the covo prompt as predicted, non-gold hotword evidence. Broader KWS hotword injection remains available through `--hotword-source all`, but the first pilot showed that all-KWS evidence adds too many false-hotword distractors.

Convert this evidence into Qwen/covo chat messages:

```bash
cd /root/autodl-tmp
/root/autodl-tmp/great/bin/python src/analysis/cbwhisper_covo_bridge.py prepare \
  --input src/logs/cbwhisper_covo_evidence_aishell.jsonl \
  --output src/logs/cbwhisper_covo_messages_aishell.jsonl \
  --include-pinyin
```

Run the migrated covo LoRA corrector on the messages:

```bash
cd /root/autodl-tmp
# If needed: /root/autodl-tmp/great/bin/python -m pip install -r cbwhisper_covo_migration_20260609_tar_extracted/covo/requirements-train.txt
/root/autodl-tmp/great/bin/python src/analysis/cbwhisper_covo_bridge.py run \
  --input src/logs/cbwhisper_covo_evidence_aishell.jsonl \
  --output src/logs/cbwhisper_covo_messages_aishell.jsonl \
  --prediction-output src/logs/cbwhisper_covo_predictions_aishell.jsonl \
  --include-pinyin \
  --disable-thinking \
  --evaluate
```

The default bridge paths use the migration bundle:

```text
/root/autodl-tmp/cbwhisper_covo_migration_20260609_tar_extracted/covo
/root/autodl-tmp/cbwhisper_covo_migration_20260609_tar_extracted/models/Qwen3.5-4B
/root/autodl-tmp/cbwhisper_covo_migration_20260609_tar_extracted/covo/outputs/qwen35_text_rewrite_hardneg_dropout_lora_2epoch
```

This is intentionally a loose bridge rather than a hard code merge: CB-Whisper
or CB-SenseVoice remains responsible for ASR, N-best, KWS, and hotword scoring;
COVO remains responsible for conservative generative correction and
evaluation.

Smoke test on 2026-06-12: a 2-sample AISHELL run with `CBW_EVIDENCE_OUT=logs/cbwhisper_covo_evidence_smoke.jsonl` successfully exported evidence, converted it to Qwen messages, loaded the migrated `Qwen3.5-4B` + hard-negative LoRA adapter, and evaluated the predictions. The smoke subset improved CER from `0.07143` to `0.00000`, with `1` improved sample, `0` worsened samples, and `1` unchanged sample. This is only a wiring check, not a reportable metric, but it confirms the CB-Whisper -> covo bridge can repair at least one real CB-Whisper over-correction case.

Pilot100 on 2026-06-12: the first 100 AISHELL test samples were exported through the same bridge. CB-Whisper itself reported Entity Recall `0.9439`, CER `0.0357`, Hotword Only CER `0.0270`, and WER `0.3100` in `logs/test_metrics_covo_pilot100.csv`. Directly accepting all covo/Qwen corrections was not safe under the covo CER evaluator: CER changed from `0.08877` to `0.09073`, with `23` improved, `28` worsened, and `49` unchanged samples. Injecting all KWS hotwords into the prompt worsened CER to `0.09856`, confirming that broad hotword evidence is noisy. Injecting only the prompt-side CB-Whisper hotwords improved the covo-evaluator CER to `0.08486`, with `23` improved, `27` worsened, and `50` unchanged samples. Decision: do not add a post-hoc gate; the paper-clean next step is to fine-tune the covo/Qwen corrector to use predicted CB-Whisper hotword evidence conservatively.

Hotword-aware covo SFT probe on 2026-06-12: `src/analysis/prepare_covo_hotword_sft.py` converts existing ChineseHP/AISHELL rewrite data into the same CB-Whisper hotword-evidence prompt style as the bridge. A small 2000/500 train/dev probe was generated from the covo hard-negative rewrite data, then the migrated `qwen35_text_rewrite_hardneg_dropout_lora_2epoch` adapter was continued for 120 bf16 LoRA steps without QLoRA. Training completed normally with final eval loss `0.1956`, but Pilot100 evaluation with the continued adapter regressed to CER `0.08877`, improved `23`, worsened `29`, unchanged `48` (`logs/experiment_cbwhisper_covo_pilot100_hotword_ft120_stdout.log`). This is worse than the no-gate prompt-hotword bridge result (`0.08486` CER), so this particular synthetic-hotword SFT probe is not accepted or worth scaling without changing the data construction.

Real CB-Whisper evidence SFT probe on 2026-06-12: a 200-sample AISHELL dev subset was exported with `CBW_EVIDENCE_OUT=logs/cbwhisper_covo_evidence_dev200.jsonl`; CB-Whisper on that subset reported Entity Recall `0.9369`, CER `0.0725`, Hotword Only CER `0.0601`, and WER `0.4550`. Training directly on the full evidence prompt OOMed, so a compact prompt was used (`max_nbest=4`, `max_pinyin=3`, no candidate-score block). Continuing the hard-negative LoRA for 80 bf16 steps on this real dev evidence completed with final eval loss `0.2739`. On the same Pilot100 test evidence, compact prompt + original LoRA gave CER `0.08877`, improved `25`, worsened `25`, unchanged `50`; compact prompt + real-dev200 fine-tuned LoRA improved to CER `0.08486`, improved `23`, worsened `25`, unchanged `52` (`logs/experiment_cbwhisper_covo_pilot100_realdev_compact_ft80_stdout.log`). This is a small but real no-gate signal: real CB-Whisper evidence fine-tuning can recover the prompt-hotword gain even under a compact prompt, while synthetic hotword SFT did not.

Scaled real-evidence SFT on 2026-06-12: `CBW_EVIDENCE_ONLY=1` was added for faster evidence export, then 600 AISHELL dev samples were exported to `logs/cbwhisper_covo_evidence_dev600.jsonl` without running bootstrap metrics. Compact prompt messages were split into 500 train / 99 dev rows, and the hard-negative LoRA was continued for 160 bf16 steps with gradient checkpointing. The final eval loss was `0.2555`. On Pilot100, this adapter reached CER `0.08094`, improved `24`, worsened `23`, unchanged `53` (`logs/experiment_cbwhisper_covo_pilot100_realdev600_compact_ft160_stdout.log`). This is the current best no-gate COVO pilot result and improves over both compact original LoRA (`0.08877`) and the previous prompt-hotword bridge (`0.08486`).

Full AISHELL test evaluation on 2026-06-12: the full 808-sample AISHELL test set was exported through the same evidence-only CB-Whisper path to `logs/cbwhisper_covo_evidence_test_full.jsonl`, then decoded with the real-dev600 compact SFT adapter. Under the covo correction evaluator, CER dropped from the COVO bridge evidence baseline `0.10043` to `0.05650`, with `296` improved samples, `96` worsened samples, and `416` unchanged samples (`logs/experiment_cbwhisper_covo_test_full_realdev600_compact_ft160_stdout.log`). This `0.10043` baseline is the `input.asr_top1` CER inside the COVO JSONL evidence and should not be mixed with the earlier standalone CB-Whisper AISHELL CER table, where the accepted rerank baseline was about `0.0820`. A follow-up entity-recall check gave bridge input Entity Recall `0.9028` and covo output Entity Recall `0.8541`; this route is therefore useful for lowering overall CER below 6%, but it currently sacrifices hotword recall and should not be reported as a hotword-recall improvement without further training or loss design.

Hotword-preservation oversampling on 2026-06-12: `src/analysis/build_covo_preserve_sft_split.py` creates SFT splits from real CB-Whisper evidence and oversamples train rows where ASR top-1 already contains a true hotword mention. Using the same dev600 compact evidence, the 500-row train base became 1422 rows after repeating 461 hotword-preserved rows twice. Continuing the real-dev600 adapter for 120 bf16 LoRA steps gave final eval loss `0.2139`. On Pilot100, CER improved from the previous best `0.08094` to `0.03916`, and Entity Recall improved from `0.7477` to `0.8037`. On the full AISHELL test set, covo-evaluator CER improved from `0.05650` to `0.04400`, with `309` improved, `62` worsened, and `437` unchanged samples (`logs/experiment_cbwhisper_covo_test_full_realdev600_preserve2_ft120_stdout.log`). Full-test Entity Recall improved from `0.8541` to `0.8785`; lost hotwords dropped from `89` to `65`, while gained hotwords stayed essentially unchanged (`35` to `36`). This is the current best no-gate CB-Whisper + covo result: it beats the 6% CER target and partially repairs the recall loss, but still trails the CB-Whisper input recall `0.9028`.

Full AISHELL-train hotword no-op SFT on 2026-06-14: `src/analysis/build_aishell_train_hotword_noop_sft.py` converts the full AISHELL train hotword alignments (`datasets/aishell/train/aligned.txt`) and transcripts into 17301 no-op hotword-preservation Qwen/COVO rows. These rows were mixed with the real CB-Whisper dev600 preserve2 rows for a 18723-row training set, then the preserve2 adapter was continued for one full epoch with batch size 7. Training completed in about 62 minutes with final eval loss `0.2208` and train loss `0.1704`. On Pilot100, CER improved to `0.03721` and Entity Recall to `0.8318`. On the full AISHELL test set, covo-evaluator CER improved to `0.04354`, with `300` improved, `41` worsened, and `467` unchanged samples (`logs/experiment_cbwhisper_covo_test_full_aishell_train_noop_bs7_stdout.log`). Full-test Entity Recall reached `0.9039`, slightly above the CB-Whisper input recall `0.9028`; lost hotwords dropped to `39`. This is now the best no-gate result and meets both targets: recall is above 90% and CER is below 6% under the current evaluation setup.

Train-side real-error COVO probe on 2026-06-14: `src/analysis/build_aishell_hotword_train_split.py` materializes AISHELL train as a CB-Whisper hotword split by reusing existing KWS hidden states. A full `hotword/train` split was generated with `17301` usable utterances and `20000` keywords. Because loading all 20000 keyword hidden states is expensive, a first `train_probe1000` split used `1000` train utterances and `2000` keywords. Exporting 200 CB-Whisper evidence rows from this split took about `9m41s` and produced `logs/cbwhisper_covo_evidence_train_probe200.jsonl`. The raw CB-Whisper top1 on these 200 train rows had CER `0.1252` and keyword recall `0.8686`, so it does contain real correction signal. However, continuing the current best adapter for 40 steps on only these 200 real-error rows regressed Pilot100 from CER `0.03721` / keyword recall `0.8136` to CER `0.04178` / keyword recall `0.7542`. Decision: the route is clean and should be scaled with more real train evidence, but the 200-row short run is too small and overfits, so do not promote `outputs/qwen35_cbwhisper_train_probe200_realerr_40steps_bf16`.

Full train-side real-error COVO SFT on 2026-06-15: all `17301` AISHELL train hotword utterances were exported as real CB-Whisper evidence in 18 shards and merged to `logs/cbwhisper_covo_evidence_train_full.jsonl`. The train evidence has raw CB-Whisper top1 CER `0.1155`, keyword recall `0.9044`, exact rate `0.4041`, and average n-best size `5.42`. A compact COVO training set was mixed from full train real-error evidence (`17301` rows), full AISHELL train no-op preservation rows (`17301` rows), and dev600 preserve2 rows (`1422` rows), for `36024` rows total. Continuing the current best no-op adapter for one epoch at lr `1e-5` produced `outputs/qwen35_cbwhisper_train_full_real_noop_preserve2_1epoch_bf16_bs7`, with final eval loss `0.2133` and train loss `0.2005`. Pilot100 regressed from the current best CER `0.03721` / keyword recall `0.8136` to CER `0.04634` / keyword recall `0.7034`, with `30` improved, `18` worsened, and `52` unchanged samples. Full AISHELL test also regressed: CER `0.04680` vs current best `0.04354`, keyword recall `0.8316` vs `0.9057`, lost hotwords `108` vs `39`, with `312` improved, `95` worsened, and `401` unchanged samples. Decision: do not promote this adapter. Real-error SFT at this scale makes the corrector more aggressive and damages hotword preservation; the next variant should either reduce real-error weight, train fewer steps, or use an explicit preservation-balanced sampling schedule rather than a full real-error epoch.

Hotword-anchored real-error COVO SFT on 2026-06-15: to avoid teaching the model to rewrite already-correct hotwords, a stricter subset was built from the full train evidence: keep only real-error rows where all true keyword mentions already appear in ASR top-1, but the sentence is still not exactly correct. This produced `6408` train-real-error-hotword-anchored rows, mixed with `17301` full AISHELL train no-op preservation rows and `1422` dev600 preserve2 rows (`25131` rows total). Continuing the current best no-op adapter for `300` steps at lr `5e-6` produced `outputs/qwen35_cbwhisper_hotword_anchored_real_300steps_bf16_bs7`, with final eval loss `0.2176` and train loss `0.1948`. Pilot100 still regressed: CER `0.04504` vs current best `0.03721`, keyword recall `0.7373` vs `0.8136`, lost hotwords `22` vs `13`, with `30` improved, `16` worsened, and `54` unchanged samples. Error inspection showed the model changing already-present rare proper nouns into more common homophones or variants, e.g. `许玮甯 -> 许玮宁`, `今久 -> 金九`, `宋芳 -> 孙芳`, `杨锋 -> 杨峰`. A prompt-only strict preservation probe with the current best adapter did not improve recall and slightly worsened Pilot100 CER (`0.03786`). Decision: do not promote this route. The best result remains `qwen35_cbwhisper_preserve2_aishell_train_noop_1epoch_bf16_bs7`; further gains likely need better target construction or contrastive preservation training, not more generic real-error SFT.

Contrastive hotword-preservation DPO probe on 2026-06-16: `src/analysis/build_covo_hotword_dpo_pairs.py` builds preference pairs from full AISHELL train CB-Whisper evidence. For each row, chosen is the reference text and rejected is a close n-best candidate that drops at least one prompt-side hotword mention; this yielded `2583` real n-best preference pairs. Continuing the current best no-op adapter with DPO for `30` steps produced `outputs/qwen35_cbwhisper_hotword_preserve_dpo_30steps_bf16`. Pilot100 improved over the current best on both CER and hotword preservation: CER `0.03460` vs `0.03721`, keyword recall `0.8475` vs `0.8136`, and lost hotwords `9` vs `13`. Full AISHELL test showed the expected trade-off: keyword recall improved from `0.9057` to `0.9227`, lost hotwords dropped from `39` to `22`, but CER worsened from `0.04354` to `0.04501`. A weaker `15`-step DPO probe was not better on Pilot100 (CER `0.03786`, keyword recall `0.8305`) and was not promoted to full test. Decision: DPO is effective as a recall-oriented variant and validates the contrastive-preservation idea, but it does not replace the current CER-best adapter yet. The next useful variant should reduce DPO strength or mix in CER-preserving preferences so the recall gain does not cost overall CER.

Full-pass contrastive DPO on 2026-06-16: to test whether the 30-step signal scales, the same `2583` preference pairs were trained for `650` DPO steps, roughly one full pass over the pair set with gradient accumulation 4. To avoid over-strengthening the recall bias, lr and beta were lowered (`lr=2e-6`, `beta=0.03`, `sft_weight=0.03`). The output adapter was `outputs/qwen35_cbwhisper_hotword_preserve_dpo_full_lr2e6_beta003_bf16`. Full AISHELL test did not improve: CER regressed to `0.05658`, keyword recall was only `0.9089`, lost hotwords `32`, with `244` improved, `58` worsened, and `506` unchanged samples. This is worse than both the CER-best no-op adapter (`0.04354` CER, `0.9057` recall) and the short DPO high-recall adapter (`0.04501` CER, `0.9227` recall). Decision: do not promote full-pass DPO. The useful region is a very short preference nudge, not full convergence on the hotword-preservation pairs.

Mixed DPO preference probes on 2026-06-16: `src/analysis/build_covo_mixed_dpo_pairs.py` adds CER-candidate and no-op conservative preference pairs to the hotword-preservation pairs. A balanced three-way set (`2583` hotword + `2583` CER + `2583` no-op pairs) was trained for `20` and `60` steps. Both became too conservative: mixed20 reached CER `0.05728`, keyword recall `0.9142`, lost hotwords `15`; mixed60 reached CER `0.08188`, keyword recall `0.9121`, lost hotwords `1`. A two-way hotword+CER set (`2583` + `2583`, no no-op) trained for `30` steps also over-constrained output: CER `0.06271`, keyword recall `0.9206`, lost hotwords `5`. Decision: mixed DPO does not beat the short hotword-only DPO or the no-op CER-best adapter. The added conservative preferences suppress useful corrections more than they recover CER.

AISHELL COVO error audit on 2026-06-16: `src/analysis/covo_error_audit.py` audits the current best full-test predictions (`qwen35_cbwhisper_preserve2_aishell_train_noop_1epoch_bf16_bs7`) against the COVO bridge input and n-best. The current best has CER `0.04354` vs bridge-input CER `0.10043`. This bridge-input CER is computed from `input.asr_top1` in the COVO evidence JSONL and is not the same measurement as the earlier standalone CB-Whisper AISHELL CER (`~0.0820`). N-best oracle alone is only `0.06178`, so COVO is already better than selecting among CB-Whisper candidates under this bridge-evidence setup. However, the oracle that can choose between COVO output and n-best reaches `0.03097`, showing some remaining recoverable errors but also implying that the current evidence pool does not naturally explain the ChineseHP-style `0.0277` result. Exact-match counts: bridge input `368/808`, n-best oracle `486/808`, current COVO `523/808`. COVO fixed `181` samples exactly and partially improved `119`, but left `125` erroneous samples unchanged, broke `26` originally correct samples, and worsened `15` already-wrong samples. True hotword recall is stable around `0.9055`; COVO loses `39` base-hit hotwords and gains `35`, while reducing false prompt-hotword insertions from `24` in the bridge input to `8`. OpenCC traditional-to-simplified normalization only reduces current CER from `0.04354` to `0.04261`, so the remaining gap is not mainly a繁简 issue.

Legacy CB-Whisper-style CER recalculation on 2026-06-17: to avoid mixing evaluator definitions, the full-test COVO predictions were also recalculated with the same structural CER used by `src/model/cb_whisper.py` test metrics: normalize surface text, compute per-sample `edit_distance(ref, pred) / len(ref)`, then average over samples. Under this mean-sample CER口径, the bridge input baseline for the COVO evidence is `0.06600`, the current best no-op adapter is `0.04297`, the short hotword DPO adapter is `0.04417`, and the full-real 500-step continuation is `0.04943`. The corresponding corpus-level normalized CERs are `0.06750`, `0.04149`, `0.04234`, and `0.04583`. Going forward, COVO results should report this legacy mean-sample CER alongside the COVO correction evaluator CER.

Complete AISHELL test-set check on 2026-06-17: the previous COVO "full test" files cover the 808-sample AISHELL hotword subset, not the complete AISHELL test split. A separate full-split decoder was added in `src/analysis/aishell_full_whisper_decode.py` and run on all `7176` wavs under `datasets/aishell/data_aishell/wav/test` with `openai/whisper-large-v3` greedy decoding. The raw Whisper large-v3 full-test baseline is mean-sample CER `0.09060` and corpus CER `0.09023` (`src/logs/aishell_full_whisper_large_v3_test_full.jsonl`). Feeding the same 7176 outputs through the current best no-gate COVO adapter `qwen35_cbwhisper_preserve2_aishell_train_noop_1epoch_bf16_bs7` gives COVO evaluator CER `0.06137` versus baseline `0.09288`, with `1184` improved, `20` worsened, and `5972` unchanged samples. Recomputed with the older stripped-text mean-sample口径, the baseline is mean CER `0.08070` / corpus CER `0.08044`, and the COVO output is mean CER `0.05752` / corpus CER `0.05589`. This is the first true complete-AISHELL result and should be kept separate from all 808-row hotword-subset recall/CER tables.

Full AISHELL n-best sanity check on 2026-06-17: `src/analysis/aishell_full_whisper_decode.py` was extended with `--num-return-sequences` and the COVO bridge was updated to copy top-level `nbest`. A full `openai/whisper-large-v3` `beam5/return5` decode was then run on all 7176 AISHELL test wavs (`src/logs/aishell_full_whisper_large_v3_nbest5_test_full.jsonl`). The beam top1 improved slightly over greedy, with mean-sample CER `0.08735` and corpus CER `0.08730`, but every sample collapsed to one unique normalized hypothesis after de-duplication: average unique n-best `1.0000`, samples with more than one unique candidate `0`, and n-best oracle CER equal to top1 CER. Therefore the missing n-best in the full-split COVO run is not merely a bridge bug; ordinary deterministic HuggingFace Whisper beam search does not provide useful diverse candidates here. The useful CB-Whisper n-best observed in earlier experiments comes from the original `PBAWhisper` path with KWS prompt injection and shortform candidate processing, which is currently tied to the hotword split rather than the 7176-row full AISHELL split.

Sampled diverse full-AISHELL n-best check on 2026-06-17: after deterministic beam search collapsed, `src/analysis/aishell_full_whisper_decode.py` was extended with sampled n-best de-duplication. A full `openai/whisper-large-v3` run with greedy top1 plus temperature `0.8` sampling generated `src/logs/aishell_full_whisper_large_v3_sample_t08_nbest5_test_full.jsonl` for all `7176` AISHELL test wavs. The output is complete. It produced average unique n-best size `2.6506`, with `4332` samples having more than one unique candidate and `1943` samples reaching five unique candidates. However, top1 CER worsened to mean-sample `0.09050` / corpus `0.09012`, while n-best oracle CER was mean-sample `0.06471` / corpus `0.06558`. Interpretation: sampling finally creates candidate diversity, but the oracle ceiling is still weaker than the complete-AISHELL COVO result (`0.05752` old mean-sample CER / `0.06137` COVO evaluator CER). This route may help only if COVO can use diverse candidates conservatively; it is not enough by itself to explain or exceed the best COVO result.

AISHELL 808 hotword multi-prompt n-best check on 2026-06-18: `src/analysis/aishell_hotword_multiprompt_nbest.py` was added to test whether Whisper large-v3 can generate useful alternatives by itself when prompted with CB-Whisper/KWS hotwords. The run used the 808-row AISHELL hotword test evidence, four prompt variants (`none`, parenthesized top-3 hotwords, natural top-3, natural top-6), and temperatures `0` and `0.4` with `beam5/return5`, writing `src/logs/aishell_hotword_multiprompt_nbest_test808_t04.jsonl`. Multi-prompt candidates alone produced average unique n-best `3.9295`, `634/808` samples with more than one unique candidate, top1 mean CER `0.11230`, oracle mean CER `0.03842`, and oracle hotword recall `0.9364`. This is not good enough as a standalone ASR output because top1 quality is poor. However, when these candidates are unioned with the existing CB-Whisper evidence n-best, the candidate pool improves from CB-only oracle mean/corpus CER `0.03293 / 0.03344` to union oracle `0.02711 / 0.02725`; exact-reference-in-candidates improves from `576/808` to `606/808`, and average unique n-best rises from `6.0965` to `8.0087`. Interpretation: multi-prompt Whisper is useful as a complementary candidate generator, not as a replacement for CB-Whisper decoding. The next meaningful experiment is to feed the union candidate pool to COVO/reranking conservatively.

Targeted and longer COVO capability probes on 2026-06-16: `src/analysis/build_covo_targeted_sft.py` builds targeted SFT rows from full AISHELL train CB-Whisper evidence. Three follow-up variants were tried from the current best no-op adapter. First, `false_hotword_rejection + nbest_local_repair + noop` (`2800/2800/2800`, 120 steps) badly damaged hotword preservation: full-test CER `0.05029`, keyword recall `0.8432`, lost hotwords `98`. Removing false-hotword rejection and training only `nbest_local_repair + noop` improved stability but still did not beat the current best: 60 steps reached CER `0.04463`, recall `0.8888`; a safer 1:5 repair/noop ratio trained for 200 steps with visible loss logging reached CER `0.04447`, recall `0.8919`. Finally, a larger full-real continuation on `train_full_real_plus_noop_preserve2.jsonl` (`36024` rows) was run for 500 steps with constant lr `5e-6`; eval loss improved from `0.2187` at step 125 to `0.2163` at step 500, but full-test metrics regressed to CER `0.04711`, recall `0.8379`. Its checkpoint-250 was also worse (CER `0.04563`, recall `0.8496`). Decision: more generic real-error SFT does train, but it teaches the generator to rewrite too aggressively and sacrifices already-present hotwords. Do not promote these adapters; the main adapter remains `qwen35_cbwhisper_preserve2_aishell_train_noop_1epoch_bf16_bs7`.

CB-Whisper candidate-quality update on 2026-06-26: `src/analysis/build_cbwhisper_candidate_pool.py` was added to diagnose ChineseHP-like 10-best candidate construction. On the 808-row AISHELL hotword subset, merging CB-Whisper evidence candidates with complementary multi-prompt candidates produced average unique n-best `7.4022`, `287/808` samples with 10 candidates, exact reference in pool `595/808`, and oracle mean/corpus CER `0.03003 / 0.03058`. CB-only evidence was average unique `6.0965` with oracle mean/corpus CER `0.03293 / 0.03344`, so the complementary candidates help but still do not fully reproduce ChineseHP's near-10 unique n-best. `CBWhisper` now exposes `rescore_generation_factor`, `rescore_generation_cap`, and `covo_nbest`; the AISHELL config targets 10 retained candidates and up to 32 generated candidates before de-duplication.

Integrated AISHELL v3 10-best run on 2026-06-26: `src/configs/cb-whisper-aishell-v3-kws.yaml` was updated to `rescore_nbest=10`, `rescore_generation_cap=32`, and `covo_nbest=10`, then rerun on the 808-row AISHELL hotword test set. The exported evidence reached average unique n-best `8.8205`, `524/808` rows with 10 unique candidates, exact reference in n-best `612/808`, and n-best oracle corpus CER `0.02864`. Standalone CB-Whisper top1 metrics were Entity Recall `0.92707`, CER `0.07102`, Hotword Only CER `0.04531`, and WER `0.46535`; therefore this route is best treated as a candidate-quality improvement for downstream COVO/reranker input, not as a standalone top1 CER improvement.

10-best cleanup on 2026-06-26: `CBWhisper` now has a conservative `enable_covo_candidate_quality_filter` for exported COVO/evidence n-best candidates, and `src/analysis/clean_covo_nbest_quality.py` can clean existing evidence files. On `cbwhisper_covo_evidence_aishell_v3_nbest10_gen32.jsonl`, the filter removed `28` obvious outliers (`too_long=13`, `too_short=12`, `repeat_heavy=1`, `latin_tail=2`) while preserving exact-reference coverage `612/808`, oracle corpus CER `0.02864`, and oracle hotword recall `0.9448`. The cleaned file is `src/logs/cbwhisper_covo_evidence_aishell_v3_nbest10_gen32_clean.jsonl`.

Targeted n-best supplement on 2026-06-26: `src/analysis/targeted_supplement_covo_nbest.py` was added to supplement only rows with fewer than 10 unique candidates. Merging cleaned CB-Whisper 10-best evidence with existing multi-prompt and full-AISHELL sampled candidates reached average unique n-best `9.2649`; one targeted supplement pass over 195 low-diversity rows reached `9.6720`; a second higher-temperature pass over the remaining 86 rows reached `10.0000`. A stronger cleaned version removed 73 long-suffix artifacts and still reached average unique n-best `9.9097`, rows with 10 candidates `789/808`, and oracle corpus CER `0.02864`. This confirms that the ChineseHP-like `9.8+` candidate-diversity target is reachable, though the clean and unclean supplemented pools should both be compared before downstream COVO use.

Cleanliness-first candidate pool on 2026-06-26: the quality filter was tightened so severe pollution is removed before length statistics are computed, including video/platform tails, Unicode replacement characters, long Latin tails, and repeated-heavy strings. The final preferred clean file is `src/logs/cbwhisper_candidate_pool_v3_nbest10_targeted_supplement_round2_clean_strict_slack5.jsonl`: average unique n-best `9.9332`, rows with 10 candidates `779/808`, exact reference in n-best `616/808`, oracle corpus CER `0.03019`, and oracle hotword recall `0.9459`. A strict audit found `0` bad phrase tails, `0` long Latin tails, `0` replacement-character candidates, `0` repeated-heavy candidates, and `0` extreme length outliers.

COVO check on the cleanliness-first 9.8+ pool on 2026-06-26: using the current best no-gate adapter `qwen35_cbwhisper_preserve2_aishell_train_noop_1epoch_bf16_bs7` with `max_nbest=10` produced COVO evaluator CER `0.04594` versus baseline `0.10446`, with `310` improved, `48` worsened, and `450` unchanged samples. Under the legacy CB-Whisper-style normalization, prediction mean/corpus CER was `0.04564 / 0.04327`. Hotword recall fell from base `0.9151` to prediction `0.8907`. This is slightly worse than the current best no-op COVO result (`0.04354` evaluator CER), so it is not promoted; the existing COVO adapter does not yet exploit the richer clean 10-best evidence well enough.

COVO reliability-label probe on 2026-06-27: `src/analysis/cbwhisper_covo_bridge.py` now labels each n-best hypothesis as `trusted_scored` when it matches a CB-Whisper scored candidate, or `supplemental_unscored` when it comes only from candidate supplementation, and annotates which prompt/context hotwords each hypothesis preserves. The prompt also tells COVO not to let supplemental candidates alone override ASR top-1 or high-score trusted candidates, especially when this would replace an already-present prompt hotword with a common homophone. On the cleanliness-first pool, full-test `max_nbest=6` with reliability labels improved over the previous `max_nbest=10` COVO run but still did not beat the current best adapter result: COVO evaluator CER `0.04494` vs `0.04594`, improved/worsened/unchanged `314/43/451`, legacy mean/corpus CER `0.04671 / 0.04494`, and hotword recall `0.9034` vs `0.8907`. This confirms the diagnosis: the richer pool is useful, but the current COVO adapter needs training on reliability-labeled 10-best evidence to exploit it fully.

Compact reliability-label COVO training probe on 2026-06-27: the reliability prompt was shortened by keeping `max_nbest=6`, `max_pinyin=3`, `max_hotwords=6`, and removing the duplicate candidate-score block. Without retraining, this compact prompt improved the cleanliness-first pool to COVO evaluator CER `0.04408`, legacy mean/corpus CER `0.04625 / 0.04408`, and hotword recall `0.9108`. Continuing the current best adapter on AISHELL train reliability-labeled compact evidence for 40 and 80 steps both reached COVO evaluator CER `0.04362`; 80 steps had slightly better legacy mean CER (`0.04564` vs `0.04574`) while 40 steps had slightly higher hotword recall (`0.9087` vs `0.9076`). This nearly matches the old CER-best no-gate adapter (`0.04354`) while using the richer clean n-best pool and retaining higher recall than the old result. It is a useful direction, but not yet a clean replacement for the current main result.

Full Shuili CB-SenseVoice + COVO comparison on 2026-07-15: the 990-row
`data_shuil_videos_largev3` test set was decoded by end-to-end CB-SenseVoice,
then passed to three historical COVO adapters with identical ChineseHP-style
evidence (`6` n-best, consensus/uncertain spans, pinyin, and predicted
hotwords). No gate or post-hoc fallback was used. Under the CB normalization,
the CB-SenseVoice input has mean/corpus CER `0.06522 / 0.05520`, Entity Recall
`0.96847`, and `671/990` exact rows. The original ChineseHP hard-negative COVO
adapter improves these to mean/corpus CER `0.06399 / 0.05331`, Entity Recall
`0.97748`, and `701/990` exact rows. The prior AISHELL-best preserve2/no-op
adapter increases recall to `0.97973`, but regresses mean/corpus CER to
`0.08896 / 0.08520`, with only `642/990` exact rows. The later hotword-use
adapter behaves almost identically (`0.08910 / 0.08520` CER, `0.97973` recall,
`641/990` exact). Therefore the original COVO is the accepted model for this
integrated Shuili path; both AISHELL hotword-specialized adapters are rejected
because their small recall gain is outweighed by generic over-correction.
Machine-readable details are in
`src/logs/cb_sensevoice_covo_shuili_videos_model_comparison_20260715.json`.
When Chinese and Arabic number forms are treated as equivalent, the current
CB-SenseVoice input corpus CER is `0.04798`; original COVO improves it to
`0.04565`, while preserve2 and hotword-use remain worse at `0.05562` and
`0.05553`. The historical external SenseVoice-anchor route remains best at
`0.04304`, but it includes same-length and digit post-filters. Shuili results
should therefore report number-normalized CER as the semantic metric and raw
CER as a surface-form diagnostic.

Shuili error-hotword diagnostic on 2026-07-15: an explicit, corpus-backed
hotword extension path was added to `src/analysis/build_shuili_video_dataset.py`.
Adding 79 terms identified from current test errors expands the lexicon from
180 to 259 words and lowers number-normalized CB-SenseVoice corpus CER from
`0.04798` to `0.04072` (`515 -> 437` edits; `696 -> 745` exact rows). This run
is a diagnostic upper bound and must not be reported as an untuned paper test
because its lexicon was derived from test errors. The original no-gate COVO
regresses the stronger input to `0.04304` CER, so CB-SenseVoice top1 is the
accepted output for this diagnostic. A reportable version must freeze a
lexicon built only from train/dev data or an external domain glossary.

Listwise COVO update on 2026-07-15: the expanded CB-SenseVoice candidate pool
has a number-normalized oracle CER of `0.02227`, so COVO was changed from
free-form rewriting to conditional-likelihood N-best scoring. Oracle-candidate
SFT on independent AISHELL CB-SenseVoice evidence lowers Shuili diagnostic CER
to `0.03764`; a 60-step near-miss/no-op DPO continuation improves it to
`0.03736` (`437 -> 401` edits, `745 -> 768` exact rows). An independent
1152-row old-Shuili source-domain continuation regresses to `0.03960` and is
rejected. The DPO listwise model is the current best, but the sub-3% target has
not been reached without test-label tuning.

Independent course-recording transfer on 2026-07-15: the AISHELL-trained
DPO60 listwise COVO was evaluated on all `1152` utterances of the original
`data_shuil_largev3` classroom recording, without training on this test set or
removing spoken fillers. Number-normalized corpus CER changes from naked
SenseVoice `0.04636`, to CB-SenseVoice `0.04059`, and finally to listwise COVO
`0.03887`. Aligned hotword recall changes from `0.82884`, to `0.93050`, and
then `0.93136`. COVO reduces edits from `640` to `613`, with `61/40/1051`
improved/worsened/unchanged rows. The candidate oracle is `0.01820`, leaving
substantial candidate-selection headroom.

Explicit test-leak diagnostic on 2026-07-15: `259` terms and oracle candidate
labels from the 990-row Shuili-video test set were used to build `344` domain
DPO pairs. A 43-step continuation does not improve the fixed AISHELL-selected
score interpolation (`0.03722 -> 0.03769` filler+number-normalized CER). When
the CB interpolation weight is also tuned on the same test set, the leaked
model reaches `0.03636` versus the original model's test-tuned `0.03703`, a
gain of only seven edits. This result is diagnostic only and must never be
reported as an untuned paper result.

Test-leaked confusion coverage was then tightened with 20 manually checked
professional phrases injected directly into the COVO evidence and 408 repeated
exact-homophone hard pairs. A 70-step DPO continuation lowers filler+number
normalized CER to `0.03522` at weight `0.5`, or **`0.03484`** after leaked
test-set interpolation tuning (`366` edits). Pure-homophone misses fall from
`42` to `36`, and exact-reference homophone misses from `23` to `15`.
Aligned hotword recall is `790/826 = 0.95642`, compared with CB-SenseVoice
`786/826 = 0.95157` and naked SenseVoice `682/826 = 0.82567`.
Suspected reference errors such as `南路/南麓` and `饮水量/引水量` were excluded
from the injected phrase list. This remains a deliberately invalid test-leak
diagnostic, not a paper result.

## Current consolidated results (2026-07-15)

Metrics with different normalization rules are not directly interchangeable.
AISHELL uses the original CB mean-sample CER convention. The two Shuili tables
use corpus CER with Chinese/Arabic numbers normalized; filler removal is shown
only where explicitly stated. `Independent` means the evaluated references
were not used for model/lexicon fitting, while `Leaked diagnostic` must never
be reported as a paper test result.

### AISHELL hotword test (808 utterances)

| System | CER | Hotword recall | Hotword CER | WER | Status |
|---|---:|---:|---:|---:|---|
| Naked SenseVoice | 8.554% | - | - | - | Independent |
| SenseVoice KWS + Whisper-v3 | 7.174% | 92.155% | 5.228% | 46.906% | Independent |
| End-to-end CB-SenseVoice | 6.202% | 83.204% | 10.979% | 40.099% | Independent |
| CB-SenseVoice + preserve2 COVO | **4.379%** | 83.978% | - | - | Independent |
| Candidate oracle | 3.686% | 85.128% | - | - | Upper bound |

The accepted COVO row has `541/808` exact utterances. Under COVO's separate
corpus evaluator, the same prediction is `4.137%` CER; this value must not be
mixed with the CB mean-sample CER column above.

### Original Shuili course recording (1152 utterances)

| System | Number-normalized CER | Raw CER | Hotword recall | Edits | Exact | Status |
|---|---:|---:|---:|---:|---:|---|
| Naked SenseVoice | 4.636% | 4.654% | 82.884% | 731 | 722 | Independent |
| CB-SenseVoice | 4.059% | 4.076% | 93.050% | 640 | 763 | Independent |
| AISHELL DPO60 listwise COVO | **3.887%** | **3.905%** | **93.136%** | **613** | **774** | Independent |
| Candidate oracle | 1.820% | - | - | - | - | Upper bound |

### New Shuili video set (990 utterances)

| System | CER | Filler+number CER | Hotword recall | Edits | Exact | Status |
|---|---:|---:|---:|---:|---:|---|
| Naked SenseVoice | 4.873% | 4.902% | 82.567% (259-term mentions) | 523 | 699 | Independent |
| CB-SenseVoice, original 180 terms | 4.798% | - | 97.849% (historical 180-term mentions) | 515 | 696 | Independent |
| CB-SenseVoice, expanded 259 terms | 4.072% | 4.074% | 95.157% | 437 | 745 | Leaked diagnostic |
| Original COVO free generation | 4.304% | - | - | 462 | 748 | Rejected, leaked pool |
| AISHELL oracle SFT free generation | 4.090% | - | - | 439 | 761 | Rejected, leaked pool |
| SFT output projected to N-best | 3.950% | - | - | 424 | 764 | Diagnostic, leaked pool |
| AISHELL SFT listwise | 3.764% | - | - | 404 | 767 | Superseded, leaked pool |
| AISHELL DPO60 listwise | 3.736% | 3.722% | - | 401 | 768 | Best before direct test-label training |
| Old-Shuili source continuation | 3.960% | - | - | 425 | 759 | Rejected distribution shift |
| First direct test-term DPO, test-tuned | - | 3.636% | - | 382 | - | Explicit test leak |
| Confusion-term DPO70, weight 0.5 | 3.540% | 3.522% | - | 380 | 787 | Explicit test leak |
| **Confusion-term DPO70, test-tuned weight 0.3** | **3.503%** | **3.484%** | **95.642%** | **376** | **790** | **Explicit test leak** |
| Candidate oracle | 2.227% | 2.208% | - | 239 | - | Upper bound on leaked pool |

The final test-tuned row has `796` exact utterances after filler removal.
Current 259-term aligned recall uses 826 mentions:
naked SenseVoice `682/826`, CB-SenseVoice `786/826`, and the final leaked COVO
`790/826`.

## License

See the [LICENSE.md](LICENSE.md) file for details.

## Citation

If you use any of the resources in this repository, please cite the following paper:

Citation will be added in the future.
