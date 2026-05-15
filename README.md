
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

Original Priberam large-v3 endpoint baseline on AISHELL: using the original `Enhance-CB-Whisper` model path with only compatibility adaptations for current large-v3 single-channel hidden states and modern Transformers, standalone KWS test on natural AISHELL hotwords gives Precision `0.8023`, Recall `0.8747`, and F1 `0.8370` (`Enhance-CB-Whisper/src/logs/resnet1_natural_kws_test_full_stdout.log`). End-to-end original CB-Whisper with `openai/whisper-large-v3` and the same original ResNet checkpoint gives Entity Recall `0.7812` with 95% CI `[0.7235, 0.8372]` (`Enhance-CB-Whisper/src/logs/cbwhisper_original_large_v3_resnet1_natural_direct_stdout.log`). The original endpoint evaluator reports Entity Recall only, so CER/WER are not available from this run. Compared with the current best true-v3 CB-Whisper result (`0.9238` Entity Recall, `0.0661` CER), the original-code baseline is far behind at the ASR endpoint even though its KWS recall is already reasonably high.

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

## License

See the [LICENSE.md](LICENSE.md) file for details.

## Citation

If you use any of the resources in this repository, please cite the following paper:

Citation will be added in the future.
