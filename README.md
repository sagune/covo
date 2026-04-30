
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

Experiment bookkeeping rule: after each code change, commit to git; after each experiment, append the result here and compare against the previous run or best known run.

| Date | Config | Checkpoint | Key settings | Entity Recall | CER | Hotword Only CER | WER | Notes |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| 2026-04-27 | `configs/cb-whisper-aishell.yaml` | `src/mlruns/641753688314575260/d9b9fafe87d64bc98400705d2c58525c/checkpoints/f1G-epoch=7-step=48040.ckpt` | KWS prompt, n-best=8, pinyin rescore, surface/consensus repair, consensus rerank | 0.8475 | 0.0822 | 0.1087 | 0.5161 | Current baseline. Oracle n-best top1 recall is 0.8455, oracle recall is 0.8650, leaving about +0.0194 recall headroom in the candidate pool. |

## License

See the [LICENSE.md](LICENSE.md) file for details.

## Citation

If you use any of the resources in this repository, please cite the following paper:

Citation will be added in the future.
