import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
sys.path.insert(1, os.path.join(os.path.dirname(__file__), "data"))

import pytorch_lightning as pl

from data.data_module import DatasetInfo, KWSDataMod
from model.cb_whisper import CBWhisper


ROOT = "/root/autodl-tmp/datasets/shuili/data_shuil_largev3"
KWS_CKPT = (
    "/root/autodl-tmp/src/outputs/aishell_large_v3_kws_true/"
    "checkpoints/f1G/f1G-epoch=11-step=72060.ckpt"
)


def main():
    pl.seed_everything(123)
    data = KWSDataMod(
        batch_size=1,
        sampling="random",
        num_workers=4,
        train_info=[DatasetInfo(name="aishell", root="/root/autodl-tmp/datasets/aishell/data_aishell", kw_type="tts")],
        val_info=[DatasetInfo(name="aishell", root="/root/autodl-tmp/datasets/aishell/data_aishell", kw_type="tts")],
        test_info=DatasetInfo(name="shuili", root=ROOT, kw_type="natural"),
        hotwords_per_group=100,
        features_size=(150, 750),
        test_split="test",
        whisper_ckpt="openai/whisper-large-v3",
    )
    model = CBWhisper(
        dataset="shuili",
        split="test",
        root=os.path.join(ROOT, "hotword"),
        kw_type="natural",
        encoder_ckpt="openai/whisper-large-v3",
        whisper_ckpt="openai/whisper-large-v3",
        kws_ckpt=KWS_CKPT,
        language="chinese",
        force_decoder_prompt_ids=True,
        prompt=True,
        oracle="kws",
        kws_features_size=(150, 750),
        keyword_prompt_prepend="(",
        keyword_prompt_append=")",
        keyword_separator=" ",
        keywords_per_group=100,
        kws_positive_threshold=0.05,
        kws_topk_per_group=50,
        kws_max_prompt_keywords=200,
        kws_infer_chunk_size=16,
        prompt_max_injected_keywords=4,
        prompt_score_threshold=0.55,
        prompt_relative_threshold=0.80,
        enable_nested_keyword_promotion=True,
        nested_keyword_promotion_score_ratio=0.95,
        nested_keyword_promotion_min_long_chars=3,
        nested_keyword_promotion_max_extra=2,
        keyword_perturb_prob=0.0,
        enable_phonetic_rescore=True,
        rescore_nbest=8,
        rescore_use_asr_score=True,
        rescore_asr_weight=1.0,
        rescore_keyword_weight=1.8,
        rescore_phonetic_weight=0.7,
        rescore_prefix_penalty_weight=1.0,
        shortform_no_repeat_ngram_size=3,
        rescore_max_keywords=160,
        enable_phonetic_surface_repair=True,
        surface_repair_score_threshold=0.95,
        surface_repair_ambiguity_gap=0.05,
        surface_repair_max_keywords=8,
        surface_repair_max_edits=2,
        surface_repair_min_keyword_chars=2,
        surface_repair_max_keyword_chars=6,
        enable_phonetic_consensus_repair=True,
        consensus_repair_score_threshold=0.90,
        consensus_repair_min_support=2,
        enable_consensus_rerank=True,
        consensus_rerank_weight=0.35,
        consensus_rerank_min_support=2,
        neutral_anchor=True,
        neutral_anchor_as_top1=False,
        neutral_anchor_include_in_nbest=True,
        neutral_anchor_num_beams=5,
        neutral_anchor_skip_surface_repair=True,
        oracle_nbest_diagnostic=True,
        oracle_nbest_detail_path="logs/oracle_nbest_detail_shuili_v3_neutral_anchor.csv",
        oracle_nbest_summary_path="logs/oracle_nbest_summary_shuili_v3_neutral_anchor.csv",
    )
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        inference_mode=True,
    )
    trainer.test(model=model, datamodule=data)


if __name__ == "__main__":
    main()
