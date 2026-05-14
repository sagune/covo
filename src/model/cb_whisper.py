import re
from typing import List, Union, Optional, Tuple, Dict, Any
import os
import json
import unicodedata
from datetime import datetime
import torch
import torchvision
import pytorch_lightning as pl
from transformers import WhisperProcessor, WhisperModel
from .model import KWSModel
from .pba_whisper import PBAWhisper
import sys
sys.path.insert(1, '../data')
from dataset import AishellHotwordDataset, ACL6060KeywordDataset
from whisper.audio import N_FRAMES
from scorer import entity_recall
import random
import pandas as pd
from confidence_intervals import evaluate_with_conf_int
from tqdm.auto import tqdm


class CBWhisper(pl.LightningModule):
    @staticmethod
    def _infer_kws_model_kwargs_from_ckpt(ckpt_path: str) -> dict:
        """
        Infer KWSModel constructor kwargs from a checkpoint.
        We only support the current TCResNet-based KWS checkpoints here.
        """
        kwargs = {
            "backbone": "tcresnet",
            "tcresnet_channels": 96,
            "tcresnet_blocks": 8,
            "tcresnet_dropout": 0.15,
            "subsequence_aux": False,
            "subsequence_aux_weight": 0.02,
            "subsequence_aux_temperature": 0.5,
            "adversarial_training": False,
            "num_domains": 2,
        }
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu")
            hparams = ckpt.get("hyper_parameters", {}) or {}
            state_dict = ckpt.get("state_dict", {}) or {}

            # Prefer explicit hyperparameters if present.
            if "tcresnet_channels" in hparams:
                kwargs["tcresnet_channels"] = int(hparams["tcresnet_channels"])
            if "tcresnet_blocks" in hparams:
                kwargs["tcresnet_blocks"] = int(hparams["tcresnet_blocks"])
            if "tcresnet_dropout" in hparams:
                kwargs["tcresnet_dropout"] = float(hparams["tcresnet_dropout"])
            if "subsequence_aux" in hparams:
                kwargs["subsequence_aux"] = bool(hparams["subsequence_aux"])

            # Infer subsequence auxiliary head presence from checkpoint tensors.
            has_subseq_head = any(k.startswith("subseq_head.") for k in state_dict.keys())
            if has_subseq_head:
                kwargs["subsequence_aux"] = True

            if "tcresnet_channels" not in kwargs and "model.feature_extractor.stem.3.weight" in state_dict:
                # stem.3: Conv2d(stem_channels -> channels)
                kwargs["tcresnet_channels"] = int(state_dict["model.feature_extractor.stem.3.weight"].shape[0])
            if "tcresnet_blocks" not in kwargs:
                block_ids = set()
                prefix = "model.feature_extractor.blocks."
                for k in state_dict.keys():
                    if k.startswith(prefix):
                        parts = k.split(".")
                        if len(parts) > 3 and parts[3].isdigit():
                            block_ids.add(int(parts[3]))
                if len(block_ids) > 0:
                    kwargs["tcresnet_blocks"] = max(block_ids) + 1
        except Exception:
            # Keep safe defaults if inference fails.
            pass
        return kwargs

    @staticmethod
    def _load_kws_model_from_ckpt(ckpt_path: str) -> KWSModel:
        """
        Load the KWS model without Lightning's hyperparameter re-instantiation path.
        This avoids stale serialized hparams rebuilding an old ResNet-shaped model.
        """
        ckpt = torch.load(ckpt_path, map_location="cpu")
        kws_kwargs = CBWhisper._infer_kws_model_kwargs_from_ckpt(ckpt_path)
        model = KWSModel(**kws_kwargs)
        print("DEBUG kws_kwargs =", kws_kwargs)
        print("DEBUG kws backbone =", getattr(model.hparams, "backbone", None), getattr(model.model, "backbone", None))
        assert getattr(model.model, "backbone", None) in {"tcresnet", "tc-resnet"}, getattr(model.model, "backbone", None)
        model.on_load_checkpoint(ckpt)
        model.load_state_dict(ckpt["state_dict"], strict=True)
        return model

    def __init__(
        self,
        dataset: str,
        split: str,
        root: str,
        kw_type: str,
        encoder_ckpt: str,
        whisper_ckpt: str,
        kws_ckpt: str,
        language: str,
        force_decoder_prompt_ids: bool = False,
        prompt: bool = True,
        oracle: Union[bool, str] = 'kws',
        kws_features_size: Optional[Tuple[int, int]] = (150, 750),
        keyword_prompt_prepend: str = '(',
        keyword_prompt_append: str = ')',
        keyword_separator: str = ' ',
        keywords_per_group: int = 100,
        kws_positive_threshold: float = 0.35,
        kws_topk_per_group: int = 3,
        kws_max_prompt_keywords: int = 24,
        kws_infer_chunk_size: int = 16,
        keyword_perturb_prob: float = 0.0,
        keyword_perturb_rules: Optional[List[str]] = None,
        prompt_max_injected_keywords: int = 4,
        prompt_score_threshold: float = 0.55,
        prompt_relative_threshold: float = 0.8,
        enable_nested_keyword_promotion: bool = False,
        nested_keyword_promotion_score_ratio: float = 0.95,
        nested_keyword_promotion_min_long_chars: int = 3,
        nested_keyword_promotion_max_extra: int = 2,
        enable_phonetic_rescore: bool = False,
        rescore_nbest: int = 5,
        rescore_use_asr_score: bool = True,
        rescore_asr_weight: float = 1.0,
        rescore_keyword_weight: float = 2.0,
        rescore_phonetic_weight: float = 1.0,
        rescore_prefix_penalty_weight: float = 1.0,
        shortform_no_repeat_ngram_size: int = 3,
        rescore_max_keywords: int = 12,
        enable_phonetic_surface_repair: bool = False,
        surface_repair_score_threshold: float = 0.95,
        surface_repair_ambiguity_gap: float = 0.05,
        surface_repair_max_keywords: int = 8,
        surface_repair_max_edits: int = 2,
        surface_repair_min_keyword_chars: int = 2,
        surface_repair_max_keyword_chars: int = 6,
        enable_phonetic_consensus_repair: bool = False,
        consensus_repair_score_threshold: float = 0.90,
        consensus_repair_min_support: int = 2,
        enable_consensus_rerank: bool = False,
        consensus_rerank_weight: float = 0.35,
        consensus_rerank_min_support: int = 2,
        oracle_nbest_diagnostic: bool = False,
        oracle_nbest_detail_path: str = "logs/oracle_nbest_detail.csv",
        oracle_nbest_summary_path: str = "logs/oracle_nbest_summary.csv",
    ):
        super().__init__()

        # save hyperparameters
        self.save_hyperparameters()  

        # tokenizer/processor for prompting and decoding
        self.processor_whisper = WhisperProcessor.from_pretrained(
            self.hparams.whisper_ckpt,
            task = 'transcribe'
        )   

        # create an instance of PBAWhisper
        self.whisper = PBAWhisper.from_pretrained(self.hparams.whisper_ckpt)

        # create an instance of a KWSModel
        # some older checkpoints can miss hyperparameters such as
        # `adversarial_training`, so provide safe defaults here.
        self.kws = self._load_kws_model_from_ckpt(self.hparams.kws_ckpt)
        self.kws.eval()
        self.kws.requires_grad_(False)

        # create an instance of KeywordDatabase
        self.kw_database = DatabaseLite(
            dataset = self.hparams.dataset,
            split = self.hparams.split,
            root = self.hparams.root,
            kw_type = self.hparams.kw_type,
            keywords_per_group = self.hparams.keywords_per_group
        )

        # instantiate WhisperModel object and get encoder
        self.encoder = WhisperModel.from_pretrained(self.hparams.encoder_ckpt).encoder
        self.encoder.eval()
        self.encoder.requires_grad_(False)
        encoder_dim = int(getattr(getattr(self.encoder, "config", None), "d_model", 0))
        database_dim = self.kw_database.hidden_dim()
        if encoder_dim > 0 and database_dim is not None and int(database_dim) != encoder_dim:
            raise ValueError(
                "KWS encoder/database hidden-state dimension mismatch: "
                f"encoder_ckpt={self.hparams.encoder_ckpt} has d_model={encoder_dim}, "
                f"but keyword database under root={self.hparams.root}, split={self.hparams.split}, "
                f"kw_type={self.hparams.kw_type} has dim={database_dim}. "
                "Use the same Whisper encoder/profile to extract hotword keyword hidden states "
                "and to run online KWS encoding."
            )

        # check if oracle is valid
        if isinstance(self.hparams.oracle, bool):
            self.hparams.oracle = 'gold' if oracle else 'kws'
        assert self.hparams.oracle in ['gold', 'kws', 'random'], f'the provided oracle type is not supported, got f{oracle}'    

        # initialize oracle buffer
        self.oracle_buffer = []
        self._latest_keywords = []
        self._latest_keyword_scores = []
        self._latest_prompt_keywords = []
        self._latest_prompt_keyword_scores = []
        self._latest_prompt_token_lists = []
        self._latest_prompt_weights = []
        self._latest_full_prompt_ids = []
        self._keyword_spotting_cache = {}
        self._debug_log_path = os.getenv("CBW_DEBUG_LOG", "logs/runtime_probe.jsonl")
        if self._debug_log_path.lower() in {"", "none", "off", "0"}:
            self._debug_log_path = ""
        elif os.path.dirname(self._debug_log_path):
            os.makedirs(os.path.dirname(self._debug_log_path), exist_ok=True)
        # Clear debug log once at startup to avoid mixing runs.
        # Set CBW_DEBUG_CLEAR_ON_START=0 to keep appending across runs.
        self._debug_clear_on_start = os.getenv("CBW_DEBUG_CLEAR_ON_START", "1").lower() not in {"0", "false", "no", "off"}
        if self._debug_log_path and self._debug_clear_on_start:
            with open(self._debug_log_path, "w", encoding="utf-8"):
                pass
        self._debug_max_samples = int(os.getenv("CBW_DEBUG_MAX_SAMPLES", "100"))
        self._debug_seed = int(os.getenv("CBW_DEBUG_SEED", "12345"))
        self._debug_selected_indices = None
        self._metrics_output_path = os.getenv("CBW_METRICS_OUT", "logs/test_metrics.csv")
        self._oracle_nbest_diagnostic = bool(getattr(self.hparams, "oracle_nbest_diagnostic", False))
        self._oracle_nbest_detail_path = str(getattr(self.hparams, "oracle_nbest_detail_path", "logs/oracle_nbest_detail.csv"))
        self._oracle_nbest_summary_path = str(getattr(self.hparams, "oracle_nbest_summary_path", "logs/oracle_nbest_summary.csv"))
        self._post_test_progress_bar = None
        self._debug_counters = {
            "test_step": 0,
            "forward": 0,
            "keyword_spotting": 0,
            "test_output": 0,
        }
        self._latest_forward_candidates = []
        self._keyword_perturb_rules = list(keyword_perturb_rules) if keyword_perturb_rules is not None else []
        if len(self._keyword_perturb_rules) == 0:
            # Lightweight defaults for Mandarin homophone-like confusions.
            self._keyword_perturb_rules = [
                "\u59ca=>\u6b7b",  # ?=>?
                "\u73ae=>\u552f",  # ?=>?
                "\u752f=>\u5b81",  # ?=>?
                "\u7ecf=>\u91d1",  # ?=>?
                "\u62d3=>\u7279",  # ?=>?
            ]
        self._lazy_pinyin = None
        self._phonetic_backend = "char_fallback"
        self._phonetic_probe_examples = {}
        self._text_to_simplified = lambda s: str(s)
        self._text_normalizer_backend = "identity"
        self._init_phonetic_backend()
        self._init_text_normalizer()
        if self._debug_enabled():
            self._debug_log(
                "cb_debug_init",
                debug_log_path=self._debug_log_path,
                debug_max_samples=self._debug_max_samples,
                debug_seed=self._debug_seed,
                debug_clear_on_start=self._debug_clear_on_start,
                metrics_output_path=self._metrics_output_path,
                phonetic_backend=self._phonetic_backend,
                phonetic_probe_examples=self._phonetic_probe_examples,
                text_normalizer_backend=self._text_normalizer_backend,
            )

    def _maybe_perturb_keyword(self, kw: str) -> str:
        p = float(getattr(self.hparams, "keyword_perturb_prob", 0.0))
        if p <= 0.0:
            return kw
        if random.random() >= p:
            return kw
        out = str(kw)
        for rule in self._keyword_perturb_rules:
            if "=>" not in str(rule):
                continue
            src, dst = str(rule).split("=>", 1)
            src = src.strip()
            dst = dst.strip()
            if not src:
                continue
            out = out.replace(src, dst)
        return out

    @staticmethod
    def _edit_distance(seq1, seq2) -> int:
        m, n = len(seq1), len(seq2)
        dp = [list(range(n + 1))]
        for i in range(1, m + 1):
            dp.append([i] + [0] * n)
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                cost = 0 if seq1[i - 1] == seq2[j - 1] else 1
                dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
        return dp[m][n]

    def _init_phonetic_backend(self):
        try:
            from pypinyin import lazy_pinyin

            self._lazy_pinyin = lazy_pinyin
            self._phonetic_backend = "pypinyin"
            probe_examples = ["成飞", "程飞", "郫县", "辟县"]
            self._phonetic_probe_examples = {
                text: [u for u in lazy_pinyin(text) if str(u).strip() != ""]
                for text in probe_examples
            }
        except Exception as exc:
            self._lazy_pinyin = None
            self._phonetic_backend = "char_fallback"
            self._phonetic_probe_examples = {
                "error": str(exc),
                "成飞": list("成飞"),
                "程飞": list("程飞"),
                "郫县": list("郫县"),
                "辟县": list("辟县"),
            }

    def _init_text_normalizer(self):
        try:
            from opencc import OpenCC

            cc = OpenCC('t2s')
            self._text_to_simplified = lambda s: cc.convert(str(s))
            self._text_normalizer_backend = "opencc_t2s"
        except Exception:
            try:
                import zhconv

                self._text_to_simplified = lambda s: zhconv.convert(str(s), 'zh-cn')
                self._text_normalizer_backend = "zhconv_zh-cn"
            except Exception:
                self._text_to_simplified = lambda s: str(s)
                self._text_normalizer_backend = "identity"

    @staticmethod
    def _normalize_cn_numeric_chunk_text(s: str) -> str:
        cn_digit = {
            "零": 0, "〇": 0, "○": 0, "洞": 0,
            "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
            "五": 5, "六": 6, "七": 7, "八": 8, "九": 9,
        }
        cn_small_unit = {"十": 10, "百": 100, "千": 1000}
        cn_large_unit = {"万": 10000, "亿": 100000000}
        cn_unit = {**cn_small_unit, **cn_large_unit}

        def _cn_digit_seq_to_str(text: str):
            try:
                return "".join(str(cn_digit[ch]) for ch in text)
            except Exception:
                return None

        def _cn_with_units_to_int(text: str):
            if not text:
                return None
            if all(ch in cn_small_unit or ch in cn_large_unit for ch in text):
                return None

            total = 0
            section = 0
            number = 0
            seen_digit = False

            for ch in text:
                if ch in cn_digit:
                    number = cn_digit[ch]
                    seen_digit = True
                    continue
                if ch in cn_small_unit:
                    if number == 0:
                        number = 1
                    section += number * cn_small_unit[ch]
                    number = 0
                    continue
                if ch in cn_large_unit:
                    section += number
                    if section == 0:
                        section = 1
                    total += section * cn_large_unit[ch]
                    section = 0
                    number = 0
                    continue
                return None

            if not seen_digit:
                return None
            return total + section + number

        if not s:
            return s
        if any(ch in cn_unit for ch in s):
            val = _cn_with_units_to_int(s)
            return str(val) if val is not None else s
        val = _cn_digit_seq_to_str(s)
        return val if val is not None else s

    def _normalize_numbers_in_text(self, text: str) -> str:
        t = unicodedata.normalize("NFKC", str(text))
        t = re.sub(r"％", "%", t)
        quantity_units = (
            "美元|元|块|港元|欧元|日元|人民币|%|％|厘米|米|公斤|千克|岁|届|次|项|家|名|位|人|所|个|小时|分钟|秒"
        )

        def repl_cn_percent(m):
            num = self._normalize_cn_numeric_chunk_text(m.group(1))
            return f"{num}%"

        def repl_arabic_percent(m):
            try:
                num = str(int(m.group(1)))
            except Exception:
                num = m.group(1)
            return f"{num}%"

        t = re.sub(
            r"百分之([零〇○一二两三四五六七八九十百千万亿]+)",
            repl_cn_percent,
            t,
        )
        t = re.sub(
            r"百分之(\d+)",
            repl_arabic_percent,
            t,
        )
        t = re.sub(
            r"([零〇○一二两三四五六七八九]{2,4})(?=年)",
            lambda m: self._normalize_cn_numeric_chunk_text(m.group(1)),
            t,
        )
        t = re.sub(
            r"([零〇○一二两三四五六七八九十]{1,3})(?=[月日号岁])",
            lambda m: self._normalize_cn_numeric_chunk_text(m.group(1)),
            t,
        )
        t = re.sub(
            rf"([零〇○一二两三四五六七八九十百千万亿]+)(?=({quantity_units}))",
            lambda m: self._normalize_cn_numeric_chunk_text(m.group(1)),
            t,
        )

        arabic_large_unit_scale = {"万": 10000, "亿": 100000000}

        def repl_arabic_large_unit(m):
            try:
                base = int(m.group(1))
            except Exception:
                return m.group(0)
            unit = m.group(2)
            return str(base * arabic_large_unit_scale[unit])

        t = re.sub(
            r"(\d+)([万亿])(?=(美元|元|块|港元|欧元|日元|人民币))",
            repl_arabic_large_unit,
            t,
        )
        t = re.sub(r"\d+", lambda m: str(int(m.group(0))), t)
        return t

    def _normalize_surface_text(self, text: str) -> str:
        text = unicodedata.normalize("NFKC", str(text))
        text = self._text_to_simplified(text)
        text = self._normalize_numbers_in_text(text)
        # Remove punctuation so evaluation and n-best dedup focus on lexical content.
        text = "".join(ch for ch in text if not unicodedata.category(ch).startswith("P"))
        text = re.sub(r"[·•‧・]", "", text)
        # Remove spacing artifacts before downstream use.
        text = re.sub(r"\s+", "", text)
        return text.strip()

    def _normalize_text_for_rescore(self, text: str) -> str:
        return self._normalize_surface_text(text)

    def _normalize_text_for_logging(self, text: str) -> str:
        return self._normalize_surface_text(text)

    def _to_phonetic_units(self, text: str) -> List[str]:
        text = str(text)
        if self._lazy_pinyin is not None:
            units = [u for u in self._lazy_pinyin(text) if u.strip() != ""]
            if len(units) > 0:
                return units
        return list(text)

    def _phonetic_surface_repair(
        self,
        pred_text: str,
        keywords: List[str],
        kw_scores: dict,
        candidate_texts: Optional[List[str]] = None,
    ) -> Tuple[str, dict]:
        if not bool(getattr(self.hparams, "enable_phonetic_surface_repair", False)):
            return pred_text, {"changed": False, "repairs": []}
        if not isinstance(pred_text, str) or pred_text.strip() == "":
            return pred_text, {"changed": False, "repairs": []}
        if not isinstance(keywords, list) or len(keywords) == 0:
            return pred_text, {"changed": False, "repairs": []}

        score_threshold = float(getattr(self.hparams, "surface_repair_score_threshold", 0.95))
        consensus_enabled = bool(getattr(self.hparams, "enable_phonetic_consensus_repair", False))
        consensus_score_threshold = float(getattr(self.hparams, "consensus_repair_score_threshold", 0.90))
        consensus_min_support = max(1, int(getattr(self.hparams, "consensus_repair_min_support", 2)))
        ambiguity_gap = float(getattr(self.hparams, "surface_repair_ambiguity_gap", 0.05))
        max_keywords = max(1, int(getattr(self.hparams, "surface_repair_max_keywords", 8)))
        max_edits = max(0, int(getattr(self.hparams, "surface_repair_max_edits", 2)))
        min_chars = max(1, int(getattr(self.hparams, "surface_repair_min_keyword_chars", 2)))
        max_chars = max(min_chars, int(getattr(self.hparams, "surface_repair_max_keyword_chars", 6)))
        if max_edits <= 0:
            return pred_text, {"changed": False, "repairs": []}

        text = unicodedata.normalize("NFKC", str(pred_text))
        chars = list(text)
        normalized_text = self._normalize_text_for_rescore(text)
        ranked_keywords = self._sort_keywords_by_score(keywords, kw_scores)[:max_keywords]
        repairs = []
        normalized_candidates = []
        if isinstance(candidate_texts, list):
            seen_candidates = set()
            for candidate_text in candidate_texts:
                norm_candidate = self._normalize_text_for_rescore(str(candidate_text))
                if norm_candidate == "" or norm_candidate in seen_candidates:
                    continue
                seen_candidates.add(norm_candidate)
                normalized_candidates.append(norm_candidate)

        def _same_phonetic_competitor_exists(kw: str, kw_units: List[str], score: float) -> bool:
            for other in ranked_keywords:
                other = str(other)
                if other == kw:
                    continue
                other_score = float(kw_scores.get(other, 0.0))
                if other_score < score - ambiguity_gap:
                    continue
                other_norm = self._normalize_text_for_rescore(other)
                if other_norm == "":
                    continue
                if self._to_phonetic_units(other_norm) == kw_units:
                    return True
            return False

        def _candidate_support_count(norm_kw: str, kw_units: List[str]) -> int:
            support = 0
            win_len = len(kw_units)
            for norm_candidate in normalized_candidates:
                if norm_kw in norm_candidate:
                    support += 1
                    continue
                candidate_chars = list(norm_candidate)
                if len(candidate_chars) < win_len:
                    continue
                matched = False
                for start in range(0, len(candidate_chars) - win_len + 1):
                    window = "".join(candidate_chars[start:start + win_len])
                    if not any(self._looks_like_cjk(ch) for ch in window):
                        continue
                    if self._to_phonetic_units(window) == kw_units:
                        matched = True
                        break
                support += int(matched)
            return int(support)

        for kw in ranked_keywords:
            raw_kw = str(kw)
            score = float(kw_scores.get(raw_kw, 0.0))
            norm_kw = self._normalize_text_for_rescore(raw_kw)
            if norm_kw == "" or norm_kw in normalized_text:
                continue
            kw_chars = list(norm_kw)
            if len(kw_chars) < min_chars or len(kw_chars) > max_chars:
                continue
            if not any(self._looks_like_cjk(ch) for ch in kw_chars):
                continue
            kw_units = self._to_phonetic_units(norm_kw)
            if len(kw_units) != len(kw_chars):
                continue
            consensus_support = _candidate_support_count(norm_kw, kw_units) if consensus_enabled else 0
            has_consensus = bool(
                consensus_enabled
                and score >= consensus_score_threshold
                and consensus_support >= consensus_min_support
            )
            if score < score_threshold and not has_consensus:
                continue
            if _same_phonetic_competitor_exists(raw_kw, kw_units, score):
                continue

            windows = []
            win_len = len(kw_chars)
            for start in range(0, len(chars) - win_len + 1):
                end = start + win_len
                window_text = "".join(chars[start:end])
                if not any(self._looks_like_cjk(ch) for ch in window_text):
                    continue
                norm_window = self._normalize_text_for_rescore(window_text)
                if len(norm_window) != win_len or norm_window == norm_kw:
                    continue
                window_units = self._to_phonetic_units(norm_window)
                if window_units != kw_units:
                    continue
                changed_chars = sum(1 for a, b in zip(norm_window, norm_kw) if a != b)
                if changed_chars <= 0:
                    continue
                windows.append((start, end, window_text, norm_window, changed_chars))

            # Only repair unique, same-length, exact-pinyin substitutions. This
            # keeps the operation local and avoids broad hotword hallucination.
            if len(windows) != 1:
                continue
            start, end, window_text, norm_window, changed_chars = windows[0]
            chars[start:end] = kw_chars
            repairs.append(
                {
                    "keyword": norm_kw,
                    "score": float(score),
                    "start": int(start),
                    "end": int(end),
                    "from": norm_window,
                    "to": norm_kw,
                    "changed_chars": int(changed_chars),
                    "consensus_support": int(consensus_support),
                    "used_consensus": bool(has_consensus and score < score_threshold),
                }
            )
            normalized_text = self._normalize_text_for_rescore("".join(chars))
            if len(repairs) >= max_edits:
                break

        repaired_text = "".join(chars)
        return repaired_text, {
            "changed": bool(repaired_text != text),
            "repairs": repairs,
        }

    def _best_phonetic_window_match(self, pred_units: List[str], kw_units: List[str]) -> Dict[str, Any]:
        if len(pred_units) == 0 or len(kw_units) == 0:
            return {
                "sim": 0.0,
                "runner_up_sim": 0.0,
                "start": -1,
                "end": -1,
            }

        candidate_lengths = sorted({max(1, len(kw_units) - 1), len(kw_units), len(kw_units) + 1})
        window_scores = []
        for win_len in candidate_lengths:
            if len(pred_units) < win_len:
                dist = self._edit_distance(pred_units, kw_units)
                sim = 1.0 - (dist / float(max(len(pred_units), len(kw_units), 1)))
                window_scores.append((float(sim), 0, len(pred_units)))
                continue
            for start in range(0, len(pred_units) - win_len + 1):
                end = start + win_len
                dist = self._edit_distance(pred_units[start:end], kw_units)
                sim = 1.0 - (dist / float(max(win_len, len(kw_units), 1)))
                window_scores.append((float(sim), start, end))

        if len(window_scores) == 0:
            return {
                "sim": 0.0,
                "runner_up_sim": 0.0,
                "start": -1,
                "end": -1,
            }
        window_scores.sort(key=lambda x: x[0], reverse=True)
        best_sim, best_start, best_end = window_scores[0]
        runner_up_sim = float(window_scores[1][0]) if len(window_scores) > 1 else 0.0
        return {
            "sim": max(0.0, min(1.0, float(best_sim))),
            "runner_up_sim": max(0.0, min(1.0, float(runner_up_sim))),
            "start": int(best_start),
            "end": int(best_end),
        }

    def _phonetic_keyword_stats(self, pred_text: str, keywords: List[str], kw_scores: dict) -> dict:
        pred_text = str(pred_text)
        for kw in keywords:
            kw = str(kw)
            if kw and kw in pred_text:
                weight = float(kw_scores.get(kw, 1.0))
                weighted = float(torch.log1p(torch.tensor(weight, dtype=torch.float32)).item())
                return {
                    "score": float(weighted),
                    "support": float(weighted),
                    "top_sim": 1.0,
                    "runner_up_sim": 0.0,
                    "num_survivors": 1,
                    "best_keyword": kw,
                    "best_window_start": int(pred_text.find(kw)),
                    "best_window_end": int(pred_text.find(kw) + len(kw)),
                    "backend": self._phonetic_backend,
                    "survivor_preview": [
                        {
                            "keyword": kw,
                            "sim": 1.0,
                            "weighted_sim": float(weighted),
                            "start": int(pred_text.find(kw)),
                            "end": int(pred_text.find(kw) + len(kw)),
                            "runner_up_sim": 0.0,
                        }
                    ],
                }
        pred_units = self._to_phonetic_units(pred_text)
        if len(pred_units) == 0:
            return {
                "score": 0.0,
                "support": 0.0,
                "top_sim": 0.0,
                "runner_up_sim": 0.0,
                "num_survivors": 0,
                "best_keyword": "",
                "best_window_start": -1,
                "best_window_end": -1,
                "backend": self._phonetic_backend,
                "survivor_preview": [],
            }
        survivors = []
        sim_threshold = 0.72
        for kw in keywords:
            kw_units = self._to_phonetic_units(kw)
            if len(kw_units) == 0:
                continue
            match = self._best_phonetic_window_match(pred_units, kw_units)
            sim = float(match["sim"])
            if sim < sim_threshold:
                continue
            weight = float(kw_scores.get(kw, 1.0))
            survivors.append(
                (
                    sim,
                    sim * weight,
                    kw,
                    int(match["start"]),
                    int(match["end"]),
                    float(match["runner_up_sim"]),
                )
            )
        if len(survivors) == 0:
            return {
                "score": 0.0,
                "support": 0.0,
                "top_sim": 0.0,
                "runner_up_sim": 0.0,
                "num_survivors": 0,
                "best_keyword": "",
                "best_window_start": -1,
                "best_window_end": -1,
                "backend": self._phonetic_backend,
                "survivor_preview": [],
            }
        # Aggregate only sufficiently similar phonetic matches. This keeps the
        # number of active hotwords adaptive instead of forcing a fixed top-k.
        survivors.sort(key=lambda x: x[0], reverse=True)
        score_sum = float(sum(weighted for _, weighted, _, _, _, _ in survivors))
        sim_sum = float(sum(sim for sim, _, _, _, _, _ in survivors))
        support = score_sum / max(sim_sum, 1e-6)
        top_sim = float(survivors[0][0])
        global_runner_up_sim = float(survivors[1][0]) if len(survivors) > 1 else 0.0
        local_runner_up_sim = float(survivors[0][5])
        runner_up_sim = max(global_runner_up_sim, local_runner_up_sim)
        # Mildly prefer candidates whose best hotword match is clearly more
        # convincing than the next-best phonetic alternative.
        margin = max(0.0, top_sim - runner_up_sim)
        support = float(torch.log1p(torch.tensor(support, dtype=torch.float32)).item())
        score = support * (1.0 + 0.5 * margin)
        survivor_preview = [
            {
                "keyword": str(kw),
                "sim": float(sim),
                "weighted_sim": float(weighted),
                "start": int(start),
                "end": int(end),
                "runner_up_sim": float(local_runner_up),
            }
            for sim, weighted, kw, start, end, local_runner_up in survivors[:5]
        ]
        return {
            "score": float(score),
            "support": float(support),
            "top_sim": top_sim,
            "runner_up_sim": runner_up_sim,
            "num_survivors": len(survivors),
            "best_keyword": str(survivors[0][2]),
            "best_window_start": int(survivors[0][3]),
            "best_window_end": int(survivors[0][4]),
            "backend": self._phonetic_backend,
            "survivor_preview": survivor_preview,
        }

    def _phonetic_keyword_score(self, pred_text: str, keywords: List[str], kw_scores: dict) -> float:
        return float(self._phonetic_keyword_stats(pred_text, keywords, kw_scores)["score"])

    def _keyword_exact_stats(self, pred_text: str, keywords: List[str], kw_scores: dict) -> dict:
        text = str(pred_text)
        matched_keywords = []
        total_keywords = 0
        for kw in keywords:
            kw = str(kw)
            if not kw:
                continue
            total_keywords += 1
            score = float(kw_scores.get(kw, 1.0))
            if kw in text:
                matched_keywords.append(str(kw))
        matched_count = len(matched_keywords)
        coverage_ratio = float(matched_count / max(total_keywords, 1)) if total_keywords > 0 else 0.0
        total_weight_sum = float(sum(float(kw_scores.get(str(kw), 1.0)) for kw in keywords if str(kw)))
        matched_weight_sum = float(sum(float(kw_scores.get(kw, 1.0)) for kw in matched_keywords))
        weighted_coverage = float(matched_weight_sum / max(total_weight_sum, 1e-9)) if total_keywords > 0 else 0.0
        return {
            "score": float(matched_count),
            "matched_count": int(matched_count),
            "total_keywords": int(total_keywords),
            "coverage_ratio": float(coverage_ratio),
            "weighted_coverage": float(weighted_coverage),
            "matched_keywords": matched_keywords[:8],
        }

    def _char_phonetic_match_ratio(self, pred_chars: List[str], kw_chars: List[str]) -> float:
        denom = max(len(pred_chars), len(kw_chars), 1)
        hits = 0.0
        for pred_ch, kw_ch in zip(pred_chars, kw_chars):
            if pred_ch == kw_ch:
                hits += 1.0
                continue
            pred_units = self._to_phonetic_units(pred_ch)
            kw_units = self._to_phonetic_units(kw_ch)
            if len(pred_units) > 0 and pred_units == kw_units:
                hits += 0.6
        return float(hits / float(denom))

    def _sort_keywords_by_score(self, keywords: List[str], kw_scores: dict) -> List[str]:
        uniq = list(dict.fromkeys([str(k) for k in keywords if str(k).strip() != ""]))
        uniq.sort(key=lambda k: float(kw_scores.get(k, 0.0)), reverse=True)
        return uniq

    def _promote_nested_phonetic_keywords(
        self,
        ranked_keywords: List[str],
        selected_keywords: List[str],
        kw_scores: dict,
        max_k: int,
    ) -> List[str]:
        if not bool(getattr(self.hparams, "enable_nested_keyword_promotion", False)):
            return selected_keywords[:max_k]
        if len(ranked_keywords) == 0 or len(selected_keywords) == 0 or max_k <= 0:
            return selected_keywords[:max_k]

        ratio = max(0.0, float(getattr(self.hparams, "nested_keyword_promotion_score_ratio", 0.95)))
        min_long_chars = max(2, int(getattr(self.hparams, "nested_keyword_promotion_min_long_chars", 3)))
        max_extra = max(0, int(getattr(self.hparams, "nested_keyword_promotion_max_extra", 2)))
        selected = list(dict.fromkeys(selected_keywords))[:max_k]
        selected_set = set(selected)
        normalized = {kw: self._normalize_text_for_rescore(kw) for kw in ranked_keywords}
        units = {kw: self._to_phonetic_units(normalized[kw]) for kw in ranked_keywords}

        promotions = []
        for short_kw in selected:
            short_norm = normalized.get(short_kw, self._normalize_text_for_rescore(short_kw))
            short_units = units.get(short_kw, self._to_phonetic_units(short_norm))
            if short_norm == "" or len(short_units) == 0:
                continue
            short_score = float(kw_scores.get(short_kw, 0.0))
            for long_kw in ranked_keywords:
                if long_kw in selected_set or long_kw == short_kw:
                    continue
                long_norm = normalized.get(long_kw, "")
                long_units = units.get(long_kw, [])
                if len(long_norm) < min_long_chars or len(long_units) <= len(short_units):
                    continue
                long_score = float(kw_scores.get(long_kw, 0.0))
                if long_score + 1e-9 < short_score * ratio:
                    continue
                text_nested = short_norm in long_norm
                phon_prefix = long_units[: len(short_units)] == short_units
                if text_nested or phon_prefix:
                    promotions.append((long_score, long_kw))
        promotions = [kw for _, kw in sorted(promotions, key=lambda item: item[0], reverse=True)]
        promotions = list(dict.fromkeys(promotions))[:max_extra]

        for kw in promotions:
            if kw in selected_set:
                continue
            if len(selected) < max_k:
                selected.append(kw)
                selected_set.add(kw)
                continue
            replace_idx = min(range(len(selected)), key=lambda i: float(kw_scores.get(selected[i], 0.0)))
            selected_set.discard(selected[replace_idx])
            selected[replace_idx] = kw
            selected_set.add(kw)
        selected.sort(key=lambda k: float(kw_scores.get(k, 0.0)), reverse=True)
        return selected[:max_k]

    def _select_prompt_keywords(self, keywords: List[str], kw_scores: dict) -> List[str]:
        ranked = self._sort_keywords_by_score(keywords, kw_scores)
        if len(ranked) == 0:
            return []
        max_prompt_k = max(1, int(getattr(self.hparams, "prompt_max_injected_keywords", 4)))
        abs_threshold = float(getattr(self.hparams, "prompt_score_threshold", 0.55))
        rel_threshold = float(getattr(self.hparams, "prompt_relative_threshold", 0.8))
        top_score = float(kw_scores.get(ranked[0], 0.0))
        keep_threshold = max(abs_threshold, top_score * rel_threshold)
        selected = [
            kw for kw in ranked
            if float(kw_scores.get(kw, 0.0)) >= keep_threshold
        ]
        if len(selected) == 0:
            selected = [ranked[0]]
        return self._promote_nested_phonetic_keywords(ranked, selected, kw_scores, max_prompt_k)

    @staticmethod
    def _normalize_prompt_weights(keywords: List[str], kw_scores: dict) -> List[float]:
        if len(keywords) == 0:
            return []
        raw = [max(0.0, float(kw_scores.get(kw, 0.0))) for kw in keywords]
        total = float(sum(raw))
        if total <= 0.0:
            return [1.0 / len(keywords) for _ in keywords]
        return [v / total for v in raw]

    @staticmethod
    def _module_param_dtype(module: torch.nn.Module) -> Optional[torch.dtype]:
        try:
            return next(module.parameters()).dtype
        except Exception:
            return None

    def _decoder_vocab_size(self) -> int:
        try:
            return int(self.whisper.config.vocab_size)
        except Exception:
            try:
                return int(self.processor_whisper.tokenizer.vocab_size)
            except Exception:
                return 0

    def _special_token_id_set(self) -> set:
        try:
            return set(int(tok) for tok in self.processor_whisper.tokenizer.all_special_ids)
        except Exception:
            return set()

    def _cast_features_for_module(self, input_features: torch.Tensor, module: torch.nn.Module) -> torch.Tensor:
        if not isinstance(input_features, torch.Tensor):
            return input_features
        target_dtype = self._module_param_dtype(module)
        if target_dtype is None or input_features.dtype == target_dtype:
            return input_features
        return input_features.to(dtype=target_dtype)

    def _sanitize_generated_sequence(self, seq: torch.Tensor) -> torch.Tensor:
        if not isinstance(seq, torch.Tensor):
            return seq
        pad_id = int(self.processor_whisper.tokenizer.pad_token_id or 0)
        vocab_size = self._decoder_vocab_size()
        seq = seq[seq != pad_id]
        if vocab_size <= 0 or seq.numel() == 0:
            return seq
        valid_mask = (seq >= 0) & (seq < vocab_size)
        return seq[valid_mask]

    def _select_rescore_keywords(self, keywords: List[str], kw_scores: dict) -> List[str]:
        max_k = max(1, int(getattr(self.hparams, "rescore_max_keywords", 12)))
        uniq = list(dict.fromkeys([str(k) for k in keywords if str(k).strip() != ""]))
        uniq.sort(key=lambda k: float(kw_scores.get(k, 0.0)), reverse=True)
        return self._promote_nested_phonetic_keywords(uniq, uniq[:max_k], kw_scores, max_k)

    def _normalize_rescore_keyword_inputs(self, keywords: List[str], kw_scores: dict) -> Tuple[List[str], dict]:
        norm_scores = {}
        for kw in keywords:
            raw_kw = str(kw)
            norm_kw = self._normalize_text_for_rescore(raw_kw)
            if norm_kw == "":
                continue
            score = float(kw_scores.get(raw_kw, kw_scores.get(norm_kw, 1.0)))
            if norm_kw not in norm_scores or score > float(norm_scores[norm_kw]):
                norm_scores[norm_kw] = score
        return list(norm_scores.keys()), norm_scores

    @staticmethod
    def _scale_scores_within_candidates(values: List[float], neutral_if_flat: bool = True) -> List[float]:
        if len(values) == 0:
            return []
        vals = [float(v) for v in values]
        min_v = min(vals)
        max_v = max(vals)
        spread = max_v - min_v
        if spread <= 1e-9:
            fill = 0.0 if neutral_if_flat else (1.0 if max_v > 0.0 else 0.0)
            return [float(fill) for _ in vals]
        return [float((v - min_v) / spread) for v in vals]

    @staticmethod
    def _prompt_weight_concentration(weights: List[float]) -> float:
        if not isinstance(weights, list) or len(weights) == 0:
            return 0.0
        usable = [max(0.0, float(w)) for w in weights]
        total = float(sum(usable))
        if total <= 0.0:
            return 0.0
        return float(max(usable) / total)

    @staticmethod
    def _safe_sequence_length(value) -> int:
        if value is None:
            return 0
        if isinstance(value, torch.Tensor):
            try:
                return int(value.numel())
            except Exception:
                return 0
        try:
            return int(len(value))
        except Exception:
            return 0

    @staticmethod
    def _looks_like_cjk(ch: str) -> bool:
        if not ch:
            return False
        code = ord(ch)
        return (
            0x4E00 <= code <= 0x9FFF
            or 0x3400 <= code <= 0x4DBF
            or 0xF900 <= code <= 0xFAFF
        )

    def _prefix_pollution_stats(self, text: str) -> dict:
        text = str(text or "")
        stripped = text.lstrip()
        if stripped == "":
            return {
                "penalty": 0.0,
                "ascii_prefix": "",
                "ascii_prefix_len": 0,
                "has_cjk_after_ascii": False,
                "suspicious_single_letter_prefix": False,
            }

        ascii_match = re.match(r"^[A-Za-z]+", stripped)
        ascii_prefix = ascii_match.group(0) if ascii_match is not None else ""
        ascii_prefix_len = len(ascii_prefix)
        next_char = stripped[ascii_prefix_len:ascii_prefix_len + 1]
        has_cjk_after_ascii = bool(ascii_prefix_len > 0 and self._looks_like_cjk(next_char))
        suspicious_single_letter_prefix = bool(
            len(stripped) >= 2
            and stripped[0] in "rjvVonJPRV"
            and self._looks_like_cjk(stripped[1])
        )

        penalty = 0.0
        if has_cjk_after_ascii:
            penalty += min(1.5, 0.9 + 0.12 * max(0, ascii_prefix_len - 1))
        elif suspicious_single_letter_prefix:
            penalty += 0.8

        return {
            "penalty": float(penalty),
            "ascii_prefix": ascii_prefix,
            "ascii_prefix_len": int(ascii_prefix_len),
            "has_cjk_after_ascii": bool(has_cjk_after_ascii),
            "suspicious_single_letter_prefix": bool(suspicious_single_letter_prefix),
        }

    def _sequence_logprob(
        self, input_features: torch.Tensor, attention_mask: Optional[torch.Tensor], seq: torch.Tensor
    ) -> float:
        seq = self._sanitize_generated_sequence(seq)
        if seq.numel() < 2:
            return -1e9
        decoder_input_ids = seq[:-1].unsqueeze(0)
        target_ids = seq[1:].unsqueeze(0)
        input_features = self._cast_features_for_module(input_features, self.whisper)
        with torch.no_grad():
            out = self.whisper(
                input_features=input_features,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                use_cache=False,
                return_dict=True,
            )
            logits = out.logits
            logp = torch.log_softmax(logits, dim=-1)
            vocab_size = logp.size(-1)
            valid_targets = (target_ids >= 0) & (target_ids < vocab_size)
            special_token_ids = self._special_token_id_set()
            if len(special_token_ids) > 0:
                special_mask = torch.zeros_like(valid_targets)
                for tok_id in special_token_ids:
                    special_mask |= target_ids == int(tok_id)
                text_target_mask = valid_targets & ~special_mask
                if bool(torch.any(text_target_mask).item()):
                    valid_targets = text_target_mask
            safe_target_ids = target_ids.masked_fill(~valid_targets, 0)
            tok = logp.gather(-1, safe_target_ids.unsqueeze(-1)).squeeze(-1)
            tok = tok.masked_fill(~valid_targets, 0.0)
            val = float(tok.sum().item() / max(int(valid_targets.sum().item()), 1))
        return val

    def _sequence_logprobs_batch(
        self,
        input_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        seqs: torch.Tensor,
    ) -> List[float]:
        if not isinstance(seqs, torch.Tensor) or seqs.dim() != 2 or seqs.size(0) == 0:
            return []
        pad_id = int(self.processor_whisper.tokenizer.pad_token_id or 0)
        valid_rows = []
        lengths = []
        for row in seqs:
            row = self._sanitize_generated_sequence(row)
            if row.numel() < 2:
                valid_rows.append(None)
                lengths.append(0)
            else:
                valid_rows.append(row)
                lengths.append(int(row.numel() - 1))
        max_len = max(lengths) if len(lengths) > 0 else 0
        if max_len <= 0:
            return [-1e9 for _ in range(seqs.size(0))]

        bsz = seqs.size(0)
        decoder_input_ids = torch.full(
            (bsz, max_len),
            pad_id,
            device=seqs.device,
            dtype=seqs.dtype,
        )
        target_ids = torch.full(
            (bsz, max_len),
            pad_id,
            device=seqs.device,
            dtype=seqs.dtype,
        )
        target_mask = torch.zeros((bsz, max_len), device=seqs.device, dtype=torch.bool)
        special_token_ids = self._special_token_id_set()

        for idx, row in enumerate(valid_rows):
            if row is None:
                continue
            cur_len = row.numel() - 1
            decoder_input_ids[idx, :cur_len] = row[:-1]
            target_ids[idx, :cur_len] = row[1:]
            target_mask[idx, :cur_len] = True

        expanded_input_features = input_features.expand(bsz, -1, -1)
        expanded_attention_mask = None
        if isinstance(attention_mask, torch.Tensor):
            expanded_attention_mask = attention_mask.expand(bsz, -1)
        expanded_input_features = self._cast_features_for_module(expanded_input_features, self.whisper)

        with torch.no_grad():
            out = self.whisper(
                input_features=expanded_input_features,
                attention_mask=expanded_attention_mask,
                decoder_input_ids=decoder_input_ids,
                use_cache=False,
                return_dict=True,
            )
            logits = out.logits
            logp = torch.log_softmax(logits, dim=-1)
            vocab_size = logp.size(-1)
            valid_targets = (target_ids >= 0) & (target_ids < vocab_size)
            if len(special_token_ids) > 0:
                special_mask = torch.zeros_like(valid_targets)
                for tok_id in special_token_ids:
                    special_mask |= target_ids == int(tok_id)
                text_target_mask = target_mask & valid_targets & ~special_mask
                rows_with_text = text_target_mask.any(dim=1)
                valid_targets = torch.where(rows_with_text.unsqueeze(1), text_target_mask, valid_targets)
            safe_target_ids = target_ids.masked_fill(~valid_targets, 0)
            tok = logp.gather(-1, safe_target_ids.unsqueeze(-1)).squeeze(-1)
            tok = tok.masked_fill(~target_mask, 0.0)
            tok = tok.masked_fill(~valid_targets, 0.0)
            denom = (target_mask & valid_targets).sum(dim=1).clamp_min(1)
            vals = tok.sum(dim=1) / denom

        scores = []
        for idx in range(bsz):
            if lengths[idx] <= 0:
                scores.append(-1e9)
            else:
                scores.append(float(vals[idx].item()))
        return scores

    def _build_shortform_candidate_stats(
        self,
        candidates: List[str],
        pred_sequences: torch.Tensor,
        input_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        keywords: List[str],
        keyword_scores: dict,
    ) -> List[dict]:
        asr_scores = self._sequence_logprobs_batch(input_features, attention_mask, pred_sequences)
        rescore_keywords, rescore_keyword_scores = self._normalize_rescore_keyword_inputs(keywords, keyword_scores)
        candidate_stats = []
        for idx, candidate in enumerate(candidates):
            asr_score = float(asr_scores[idx]) if idx < len(asr_scores) else self._sequence_logprob(
                input_features,
                attention_mask,
                pred_sequences[idx],
            )
            rescore_candidate = self._normalize_text_for_rescore(candidate)
            exact_stats = self._keyword_exact_stats(rescore_candidate, rescore_keywords, rescore_keyword_scores)
            phon_stats = self._phonetic_keyword_stats(rescore_candidate, rescore_keywords, rescore_keyword_scores)
            candidate_stats.append(
                {
                    "candidate": str(candidate),
                    "rescore_candidate": str(rescore_candidate),
                    "asr_score": float(asr_score),
                    "exact_score": float(exact_stats["score"]),
                    "exact_weighted_score": float(exact_stats.get("weighted_coverage", 0.0)),
                    "exact_stats": exact_stats,
                    "phonetic_score": float(phon_stats["score"]),
                    "phonetic_stats": phon_stats,
                }
            )
        return candidate_stats

    def _score_shortform_candidates(
        self,
        candidate_stats: List[dict],
        keyword_weight: float,
    ) -> Tuple[List[dict], int, float]:
        if len(candidate_stats) == 0:
            return [], -1, 0.0

        baseline_idx = max(
            range(len(candidate_stats)),
            key=lambda i: candidate_stats[i]["asr_score"],
        )
        baseline_phonetic_score = float(candidate_stats[baseline_idx]["phonetic_score"])
        phonetic_weight = float(getattr(self.hparams, "rescore_phonetic_weight", 1.0))
        use_asr_score = bool(getattr(self.hparams, "rescore_use_asr_score", True))
        asr_weight = float(getattr(self.hparams, "rescore_asr_weight", 1.0)) if use_asr_score else 0.0
        prefix_penalty_weight = float(getattr(self.hparams, "rescore_prefix_penalty_weight", 1.0))
        consensus_enabled = bool(getattr(self.hparams, "enable_consensus_rerank", False))
        consensus_weight = float(getattr(self.hparams, "consensus_rerank_weight", 0.35)) if consensus_enabled else 0.0
        consensus_min_support = max(1, int(getattr(self.hparams, "consensus_rerank_min_support", 2)))
        exact_raw_scores = [
            float((item.get("exact_stats", {}) or {}).get("weighted_coverage", 0.0))
            for item in candidate_stats
        ]
        max_exact_score = max(exact_raw_scores) if len(exact_raw_scores) > 0 else 0.0
        has_exact_evidence = max_exact_score > 0.0
        phonetic_raw_scores = [
            float(item["phonetic_score"]) if (has_exact_evidence and float(item.get("exact_score", 0.0)) > 0.0) else 0.0
            for item in candidate_stats
        ]
        asr_scaled_scores = self._scale_scores_within_candidates(
            [float(item["asr_score"]) for item in candidate_stats],
            neutral_if_flat=True,
        )
        exact_scaled_scores = self._scale_scores_within_candidates(exact_raw_scores, neutral_if_flat=False)
        phonetic_scaled_scores = self._scale_scores_within_candidates(phonetic_raw_scores, neutral_if_flat=False)

        exact_keyword_support = {}
        for stats in candidate_stats:
            exact_stats = stats.get("exact_stats", {}) or {}
            matched_keywords = list(exact_stats.get("matched_keywords", []) or [])
            for kw in set(str(k) for k in matched_keywords if str(k).strip() != ""):
                exact_keyword_support[kw] = exact_keyword_support.get(kw, 0) + 1

        def _candidate_consensus_score(stats: dict) -> Tuple[float, int, List[str]]:
            if not consensus_enabled or len(candidate_stats) <= 1:
                return 0.0, 0, []
            exact_stats = stats.get("exact_stats", {}) or {}
            matched_keywords = [str(k) for k in list(exact_stats.get("matched_keywords", []) or []) if str(k).strip() != ""]
            supported = [
                kw for kw in matched_keywords
                if int(exact_keyword_support.get(kw, 0)) >= consensus_min_support
            ]
            if len(supported) == 0:
                return 0.0, 0, []
            max_support = max(int(exact_keyword_support.get(kw, 0)) for kw in supported)
            denom = max(len(candidate_stats) - 1, 1)
            score = float((max_support - 1) / denom)
            return score, max_support, supported[:5]

        scored_candidates = []
        for stats_idx, stats in enumerate(candidate_stats):
            exact_score = float(stats["exact_score"])
            exact_weighted_score = float(exact_raw_scores[stats_idx])
            phonetic_score = float(phonetic_raw_scores[stats_idx])
            asr_score_scaled = float(asr_scaled_scores[stats_idx])
            exact_score_scaled = float(exact_scaled_scores[stats_idx])
            phonetic_score_scaled = float(phonetic_scaled_scores[stats_idx])
            prefix_stats = self._prefix_pollution_stats(str(stats.get("candidate", "")))
            prefix_penalty = float(prefix_stats.get("penalty", 0.0))
            prefix_penalty_score = prefix_penalty_weight * prefix_penalty
            consensus_score, consensus_support, consensus_keywords = _candidate_consensus_score(stats)
            consensus_score_used = consensus_weight * consensus_score
            total_score = (
                asr_weight * asr_score_scaled
                + float(keyword_weight) * exact_score_scaled
                + float(phonetic_weight) * phonetic_score_scaled
                + float(consensus_score_used)
                - float(prefix_penalty_score)
            )
            scored_candidates.append(
                {
                    **stats,
                    "asr_score_scaled": float(asr_score_scaled),
                    "exact_weighted_score": float(exact_weighted_score),
                    "exact_score_scaled": float(exact_score_scaled),
                    "hotword_score": float(keyword_weight * exact_score_scaled),
                    "prefix_penalty": float(prefix_penalty),
                    "prefix_penalty_score": float(prefix_penalty_score),
                    "prefix_stats": prefix_stats,
                    "phonetic_score_used": float(phonetic_score),
                    "phonetic_score_scaled": float(phonetic_score_scaled),
                    "consensus_score": float(consensus_score),
                    "consensus_score_used": float(consensus_score_used),
                    "consensus_support": int(consensus_support),
                    "consensus_keywords": list(consensus_keywords),
                    "total_score": float(total_score),
                    "phon_gain_vs_baseline": float(stats["phonetic_score"] - baseline_phonetic_score),
                }
            )

        scored_candidates.sort(key=lambda item: item["total_score"], reverse=True)
        return scored_candidates, int(baseline_idx), float(baseline_phonetic_score)

    def _shortform_rescore_debug_preview(self, scored_candidates: List[dict], topk: int = 5) -> List[dict]:
        preview = []
        for rank, item in enumerate(scored_candidates[: min(topk, len(scored_candidates))], start=1):
            phon_stats = item.get("phonetic_stats", {}) or {}
            exact_stats = item.get("exact_stats", {}) or {}
            preview.append(
                {
                    "rank": int(rank),
                    "total": float(item["total_score"]),
                    "hotword": float(item.get("hotword_score", 0.0)),
                    "asr": float(item["asr_score"]),
                    "asr_scaled": float(item.get("asr_score_scaled", 0.0)),
                    "exact": float(item["exact_score"]),
                    "exact_weighted": float(item.get("exact_weighted_score", 0.0)),
                    "exact_scaled": float(item.get("exact_score_scaled", 0.0)),
                    "exact_match_count": int(exact_stats.get("matched_count", 0)),
                    "exact_total_keywords": int(exact_stats.get("total_keywords", 0)),
                    "exact_coverage_ratio": float(exact_stats.get("coverage_ratio", 0.0)),
                    "exact_weighted_coverage": float(exact_stats.get("weighted_coverage", 0.0)),
                    "exact_matched_keywords": list(exact_stats.get("matched_keywords", [])),
                    "phon": float(item["phonetic_score"]),
                    "phon_used": float(item.get("phonetic_score_used", 0.0)),
                    "phon_scaled": float(item.get("phonetic_score_scaled", 0.0)),
                    "consensus": float(item.get("consensus_score", 0.0)),
                    "consensus_used": float(item.get("consensus_score_used", 0.0)),
                    "consensus_support": int(item.get("consensus_support", 0)),
                    "consensus_keywords": list(item.get("consensus_keywords", [])),
                    "phon_gain_vs_baseline": float(item.get("phon_gain_vs_baseline", 0.0)),
                    "prefix_penalty": float(item.get("prefix_penalty", 0.0)),
                    "prefix_penalty_score": float(item.get("prefix_penalty_score", 0.0)),
                    "prefix_ascii": str((item.get("prefix_stats", {}) or {}).get("ascii_prefix", "")),
                    "prefix_ascii_len": int((item.get("prefix_stats", {}) or {}).get("ascii_prefix_len", 0)),
                    "prefix_has_cjk_after_ascii": bool((item.get("prefix_stats", {}) or {}).get("has_cjk_after_ascii", False)),
                    "phon_top_sim": float(phon_stats.get("top_sim", 0.0)),
                    "phon_runner_up_sim": float(phon_stats.get("runner_up_sim", 0.0)),
                    "phon_survivors": int(phon_stats.get("num_survivors", 0)),
                    "phon_best_keyword": str(phon_stats.get("best_keyword", "")),
                    "phon_best_window_start": int(phon_stats.get("best_window_start", -1)),
                    "phon_best_window_end": int(phon_stats.get("best_window_end", -1)),
                    "phon_backend": str(phon_stats.get("backend", "")),
                    "phon_survivor_preview": list(phon_stats.get("survivor_preview", [])),
                    "text": str(item["candidate"])[:120],
                }
            )
        return preview

    def _debug_enabled(self) -> bool:
        return bool(self._debug_log_path)

    def _setup_debug_index_sampling(self):
        self._debug_selected_indices = None
        if not self._debug_enabled():
            return
        try:
            total_batches = self.trainer.num_test_batches
            if isinstance(total_batches, (list, tuple)):
                total_batches = int(total_batches[0]) if len(total_batches) > 0 else 0
            else:
                total_batches = int(total_batches)
        except Exception:
            total_batches = 0
        if total_batches <= 0:
            return
        sample_n = min(int(self._debug_max_samples), total_batches)
        rng = random.Random(self._debug_seed)
        self._debug_selected_indices = set(rng.sample(range(total_batches), sample_n))
        self._debug_log(
            "cb_debug_sample_plan",
            total_batches=total_batches,
            selected_count=sample_n,
            selected_indices_preview=sorted(list(self._debug_selected_indices))[:20],
        )

    def _debug_should_log_idx(self, idx: int) -> bool:
        if not self._debug_enabled():
            return False
        if self._debug_selected_indices is None:
            return idx < self._debug_max_samples
        return int(idx) in self._debug_selected_indices

    def _debug_log(self, event: str, **payload):
        if not self._debug_enabled():
            return
        line = {
            "ts": datetime.now().isoformat(),
            "event": event,
            **payload,
        }
        with open(self._debug_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

    def _keyword_spotting_cache_key(
        self,
        input_features: torch.Tensor,
        start_of_prev: bool,
    ):
        if not isinstance(input_features, torch.Tensor):
            return None
        try:
            return (
                bool(start_of_prev),
                str(input_features.device),
                tuple(int(x) for x in input_features.shape),
                int(input_features.data_ptr()),
            )
        except Exception:
            return None

    def _restore_keyword_spotting_cache(self, cache_key) -> Optional[List[List[int]]]:
        if cache_key is None or cache_key not in self._keyword_spotting_cache:
            return None
        cached = self._keyword_spotting_cache[cache_key]
        self._latest_keywords = cached["keywords"]
        self._latest_keyword_scores = cached["keyword_scores"]
        self._latest_prompt_keywords = cached["prompt_keywords"]
        self._latest_prompt_keyword_scores = cached["prompt_keyword_scores"]
        self._latest_prompt_token_lists = cached["prompt_token_lists"]
        self._latest_prompt_weights = cached["prompt_weights"]
        return cached["kw_ids"]

    def _set_empty_keyword_prompt_state(self, num_segments: int):
        self._latest_keywords = [[] for _ in range(num_segments)]
        self._latest_keyword_scores = [dict() for _ in range(num_segments)]
        self._latest_prompt_keywords = [[] for _ in range(num_segments)]
        self._latest_prompt_keyword_scores = [dict() for _ in range(num_segments)]
        self._latest_prompt_token_lists = [[] for _ in range(num_segments)]
        self._latest_prompt_weights = [[] for _ in range(num_segments)]

    def _kws_keywords_from_features(
        self,
        input_features: torch.Tensor,
        num_segments: int,
    ) -> Tuple[List[List[str]], List[dict]]:
        num_groups = self.kw_database.num_groups()
        keywords = [[] for _ in range(num_segments)]
        keyword_scores = [dict() for _ in range(num_segments)]
        utt_hs = None

        if num_groups > 0:
            try:
                input_features = self._cast_features_for_module(input_features, self.encoder)
                with torch.inference_mode():
                    utt_hs = self.encoder(
                        input_features=input_features,
                        output_hidden_states=True,
                        return_dict=True,
                    )["hidden_states"][-1].unsqueeze(dim=1)
                utt_hs = utt_hs.float()
                utt_hs = utt_hs / torch.linalg.norm(utt_hs, dim=-1, keepdim=True)
            except Exception:
                utt_hs = None

        if utt_hs is None:
            return keywords, keyword_scores

        for idx in range(num_groups):
            kw_group = self.kw_database.group(idx, device=self.device)
            cossim_matrices = self._calculate_cosine_similarity_matrices_(
                utt_hs=utt_hs,
                kwd_hs=kw_group["hidden_states"],
            )
            for seg_idx, matrices in enumerate(cossim_matrices):
                if matrices is None:
                    continue
                chunk_size = max(1, int(getattr(self.hparams, "kws_infer_chunk_size", 16)))
                probs_parts = []
                with torch.inference_mode():
                    for matrices_chunk in torch.split(matrices, chunk_size, dim=0):
                        kws_chunk = self.kws.forward(input_features=matrices_chunk)
                        probs_parts.append(torch.softmax(kws_chunk.logits, dim=1)[:, 1])
                probs_pos = torch.cat(probs_parts, dim=0)
                topk = max(1, int(self.hparams.kws_topk_per_group))
                topk = min(topk, probs_pos.numel())
                top_vals, top_idx = torch.topk(probs_pos, k=topk)
                selected_idx = top_idx[top_vals >= float(self.hparams.kws_positive_threshold)]

                if selected_idx.numel() == 0 and top_idx.numel() > 0:
                    selected_idx = top_idx[:1]

                for cand_idx in selected_idx.tolist():
                    kw = kw_group["keywords"][cand_idx]
                    keywords[seg_idx].append(kw)
                    score_v = float(probs_pos[cand_idx].item())
                    prev = keyword_scores[seg_idx].get(kw, 0.0)
                    if score_v > prev:
                        keyword_scores[seg_idx][kw] = score_v

        keywords = [list(dict.fromkeys(kwds)) for kwds in keywords]
        keywords = [self._sort_keywords_by_score(kwds, keyword_scores[i]) for i, kwds in enumerate(keywords)]
        max_kw = int(self.hparams.kws_max_prompt_keywords)
        if max_kw > 0:
            keywords = [kwds[:max_kw] for kwds in keywords]
            keyword_scores = [
                {kw: float(keyword_scores[i].get(kw, 0.0)) for kw in keywords[i]}
                for i in range(len(keywords))
            ]
        return keywords, keyword_scores

    def _oracle_keywords_from_buffer(self, num_segments: int) -> Tuple[List[List[str]], List[dict]]:
        keywords = [list(self.oracle_buffer) for _ in range(num_segments)]
        keyword_scores = [{kw: 1.0 for kw in self.oracle_buffer} for _ in range(num_segments)]
        return keywords, keyword_scores

    def _build_prompt_state(
        self,
        keywords: List[List[str]],
        keyword_scores: List[dict],
    ) -> Tuple[List[List[str]], List[dict], List[List[List[int]]], List[List[float]]]:
        prompt_keywords = [self._select_prompt_keywords(kwds, keyword_scores[i]) for i, kwds in enumerate(keywords)]
        prompt_keyword_scores = [
            {kw: float(keyword_scores[i].get(kw, 0.0)) for kw in prompt_keywords[i]}
            for i in range(len(prompt_keywords))
        ]
        prompt_weights = [
            self._normalize_prompt_weights(prompt_keywords[i], prompt_keyword_scores[i])
            for i in range(len(prompt_keywords))
        ]
        prompt_token_lists = [
            [self.processor_whisper.get_prompt_ids(str(kw))[1:] for kw in prompt_keywords[i] if str(kw).strip() != ""]
            for i in range(len(prompt_keywords))
        ]
        return prompt_keywords, prompt_keyword_scores, prompt_token_lists, prompt_weights

    def _set_keyword_prompt_state(
        self,
        keywords: List[List[str]],
        keyword_scores: List[dict],
        prompt_keywords: List[List[str]],
        prompt_keyword_scores: List[dict],
        prompt_token_lists: List[List[List[int]]],
        prompt_weights: List[List[float]],
    ):
        self._latest_keywords = keywords
        self._latest_keyword_scores = keyword_scores
        self._latest_prompt_keywords = prompt_keywords
        self._latest_prompt_keyword_scores = prompt_keyword_scores
        self._latest_prompt_token_lists = prompt_token_lists
        self._latest_prompt_weights = prompt_weights

    def _build_prompt_ids(
        self,
        prompt_keywords: List[List[str]],
        start_of_prev: bool,
    ) -> List[List[int]]:
        prompt_texts = [
            self.hparams.keyword_prompt_prepend + self.hparams.keyword_separator.join(kwds) + self.hparams.keyword_prompt_append
            if kwds != [] else []
            for kwds in [[self._maybe_perturb_keyword(kw) for kw in kwds] for kwds in prompt_keywords]
        ]
        if start_of_prev:
            return [
                self.processor_whisper.get_prompt_ids(prompt_text) if prompt_text != [] else []
                for prompt_text in prompt_texts
            ]
        return [
            self.processor_whisper.get_prompt_ids(prompt_text)[1:] if prompt_text != [] else []
            for prompt_text in prompt_texts
        ]

    def _generate_shortform_candidates(
        self,
        input_features: torch.Tensor,
        attention_mask: torch.Tensor,
        num_beams: int,
        num_return_sequences: int,
        no_repeat_ngram_size: int,
    ) -> torch.Tensor:
        self._latest_full_prompt_ids = []
        input_features = self._cast_features_for_module(input_features, self.whisper)
        return self.whisper.generate(
            input_features=input_features,
            attention_mask=attention_mask,
            task='transcribe',
            language=self.hparams.language,
            return_timestamps=False,
            condition_on_prev_tokens=False,
            return_segments=False,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
            do_sample=False,
            temperature=0,
            no_repeat_ngram_size=no_repeat_ngram_size,
            keyword_spotting=self.keyword_spotting,
            force_decoder_prompt_ids=bool(getattr(self.hparams, "force_decoder_prompt_ids", False)),
        )

    def _dedup_shortform_predictions(
        self,
        pred: torch.Tensor,
        desired_num_return_sequences: int,
    ) -> torch.Tensor:
        if not isinstance(pred, torch.Tensor) or pred.dim() != 2 or pred.size(0) <= 1:
            return pred
        kept_rows = []
        seen_keys = set()
        decoded = self.processor_whisper.tokenizer.batch_decode(pred, skip_special_tokens=True)
        for row_idx, text in enumerate(decoded):
            raw_text = str(text).strip()
            norm_key = self._normalize_surface_text(raw_text)
            if norm_key == "":
                norm_key = raw_text
            if norm_key in seen_keys:
                continue
            seen_keys.add(norm_key)
            kept_rows.append(pred[row_idx:row_idx + 1])
            if len(kept_rows) >= desired_num_return_sequences:
                break
        if len(kept_rows) == 0:
            return pred[:desired_num_return_sequences]
        return torch.cat(kept_rows, dim=0)

    def _debug_sampled_test_outputs(self) -> List[dict]:
        if self._debug_selected_indices is None:
            sampled_outputs = self.test_step_outputs[: self._debug_max_samples]
        else:
            sampled_outputs = [
                out for out in self.test_step_outputs
                if int(out.get("idx", -1)) in self._debug_selected_indices
            ]
        return sorted(sampled_outputs, key=lambda out: int(out.get("idx", -1)))

    def _empty_oracle_diag(self) -> dict:
        return {
            "enabled": bool(self._oracle_nbest_diagnostic),
            "is_shortform": False,
            "nbest_size": 0,
            "selected_rank": 1,
            "candidates": [],
        }

    def _normalize_eval_texts(
        self,
        preds: List[str],
        refs: List[str],
        keywords: List[List[dict]],
    ) -> Tuple[List[str], List[str], List[List[dict]], str]:
        simplifier_name = str(getattr(self, "_text_normalizer_backend", "identity"))
        preds = [self._normalize_surface_text(p) for p in preds]
        refs = [self._normalize_surface_text(r) for r in refs]
        keywords = [[{
            **kw,
            'mention': self._normalize_surface_text(kw.get('mention', ''))
        } for kw in kw_list] for kw_list in keywords]
        return preds, refs, keywords, simplifier_name

    def _single_sample_hotword_only_cer(self, ref_text: str, pred_text: str, kw_list: List[dict]) -> float:
        total_err = 0.0
        total_chars = 0
        ref_text = self._normalize_surface_text(ref_text)
        pred_text = self._normalize_surface_text(pred_text)
        for kw in kw_list:
            mention = self._normalize_surface_text(kw.get("mention", "")).strip()
            if mention == "":
                continue
            occurrences = list(re.finditer(re.escape(mention), ref_text))
            if len(occurrences) == 0:
                continue
            for match in occurrences:
                start = int(match.start())
                end = int(match.end())
                ref_span = ref_text[start:end]
                pred_span = pred_text[start:min(end, len(pred_text))]
                total_err += float(self._metric_edit_distance(list(ref_span), list(pred_span)))
                total_chars += len(ref_span)
        if total_chars == 0:
            return 0.0
        return float(total_err / total_chars)

    def _build_oracle_nbest_tables(
        self,
        refs: List[str],
        keywords: List[List[dict]],
        raw_refs: List[str],
        raw_keywords: List[List[dict]],
    ) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        if not self._oracle_nbest_diagnostic:
            return None, None

        detail_rows = []
        summary_rows = []
        oracle_hits = 0
        top1_hits = 0
        oracle_better_count = 0

        for i, step_output in enumerate(self.test_step_outputs):
            diag = step_output.get("oracle_diag", {}) or {}
            candidates = diag.get("candidates", []) or []
            if len(candidates) == 0:
                continue

            raw_candidate_texts = [str(c.get("text", "")) for c in candidates]
            norm_candidates, _, norm_keywords, _ = self._normalize_eval_texts(
                preds=raw_candidate_texts,
                refs=[raw_refs[i]] * len(raw_candidate_texts),
                keywords=[raw_keywords[i]] * len(raw_candidate_texts),
            )
            norm_kw_list = norm_keywords[0] if len(norm_keywords) > 0 else keywords[i]
            ref_text = str(refs[i])

            candidate_rows = []
            for cand_idx, cand in enumerate(candidates):
                pred_text = str(norm_candidates[cand_idx]) if cand_idx < len(norm_candidates) else str(cand.get("text", ""))
                ent_recall = float(entity_recall(
                    preds=[pred_text],
                    refs=[ref_text],
                    mentions=[norm_kw_list],
                    ner_tags='ALL',
                    char_split=True,
                )['ALL'])
                cer = float(self._metric_cer_single(ref_text, pred_text))
                hot_cer = float(self._single_sample_hotword_only_cer(ref_text, pred_text, norm_kw_list))
                candidate_rows.append({
                    "rank": int(cand_idx + 1),
                    "pred": pred_text,
                    "entity_recall": ent_recall,
                    "cer": cer,
                    "hotword_only_cer": hot_cer,
                    "total_score": float(cand.get("total_score", 0.0)),
                    "asr_score": float(cand.get("asr_score", 0.0)),
                    "exact_score": float(cand.get("exact_score", 0.0)),
                    "exact_match_count": int(((cand.get("exact_stats", {}) or {}).get("matched_count", 0))),
                    "exact_total_keywords": int(((cand.get("exact_stats", {}) or {}).get("total_keywords", 0))),
                    "exact_coverage_ratio": float(((cand.get("exact_stats", {}) or {}).get("coverage_ratio", 0.0))),
                    "phonetic_score": float(cand.get("phonetic_score", 0.0)),
                    "consensus_score": float(cand.get("consensus_score", 0.0)),
                    "consensus_score_used": float(cand.get("consensus_score_used", 0.0)),
                    "consensus_support": int(cand.get("consensus_support", 0)),
                    "consensus_keywords": ";".join(str(k) for k in list(cand.get("consensus_keywords", []) or [])),
                })

            oracle_row = sorted(
                candidate_rows,
                key=lambda row: (-row["entity_recall"], row["hotword_only_cer"], row["cer"], row["rank"]),
            )[0]
            top1_row = candidate_rows[0]

            top1_hits += int(top1_row["entity_recall"] >= 0.999999)
            oracle_hits += int(oracle_row["entity_recall"] >= 0.999999)
            oracle_better_count += int(
                (oracle_row["entity_recall"] > top1_row["entity_recall"])
                or (
                    abs(oracle_row["entity_recall"] - top1_row["entity_recall"]) < 1e-9
                    and oracle_row["hotword_only_cer"] < top1_row["hotword_only_cer"]
                )
            )

            summary_rows.append({
                "idx": int(step_output.get("idx", i)),
                "nbest_size": int(len(candidate_rows)),
                "top1_entity_recall": float(top1_row["entity_recall"]),
                "oracle_entity_recall": float(oracle_row["entity_recall"]),
                "top1_hotword_only_cer": float(top1_row["hotword_only_cer"]),
                "oracle_hotword_only_cer": float(oracle_row["hotword_only_cer"]),
                "top1_cer": float(top1_row["cer"]),
                "oracle_cer": float(oracle_row["cer"]),
                "oracle_rank": int(oracle_row["rank"]),
                "oracle_beats_top1": bool(
                    (oracle_row["entity_recall"] > top1_row["entity_recall"])
                    or (
                        abs(oracle_row["entity_recall"] - top1_row["entity_recall"]) < 1e-9
                        and oracle_row["hotword_only_cer"] < top1_row["hotword_only_cer"]
                    )
                ),
            })

            for row in candidate_rows:
                detail_rows.append({
                    "idx": int(step_output.get("idx", i)),
                    "ref": ref_text,
                    "rank": int(row["rank"]),
                    "is_top1": bool(row["rank"] == 1),
                    "is_oracle": bool(row["rank"] == oracle_row["rank"]),
                    "entity_recall": float(row["entity_recall"]),
                    "cer": float(row["cer"]),
                    "hotword_only_cer": float(row["hotword_only_cer"]),
                    "total_score": float(row["total_score"]),
                    "asr_score": float(row["asr_score"]),
                    "exact_score": float(row["exact_score"]),
                    "phonetic_score": float(row["phonetic_score"]),
                    "consensus_score": float(row["consensus_score"]),
                    "consensus_score_used": float(row["consensus_score_used"]),
                    "consensus_support": int(row["consensus_support"]),
                    "consensus_keywords": str(row["consensus_keywords"]),
                    "pred": str(row["pred"]),
                })

        if len(summary_rows) == 0:
            return None, None

        num_samples = len(summary_rows)
        aggregate_row = {
            "idx": "ALL",
            "nbest_size": float(sum(r["nbest_size"] for r in summary_rows) / max(num_samples, 1)),
            "top1_entity_recall": float(sum(r["top1_entity_recall"] for r in summary_rows) / max(num_samples, 1)),
            "oracle_entity_recall": float(sum(r["oracle_entity_recall"] for r in summary_rows) / max(num_samples, 1)),
            "top1_hotword_only_cer": float(sum(r["top1_hotword_only_cer"] for r in summary_rows) / max(num_samples, 1)),
            "oracle_hotword_only_cer": float(sum(r["oracle_hotword_only_cer"] for r in summary_rows) / max(num_samples, 1)),
            "top1_cer": float(sum(r["top1_cer"] for r in summary_rows) / max(num_samples, 1)),
            "oracle_cer": float(sum(r["oracle_cer"] for r in summary_rows) / max(num_samples, 1)),
            "oracle_rank": float(sum(r["oracle_rank"] for r in summary_rows) / max(num_samples, 1)),
            "oracle_beats_top1": float(oracle_better_count / max(num_samples, 1)),
            "top1_hit_rate": float(top1_hits / max(num_samples, 1)),
            "oracle_hit_rate": float(oracle_hits / max(num_samples, 1)),
        }
        summary_df = pd.concat([pd.DataFrame(summary_rows), pd.DataFrame([aggregate_row])], ignore_index=True)
        detail_df = pd.DataFrame(detail_rows)
        return detail_df, summary_df

    @staticmethod
    def _metric_edit_distance(seq1, seq2):
        m, n = len(seq1), len(seq2)
        dp = [list(range(n + 1))]
        for i in range(1, m + 1):
            row = [i] + [0] * n
            dp.append(row)
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                cost = 0 if seq1[i - 1] == seq2[j - 1] else 1
                dp[i][j] = min(
                    dp[i - 1][j] + 1,
                    dp[i][j - 1] + 1,
                    dp[i - 1][j - 1] + cost,
                )
        return dp[m][n]

    def _metric_cer_single(self, ref: str, pred: str) -> float:
        ref = self._normalize_surface_text(ref)
        pred = self._normalize_surface_text(pred)
        n = len(ref)
        if n == 0:
            return 0.0 if len(pred) == 0 else 1.0
        return self._metric_edit_distance(list(ref), list(pred)) / n

    def _metric_wer_single(self, ref: str, pred: str) -> float:
        ref_words = str(ref).split()
        pred_words = str(pred).split()
        n = len(ref_words)
        if n == 0:
            return 0.0 if len(pred_words) == 0 else 1.0
        return self._metric_edit_distance(ref_words, pred_words) / n

    def _evaluation_conditions(self) -> Optional[List[int]]:
        speakers = [step_output['speaker'] for step_output in self.test_step_outputs]
        if speakers[0] is None:
            return None
        speaker2id = {speaker: speaker_id for speaker_id, speaker in enumerate(set(speakers))}
        return [speaker2id[speaker] for speaker in speakers]

    def _evaluate_test_metrics(
        self,
        preds: List[str],
        refs: List[str],
        keywords: List[List[dict]],
        conditions: Optional[List[int]],
    ) -> dict:
        def f_entity_recall(labels, samples, samples2=None):
            preds_ = samples
            refs_, keywords_ = list(zip(*labels))
            return entity_recall(
                preds=preds_,
                refs=refs_,
                mentions=keywords_,
                ner_tags='ALL',
                char_split=True
            )['ALL']

        def f_cer(labels, samples, samples2=None):
            preds_ = samples
            refs_ = labels
            return float(sum(self._metric_cer_single(r, p) for r, p in zip(refs_, preds_)) / max(len(refs_), 1))

        def f_hotword_sentence_cer(labels, samples, samples2=None):
            preds_ = samples
            pairs = [
                (ref, pred)
                for (ref, kw_list), pred in zip(labels, preds_)
                if len(kw_list) > 0
            ]
            if len(pairs) == 0:
                return 0.0
            return float(sum(self._metric_cer_single(ref, pred) for ref, pred in pairs) / len(pairs))

        def f_hotword_only_cer(labels, samples, samples2=None):
            preds_ = samples
            total_err = 0.0
            total_chars = 0
            for (ref, kw_list), pred in zip(labels, preds_):
                ref_text = str(ref)
                pred_text = str(pred)
                for kw in kw_list:
                    mention = str(kw.get("mention", "")).strip()
                    if mention == "":
                        continue
                    occurrences = list(re.finditer(re.escape(mention), ref_text))
                    if len(occurrences) == 0:
                        continue
                    for match in occurrences:
                        start = int(match.start())
                        end = int(match.end())
                        ref_span = ref_text[start:end]
                        pred_span = pred_text[start:min(end, len(pred_text))]
                        total_err += float(self._metric_edit_distance(list(ref_span), list(pred_span)))
                        total_chars += len(ref_span)
            if total_chars == 0:
                return 0.0
            return float(total_err / total_chars)

        def f_wer(labels, samples, samples2=None):
            preds_ = samples
            refs_ = labels
            return float(sum(self._metric_wer_single(r, p) for r, p in zip(refs_, preds_)) / max(len(refs_), 1))

        def _update_progress(desc: str):
            if self._post_test_progress_bar is not None:
                self._post_test_progress_bar.set_description(desc)
                self._post_test_progress_bar.update(1)
                self._post_test_progress_bar.refresh()

        samples = Flexlist(preds)
        labels = Flexlist(zip(refs, keywords))
        refs_samples = Flexlist(refs)
        metrics = {}
        metrics["recall"] = evaluate_with_conf_int(samples, f_entity_recall, labels, conditions, num_bootstraps=1000, alpha=5)
        _update_progress("Post-test eval recall")
        metrics["cer"] = evaluate_with_conf_int(samples, f_cer, refs_samples, conditions, num_bootstraps=1000, alpha=5)
        _update_progress("Post-test eval cer")
        metrics["hotword_sentence_cer"] = evaluate_with_conf_int(samples, f_hotword_sentence_cer, labels, conditions, num_bootstraps=1000, alpha=5)
        _update_progress("Post-test eval hotword-sent")
        metrics["hotword_only_cer"] = evaluate_with_conf_int(samples, f_hotword_only_cer, labels, conditions, num_bootstraps=1000, alpha=5)
        _update_progress("Post-test eval hotword-only")
        metrics["wer"] = evaluate_with_conf_int(samples, f_wer, refs_samples, conditions, num_bootstraps=1000, alpha=5)
        _update_progress("Post-test eval wer")
        return metrics

    def _build_results_dataframe(self, metrics: dict) -> pd.DataFrame:
        recall = metrics["recall"]
        cer = metrics["cer"]
        hotword_sentence_cer = metrics["hotword_sentence_cer"]
        hotword_only_cer = metrics["hotword_only_cer"]
        wer = metrics["wer"]
        return pd.DataFrame(
            [[
                recall[0], recall[1][0], recall[1][1],
                cer[0], cer[1][0], cer[1][1],
                hotword_sentence_cer[0], hotword_sentence_cer[1][0], hotword_sentence_cer[1][1],
                hotword_only_cer[0], hotword_only_cer[1][0], hotword_only_cer[1][1],
                wer[0], wer[1][0], wer[1][1],
            ]],
            index=[('w/ prompt' if self.hparams.prompt else 'w/o prompt') + ' - ' + self.hparams.oracle],
            columns=[
                "Entity Recall", "Entity Recall LB", "Entity Recall UB",
                "CER", "CER LB", "CER UB",
                "Hotword Sentence CER", "Hotword Sentence CER LB", "Hotword Sentence CER UB",
                "Hotword Only CER", "Hotword Only CER LB", "Hotword Only CER UB",
                "WER", "WER LB", "WER UB",
            ],
        )

    def keyword_spotting(
        self,
        input_features: torch.Tensor,
        start_of_prev: bool = False
    ):
        cache_key = self._keyword_spotting_cache_key(input_features, start_of_prev)
        cached_kw_ids = self._restore_keyword_spotting_cache(cache_key)
        if cached_kw_ids is not None:
            return cached_kw_ids

        dbg_idx = self._debug_counters["keyword_spotting"]
        self._debug_counters["keyword_spotting"] += 1

        num_segments = input_features.size(dim=0)

        # case no prompt to provide
        if not self.hparams.prompt:
            self._set_empty_keyword_prompt_state(num_segments)
            self._latest_full_prompt_ids = []
            return [[]] * num_segments

        if self.hparams.oracle == 'kws':
            keywords, keyword_scores = self._kws_keywords_from_features(input_features, num_segments)
        else:
            keywords, keyword_scores = self._oracle_keywords_from_buffer(num_segments)

        prompt_keywords, prompt_keyword_scores, prompt_token_lists, prompt_weights = self._build_prompt_state(
            keywords,
            keyword_scores,
        )
        self._set_keyword_prompt_state(
            keywords,
            keyword_scores,
            prompt_keywords,
            prompt_keyword_scores,
            prompt_token_lists,
            prompt_weights,
        )
        kw_ids = self._build_prompt_ids(prompt_keywords, start_of_prev)
        if start_of_prev and len(kw_ids) > 0 and isinstance(kw_ids[0], list):
            self._latest_full_prompt_ids = list(kw_ids[0])
        elif start_of_prev:
            self._latest_full_prompt_ids = []

        if self._debug_should_log_idx(dbg_idx):
            self._debug_log(
                "keyword_spotting",
                idx=dbg_idx,
                input_features_shape=list(input_features.shape),
                start_of_prev=bool(start_of_prev),
                num_segments=num_segments,
                keyword_counts=[len(k) for k in keywords],
                prompt_keyword_counts=[len(k) for k in prompt_keywords],
                kw_ids_lengths=[len(x) for x in kw_ids],
                keyword_perturb_prob=float(getattr(self.hparams, "keyword_perturb_prob", 0.0)),
                keyword_score_preview=[
                    {k: round(v, 4) for k, v in list(keyword_scores[i].items())[:5]}
                    for i in range(min(3, len(keyword_scores)))
                ],
                keyword_preview=[k[:5] for k in keywords[:3]],
                prompt_keyword_preview=[k[:5] for k in prompt_keywords[:3]],
                prompt_weight_preview=[w[:5] for w in prompt_weights[:3]],
            )

        if cache_key is not None:
            self._keyword_spotting_cache = {
                cache_key: {
                    "keywords": keywords,
                    "keyword_scores": keyword_scores,
                    "prompt_keywords": prompt_keywords,
                    "prompt_keyword_scores": prompt_keyword_scores,
                    "prompt_token_lists": prompt_token_lists,
                    "prompt_weights": prompt_weights,
                    "kw_ids": kw_ids,
                }
            }

        return kw_ids    

    def forward(
        self,
        input_features: torch.Tensor,
        attention_mask: torch.Tensor,
        oracle: List[str] = []
    ):
        expected_mels = int(getattr(self.processor_whisper.feature_extractor, "feature_size", 0))
        if expected_mels > 0 and int(input_features.size(1)) != expected_mels:
            raise ValueError(
                "ASR feature dimension mismatch: "
                f"model whisper_ckpt={self.hparams.whisper_ckpt} expects {expected_mels} mel bins, "
                f"but dataloader provided {int(input_features.size(1))}. "
                "Set data.init_args.whisper_ckpt to the same Whisper family as model.init_args.whisper_ckpt."
            )
        dbg_idx = self._debug_counters["forward"]
        self._debug_counters["forward"] += 1

        # set buffer of the oracle
        self.oracle_buffer = oracle           
        # Only reuse keyword spotting results within the current forward pass.
        # Cross-sample reuse is unsafe because tensor storage pointers can be
        # recycled across batches.
        self._keyword_spotting_cache = {}

        # Whether audio is shorter than 30s (Whisper frame axis is the last dim).
        # input_features shape: [batch, n_mels, n_frames]
        is_shortform = input_features.size(dim=-1) <= N_FRAMES

        if self._debug_should_log_idx(dbg_idx):
            self._debug_log(
                "forward_in",
                idx=dbg_idx,
                input_features_shape=list(input_features.shape),
                attention_mask_shape=list(attention_mask.shape) if isinstance(attention_mask, torch.Tensor) else None,
                is_shortform=bool(is_shortform),
                oracle_count=len(oracle),
                oracle_preview=oracle[:5],
            )

        do_rescore = bool(is_shortform and getattr(self.hparams, "enable_phonetic_rescore", False))
        nbest = max(1, int(getattr(self.hparams, "rescore_nbest", 5)))
        gen_nbest = min(max(nbest * 3, nbest), 16) if do_rescore else 1
        num_beams = max(5, gen_nbest) if do_rescore else 5
        num_return_sequences = gen_nbest if do_rescore else 1
        # generate transcript candidates
        if is_shortform:
            pred = self._generate_shortform_candidates(
                input_features=input_features,
                attention_mask=attention_mask,
                num_beams=num_beams,
                num_return_sequences=num_return_sequences,
                no_repeat_ngram_size=max(0, int(getattr(self.hparams, "shortform_no_repeat_ngram_size", 3))),
            )
            if do_rescore:
                pred = self._dedup_shortform_predictions(pred, desired_num_return_sequences=gen_nbest)
        else:
            pred = self.whisper.generate(
                input_features = input_features,
                attention_mask = attention_mask,
                task = 'transcribe',
                language = self.hparams.language,
                return_timestamps = True,
                condition_on_prev_tokens = True,
                return_segments = True,
                num_beams = num_beams,
                num_return_sequences = num_return_sequences,
                do_sample = False,
                temperature = 0,
                no_repeat_ngram_size = 0,
                keyword_spotting = self.keyword_spotting,
                force_decoder_prompt_ids=bool(getattr(self.hparams, "force_decoder_prompt_ids", False)),
            )

        if self._debug_should_log_idx(dbg_idx):
            prompt_keywords = self._latest_prompt_keywords[0] if len(self._latest_prompt_keywords) > 0 else []
            prompt_weights = self._latest_prompt_weights[0] if len(self._latest_prompt_weights) > 0 else []
            self._debug_log(
                "forward_prompt_state",
                idx=dbg_idx,
                prompt_keyword_count=len(prompt_keywords),
                prompt_token_count=self._safe_sequence_length(self._latest_full_prompt_ids),
                prompt_weight_preview=[float(w) for w in prompt_weights[:5]],
                prompt_weight_concentration=float(self._prompt_weight_concentration(prompt_weights)),
                prompt_keywords_preview=prompt_keywords[:5],
                prompt_active=bool(len(prompt_keywords) > 0),
            )

        # decode prediction / select best candidate
        surface_candidate_texts = []
        if is_shortform:
            if isinstance(pred, torch.Tensor) and pred.dim() == 2 and pred.size(0) > 1 and do_rescore:
                candidates = self.processor_whisper.tokenizer.batch_decode(pred, skip_special_tokens=True)
                surface_candidate_texts = list(candidates)
                kws = self._latest_keywords[0] if len(self._latest_keywords) > 0 else []
                kws_score_map = self._latest_keyword_scores[0] if len(self._latest_keyword_scores) > 0 else {}
                kws = self._select_rescore_keywords(kws, kws_score_map)
                effective_keyword_weight = float(getattr(self.hparams, "rescore_keyword_weight", 2.0))
                candidate_stats = self._build_shortform_candidate_stats(
                    candidates=candidates,
                    pred_sequences=pred,
                    input_features=input_features,
                    attention_mask=attention_mask,
                    keywords=kws,
                    keyword_scores=kws_score_map,
                )
                scored_candidates, baseline_idx, baseline_raw_phon = self._score_shortform_candidates(
                    candidate_stats=candidate_stats,
                    keyword_weight=effective_keyword_weight,
                )
                baseline_total_score = float(candidate_stats[baseline_idx].get("asr_score", 0.0))
                for item in scored_candidates:
                    if str(item.get("candidate", "")) == str(candidate_stats[baseline_idx].get("candidate", "")):
                        baseline_total_score = float(item.get("total_score", baseline_total_score))
                        break
                self._latest_forward_candidates = [
                    {
                        "rank": int(rank + 1),
                        "text": str(item.get("candidate", "")),
                        "total_score": float(item.get("total_score", 0.0)),
                        "score_delta_vs_baseline": float(item.get("total_score", 0.0) - baseline_total_score),
                        "hotword_score": float(item.get("hotword_score", 0.0)),
                        "asr_score": float(item.get("asr_score", 0.0)),
                        "asr_score_scaled": float(item.get("asr_score_scaled", 0.0)),
                        "exact_score": float(item.get("exact_score", 0.0)),
                        "exact_weighted_score": float(item.get("exact_weighted_score", 0.0)),
                        "exact_score_scaled": float(item.get("exact_score_scaled", 0.0)),
                        "exact_stats": dict(item.get("exact_stats", {}) or {}),
                        "phonetic_score": float(item.get("phonetic_score", 0.0)),
                        "phonetic_score_used": float(item.get("phonetic_score_used", 0.0)),
                        "phonetic_score_scaled": float(item.get("phonetic_score_scaled", 0.0)),
                        "consensus_score": float(item.get("consensus_score", 0.0)),
                        "consensus_score_used": float(item.get("consensus_score_used", 0.0)),
                        "consensus_support": int(item.get("consensus_support", 0)),
                        "consensus_keywords": list(item.get("consensus_keywords", [])),
                        "prefix_penalty_score": float(item.get("prefix_penalty_score", 0.0)),
                    }
                    for rank, item in enumerate(scored_candidates)
                ]
                pred = str(scored_candidates[0]["candidate"]).strip()
                if self._debug_should_log_idx(dbg_idx):
                    baseline_candidate = candidate_stats[baseline_idx] if 0 <= baseline_idx < len(candidate_stats) else {}
                    final_candidate = scored_candidates[0] if len(scored_candidates) > 0 else {}
                    baseline_text_full = str(baseline_candidate.get("candidate", ""))
                    final_text_full = str(final_candidate.get("candidate", ""))
                    final_text_norm = self._normalize_text_for_logging(final_text_full)
                    exact_bucket_candidates = [cand for cand in scored_candidates if float(cand.get("exact_score", 0.0)) > 0.0]
                    shortest_exact_candidate = None
                    if len(exact_bucket_candidates) > 0:
                        shortest_exact_candidate = min(
                            exact_bucket_candidates,
                            key=lambda cand: (
                                len(self._normalize_text_for_logging(str(cand.get("candidate", "")))),
                                -float(cand.get("total_score", 0.0)),
                            ),
                        )
                    shortest_exact_text = str(shortest_exact_candidate.get("candidate", "")) if shortest_exact_candidate is not None else ""
                    shortest_exact_text_norm = self._normalize_text_for_logging(shortest_exact_text) if shortest_exact_text != "" else ""
                    shortest_exact_rank = -1
                    if shortest_exact_candidate is not None:
                        for cand_rank, cand in enumerate(scored_candidates, start=1):
                            if cand is shortest_exact_candidate:
                                shortest_exact_rank = cand_rank
                                break
                    candidate_pool_summary = {
                        "num_candidates": int(len(scored_candidates)),
                        "num_exact_positive": int(sum(float(c.get("exact_score", 0.0)) > 0.0 for c in scored_candidates)),
                        "use_asr_score": bool(getattr(self.hparams, "rescore_use_asr_score", True)),
                        "num_phon_positive": int(sum(float(c.get("phonetic_score", 0.0)) > 0.0 for c in scored_candidates)),
                        "max_exact_score": float(max(float(c.get("exact_score", 0.0)) for c in scored_candidates)),
                        "max_phonetic_score": float(max(float(c.get("phonetic_score", 0.0)) for c in scored_candidates)),
                        "top1_has_exact": bool(float(final_candidate.get("exact_score", 0.0)) > 0.0),
                        "top1_norm_len": int(len(final_text_norm)),
                        "shortest_exact_rank": int(shortest_exact_rank),
                        "shortest_exact_norm_len": int(len(shortest_exact_text_norm)),
                        "shortest_exact_is_prefix_of_top1": bool(
                            shortest_exact_text_norm != ""
                            and final_text_norm.startswith(shortest_exact_text_norm)
                            and len(final_text_norm) > len(shortest_exact_text_norm)
                        ),
                        "top1_minus_shortest_exact_len": int(
                            max(0, len(final_text_norm) - len(shortest_exact_text_norm))
                        ) if shortest_exact_text_norm != "" else 0,
                    }
                    baseline_final_rank = -1
                    for cand_rank, cand in enumerate(scored_candidates, start=1):
                        if str(cand.get("candidate", "")) == baseline_text_full:
                            baseline_final_rank = cand_rank
                            break
                    self._debug_log(
                        "phonetic_rescore_shortform",
                        idx=dbg_idx,
                        nbest=int(len(scored_candidates)),
                        baseline_idx=int(baseline_idx),
                        baseline_raw_phon=float(baseline_raw_phon),
                        effective_keyword_weight=float(effective_keyword_weight),
                        keywords_preview=kws[:8],
                        candidate_pool_summary=candidate_pool_summary,
                        rerank_summary={
                            "baseline_text": baseline_text_full[:160],
                            "final_text": final_text_full[:160],
                            "baseline_asr_score": float(baseline_candidate.get("asr_score", 0.0)),
                            "baseline_exact_score": float(baseline_candidate.get("exact_score", 0.0)),
                            "baseline_phonetic_score": float(baseline_candidate.get("phonetic_score", 0.0)),
                            "final_total_score": float(final_candidate.get("total_score", 0.0)),
                            "final_exact_score": float(final_candidate.get("exact_score", 0.0)),
                            "final_phonetic_score": float(final_candidate.get("phonetic_score", 0.0)),
                            "shortest_exact_text": shortest_exact_text[:160],
                            "shortest_exact_rank": int(shortest_exact_rank),
                            "shortest_exact_is_prefix_of_top1": bool(
                                shortest_exact_text_norm != ""
                                and final_text_norm.startswith(shortest_exact_text_norm)
                                and len(final_text_norm) > len(shortest_exact_text_norm)
                            ),
                            "baseline_kept": bool(baseline_text_full == final_text_full),
                            "baseline_final_rank": int(baseline_final_rank),
                            "baseline_total_score": float(baseline_total_score),
                            "final_beats_baseline_total": float(final_candidate.get("total_score", 0.0) - baseline_total_score),
                        },
                        top_candidates=self._shortform_rescore_debug_preview(scored_candidates),
                    )
            else:
                self._latest_forward_candidates = []
                pred = self.processor_whisper.tokenizer.batch_decode(pred, skip_special_tokens=True)[0].strip()
        else:
            self._latest_forward_candidates = []
            pred = self.processor_whisper.tokenizer.batch_decode(pred['sequences'], skip_special_tokens=True)[0].strip()

        if is_shortform:
            repair_keywords = self._latest_keywords[0] if len(self._latest_keywords) > 0 else []
            repair_score_map = self._latest_keyword_scores[0] if len(self._latest_keyword_scores) > 0 else {}
            pred_before_repair = str(pred)
            pred, repair_info = self._phonetic_surface_repair(
                pred_text=pred_before_repair,
                keywords=repair_keywords,
                kw_scores=repair_score_map,
                candidate_texts=surface_candidate_texts,
            )
            if bool(repair_info.get("changed", False)):
                if len(self._latest_forward_candidates) > 0:
                    self._latest_forward_candidates[0]["text_before_surface_repair"] = pred_before_repair
                    self._latest_forward_candidates[0]["text"] = str(pred)
                    self._latest_forward_candidates[0]["surface_repair"] = dict(repair_info)
                if self._debug_should_log_idx(dbg_idx):
                    self._debug_log(
                        "phonetic_surface_repair",
                        idx=dbg_idx,
                        before=pred_before_repair[:160],
                        after=str(pred)[:160],
                        repairs=list(repair_info.get("repairs", [])),
                    )

        if self._debug_should_log_idx(dbg_idx):
            self._debug_log(
                "forward_out",
                idx=dbg_idx,
                pred_len=len(pred),
                pred_preview=pred[:120],
                normalized_pred_preview=self._normalize_text_for_logging(pred)[:120],
            )

        return pred
    
    def _calculate_cosine_similarity_matrices_(
        self,
        utt_hs: torch.Tensor,
        kwd_hs: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        if utt_hs is None:
            return []
        num_segments = utt_hs.size(dim=0)
        num_keywords = len(kwd_hs)
        for kw_idx, kwd_hs_ in enumerate(kwd_hs):
            if int(kwd_hs_.size(-1)) != int(utt_hs.size(-1)):
                raise ValueError(
                    "KWS hidden-state dimension mismatch before cosine similarity: "
                    f"keyword_idx={kw_idx} dim={int(kwd_hs_.size(-1))}, "
                    f"utterance dim={int(utt_hs.size(-1))}. "
                    "Re-extract utterance and keyword hidden states with the same Whisper encoder/profile."
                )
        kwd_hs = [
            kwd_hs_.to(device=utt_hs.device, dtype=utt_hs.dtype)
            for kwd_hs_ in kwd_hs
        ]
        # compute similarity matrices
        # simple inner product because vectors are normalized
        cossim_matrices = [matrices for matrices in [torch.matmul(kwd_hs_, utt_hs.transpose(2, 3)) if utt_hs != None else None for kwd_hs_ in kwd_hs]]
        cossim_matrices = [[cossim_matrices[kwd_idx][seg_idx] for kwd_idx in range(num_keywords)] for seg_idx in range(num_segments)]

        # resize edges
        if self.hparams.kws_features_size is None:
            short_edge = max([hs.size(dim=1) for hs in kwd_hs])
            long_edge = utt_hs.size(dim=2)
        else:
            short_edge = self.hparams.kws_features_size[0]
            long_edge = self.hparams.kws_features_size[1]
        cossim_matrices = [torch.stack([torchvision.transforms.functional.resize(matrices, (short_edge, long_edge), antialias=False) for matrices in cossim_matrices_], dim=0) if cossim_matrices_ != None else None for cossim_matrices_ in cossim_matrices]

        return cossim_matrices

    def on_test_epoch_start(self):
        # list for storing the outputs of each test step
        # in the past this was done automatically using the test_epoch_end hook
        # but https://github.com/Lightning-AI/lightning/pull/16520
        self.test_step_outputs = []
        self._setup_debug_index_sampling()

    def test_step(self, batch, batch_idx):  
        dbg_idx = self._debug_counters["test_step"]
        self._debug_counters["test_step"] += 1

        if self._debug_should_log_idx(dbg_idx):
            self._debug_log(
                "test_step_in",
                idx=dbg_idx,
                batch_idx=int(batch_idx),
                utterance_features_shape=list(batch["utterance"]["features"].shape),
                utterance_attention_mask_shape=list(batch["utterance"]["attention_mask"].shape),
                transcript_preview=str(batch.get("transcript", ""))[:120],
            )

        # parameters
        if self.hparams.oracle == 'gold':
            oracle = [self.kw_database[idx]['keyword'] for idx in torch.argwhere(torch.cat(batch['hotword_labels'], dim=0)).squeeze(dim=1)]
        elif self.hparams.oracle == 'random':
            oracle = [self.kw_database[idx]['keyword'] for idx in random.sample(list(set(range(len(self.kw_database))) - set(torch.argwhere(torch.cat(batch['hotword_labels'], dim=0)).squeeze(dim=1).tolist())), torch.sum(torch.cat(batch['hotword_labels'], dim=0)).item())]
        else:
            oracle = []
        
        # get predictions from the CB-Whisper for the given setting
        preds = self.forward(
            input_features = batch['utterance']['features'],
            attention_mask = batch['utterance']['attention_mask'],
            oracle = oracle
        )

        # Build keyword mentions for evaluation.
        # ACL provides `batch['keywords']`; AISHELL does not, so recover mentions from hotword_labels.
        if batch.get('keywords', None) is not None:
            keyword_mentions = batch['keywords']
        else:
            keyword_mentions = []
            transcript = str(batch.get('transcript', ''))
            for group_idx, group_labels in enumerate(batch['hotword_labels']):
                if not isinstance(group_labels, torch.Tensor):
                    group_labels = torch.tensor(group_labels)
                pos_idx = torch.nonzero(group_labels > 0, as_tuple=False).flatten().tolist()
                if len(pos_idx) == 0:
                    continue
                group_keywords = self.kw_database.group(group_idx, load_hs=False)['keywords']
                for kw_idx in pos_idx:
                    if kw_idx < 0 or kw_idx >= len(group_keywords):
                        continue
                    mention = group_keywords[kw_idx]
                    for match in re.finditer(re.escape(mention), transcript):
                        keyword_mentions.append({
                            'mention': mention,
                            'total_offset': match.start(),
                            'end_offset': match.end(),
                            'ner_tag': 'UNK'
                        })

        # add results for later evaluation
        self.test_step_outputs.append({
            'idx': int(batch_idx),
            'preds': preds,
            'target': batch['transcript'],
            'speaker': batch.get('speaker', None),
            'keywords': keyword_mentions,
            'oracle_diag': {
                'enabled': bool(self._oracle_nbest_diagnostic),
                'is_shortform': bool(batch['utterance']['features'].shape[-1] <= N_FRAMES),
                'nbest_size': int(len(self._latest_forward_candidates)),
                'selected_rank': 1,
                'candidates': list(self._latest_forward_candidates),
            } if self._oracle_nbest_diagnostic else self._empty_oracle_diag(),
        })

    def on_test_epoch_end(self):
        self._post_test_progress_bar = tqdm(
            total=7,
            desc="Post-test eval normalize",
            leave=True,
            dynamic_ncols=True,
        )
        # get list of predictions
        preds = [out['preds'] for out in self.test_step_outputs]
        # get list of reference transcripts
        refs = [out['target'] for out in self.test_step_outputs]    
        raw_preds = list(preds)
        raw_refs = list(refs)
        keywords = [[{
            **kw, **({'ner_tag': 'UNK'} if 'ner_tag' not in kw else {})
        } for kw in step_output.get('keywords', [])] for step_output in self.test_step_outputs]

        preds, refs, keywords, simplifier_name = self._normalize_eval_texts(preds, refs, keywords)
        if self._post_test_progress_bar is not None:
            self._post_test_progress_bar.set_description("Post-test eval normalize")
            self._post_test_progress_bar.update(1)
            self._post_test_progress_bar.refresh()

        if self._debug_enabled():
            if self._debug_selected_indices is None:
                preview_indices = list(range(min(self._debug_max_samples, len(refs))))
            else:
                preview_indices = [
                    i for i, step_output in enumerate(self.test_step_outputs)
                    if int(step_output.get("idx", -1)) in self._debug_selected_indices
                ]
            self._debug_log(
                "metrics_text_norm",
                simplifier=simplifier_name,
                num_samples=len(refs),
            )
            self._debug_log(
                "metrics_text_norm_preview",
                samples=[{
                    "idx": int(self.test_step_outputs[i].get("idx", i)),
                    "raw_pred": str(raw_preds[i])[:120],
                    "raw_ref": str(raw_refs[i])[:120],
                    "norm_pred": str(preds[i])[:120],
                    "norm_ref": str(refs[i])[:120],
                } for i in preview_indices],
            )
            self._debug_log(
                "test_step_output_normalized",
                samples=[{
                    "idx": int(self.test_step_outputs[i].get("idx", i)),
                    "pred_preview": str(preds[i])[:160],
                    "target_preview": str(refs[i])[:160],
                    "has_keywords": len(keywords[i]) > 0,
                    "keywords_preview": [
                        {
                            "mention": str(kw.get("mention", ""))[:80],
                            "total_offset": int(kw.get("total_offset", -1)),
                            "end_offset": int(kw.get("end_offset", -1)),
                            "ner_tag": str(kw.get("ner_tag", "UNK")),
                        }
                        for kw in keywords[i][:3]
                    ],
                } for i in preview_indices],
            )

        conditions = self._evaluation_conditions()
        metrics = self._evaluate_test_metrics(preds, refs, keywords, conditions)
        results = self._build_results_dataframe(metrics)
        oracle_detail_df, oracle_summary_df = self._build_oracle_nbest_tables(
            refs=refs,
            keywords=keywords,
            raw_refs=raw_refs,
            raw_keywords=[[{
                **kw, **({'ner_tag': 'UNK'} if 'ner_tag' not in kw else {})
            } for kw in step_output.get('keywords', [])] for step_output in self.test_step_outputs],
        )
        if self._metrics_output_path:
            metrics_dir = os.path.dirname(self._metrics_output_path)
            if metrics_dir:
                os.makedirs(metrics_dir, exist_ok=True)
            results.to_csv(self._metrics_output_path, encoding="utf-8-sig")
            if self._debug_enabled():
                self._debug_log(
                    "metrics_written",
                    path=self._metrics_output_path,
                    columns=list(results.columns),
                    rows=int(len(results)),
                )
        if oracle_detail_df is not None and self._oracle_nbest_detail_path:
            detail_dir = os.path.dirname(self._oracle_nbest_detail_path)
            if detail_dir:
                os.makedirs(detail_dir, exist_ok=True)
            oracle_detail_df.to_csv(self._oracle_nbest_detail_path, index=False, encoding="utf-8-sig")
            if self._debug_enabled():
                self._debug_log(
                    "oracle_nbest_detail_written",
                    path=self._oracle_nbest_detail_path,
                    rows=int(len(oracle_detail_df)),
                    columns=list(oracle_detail_df.columns),
                )
        if oracle_summary_df is not None and self._oracle_nbest_summary_path:
            summary_dir = os.path.dirname(self._oracle_nbest_summary_path)
            if summary_dir:
                os.makedirs(summary_dir, exist_ok=True)
            oracle_summary_df.to_csv(self._oracle_nbest_summary_path, index=False, encoding="utf-8-sig")
            if self._debug_enabled():
                self._debug_log(
                    "oracle_nbest_summary_written",
                    path=self._oracle_nbest_summary_path,
                    rows=int(len(oracle_summary_df)),
                    columns=list(oracle_summary_df.columns),
                    aggregate=oracle_summary_df.tail(1).to_dict(orient="records")[0] if len(oracle_summary_df) > 0 else {},
                )
        if self._post_test_progress_bar is not None:
            self._post_test_progress_bar.set_description("Post-test eval write")
            self._post_test_progress_bar.update(1)
            self._post_test_progress_bar.set_description("Post-test eval done")
            self._post_test_progress_bar.close()
            self._post_test_progress_bar = None
        print(results)


class Flexlist(list):
    def __getitem__(self, keys):
        if isinstance(keys, (int, slice)): return list.__getitem__(self, keys)
        return [self[int(k)] for k in keys]


class DatabaseLite:
    def __init__(
        self,
        dataset: str,
        split: str,
        root: str,
        kw_type: str,
        keywords_per_group: int = 100
    ):
        # check dataset
        assert dataset in ['aishell', 'acl', 'shuili'], f'DatabaseLite: the dataset is not supported, got {dataset}'
        # check split
        assert split in ['dev', 'test'], f'DatabaseLite: the split is not supported, got {split} for {dataset}'
        # check keyword type
        assert kw_type in ['tts', 'natural'], f'DatabaseLite: the keyword type is not supported, got {kw_type} for {dataset}'

        # get database
        if dataset in ['aishell', 'shuili']:
            self.database = AishellHotwordDataset(
                root = root,
                split = split,
                r1_only = False,
                hotwords_per_group = keywords_per_group,
                kw_type = kw_type
            ).database
        elif dataset == 'acl':
            self.database = ACL6060KeywordDataset(
                root = root,
                split = split,
                keywords_per_group = keywords_per_group,
                kw_type = kw_type
            ).database

        # set group size
        self.keywords_per_group = keywords_per_group

        # set number of keywords
        self.num_keywords = sum([len(group['keywords']) for group in self.database])

    def hidden_dim(self) -> Optional[int]:
        for group in self.database:
            for hs in group.get('hidden_states', []):
                if hs is not None:
                    return int(hs.size(-1))
        return None
        
    def __len__(
        self
    ) -> int:
        return self.num_keywords
    
    def __getitem__(
        self,
        idx: int
    ) -> dict:
        keyword = {
            'keyword': self.database[idx // self.keywords_per_group]['keywords'][idx % self.keywords_per_group],
            'hidden_states': self.database[idx // self.keywords_per_group]['hidden_states'][idx % self.keywords_per_group]
        }
        return keyword
    
    def num_groups(
        self
    ) -> int:
        return len(self.database)
    
    def group(
        self,
        idx: int,
        device: str = 'cpu',
        load_hs: bool = True
    ) -> dict:        
        kw_group = {
            'keywords' : self.database[idx]['keywords'],
            'hidden_states' : [hs.to(device) for hs in self.database[idx]['hidden_states']] if load_hs else None
        }
        return kw_group
