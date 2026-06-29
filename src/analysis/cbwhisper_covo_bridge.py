#!/usr/bin/env python
"""Bridge CB-Whisper evidence JSONL to covo/Qwen text-rewrite inference."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List


SYSTEM_MESSAGE = (
    "你是一个保守的中文 ASR 后纠错器。根据 CB-Whisper 输出、N-best 候选、"
    "KWS 热词、拼音和候选分数，只做必要的最小修改。必须只输出一个合法 JSON 对象，"
    "格式为 {\"text\":\"...\"}。不要输出解释、推理过程、Markdown 或额外字段。"
)

INSTRUCTION = (
    "任务：融合 CB-Whisper 的最终输出、N-best 候选和 KWS 热词证据进行中文 ASR 后纠错。"
    "优先保留 CB-Whisper 输出；只有当 N-best、拼音或高置信热词共同支持时才修改。"
    "N-best 中 trusted_scored 候选比 supplemental_unscored 候选更可信；"
    "补充候选只能作为纠错线索，不能单独推翻 ASR top-1 或高分 trusted_scored 候选。"
    "如果 ASR top-1 或 trusted_scored 候选已经包含 prompt 热词，不要把该热词改成同音常见词，"
    "除非多个可信候选和上下文都明确支持替换。"
    "热词证据来自 CB-Whisper/KWS，不是参考答案；它用于保护专名/领域词，"
    "但不能因为热词分数高就强行插入无上下文支持的词。"
    "如果证据不足，保持 ASR top-1 不变。"
)

SELECTOR_INSTRUCTION = (
    "任务：从 CB-Whisper 的 ASR top-1、N-best 候选、拼音和 KWS 热词证据中选择最可信的完整中文转写。"
    "不要默认保留 ASR top-1；如果 N-best 中非 top-1 候选在语义、拼音、上下文或热词上更合理，"
    "应直接采用该候选并只做必要的小修正。"
    "N-best 中 trusted_scored 候选通常可信，但 score 不是唯一依据；"
    "要比较候选是否完整、是否多字漏字、是否包含无关热词、是否符合上下文。"
    "热词证据来自 CB-Whisper/KWS，不是参考答案；不要因为热词分数高就强行插入无上下文支持的词。"
    "输出应尽量等于某个高质量 N-best 候选；只有候选存在明显局部错字时才做小幅修正。"
)

PROTECTED_HOTWORD_INSTRUCTION = (
    "受保护热词是已经出现在 ASR top-1 或 trusted_scored 候选中的 prompt 热词。"
    "最终输出必须逐字保留受保护热词；不要把它改成同音、近音、繁简异体或更常见写法。"
    "如果受保护热词附近还有其他明显 ASR 错误，只能修改热词之外的字符。"
)


def read_jsonl(path: str | Path) -> Iterable[Dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_no}: expected JSON object")
            yield value


def write_jsonl(path: str | Path, records: Iterable[Dict[str, Any]]) -> int:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def nested_get(record: Dict[str, Any], dotted: str, default: Any = "") -> Any:
    value: Any = record
    for part in dotted.split("."):
        if not isinstance(value, dict) or part not in value:
            return default
        value = value[part]
    return value


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


_PUNCT_RE = re.compile(r"[\s,，。.!！？?；;：:“”\"'‘’、（）()\[\]【】《》<>-]+")


def normalize_text(text: Any) -> str:
    return _PUNCT_RE.sub("", str(text or "").strip())


def _candidate_score_index(candidates: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    indexed: Dict[str, Dict[str, Any]] = {}
    for item in candidates:
        text = str(item.get("text", "")).strip()
        key = normalize_text(text)
        if key and key not in indexed:
            indexed[key] = item
    return indexed


def build_context_hotwords(input_block: Dict[str, Any], args: argparse.Namespace) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    source_mode = str(getattr(args, "hotword_source", "prompt")).strip().lower()
    use_kws = source_mode in {"all", "kws"}
    use_prompt = source_mode in {"all", "prompt"}
    for rank, item in enumerate(list(input_block.get("hotwords", []) or [])[: int(args.max_hotwords)], 1):
        if not use_kws:
            continue
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        row = merged.setdefault(
            text,
            {
                "text": text,
                "score": 0.0,
                "prompt_weight": 0.0,
                "sources": [],
                "kws_rank": rank,
                "in_prompt": False,
            },
        )
        row["score"] = max(float(row.get("score", 0.0)), _as_float(item.get("score", 0.0)))
        if "kws" not in row["sources"]:
            row["sources"].append("kws")

    for rank, item in enumerate(list(input_block.get("prompt_hotwords", []) or [])[: int(args.max_prompt_hotwords)], 1):
        if not use_prompt:
            continue
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        row = merged.setdefault(
            text,
            {
                "text": text,
                "score": 0.0,
                "prompt_weight": 0.0,
                "sources": [],
                "kws_rank": 999999,
                "in_prompt": False,
            },
        )
        row["prompt_weight"] = max(float(row.get("prompt_weight", 0.0)), _as_float(item.get("weight", item.get("score", 0.0))))
        row["prompt_rank"] = rank
        row["in_prompt"] = True
        if "prompt" not in row["sources"]:
            row["sources"].append("prompt")

    rows = list(merged.values())
    rows.sort(
        key=lambda row: (
            -int(bool(row.get("in_prompt", False))),
            -float(row.get("score", 0.0)),
            -float(row.get("prompt_weight", 0.0)),
            int(row.get("kws_rank", 999999)),
            -len(str(row.get("text", ""))),
            str(row.get("text", "")),
        )
    )
    return rows[: int(args.max_hotwords)]


def _matched_hotword_texts(text: str, rows: List[Dict[str, Any]], prompt_only: bool = False) -> List[str]:
    normalized = normalize_text(text)
    matched = []
    for item in rows:
        if prompt_only and not bool(item.get("in_prompt", False)):
            continue
        keyword = str(item.get("text", "")).strip()
        if keyword and normalize_text(keyword) in normalized:
            matched.append(keyword)
    return matched


def build_protected_hotwords(input_block: Dict[str, Any], rows: List[Dict[str, Any]]) -> List[str]:
    cbw = input_block.get("cbwhisper", {}) or {}
    candidates = list(cbw.get("candidates", []) or [])
    trusted_text = normalize_text(input_block.get("asr_top1", ""))
    scored_support = set()
    for item in candidates:
        trusted_text += normalize_text(item.get("text", ""))
        exact_stats = item.get("exact_stats", {}) or {}
        for keyword in list(exact_stats.get("matched_keywords", []) or []) + list(item.get("consensus_keywords", []) or []):
            normalized_keyword = normalize_text(keyword)
            if normalized_keyword:
                scored_support.add(normalized_keyword)

    protected = []
    seen = set()
    for item in rows:
        if not bool(item.get("in_prompt", False)):
            continue
        keyword = str(item.get("text", "")).strip()
        normalized = normalize_text(keyword)
        if normalized and normalized in trusted_text and normalized in scored_support and normalized not in seen:
            protected.append(keyword)
            seen.add(normalized)
    return protected


def format_context_hotwords(rows: List[Dict[str, Any]]) -> List[str]:
    output = []
    for item in rows:
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        score = _as_float(item.get("score", 0.0))
        prompt_weight = _as_float(item.get("prompt_weight", 0.0))
        sources = ",".join(str(source) for source in item.get("sources", []) or [])
        in_prompt = "yes" if item.get("in_prompt", False) else "no"
        output.append(
            f"- {text} kws_score={score:.4f} prompt_weight={prompt_weight:.4f} "
            f"in_prompt={in_prompt} source={sources}"
        )
    return output


def format_nbest(
    nbest: List[str],
    input_block: Dict[str, Any],
    hotword_rows: List[Dict[str, Any]],
    max_items: int,
) -> List[str]:
    cbw = input_block.get("cbwhisper", {}) or {}
    scored_candidates = list(cbw.get("candidates", []) or [])
    scored_index = _candidate_score_index(scored_candidates)
    output = []
    seen = set()
    for idx, hyp in enumerate(nbest[:max_items], 1):
        text = str(hyp).strip()
        if not text:
            continue
        normalized = normalize_text(text)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        scored = scored_index.get(normalized)
        if scored is None:
            source = "supplemental_unscored"
            score_bits = "score=NA"
        else:
            source = "trusted_scored"
            score_bits = (
                f"score_rank={int(scored.get('rank', idx))} "
                f"total={float(scored.get('total_score', 0.0)):.4f} "
                f"asr={float(scored.get('asr_score', 0.0)):.4f} "
                f"exact={float(scored.get('exact_score', 0.0)):.4f} "
                f"phon={float(scored.get('phonetic_score', 0.0)):.4f}"
            )
        prompt_hits = _matched_hotword_texts(text, hotword_rows, prompt_only=True)
        kws_hits = _matched_hotword_texts(text, hotword_rows, prompt_only=False)
        prompt_msg = ",".join(prompt_hits) if prompt_hits else "none"
        kws_msg = ",".join(kws_hits) if kws_hits else "none"
        output.append(
            f"{idx}. {text} | source={source} {score_bits} "
            f"keeps_prompt_hotwords={prompt_msg} keeps_context_hotwords={kws_msg}"
        )
    return output


def format_candidates(candidates: List[Dict[str, Any]], max_items: int) -> List[str]:
    output = []
    for item in candidates[:max_items]:
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        rank = int(item.get("rank", len(output) + 1))
        total = float(item.get("total_score", 0.0))
        exact = float(item.get("exact_score", 0.0))
        phon = float(item.get("phonetic_score", 0.0))
        asr = float(item.get("asr_score", 0.0))
        output.append(
            f"{rank}. {text} | total={total:.4f} asr={asr:.4f} exact={exact:.4f} phon={phon:.4f}"
        )
    return output


def build_user_prompt(record: Dict[str, Any], args: argparse.Namespace) -> str:
    input_block = record.get("input", {}) or {}
    prompt_mode = str(getattr(args, "prompt_mode", "correction")).strip().lower()
    lines = [SELECTOR_INSTRUCTION if prompt_mode == "selector" else INSTRUCTION]
    if bool(getattr(args, "protect_supported_hotwords", False)):
        lines.append(PROTECTED_HOTWORD_INSTRUCTION)
    asr_top1 = str(input_block.get("asr_top1", "")).strip()
    lines.append(f"ASR top-1: {asr_top1}")
    hotword_rows = build_context_hotwords(input_block, args)
    protected_hotwords = build_protected_hotwords(input_block, hotword_rows)
    if bool(getattr(args, "protect_supported_hotwords", False)) and protected_hotwords:
        lines.append("Protected hotwords that must be preserved exactly: " + ",".join(protected_hotwords))
    if hotword_rows:
        asr_prompt_hits = _matched_hotword_texts(asr_top1, hotword_rows, prompt_only=True)
        hit_msg = ",".join(asr_prompt_hits) if asr_prompt_hits else "none"
        lines.append(f"ASR top-1 keeps prompt hotwords: {hit_msg}")

    nbest = list(input_block.get("nbest", []) or [])[: max(1, int(args.max_nbest))]
    if nbest:
        lines.append(
            "N-best with reliability labels "
            "(trusted_scored=CB-Whisper scored beam, supplemental_unscored=extra diversity candidate):"
        )
        lines.extend(format_nbest(nbest, input_block, hotword_rows, max_items=max(1, int(args.max_nbest))))

    if bool(args.include_pinyin):
        pinyin = list(input_block.get("nbest_pinyin", []) or [])[: max(1, int(args.max_pinyin))]
        if pinyin:
            lines.append("Pinyin:")
            for idx, item in enumerate(pinyin, 1):
                lines.append(f"{idx}. {item}")

    hotword_lines = format_context_hotwords(hotword_rows)
    if hotword_lines:
        lines.append("CB-Whisper hotword evidence (predicted, not gold):")
        lines.extend(hotword_lines)

    cbw = input_block.get("cbwhisper", {}) or {}
    candidates = list(cbw.get("candidates", []) or [])
    candidate_lines = format_candidates(candidates, max_items=int(args.max_candidates_with_scores))
    if candidate_lines:
        lines.append("CB-Whisper candidate scores:")
        lines.extend(candidate_lines)

    lines.append('请输出 JSON：{"text":"纠错后的完整句子"}')
    return "\n".join(lines)


def prepare_records(args: argparse.Namespace) -> Iterable[Dict[str, Any]]:
    count = 0
    for record in read_jsonl(args.input):
        reference = str(nested_get(record, args.reference_field, "")).strip()
        input_block = dict(record.get("input", {}) or {})
        if not str(input_block.get("asr_top1", "")).strip():
            asr_top1 = str(record.get("asr_top1", "")).strip()
            if asr_top1:
                input_block["asr_top1"] = asr_top1
        if not input_block.get("nbest") and record.get("nbest"):
            input_block["nbest"] = list(record.get("nbest", []) or [])
        if not input_block.get("nbest") and input_block.get("asr_top1"):
            input_block["nbest"] = [str(input_block.get("asr_top1", "")).strip()]
        input_block["covo_hotwords"] = build_context_hotwords(input_block, args)
        output = {
            "id": str(record.get("id", "")),
            "source": record.get("source", "cbwhisper"),
            "dataset": record.get("dataset", ""),
            "split": record.get("split", ""),
            "reference": reference,
            "input": input_block,
            "messages": [
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": build_user_prompt({**record, "input": input_block}, args)},
            ],
        }
        if reference:
            output["messages"].append(
                {
                    "role": "assistant",
                    "content": json.dumps({"text": reference}, ensure_ascii=False, separators=(",", ":")),
                }
            )
        yield output
        count += 1
        if args.limit and count >= int(args.limit):
            break


def cmd_prepare(args: argparse.Namespace) -> int:
    written = write_jsonl(args.output, prepare_records(args))
    print(json.dumps({"input": args.input, "output": args.output, "written": written}, ensure_ascii=False, indent=2))
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    cmd_prepare(args)
    covo_dir = Path(args.covo_dir).resolve()
    infer_script = covo_dir / "scripts" / "infer_lora_text.py"
    eval_script = covo_dir / "scripts" / "evaluate_correction_jsonl.py"
    if not infer_script.exists():
        raise FileNotFoundError(f"missing covo infer script: {infer_script}")
    message_path = Path(args.output).resolve()
    prediction_path = Path(args.prediction_output).resolve()

    command = [
        args.python,
        str(infer_script),
        "--input",
        str(message_path),
        "--output",
        str(prediction_path),
        "--model-name-or-path",
        str(args.model_name_or_path),
        "--adapter-path",
        str(args.adapter_path),
        "--batch-size",
        str(args.batch_size),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--temperature",
        str(args.temperature),
        "--top-p",
        str(args.top_p),
        "--device",
        str(args.device),
        "--progress-every",
        str(args.progress_every),
    ]
    if args.disable_thinking:
        command.append("--disable-thinking")
    if args.infer_limit:
        command.extend(["--limit", str(args.infer_limit)])
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=str(covo_dir), check=True)

    if args.evaluate and eval_script.exists():
        eval_command = [
            args.python,
            str(eval_script),
            "--input",
            str(prediction_path),
            "--prediction-field",
            "prediction",
            "--reference-field",
            "reference",
            "--baseline-field",
            "input.asr_top1",
        ]
        print("+ " + " ".join(eval_command), flush=True)
        subprocess.run(eval_command, cwd=str(covo_dir), check=True)
    return 0


def add_prepare_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", required=True, help="CB-Whisper evidence JSONL from CBW_EVIDENCE_OUT")
    parser.add_argument("--output", required=True, help="Qwen/covo messages JSONL")
    parser.add_argument("--reference-field", default="reference")
    parser.add_argument("--max-nbest", type=int, default=8)
    parser.add_argument("--max-pinyin", type=int, default=5)
    parser.add_argument("--max-hotwords", type=int, default=12)
    parser.add_argument("--max-prompt-hotwords", type=int, default=6)
    parser.add_argument("--max-candidates-with-scores", type=int, default=8)
    parser.add_argument(
        "--hotword-source",
        choices=["prompt", "kws", "all"],
        default="prompt",
        help="Which CB-Whisper hotwords to inject into covo prompts.",
    )
    parser.add_argument("--include-pinyin", action="store_true")
    parser.add_argument(
        "--prompt-mode",
        choices=["correction", "selector"],
        default="correction",
        help="Prompt style for COVO: conservative correction or n-best selector.",
    )
    parser.add_argument(
        "--protect-supported-hotwords",
        action="store_true",
        help="Add a hard prompt constraint to preserve prompt hotwords already present in ASR/trusted candidates.",
    )
    parser.add_argument("--limit", type=int, default=0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Convert CB-Whisper evidence to Qwen messages")
    add_prepare_args(prepare)
    prepare.set_defaults(func=cmd_prepare)

    run = subparsers.add_parser("run", help="Prepare messages, then run covo LoRA inference")
    add_prepare_args(run)
    run.add_argument("--prediction-output", required=True)
    run.add_argument("--covo-dir", default="/root/autodl-tmp/cbwhisper_covo_migration_20260609_tar_extracted/covo")
    run.add_argument("--python", default=sys.executable)
    run.add_argument("--model-name-or-path", default="/root/autodl-tmp/cbwhisper_covo_migration_20260609_tar_extracted/models/Qwen3.5-4B")
    run.add_argument("--adapter-path", default="/root/autodl-tmp/cbwhisper_covo_migration_20260609_tar_extracted/covo/outputs/qwen35_text_rewrite_hardneg_dropout_lora_2epoch")
    run.add_argument("--batch-size", type=int, default=1)
    run.add_argument("--max-new-tokens", type=int, default=128)
    run.add_argument("--temperature", type=float, default=0.0)
    run.add_argument("--top-p", type=float, default=1.0)
    run.add_argument("--device", default="auto")
    run.add_argument("--progress-every", type=int, default=100)
    run.add_argument("--infer-limit", type=int, default=0)
    run.add_argument("--disable-thinking", action="store_true")
    run.add_argument("--evaluate", action="store_true")
    run.set_defaults(func=cmd_run)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
