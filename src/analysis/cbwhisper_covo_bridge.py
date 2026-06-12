#!/usr/bin/env python
"""Bridge CB-Whisper evidence JSONL to covo/Qwen text-rewrite inference."""

from __future__ import annotations

import argparse
import json
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
    "热词证据来自 CB-Whisper/KWS，不是参考答案；它用于保护专名/领域词，"
    "但不能因为热词分数高就强行插入无上下文支持的词。"
    "如果证据不足，保持 ASR top-1 不变。"
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


def build_context_hotwords(input_block: Dict[str, Any], args: argparse.Namespace) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    for rank, item in enumerate(list(input_block.get("hotwords", []) or [])[: int(args.max_hotwords)], 1):
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
    lines = [INSTRUCTION]
    asr_top1 = str(input_block.get("asr_top1", "")).strip()
    lines.append(f"ASR top-1: {asr_top1}")

    nbest = list(input_block.get("nbest", []) or [])[: max(1, int(args.max_nbest))]
    if nbest:
        lines.append("N-best:")
        for idx, hyp in enumerate(nbest, 1):
            lines.append(f"{idx}. {hyp}")

    if bool(args.include_pinyin):
        pinyin = list(input_block.get("nbest_pinyin", []) or [])[: max(1, int(args.max_pinyin))]
        if pinyin:
            lines.append("Pinyin:")
            for idx, item in enumerate(pinyin, 1):
                lines.append(f"{idx}. {item}")

    hotword_lines = format_context_hotwords(build_context_hotwords(input_block, args))
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
    guard_script = covo_dir / "scripts" / "guard_text_predictions.py"
    if not infer_script.exists():
        raise FileNotFoundError(f"missing covo infer script: {infer_script}")
    message_path = Path(args.output).resolve()
    prediction_path = Path(args.prediction_output).resolve()
    eval_input_path = prediction_path

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

    if args.guard:
        if not guard_script.exists():
            raise FileNotFoundError(f"missing covo guard script: {guard_script}")
        guard_output = (
            Path(args.guard_output).resolve()
            if args.guard_output
            else prediction_path.with_name(prediction_path.stem + ".guarded" + prediction_path.suffix)
        )
        guard_command = [
            args.python,
            str(guard_script),
            "--input",
            str(prediction_path),
            "--output",
            str(guard_output),
            "--prediction-field",
            "prediction",
            "--baseline-field",
            "input.asr_top1",
            "--max-distance",
            str(args.guard_max_distance),
            "--max-distance-ratio",
            str(args.guard_max_distance_ratio),
            "--min-output-length-ratio",
            str(args.guard_min_output_length_ratio),
        ]
        print("+ " + " ".join(guard_command), flush=True)
        subprocess.run(guard_command, cwd=str(covo_dir), check=True)
        eval_input_path = guard_output

    if args.evaluate and eval_script.exists():
        eval_command = [
            args.python,
            str(eval_script),
            "--input",
            str(eval_input_path),
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
    parser.add_argument("--include-pinyin", action="store_true")
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
    run.add_argument("--guard", action="store_true", help="Apply covo distance guard before evaluation")
    run.add_argument("--guard-output", default="", help="Optional guarded prediction JSONL path")
    run.add_argument("--guard-max-distance", type=int, default=1)
    run.add_argument("--guard-max-distance-ratio", type=float, default=0.45)
    run.add_argument("--guard-min-output-length-ratio", type=float, default=0.55)
    run.add_argument("--evaluate", action="store_true")
    run.set_defaults(func=cmd_run)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
