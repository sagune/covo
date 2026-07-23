#!/usr/bin/env python3
"""Train a lightweight hotword-aware adapter on frozen SenseVoice CTC logits."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import wave
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from pypinyin import Style, lazy_pinyin

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from analysis.extract_sensevoice_hidden_states import encode_sensevoice, load_fbank  # noqa: E402
from model.sensevoice_context_adapter import (  # noqa: E402
    SenseVoiceContextAdapter,
    SenseVoicePhraseContextAdapter,
    load_context_adapter,
    pinyin_bucket_ids,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--initial-adapter", type=Path)
    parser.add_argument("--model", default="iic/SenseVoiceSmall")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--language", default="zh")
    parser.add_argument("--textnorm", default="withitn")
    parser.add_argument("--projection-size", type=int, default=128)
    parser.add_argument("--adapter-type", choices=("frame", "phrase"), default="phrase")
    parser.add_argument("--context-layers", type=int, default=2)
    parser.add_argument("--context-heads", type=int, default=4)
    parser.add_argument("--monotonic-phrase-activation", action="store_true")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--distractors", type=int, default=7)
    parser.add_argument("--phonetic-distractors", type=int, default=3)
    parser.add_argument("--position-loss-weight", type=float, default=0.25)
    parser.add_argument("--phrase-loss-weight", type=float, default=0.15)
    parser.add_argument("--hotword-ctc-loss-weight", type=float, default=0.30)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--max-grad-norm", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=1000)
    return parser.parse_args()


def phonetic_key(text: str) -> tuple[str, ...]:
    return tuple(lazy_pinyin(text, style=Style.TONE3, neutral_tone_with_five=True, errors="ignore"))


def build_distractor_indexes(vocabulary: list[str]) -> tuple[dict, dict]:
    phonetic_index: dict[tuple[str, ...], list[str]] = defaultdict(list)
    character_index: dict[str, list[str]] = defaultdict(list)
    for word in vocabulary:
        phonetic_index[phonetic_key(word)].append(word)
        for character in set(word):
            character_index[character].append(word)
    return phonetic_index, character_index


def read_training_rows(data_root: Path, audio_root: Path) -> tuple[list[dict], list[str]]:
    transcripts = {}
    with (data_root / "text").open(encoding="utf-8-sig") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t", 1)
            if len(parts) == 2:
                transcripts[parts[0].strip()] = parts[1].strip()
    positives: dict[str, list[dict]] = defaultdict(list)
    with (data_root / "aligned.txt").open(encoding="utf-8-sig") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 4 and parts[0].strip() and parts[1].strip():
                positives[parts[1].strip()].append(
                    {"word": parts[0].strip(), "start": float(parts[2]), "end": float(parts[3])}
                )
    vocabulary = [line.strip().split("\t")[0] for line in (data_root / "hotword.txt").open(encoding="utf-8-sig") if line.strip()]
    audio_files = {path.stem: path for path in audio_root.glob("**/*.wav")}
    rows = []
    for utterance_id in (data_root / "uttid").read_text(encoding="utf-8-sig").splitlines():
        utterance_id = utterance_id.strip().split()[0]
        if utterance_id in transcripts and utterance_id in positives and utterance_id in audio_files:
            rows.append(
                {
                    "id": utterance_id,
                    "text": transcripts[utterance_id],
                    "positive_hotwords": list(dict.fromkeys(item["word"] for item in positives[utterance_id])),
                    "positive_spans": positives[utterance_id],
                    "audio": audio_files[utterance_id],
                }
            )
    return rows, list(dict.fromkeys(vocabulary))


def context_for_row(
    row: dict,
    vocabulary: list[str],
    phonetic_index: dict,
    character_index: dict,
    distractor_count: int,
    phonetic_distractor_count: int,
    rng: random.Random,
) -> tuple[list[str], list[float]]:
    context = list(row["positive_hotwords"])
    text = row["text"]
    hard_candidates = []
    for positive in row["positive_hotwords"]:
        hard_candidates.extend(phonetic_index.get(phonetic_key(positive), []))
        for character in set(positive):
            hard_candidates.extend(character_index.get(character, []))
    hard_candidates = list(dict.fromkeys(hard_candidates))
    rng.shuffle(hard_candidates)
    hard_limit = min(distractor_count, max(0, phonetic_distractor_count))
    for candidate in hard_candidates:
        if len(context) >= len(row["positive_hotwords"]) + hard_limit:
            break
        if candidate not in context and candidate not in text:
            context.append(candidate)
    attempts = 0
    while len(context) < len(row["positive_hotwords"]) + distractor_count and attempts < distractor_count * 20 + 20:
        attempts += 1
        candidate = vocabulary[rng.randrange(len(vocabulary))]
        if candidate not in context and candidate not in text:
            context.append(candidate)
    rng.shuffle(context)
    positives = set(row["positive_hotwords"])
    return context, [1.0 if word in positives else 0.0 for word in context]


def position_targets(row: dict, frame_count: int, device: str) -> torch.Tensor:
    with wave.open(str(row["audio"]), "rb") as audio_file:
        duration = float(audio_file.getnframes()) / max(float(audio_file.getframerate()), 1.0)
    target = torch.zeros((1, frame_count), dtype=torch.float32, device=device)
    if duration <= 0.0:
        return target
    for span in row["positive_spans"]:
        start = max(0, min(frame_count - 1, int(float(span["start"]) / duration * frame_count)))
        end = max(start + 1, min(frame_count, int(float(span["end"]) / duration * frame_count + 0.999)))
        target[:, max(0, start - 1):min(frame_count, end + 1)] = 1.0
    return target


def phrase_position_targets(
    row: dict,
    context: list[str],
    frame_count: int,
    device: str,
) -> torch.Tensor:
    with wave.open(str(row["audio"]), "rb") as audio_file:
        duration = float(audio_file.getnframes()) / max(float(audio_file.getframerate()), 1.0)
    target = torch.zeros((1, frame_count, len(context)), dtype=torch.float32, device=device)
    if duration <= 0.0:
        return target
    context_index = {word: index for index, word in enumerate(context)}
    for span in row["positive_spans"]:
        phrase_index = context_index.get(span["word"])
        if phrase_index is None:
            continue
        start = max(0, min(frame_count - 1, int(float(span["start"]) / duration * frame_count)))
        end = max(start + 1, min(frame_count, int(float(span["end"]) / duration * frame_count + 0.999)))
        target[:, max(0, start - 1):min(frame_count, end + 1), phrase_index] = 1.0
    return target


def localized_hotword_ctc_loss(
    row: dict,
    adapted_log_probs: torch.Tensor,
    tokenizer,
    blank_id: int,
    device: str,
) -> torch.Tensor:
    """Apply CTC directly to each aligned positive phrase and its local frames."""
    with wave.open(str(row["audio"]), "rb") as audio_file:
        duration = float(audio_file.getnframes()) / max(float(audio_file.getframerate()), 1.0)
    if duration <= 0.0:
        return adapted_log_probs.new_zeros(())
    frame_count = int(adapted_log_probs.size(1))
    vocab_size = int(adapted_log_probs.size(-1))
    losses = []
    for span in row["positive_spans"]:
        target = [
            int(token)
            for token in tokenizer.encode(span["word"])
            if int(token) != blank_id and 0 <= int(token) < vocab_size
        ]
        if not target:
            continue
        start = max(0, min(frame_count - 1, int(float(span["start"]) / duration * frame_count)))
        end = max(start + 1, min(frame_count, int(float(span["end"]) / duration * frame_count + 0.999)))
        margin = max(2, len(target) // 2)
        start = max(0, start - margin)
        end = min(frame_count, end + margin)
        if end - start < len(target):
            continue
        segment = adapted_log_probs[:, start:end, :]
        local_loss = F.ctc_loss(
                segment.transpose(0, 1),
                torch.tensor(target, dtype=torch.long, device=device),
                input_lengths=torch.tensor([segment.size(1)], dtype=torch.long),
                target_lengths=torch.tensor([len(target)], dtype=torch.long),
                blank=blank_id,
                reduction="mean",
                zero_infinity=True,
            )
        # Timestamp boundaries can be tighter than the CTC emission support.
        # Log compression keeps those hard spans useful without dominating a batch.
        losses.append(torch.log1p(local_loss))
    return torch.stack(losses).mean() if losses else adapted_log_probs.new_zeros(())


def save_checkpoint(path: Path, adapter: torch.nn.Module, metadata: dict, adapter_type: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "adapter_type": adapter_type,
            "config": adapter.config(),
            "state_dict": {key: value.detach().cpu() for key, value in adapter.state_dict().items()},
            "metadata": metadata,
        },
        path,
    )


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    rows, vocabulary = read_training_rows(args.data_root, args.audio_root)
    if not rows:
        raise RuntimeError("no aligned training rows were found")
    phonetic_index, character_index = build_distractor_indexes(vocabulary)

    from funasr import AutoModel

    auto_model = AutoModel(model=args.model, device=args.device, disable_update=True)
    model = auto_model.model
    tokenizer = auto_model.kwargs["tokenizer"]
    model.eval()
    model.requires_grad_(False)
    hidden_size = int(model.encoder_output_size)
    if args.initial_adapter is not None:
        adapter = load_context_adapter(str(args.initial_adapter))
        loaded_type = "phrase" if bool(getattr(adapter, "expects_phrases", False)) else "frame"
        if loaded_type != args.adapter_type:
            raise ValueError(
                f"initial adapter type {loaded_type!r} does not match --adapter-type {args.adapter_type!r}"
            )
        if (
            args.adapter_type == "phrase"
            and args.monotonic_phrase_activation
            and not bool(getattr(adapter, "monotonic_phrase_activation", False))
        ):
            upgraded = SenseVoicePhraseContextAdapter(
                **{**adapter.config(), "monotonic_phrase_activation": True}
            )
            upgraded.load_state_dict(adapter.state_dict(), strict=True)
            adapter = upgraded
        adapter = adapter.to(args.device)
    elif args.adapter_type == "phrase":
        adapter = SenseVoicePhraseContextAdapter(
            hidden_size=hidden_size,
            projection_size=args.projection_size,
            num_heads=args.context_heads,
            num_context_layers=args.context_layers,
            monotonic_phrase_activation=args.monotonic_phrase_activation,
        ).to(args.device)
    else:
        adapter = SenseVoiceContextAdapter(
            hidden_size=hidden_size,
            projection_size=args.projection_size,
        ).to(args.device)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    accumulation = max(1, int(args.gradient_accumulation))
    optimizer.zero_grad(set_to_none=True)
    micro_step = optimizer_step = skipped = 0
    loss_sum = 0.0
    started = time.time()
    stop = False

    print(json.dumps({"event": "train_start", "rows": len(rows), "vocabulary": len(vocabulary), "args": vars(args)}, default=str, ensure_ascii=False), flush=True)
    for epoch in range(max(1, args.epochs)):
        rng.shuffle(rows)
        for row in rows:
            context, phrase_labels = context_for_row(
                row,
                vocabulary,
                phonetic_index,
                character_index,
                args.distractors,
                args.phonetic_distractors,
                rng,
            )
            token_ids = []
            context_phrases = []
            context_pinyin = []
            for hotword in context:
                phrase = [int(token) for token in tokenizer.encode(hotword)]
                if not phrase:
                    continue
                context_phrases.append(phrase)
                context_pinyin.append(pinyin_bucket_ids(hotword, len(phrase)))
                token_ids.extend(phrase)
            target = [
                int(token)
                for token in tokenizer.encode(row["text"])
                if int(token) != int(model.blank_id) and 0 <= int(token) < int(model.ctc.ctc_lo.out_features)
            ]
            if not target or not token_ids:
                skipped += 1
                continue
            try:
                speech, speech_lengths = load_fbank(auto_model, str(row["audio"]), device=args.device)
                with torch.no_grad():
                    encoder_out, encoder_out_lens = encode_sensevoice(
                        auto_model,
                        speech,
                        speech_lengths,
                        language=args.language,
                        textnorm=args.textnorm,
                    )
                    valid_len = int(encoder_out_lens[0].item())
                    encoder_hidden = encoder_out[:, 4:valid_len, :].float()
                    base_log_probs = model.ctc.log_softmax(encoder_out)[:, 4:valid_len, :].float()
                if encoder_hidden.size(1) < len(target):
                    skipped += 1
                    continue
                if args.adapter_type == "phrase":
                    adapted, position_logits, phrase_logits = adapter(
                        encoder_hidden=encoder_hidden,
                        base_log_probs=base_log_probs,
                        ctc_token_weights=model.ctc.ctc_lo.weight.detach(),
                        ctc_token_bias=model.ctc.ctc_lo.bias.detach(),
                        context_phrases=context_phrases,
                        context_pinyin_ids=context_pinyin,
                        return_auxiliary=True,
                    )
                else:
                    adapted, position_logits = adapter(
                        encoder_hidden=encoder_hidden,
                        base_log_probs=base_log_probs,
                        ctc_token_weights=model.ctc.ctc_lo.weight.detach(),
                        context_token_ids=token_ids,
                        return_position_logits=True,
                    )
                target_tensor = torch.tensor(target, dtype=torch.long, device=args.device)
                asr_loss = F.ctc_loss(
                    adapted.transpose(0, 1),
                    target_tensor,
                    input_lengths=torch.tensor([adapted.size(1)], dtype=torch.long),
                    target_lengths=torch.tensor([len(target)], dtype=torch.long),
                    blank=int(model.blank_id),
                    reduction="mean",
                    zero_infinity=True,
                )
                frame_targets = (
                    phrase_position_targets(row, context, adapted.size(1), args.device)
                    if args.adapter_type == "phrase"
                    else position_targets(row, adapted.size(1), args.device)
                )
                positive_weight = torch.tensor(
                    max(1.0, float(frame_targets.numel() - frame_targets.sum()) / frame_targets.sum().clamp_min(1.0)),
                    device=args.device,
                ).clamp(max=20.0)
                position_loss = F.binary_cross_entropy_with_logits(
                    position_logits,
                    frame_targets,
                    pos_weight=positive_weight,
                )
                phrase_loss = adapted.new_zeros(())
                hotword_ctc_loss = adapted.new_zeros(())
                if args.adapter_type == "phrase":
                    phrase_target = torch.tensor(
                        phrase_labels,
                        dtype=phrase_logits.dtype,
                        device=args.device,
                    ).unsqueeze(0)
                    phrase_positive_weight = (
                        (phrase_target.numel() - phrase_target.sum()) / phrase_target.sum().clamp_min(1.0)
                    ).clamp(min=1.0, max=20.0)
                    phrase_loss = F.binary_cross_entropy_with_logits(
                        phrase_logits,
                        phrase_target,
                        pos_weight=phrase_positive_weight,
                    )
                    hotword_ctc_loss = localized_hotword_ctc_loss(
                        row,
                        adapted,
                        tokenizer,
                        int(model.blank_id),
                        args.device,
                    )
                loss = (
                    asr_loss
                    + float(args.position_loss_weight) * position_loss
                    + float(args.phrase_loss_weight) * phrase_loss
                    + float(args.hotword_ctc_loss_weight) * hotword_ctc_loss
                )
                if not torch.isfinite(loss):
                    skipped += 1
                    continue
                (loss / accumulation).backward()
            except Exception as exc:
                skipped += 1
                print(json.dumps({"event": "sample_error", "id": row["id"], "error": str(exc)}, ensure_ascii=False), flush=True)
                continue

            micro_step += 1
            loss_sum += float(loss.detach())
            if micro_step % accumulation != 0:
                continue
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_step += 1
            if optimizer_step % args.log_every == 0:
                elapsed = time.time() - started
                print(json.dumps({"event": "train_progress", "epoch": epoch, "step": optimizer_step, "micro_step": micro_step, "mean_loss": loss_sum / micro_step, "asr_loss": float(asr_loss.detach()), "position_loss": float(position_loss.detach()), "phrase_loss": float(phrase_loss.detach()), "hotword_ctc_loss": float(hotword_ctc_loss.detach()), "steps_per_second": optimizer_step / elapsed, "skipped": skipped}, ensure_ascii=False), flush=True)
            metadata = {"model": args.model, "epoch": epoch, "step": optimizer_step, "micro_step": micro_step, "mean_loss": loss_sum / micro_step, "seed": args.seed}
            if args.save_every > 0 and optimizer_step % args.save_every == 0:
                save_checkpoint(args.output.with_name(f"{args.output.stem}-step{optimizer_step}{args.output.suffix}"), adapter, metadata, args.adapter_type)
            if args.max_steps is not None and optimizer_step >= args.max_steps:
                stop = True
                break
        if stop:
            break

    metadata = {"model": args.model, "step": optimizer_step, "micro_step": micro_step, "mean_loss": loss_sum / max(micro_step, 1), "skipped": skipped, "seconds": time.time() - started, "seed": args.seed}
    save_checkpoint(args.output, adapter, metadata, args.adapter_type)
    print(json.dumps({"event": "train_done", "output": str(args.output), **metadata}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
