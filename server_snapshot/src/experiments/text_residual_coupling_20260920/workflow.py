#!/usr/bin/env python3
"""Independent correction-only experiment. Never dispatches GPU work implicitly."""
import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
DEFAULT_LEGACY = HERE.parent / "staged_coupling_20260917"
SOURCE_FILES = [HERE / name for name in ("connector.py", "inputs.py", "workflow.py", "diagnostics.py")]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def legacy_module(directory):
    spec = importlib.util.spec_from_file_location("legacy_coupling", Path(directory) / "run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def audio_path(p, uid):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", uid):
        raise ValueError("Unsafe sample ID")
    return p.RUN / "audio" / f"{uid}.npy"


def prepare(args):
    """CPU only. Freeze existing split, sources, all selected caches and training budget."""
    import numpy as np
    p = legacy_module(args.legacy)
    old = read(p.RUN / "protocol.json")
    lock = read(p.RUN / "implementation_lock.json")
    if sha(p.HERE / "run.py") != lock["code_sha256"]:
        raise ValueError("Legacy implementation differs from its lock")
    train_path, dev_path = p.RUN / "train.jsonl", p.RUN / "dev.jsonl"
    for name, path in (("train", train_path), ("dev", dev_path)):
        if sha(path) != old["data_hashes"][name]:
            raise ValueError(f"Locked {name} data changed")
    train, dev = p.load(train_path), p.load(dev_path)
    p.unique(train, "id"); p.unique(dev, "id")
    if len(train) != 4096 or len(dev) != 1334:
        raise ValueError("Requires existing 4096 train / complete 1334 dev protocol")
    if {r["id"] for r in train} & {r["id"] for r in dev}:
        raise ValueError("Train/dev ID overlap")
    ev = p.evaluator()
    if {ev.norm(r["reference"]) for r in train} & {ev.norm(r["reference"]) for r in dev}:
        raise ValueError("Train/dev normalized-reference overlap")
    if not all(r["split"] == "train" for r in train) or not all(r["split"] == "dev" for r in dev):
        raise ValueError("Unexpected split label")
    correct = sum(ev.norm(r["reference"]) == ev.norm(r["input"]["asr_top1"]) for r in train)
    if not 0 < correct < len(train):
        raise ValueError("Correction training must contain both correct and erroneous inputs")
    paths = [*SOURCE_FILES, p.HERE / "run.py", train_path, dev_path, p.ALIGNED,
             p.ROOT / ".dsh_checks/restore_eval.py", p.ADAPTER / "adapter_config.json",
             p.ADAPTER / "adapter_model.safetensors", p.BASE / "config.json"]
    paths += [p.BASE / name for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")
              if (p.BASE / name).is_file()]
    if args.architecture == "mlp":
        init = Path(args.mlp_init or (p.HERE / "overnight/Afull_3243")).resolve()
        if (init / "adapter").exists():
            raise ValueError("MLP control must use the original frozen checkpoint-625 adapter")
        paths.append(init / "connector.pt")
    else:
        if args.mlp_init:
            raise ValueError("--mlp-init only applies to MLP control")
        init = None
    sources = {str(path.resolve()): sha(path) for path in paths}
    cache = {}
    for i, row in enumerate(train + dev):
        path = audio_path(p, row["id"])
        value = np.load(path, mmap_mode="r", allow_pickle=False)
        if value.ndim != 2 or value.shape[1] != 512 or not len(value) or not np.isfinite(value).all():
            raise ValueError(f"Invalid audio cache: {path}")
        cache[row["id"]] = {"path": str(path), "sha256": sha(path), "frames": len(value)}
        if (i + 1) % 1000 == 0:
            print(json.dumps({"cache_audited": i + 1}), flush=True)
    config = dict(version=1, architecture=args.architecture, legacy=str(p.HERE), seed=917,
                  train=str(train_path), dev=str(dev_path), train_n=len(train), dev_n=len(dev),
                  correct_train_inputs=correct, incorrect_train_inputs=len(train) - correct,
                  label="fixed previously examined AISHELL protocol dev; diagnostic, not held-out test",
                  data_caveat="Inherited 4096 error-enriched subset, not natural-distribution full train; includes correct inputs",
                  external_context_caveat=old["context_caveat"],
                  base=str(p.BASE), adapter=str(p.ADAPTER), mlp_initialization=str(init) if init else None,
                  hidden_size=read(p.BASE / "config.json").get("hidden_size"), width=256, heads=4, pool=4,
                  steps=512, accumulation=8, lr=5e-5, weight_decay=.01, warmup=32,
                  max_sequence=4096, max_new_tokens=96, trainable="connector only; frozen 9B and checkpoint-625 LoRA",
                  objective="final JSON transcript cross entropy on audio + whitelisted ASR evidence; no restore rules",
                  selection="fixed final step 512; no dev-based checkpoint sweep or early stopping",
                  comparison="same correction data/order/optimizer steps and frozen LLM; MLP has A pretraining, residual starts at text baseline",
                  normalization=old["normalization"], sources=sources,
                  modes=["text", "true", "off", "zero", "mismatch"],
                  zero_semantics="residual: zero encoder frames with branch active; MLP: legacy zero projected prefix",
                  off_semantics="remove acoustic branch; residual prompt/tokens/positions identical to text",
                  mismatch_caveat="legacy duration-ordered derangement; frame counts can differ, not length-controlled",
                  audio_permutation=p.permutation(dev))
    # Some Qwen configs nest language dimensions; match the model's text configuration.
    if config["hidden_size"] is None:
        config["hidden_size"] = read(p.BASE / "config.json")["text_config"]["hidden_size"]
    dest = Path(args.output).resolve()
    dest.mkdir(parents=True, exist_ok=False)
    save(dest / "cache_manifest.json", cache)
    config["cache_manifest_sha256"] = sha(dest / "cache_manifest.json")
    save(dest / "protocol.json", config)
    save(dest / "status.json", dict(status="PREPARED_CPU_ONLY", gpu_smoke="NOT_RUN", training="NOT_STARTED"))
    print(json.dumps(config, ensure_ascii=False), flush=True)


def verify(directory, caches=False):
    directory = Path(directory)
    config = read(directory / "protocol.json")
    for path, expected in config["sources"].items():
        if sha(path) != expected:
            raise ValueError(f"Locked source changed: {path}")
    if sha(directory / "cache_manifest.json") != config["cache_manifest_sha256"]:
        raise ValueError("Cache manifest changed")
    if caches:
        for info in read(directory / "cache_manifest.json").values():
            if sha(info["path"]) != info["sha256"]:
                raise ValueError(f"Audio cache changed: {info['path']}")
    return config, legacy_module(config["legacy"])


@contextmanager
def gpu_guard(p):
    import fcntl
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA device: no model loading or GPU job was started")
    with (p.HERE.parent / "text_residual_coupling_gpu.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        busy = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True)
        if any(line.strip() and line.strip() != str(os.getpid()) for line in busy.splitlines()):
            raise RuntimeError("Another GPU process is active; refusing concurrent experiment")
        yield


class Runtime:
    def __init__(self, config, legacy, training=False, checkpoint=None):
        import torch
        from connector import ConnectorConfig, TextAcousticResidual
        self.c, self.p = config, legacy
        init = Path(config["mlp_initialization"]) if config["architecture"] == "mlp" else None
        self.model, self.tokenizer, old = legacy.load_model("A", init, training)
        if config["architecture"] == "residual":
            self.connector = TextAcousticResidual(ConnectorConfig(config["hidden_size"], width=config["width"],
                                                                  heads=config["heads"], pool=config["pool"])).cuda()
            del old
        else:
            self.connector = old
        if checkpoint:
            meta = read(Path(checkpoint) / "metadata.json")
            if meta["architecture"] != config["architecture"] or meta["protocol_sha256"] != config["protocol_sha256"]:
                raise ValueError("Checkpoint architecture/protocol mismatch")
            if sha(Path(checkpoint) / "connector.pt") != meta["connector_sha256"]:
                raise ValueError("Checkpoint hash mismatch")
            self.connector.load_state_dict(torch.load(Path(checkpoint) / "connector.pt", weights_only=True, map_location="cuda"))
        if any(param.requires_grad for param in self.model.parameters()):
            raise AssertionError("Backbone or LoRA unexpectedly trainable")
        self.connector.train(training)

    def batch(self, row, mode="true", training=False, audio_id=None):
        import numpy as np
        import torch
        from inputs import residual_batch
        if self.c["architecture"] == "mlp":
            return self.p.batch(self.model, self.tokenizer, self.connector, row, "B",
                                "text" if mode in {"off", "text"} else mode, audio_id, training)
        frames = None
        if mode not in {"off", "text"}:
            value = np.load(audio_path(self.p, audio_id or row["id"]), allow_pickle=False)
            frames = torch.from_numpy(value.astype(np.float32)).cuda()
        return residual_batch(self.p, self.model, self.tokenizer, self.connector, row, frames, mode, training)

    def predict(self, row, mode, audio_id=None):
        import torch
        with torch.no_grad():
            ids = self.model.generate(**self.batch(row, mode, audio_id=audio_id), do_sample=False,
                                      max_new_tokens=self.c["max_new_tokens"],
                                      pad_token_id=self.tokenizer.pad_token_id,
                                      eos_token_id=self.tokenizer.eos_token_id, use_cache=True)
        if ids.shape[1] > self.c["max_new_tokens"]:
            raise AssertionError("Unexpected inputs_embeds generation API")
        raw = self.tokenizer.decode(ids[0], skip_special_tokens=True)
        pred, parsed = self.p.parse_prediction(raw)
        return dict(id=row["id"], reference=row["reference"], input=row["input"], prediction=pred,
                    model_output=raw, parsed=parsed, audio_id=audio_id or row["id"], condition=mode)


def configuration(directory, caches=True):
    c, p = verify(directory, caches=caches)
    c["protocol_sha256"] = sha(Path(directory) / "protocol.json")
    return c, p


def smoke(directory):
    import torch
    directory = Path(directory)
    if (directory / "gpu_smoke.json").exists():
        raise FileExistsError("Preserve existing smoke evidence; use a separate prepared run for retries")
    c, p = configuration(directory)
    with gpu_guard(p):
        runtime = Runtime(c, p, training=True)
        row = p.load(c["train"])[0]  # Never use dev references for engineering optimization.
        runtime.model.eval(); runtime.connector.eval()
        b = runtime.batch(row, "off")
        old = p.batch(runtime.model, runtime.tokenizer, runtime.connector, row, "B", "text")
        assert torch.equal(b["inputs_embeds"], old["inputs_embeds"])
        if c["architecture"] == "residual":
            true = runtime.batch(row, "true")
            assert torch.equal(true["inputs_embeds"], b["inputs_embeds"])
            with torch.no_grad():
                logits_a = runtime.model(**b, use_cache=False).logits
                logits_b = runtime.model(**true, use_cache=False).logits
                assert torch.equal(logits_a, logits_b), "Zero-init logits differ from text baseline"
            del logits_a, logits_b
            assert runtime.predict(row, "true")["model_output"] == runtime.predict(row, "off")["model_output"]
        runtime.model.train(); runtime.connector.train()
        opt = torch.optim.AdamW(runtime.connector.parameters(), lr=c["lr"])
        losses = []
        for step in range(2):
            opt.zero_grad(set_to_none=True)
            batch = runtime.batch(row, training=True)
            loss = runtime.model(**batch, use_cache=False).loss
            assert torch.isfinite(loss)
            loss.backward()
            if c["architecture"] == "residual":
                assert runtime.connector.output.weight.grad.abs().sum() > 0
                if step == 1:
                    assert runtime.connector.cross_attention.in_proj_weight.grad.abs().sum() > 0
            norm = torch.nn.utils.clip_grad_norm_(runtime.connector.parameters(), 1.)
            assert torch.isfinite(norm) and norm > 0
            assert all(param.grad is None for param in runtime.model.parameters())
            opt.step(); losses.append(float(loss.detach()))
        save(directory / "gpu_smoke.json", dict(status="PASS", protocol_sha256=c["protocol_sha256"],
                                                losses=losses, training_checkpoint_saved=False,
                                                torch=torch.__version__, seed=c["seed"]))


def train(directory):
    import torch
    directory = Path(directory)
    if (directory / "training_started.json").exists():
        raise FileExistsError("Training already attempted: preserve evidence, do not overwrite or auto-restart")
    c, p = configuration(directory)
    result = read(directory / "gpu_smoke.json")
    if result["status"] != "PASS" or result["protocol_sha256"] != c["protocol_sha256"]:
        raise ValueError("Matching real-model GPU smoke required before training")
    with gpu_guard(p):
        started = time.time()
        save(directory / "training_started.json", dict(pid=os.getpid(), started=started))
        try:
            runtime = Runtime(c, p, training=True)
            rows = p.load(c["train"])
            random.Random(c["seed"]).shuffle(rows)
            if len(rows) != c["steps"] * c["accumulation"]:
                raise ValueError("Training budget/data mismatch")
            optimizer = torch.optim.AdamW(runtime.connector.parameters(), lr=c["lr"], weight_decay=c["weight_decay"])
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: min((s + 1) / c["warmup"], 1.)
                    * max(0., (c["steps"] - s) / (c["steps"] - c["warmup"])) if s >= c["warmup"] else (s + 1) / c["warmup"])
            with (directory / "training.jsonl").open("x", encoding="utf-8") as log:
                for step in range(c["steps"]):
                    optimizer.zero_grad(set_to_none=True)
                    loss_sum = 0.
                    for row in rows[step * c["accumulation"]:(step + 1) * c["accumulation"]]:
                        loss = runtime.model(**runtime.batch(row, training=True), use_cache=False).loss
                        if not torch.isfinite(loss):
                            raise FloatingPointError(f"Nonfinite loss: {row['id']}")
                        (loss / c["accumulation"]).backward()
                        loss_sum += float(loss.detach())
                    grad = torch.nn.utils.clip_grad_norm_(runtime.connector.parameters(), 1.)
                    if not torch.isfinite(grad) or grad <= 0:
                        raise FloatingPointError("Invalid connector gradient")
                    optimizer.step(); scheduler.step()
                    record = dict(step=step + 1, steps=c["steps"], loss=loss_sum / c["accumulation"],
                                  grad_norm=float(grad), seconds=time.time() - started)
                    log.write(json.dumps(record) + "\n"); log.flush()
                    save(directory / "status.json", dict(status="TRAINING", **record))
                    if (step + 1) % 10 == 0 or step == 0:
                        print(json.dumps(record), flush=True)
            checkpoint = directory / "final"
            checkpoint.mkdir(exist_ok=False)
            torch.save(runtime.connector.state_dict(), checkpoint / "connector.pt")
            save(checkpoint / "metadata.json", dict(architecture=c["architecture"], step=c["steps"],
                 protocol_sha256=c["protocol_sha256"], connector_sha256=sha(checkpoint / "connector.pt"),
                 frozen_adapter=c["adapter"], complete=True))
            save(directory / "selected_checkpoint.json", dict(checkpoint=str(checkpoint),
                 connector_sha256=sha(checkpoint / "connector.pt"), reason=c["selection"]))
            save(directory / "status.json", dict(status="TRAINED_NOT_EVALUATED", seconds=time.time() - started))
        except BaseException:
            save(directory / "status.json", dict(status="TRAINING_FAILED", error=traceback.format_exc()))
            raise


def evaluate(directory):
    import torch
    from diagnostics import diagnose
    directory = Path(directory)
    c, p = configuration(directory)
    selection = read(directory / "selected_checkpoint.json")
    checkpoint = directory / "final"
    if selection["connector_sha256"] != sha(checkpoint / "connector.pt"):
        raise ValueError("Selected checkpoint changed")
    with gpu_guard(p):
        dest = directory / "full_dev"
        dest.mkdir(exist_ok=False)  # No overwrite or silently omitting partial/failed rows.
        try:
            runtime = Runtime(c, p, checkpoint=checkpoint)
            rows = p.load(c["dev"])
            assert len(rows) == 1334
            metrics = {}
            for mode in c["modes"]:
                partial = dest / f"{mode}.predictions.jsonl.partial"
                with partial.open("x", encoding="utf-8") as stream:
                    for i, row in enumerate(rows):
                        prediction = runtime.predict(row, mode, c["audio_permutation"][row["id"]] if mode == "mismatch" else None)
                        stream.write(json.dumps(prediction, ensure_ascii=False) + "\n"); stream.flush()
                        if (i + 1) % 50 == 0:
                            print(json.dumps(dict(mode=mode, done=i + 1, total=1334)), flush=True)
                predictions = p.load(partial)
                assert [r["id"] for r in predictions] == [r["id"] for r in rows]
                assert all(a["reference"] == b["reference"] and a["input"] == b["input"] for a, b in zip(predictions, rows))
                metric = p.metrics(predictions, protected=False)
                assert metric["summary"]["chars"] == 21102 and metric["summary"]["mentions"] == 599
                metric["error_diagnostics"] = diagnose(predictions, p.evaluator().norm)
                assert sum(metric["error_diagnostics"][k] for k in ("S", "D", "I")) == metric["summary"]["errors"]
                partial.rename(dest / f"{mode}.predictions.jsonl")
                save(dest / f"{mode}.metrics.json", metric)
                metrics[mode] = metric
            text_rows, off_rows = p.load(dest / "text.predictions.jsonl"), p.load(dest / "off.predictions.jsonl")
            assert [r["model_output"] for r in text_rows] == [r["model_output"] for r in off_rows], "off must reproduce text"
            true, text = metrics["true"]["summary"], metrics["text"]["summary"]
            compare = {mode: p.bootstrap(metrics["true"]["per_utterance"], metrics[mode]["per_utterance"])
                       for mode in ("text", "zero", "mismatch")}
            summary = dict(label=c["label"], summaries={k: v["summary"] for k, v in metrics.items()},
                           error_diagnostics={k: v["error_diagnostics"] for k, v in metrics.items()},
                           paired_bootstrap=compare,
                           meets_text_joint_point_objective=true["cer"] < text["cer"] and true["hits"] >= text["hits"],
                           input_hotword_recall_preserved=true["lost"] <= true["gained"],
                           no_output_repairs=True, statistical_caveat="point comparisons are not a noninferiority guarantee",
                           predictions_sha256={f.name: sha(f) for f in dest.glob("*.predictions.jsonl")})
            save(dest / "summary.json", summary)
            save(directory / "status.json", dict(status="COMPLETED_FULL_DEV", summary=str(dest / "summary.json")))
            print(json.dumps(summary, ensure_ascii=False), flush=True)
        except BaseException:
            save(directory / "status.json", dict(status="EVALUATION_FAILED", error=traceback.format_exc()))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare", help="CPU audit and immutable protocol only")
    prep.add_argument("--legacy", type=Path, default=DEFAULT_LEGACY)
    prep.add_argument("--architecture", choices=("residual", "mlp"), required=True)
    prep.add_argument("--mlp-init", type=Path)
    prep.add_argument("--output", type=Path, required=True)
    for name in ("verify", "smoke", "train", "eval"):
        cmd = sub.add_parser(name)
        cmd.add_argument("directory", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args)
    elif args.command == "verify":
        config, _ = verify(args.directory, caches=True)
        print(json.dumps(dict(status="PASS_CPU_PROTOCOL_AND_CACHES", architecture=config["architecture"])))
    else:
        {"smoke": smoke, "train": train, "eval": evaluate}[args.command](args.directory)


if __name__ == "__main__":
    main()
