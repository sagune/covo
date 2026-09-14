#!/usr/bin/env python3
"""Patch the bridge so the transmitted n-best list cannot contradict itself.

Motivation (measured, RESULTS doc section 7.9 / 8.8): the rescorer rewrites
`asr_top1` and puts its winner on n-best line 1, but each line still prints the
ORIGINAL beam metadata looked up from cbwhisper.candidates.  On ST-CMDS 5.7% of
rows therefore show a first line labelled `rank=12 score=2.802` while line 2 says
`rank=1 score=3.800`, and the end-to-end cost of perturbing the prompt that way is
about +0.49 pp.  This adds a documented switch:

  --rank-mode original     (default) keep today's behaviour
  --rank-mode transmitted  print the position in the transmitted list as `rank=`,
                           keep the beam provenance as `beam_rank=`, and attach the
                           rescorer's own score as `rerank=` when the record carries
                           `input.asr_top1_rerank_score`

No behaviour changes unless the flag is passed.
"""
from pathlib import Path
import subprocess

p = Path("/root/autodl-tmp/src/analysis/cbsensevoice_covo_bridge.py")
s = p.read_text(encoding="utf-8")

OLD_SIG = """    max_items: int,
    compact: bool = False,
) -> List[str]:
    cbw = input_block.get("cbwhipser", {}) or {}"""  # (guarded below, never matches)
SIG_OLD = """    max_items: int,
    compact: bool = False,
) -> List[str]:
    cbw = input_block.get("cbwhisper", {}) or {}
    scored_candidates = list(cbw.get("candidates", []) or [])
    scored_index = _candidate_score_index(scored_candidates)
    output = []
    seen = set()"""
SIG_NEW = """    max_items: int,
    compact: bool = False,
    rank_mode: str = "original",
) -> List[str]:
    cbw = input_block.get("cbwhisper", {}) or {}
    scored_candidates = list(cbw.get("candidates", []) or [])
    scored_index = _candidate_score_index(scored_candidates)
    rerank_score = input_block.get("asr_top1_rerank_score")
    output = []
    seen = set()"""

BITS_OLD = """            if compact:
                bits = [
                    f"rank={int(scored.get('rank', idx))}",
                    f"score={float(scored.get('total_score', 0.0)):.3f}",
                    f"asr={float(scored.get('asr_score', 0.0)):.3f}",
                ]"""
BITS_NEW = """            if compact:
                if rank_mode == "transmitted":
                    # `rank=` is the position in the list the model is reading, so the
                    # ordering carries no hidden contradiction; the beam provenance is
                    # kept under an explicit name, and the rescorer's own score is shown
                    # when the record provides it.
                    bits = [
                        f"rank={idx}",
                        f"beam_rank={int(scored.get('rank', idx))}",
                        f"score={float(scored.get('total_score', 0.0)):.3f}",
                        f"asr={float(scored.get('asr_score', 0.0)):.3f}",
                    ]
                    if rerank_score is not None and normalized == normalize_text(
                            str(input_block.get("asr_top1") or "")):
                        bits.append(f"rerank={float(rerank_score):.3f}")
                else:
                    bits = [
                        f"rank={int(scored.get('rank', idx))}",
                        f"score={float(scored.get('total_score', 0.0)):.3f}",
                        f"asr={float(scored.get('asr_score', 0.0)):.3f}",
                    ]"""

CALL_OLD = "        lines.extend(format_nbest(nbest, input_block, hotword_rows, max_items=max(1, int(args.max_nbest)), compact=compact))"
CALL_NEW = "        lines.extend(format_nbest(nbest, input_block, hotword_rows, max_items=max(1, int(args.max_nbest)), compact=compact, rank_mode=str(getattr(args, \"rank_mode\", \"original\"))))"

ARG_OLD = '    parser.add_argument("--max-nbest", type=int, default=8)'
ARG_NEW = ('    parser.add_argument("--max-nbest", type=int, default=8)\n'
           '    parser.add_argument("--rank-mode", default="original", choices=["original", "transmitted"],\n'
           '                        help="transmitted: print the position in the transmitted n-best list as rank= "\n'
           '                             "and keep the beam rank under beam_rank=, so the prompt cannot contradict itself")')

for old, new in ((SIG_OLD, SIG_NEW), (BITS_OLD, BITS_NEW), (CALL_OLD, CALL_NEW), (ARG_OLD, ARG_NEW)):
    if old not in s:
        raise SystemExit("PATCH TARGET NOT FOUND:\n" + old[:160])
    s = s.replace(old, new, 1)

p.write_text(s, encoding="utf-8")
r = subprocess.run(["/root/autodl-tmp/great/bin/python", "-m", "py_compile", str(p)],
                   capture_output=True, text=True)
print("compiles:", r.returncode == 0, r.stderr[:300])
print("rank_mode refs:", s.count("rank_mode"), "| rerank_score refs:", s.count("rerank_score"))
