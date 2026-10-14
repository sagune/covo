#!/usr/bin/env python3
"""One-off patch: add conflict-row weighting to learned_ranker_pilot2.py."""
from pathlib import Path
import subprocess

p = Path("/root/autodl-tmp/.dsh_checks/learned_ranker_pilot2.py")
s = p.read_text(encoding="utf-8")

REPL = [
    # 1. record whether the pool CER-optimum loses a designated hotword
    ('        u["y"] = u["y_cls"].astype(np.float64) / max(1.0, u["y_cls"].sum())',
     '        u["y"] = u["y_cls"].astype(np.float64) / max(1.0, u["y_cls"].sum())\n'
     '        u["conflict"] = 1.0 if hlost[int(np.argmin(cers))] > 0 else 0.0'),
    # 2. pad a per-utterance weight
    ('    H = np.zeros((N, K), dtype=np.float32)\n    M = np.zeros((N, K), dtype=np.float32)',
     '    H = np.zeros((N, K), dtype=np.float32)\n    M = np.zeros((N, K), dtype=np.float32)\n'
     '    Wt = np.ones((N,), dtype=np.float32)'),
    ('        H[r, :k] = u["hlost"]\n        M[r, :k] = 1.0',
     '        H[r, :k] = u["hlost"]\n        M[r, :k] = 1.0\n'
     '        Wt[r] = 1.0 + cw * u.get("conflict", 0.0)'),
    ('def pad_pools(utts, idx, key):', 'def pad_pools(utts, idx, key, cw=0.0):'),
    ('    return (torch.tensor(X), torch.tensor(Y), torch.tensor(C), torch.tensor(H), torch.tensor(M))',
     '    return (torch.tensor(X), torch.tensor(Y), torch.tensor(C), torch.tensor(H),\n'
     '            torch.tensor(M), torch.tensor(Wt))'),
    ('    X, Y, C, H, M = blob', '    X, Y, C, H, M, Wt = blob'),
    # 3. per-row loss then weighted mean
    ('            loss = -(Y * logp).sum(dim=1).mean()', '            per = -(Y * logp).sum(dim=1)'),
    ('            loss = (torch.softmax(s, dim=1) * C).sum(dim=1).mean()',
     '            per = (torch.softmax(s, dim=1) * C).sum(dim=1)'),
    ('        if recall_weight > 0:\n            loss = loss + recall_weight * (torch.softmax(s, dim=1) * H).sum(dim=1).mean()',
     '        if recall_weight > 0:\n            per = per + recall_weight * (torch.softmax(s, dim=1) * H).sum(dim=1)\n'
     '        loss = (per * Wt).sum() / Wt.sum()'),
    # 4. pass the weight through
    ('blob = pad_pools(utts, tr, "Z")', 'blob = pad_pools(utts, tr, "Z", a.conflict_weight)'),
    ('blob = pad_pools(tr_utts, tridx, "Z")', 'blob = pad_pools(tr_utts, tridx, "Z", a.conflict_weight)'),
    # 5. the flag
    ('    ap.add_argument("--label-mode", default="mincer", choices=["mincer", "nohotloss"],',
     '    ap.add_argument("--conflict-weight", type=float, default=0.0,\n'
     '                    help="extra weight on utterances whose pool CER-optimum loses a designated hotword")\n'
     '    ap.add_argument("--label-mode", default="mincer", choices=["mincer", "nohotloss"],'),
]

for old, new in REPL:
    if old not in s:
        raise SystemExit("PATCH TARGET NOT FOUND:\n" + old[:120])
    s = s.replace(old, new, 1)

p.write_text(s, encoding="utf-8")
r = subprocess.run(["/root/autodl-tmp/great/bin/python", "-m", "py_compile", str(p)],
                   capture_output=True, text=True)
print("compiles:", r.returncode == 0, r.stderr[:300])
for i, ln in enumerate(s.splitlines(), 1):
    if "conflict" in ln or "Wt" in ln or "per = " in ln:
        print("%4d  %s" % (i, ln.strip()))
