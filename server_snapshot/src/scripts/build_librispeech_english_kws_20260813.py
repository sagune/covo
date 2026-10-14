#!/usr/bin/env python3
import json, os, shutil
from pathlib import Path

base=Path("/root/autodl-tmp/datasets/librispeech")
kws=Path("/root/autodl-tmp/datasets/librispeech_kws/Librispeech_KWS/train")
out=base/"cb_sensevoice_english_kws_20260813"
splits=["dev-clean","dev-other","test-clean","test-other"]
words=sorted(p.name for p in kws.iterdir() if p.is_dir() and not p.name.startswith("_"))
(out/"wav").mkdir(parents=True,exist_ok=True)
for split in splits:
    manifest=Path("/root/autodl-tmp/src/logs")/f"librispeech_{split}_manifest_20260812.jsonl"
    rows=[json.loads(x) for x in manifest.read_text().splitlines() if x.strip()]
    hot=out/"hotword"/split
    (hot/"keywords-audios"/"tts").mkdir(parents=True,exist_ok=True)
    (hot/"hs").mkdir(parents=True,exist_ok=True)
    (hot/"keywords-hs"/"tts").mkdir(parents=True,exist_ok=True)
    wav_dir=out/"wav"/split
    wav_dir.mkdir(parents=True,exist_ok=True)
    text="".join(f"{r['id']} {r['reference']}\n" for r in rows)
    (hot/"text").write_text(text)
    (hot/"uttid").write_text(text)
    (hot/"hotword.txt").write_text("\n".join(words)+"\n")
    (hot/"r1-hotword.txt").write_text("\n".join(words)+"\n")
    (hot/"aligned.txt").write_text("")
    for r in rows:
        target=wav_dir/(r["id"]+".wav")
        if not target.exists():
            target.symlink_to(os.path.relpath(Path(r["wav"]), wav_dir))
    width=len(str(len(words)-1))
    for i,w in enumerate(words):
        candidates=sorted((kws/w).glob("*.wav"))
        if not candidates: raise RuntimeError(f"no clip {w}")
        target=hot/"keywords-audios"/"tts"/f"{i:0{width}d}.wav"
        if not target.exists():
            target.symlink_to(os.path.relpath(candidates[0],target.parent))
(out/"hotword"/"words.txt").write_text("\n".join(words)+"\n")
print(json.dumps({"root":str(out),"words":len(words),"splits":{s:sum(1 for _ in open(out/"hotword"/s/"text")) for s in splits}},indent=2))
