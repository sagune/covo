#!/usr/bin/env python3
import json,re,os
from pathlib import Path
root=Path("/root/autodl-tmp/datasets/librispeech/cb_sensevoice_english_kws_20260813")
src=root/"hotword/dev-clean"
dst=root/"hotword/dev-clean-hotword-smoke"
wavsrc=root/"wav/dev-clean"
wavdst=root/"wav/dev-clean-hotword-smoke"
words=(root/"hotword/words.txt").read_text().splitlines()
priority=["cat","dog","tree","house","bird","bed","left","right","eight","seven","six","five","four","three","two","nine"]
rows=[json.loads(x) for x in Path("/root/autodl-tmp/src/logs/librispeech_dev-clean_manifest_20260812.jsonl").read_text().splitlines() if x.strip()]
chosen=[]
for word in priority:
  for r in rows:
    toks=set(re.findall(r"[a-z]+",r["reference"].lower()))
    if word in toks and r not in chosen:
      chosen.append(r);break
  if len(chosen)>=10:break
for p in [dst/"hs",dst/"keywords-audios",dst/"keywords-hs",wavdst]: p.mkdir(parents=True,exist_ok=True)
for name in ["hotword.txt","r1-hotword.txt","aligned.txt"]:
  target=dst/name
  if not target.exists(): target.symlink_to(os.path.relpath(src/name,dst))
for name in ["keywords-audios","keywords-hs"]:
  target=dst/name
  if target.exists() or target.is_symlink():
    if target.is_dir() and not target.is_symlink(): pass
  else: target.symlink_to(os.path.relpath(src/name,dst),target_is_directory=True)
text="".join(f"{r['id']} {r['reference']}\n" for r in chosen)
(dst/"text").write_text(text); (dst/"uttid").write_text(text)
for r in chosen:
  target=wavdst/(r["id"]+".wav")
  if not target.exists(): target.symlink_to(os.path.relpath(wavsrc/(r["id"]+".wav"),wavdst))
(root/"hotword/dev-clean-hotword-smoke.ids").write_text("\n".join(r["id"] for r in chosen)+"\n")
print(json.dumps([{"id":r["id"],"reference":r["reference"]} for r in chosen],indent=2))
