import json,re
from pathlib import Path
root=Path("/root/autodl-tmp/datasets/librispeech/cb_sensevoice_english_kws_20260813")
words=root.joinpath("hotword/words.txt").read_text().splitlines()
for split in ["dev-clean","dev-other","test-clean","test-other"]:
    rows=[json.loads(x) for x in Path("/root/autodl-tmp/src/logs",f"librispeech_{split}_manifest_20260812.jsonl").read_text().splitlines() if x]
    hits=[]
    for r in rows:
        tokens=set(re.findall(r"[a-z]+",r["reference"].lower()))
        found=sorted(tokens.intersection(words))
        if found: hits.append((r["id"],found))
    print(json.dumps({"split":split,"utterances":len(rows),"covered_utterances":len(hits),"coverage":len(hits)/len(rows),"unique_hotwords":sorted({w for _,ws in hits for w in ws}),"examples":hits[:5]},ensure_ascii=False))
