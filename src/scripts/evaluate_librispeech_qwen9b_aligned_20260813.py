import json,re,subprocess
from pathlib import Path
SRC=Path("/root/autodl-tmp/src")
COVO=Path("/root/autodl-tmp/cbwhisper_covo_migration_20260609_tar_extracted/covo")
PY="/root/autodl-tmp/great/bin/python"
for split in ["dev-clean","dev-other","test-clean","test-other"]:
    src=SRC/f"logs/librispeech_{split}_sensevoice_covo_eval_20260812.jsonl"
    pred=SRC/f"logs/librispeech_{split}_qwen9b_sensevoice_predictions_20260813.jsonl"
    aligned=SRC/f"logs/librispeech_{split}_qwen9b_sensevoice_aligned_20260813.jsonl"
    with src.open() as sf, pred.open() as pf, aligned.open("w") as out:
        for sl,pl in zip(sf,pf):
            s=json.loads(sl); p=json.loads(pl)
            ref=""
            for m in reversed(s["messages"]):
                if m.get("role")=="assistant":
                    try: ref=json.loads(m["content"]).get("text","")
                    except Exception: pass
                    break
            user=next((m.get("content","") for m in s["messages"] if m.get("role")=="user"),"")
            match=re.search(r"SenseVoice top-1: (.*?)(?:\\n|$)", user)
            base=match.group(1) if match else ""
            out.write(json.dumps({"prediction":p.get("prediction",""),"reference":ref,"input":{"asr_top1":base}},ensure_ascii=False))
            out.write(chr(10))
    summary=SRC/f"logs/librispeech_{split}_qwen9b_sensevoice_summary_20260813.json"
    with summary.open("w") as sh:
        subprocess.run([PY,str(COVO/"scripts/evaluate_correction_jsonl.py"),"--input",str(aligned)],check=True,stdout=sh)
    print(split, summary.read_text().strip())
