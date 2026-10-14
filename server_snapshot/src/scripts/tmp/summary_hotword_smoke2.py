import json
p="/root/autodl-tmp/src/logs/cb_sensevoice_librispeech_english_hotword_smoke_evidence_20260813.jsonl"
for l in open(p):
 r=json.loads(l); h=[x["text"] for x in r["input"]["hotwords"]]; q=[x["text"] for x in r["input"]["prompt_hotwords"]]
 print(json.dumps({"id":r["id"],"kws":h,"prompt":q,"top1":r["input"]["asr_top1"][:90]},ensure_ascii=False))
