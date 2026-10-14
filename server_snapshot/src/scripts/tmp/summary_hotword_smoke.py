import json,re
p="/root/autodl-tmp/src/logs/cb_sensevoice_librispeech_english_hotword_smoke_evidence_20260813.jsonl"
for line in open(p):
 r=json.loads(line)
 ref=str(r.get("reference") or r.get("target") or r.get("transcript") or "")
 kws=r.get("kws_hotwords") or r.get("keywords") or []
 prompt=r.get("prompt_hotwords") or []
 print(json.dumps({"id":r.get("id"),"ref":ref[:120],"kws":kws,"prompt":prompt,"prediction":str(r.get("prediction") or r.get("cb_sensevoice_prediction") or "")[:120]},ensure_ascii=False))
