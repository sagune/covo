#!/usr/bin/env python3
import hashlib, os
from pathlib import Path

source=Path("/root/autodl-tmp/datasets/librispeech_kws/Librispeech_KWS")
target=Path("/root/autodl-tmp/datasets/librispeech/data_librispeech_kws_english_20260813")
eval_words=Path("/root/autodl-tmp/datasets/librispeech/cb_sensevoice_english_kws_20260813/hotword/words.txt").read_text().splitlines()
words=sorted(eval_words)
for path in [target/"kws/hs",target/"kws/keywords-hs/tts",target/"kws/keywords-audios/tts",target/"kws/wav",target/"hotword/dev/hs",target/"hotword/dev/keywords-hs",target/"hotword/dev/keywords-audios",target/"wav/dev",target/"hotword/test/hs",target/"hotword/test/keywords-hs",target/"hotword/test/keywords-audios",target/"wav/test"]:
    path.mkdir(parents=True,exist_ok=True)
(target/"kws/keywords.txt").write_text("\n".join(words)+"\n")
width=len(str(len(words)-1))
reverse={word:sorted(words,key=lambda x:x[::-1]).index(word) for word in words}
train_rows=[]; dev_rows=[]
for idx,word in enumerate(words):
    clips=sorted((source/"train"/word).glob("*.wav"))
    if not clips: raise RuntimeError(word)
    prototype=target/"kws/keywords-audios/tts"/f"{idx:0{width}d}.wav"
    if not prototype.exists(): prototype.symlink_to(os.path.relpath(clips[0],prototype.parent))
    for clip in clips:
        code=f"{word}__{clip.stem}"
        dst=target/"kws/wav"/f"{code}.wav"
        if not dst.exists(): dst.symlink_to(os.path.relpath(clip,dst.parent))
        row=f"{code}\t{word}\t{idx}\t{reverse[word]}"
        bucket=int(hashlib.sha1(code.encode()).hexdigest()[:8],16)%10
        (dev_rows if bucket==0 else train_rows).append(row)
(target/"kws/positives.tsv").write_text("\n".join(train_rows)+"\n")
def build_hotword(split, rows, clips_root):
    hot=target/"hotword"/split; wav=target/"wav"/split
    for name in ["hotword.txt","r1-hotword.txt"]:
        (hot/name).write_text("\n".join(words)+"\n")
    for name in ["keywords-hs","keywords-audios"]:
        link=hot/name
        if not link.exists(): link.symlink_to(os.path.relpath(target/"kws"/name,hot),target_is_directory=True)
    texts=[]
    for row in rows:
        code,word,*_=row.split("\t")
        src=clips_root/f"{code}.wav"
        dst=wav/f"{code}.wav"
        if not dst.exists(): dst.symlink_to(os.path.relpath(src,dst.parent))
        texts.append(f"{code} {word}")
    content="\n".join(texts)+"\n"
    (hot/"text").write_text(content); (hot/"uttid").write_text(content); (hot/"aligned.txt").write_text("")
build_hotword("dev",dev_rows,target/"kws/wav")
test_rows=[]
for idx,word in enumerate(words):
    for clip in sorted((source/"test"/word).glob("*.wav")):
        code=f"{word}__{clip.stem}"
        dst=target/"wav/test"/f"{code}.wav"
        if not dst.exists(): dst.symlink_to(os.path.relpath(clip,dst.parent))
        test_rows.append(f"{code}\t{word}\t{idx}\t{reverse[word]}")
build_hotword("test",test_rows,target/"wav/test")
print({"words":len(words),"train":len(train_rows),"dev":len(dev_rows),"test":len(test_rows),"root":str(target)})
