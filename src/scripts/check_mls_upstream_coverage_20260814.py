from pathlib import Path

import pyarrow.parquet as pq


parquet = "/root/autodl-tmp/datasets/mls_hf_parquet/train-00000-of-01416.parquet"
table = pq.read_table(parquet, columns=["audio", "speaker_id", "book_id"])
available = {Path(item["path"]).stem for item in table.column("audio").to_pylist()}
required = {
    line.split()[0]
    for line in open("/root/autodl-tmp/datasets/mls/train/mls_english_opus/uttid", encoding="utf-8")
    if line.strip()
}
required_speakers = {code.split("_")[0] for code in required}
required_books = {"_".join(code.split("_")[:2]) for code in required}
print({
    "available": len(available),
    "required": len(required),
    "intersection": len(available & required),
    "missing": len(required - available),
    "required_speakers": len(required_speakers),
    "required_books": len(required_books),
})
print("examples", sorted(available & required)[:10])
speakers = table.column("speaker_id").to_pylist()
books = table.column("book_id").to_pylist()
print({
    "unique_speakers": len(set(speakers)),
    "speaker_first": speakers[:5],
    "speaker_last": speakers[-5:],
    "speaker_min": min(map(int, speakers)),
    "speaker_max": max(map(int, speakers)),
    "unique_books": len(set(books)),
})
