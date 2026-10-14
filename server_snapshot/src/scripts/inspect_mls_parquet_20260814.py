import json

import pyarrow.parquet as pq


path = "/root/autodl-tmp/datasets/mls_hf_parquet/train-00000-of-01416.parquet"
parquet = pq.ParquetFile(path)
print("rows", parquet.metadata.num_rows)
print("row_groups", parquet.metadata.num_row_groups)
print("schema", parquet.schema)
table = parquet.read_row_group(0)
print("columns", table.column_names)
for row in table.slice(0, 2).to_pylist():
    summary = {}
    for key, value in row.items():
        if isinstance(value, dict) and isinstance(value.get("bytes"), bytes):
            summary[key] = {
                **{k: v for k, v in value.items() if k != "bytes"},
                "bytes_len": len(value["bytes"]),
            }
        elif isinstance(value, bytes):
            summary[key] = {"bytes_len": len(value)}
        else:
            summary[key] = value
    print(json.dumps(summary, ensure_ascii=True, default=str))
