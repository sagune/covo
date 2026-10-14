from huggingface_hub import hf_hub_download

path = hf_hub_download(
    repo_id="openslr/librispeech_asr",
    repo_type="dataset",
    filename="clean/train.100/0000.parquet",
    local_dir="/root/autodl-tmp/datasets/librispeech_hf",
)
print(path)
