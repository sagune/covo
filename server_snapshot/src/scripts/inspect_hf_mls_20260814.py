from datasets import get_dataset_config_names, get_dataset_split_names, load_dataset

dataset_name = "parler-tts/mls_eng"
print("configs", get_dataset_config_names(dataset_name))
configs = get_dataset_config_names(dataset_name)
config = configs[0] if len(configs) == 1 and configs[0] != "default" else None
print("splits", get_dataset_split_names(dataset_name, config))
for split in ("train", "train.10h", "train_10h"):
    try:
        dataset = load_dataset(dataset_name, config, split=split, streaming=True)
        row = next(iter(dataset))
        print("selected_split", split)
        print("keys", sorted(row))
        print("row", {key: value for key, value in row.items() if key != "audio"})
        print("audio", row.get("audio"))
        break
    except Exception as exc:
        print("failed", split, type(exc).__name__, str(exc)[:300])
