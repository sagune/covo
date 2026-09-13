import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.data_collator import KWSDataCollator
from data.dataset import MLSKWSDataset
from model.model import KWSModel


root = "/root/autodl-tmp/datasets/mls_sensevoice_english_10h_20260814"
dataset = MLSKWSDataset(root=root, languages=["English"], kw_type="tts")
hs_codes = {
    os.path.splitext(name)[0]
    for name in os.listdir(os.path.join(root, "mls_english_opus", "train", "hs"))
    if name.endswith(".bin")
}
utterance_index = next(
    index for index, item in enumerate(dataset.metadata[0]["data"]) if item["code"] in hs_codes
)
keyword_index = next(
    index
    for index in range(len(dataset.keywords["English"]))
    if os.path.exists(
        os.path.join(
            root,
            "mls_english_opus",
            "train",
            "keywords-hs",
            "tts",
            f"{index:0{len(str(len(dataset.keywords['English']) - 1))}d}.bin",
        )
    )
)
item = dataset[utterance_index * dataset.n_keywords[-1] + keyword_index]
batch = KWSDataCollator(size=(150, 750))([item])
model = KWSModel(backbone="resnet", adversarial_training=False).cuda().eval()
with torch.inference_mode():
    output = model(input_features=batch["features"].cuda())
print(
    {
        "utterance_index": utterance_index,
        "keyword_index": keyword_index,
        "raw_shape": tuple(item["features"].shape),
        "batch_shape": tuple(batch["features"].shape),
        "logits_shape": tuple(output.logits.shape),
        "logits": output.logits.cpu().tolist(),
    }
)
