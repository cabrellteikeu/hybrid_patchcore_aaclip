# dataset/custom_jsonl.py
import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class CustomJsonlDataset(Dataset):
    """
    Erwartet JSONL-Zeilen mit Feldern:
      {
        "image": "/abs/or/rel/path/to/image.png",
        "caption": "bucket",               # optional/ignoriert
        "label": 0 or 1,                   # optional, falls Training es braucht
        "mask_path": "/path/to/mask.png"   # optional; wenn fehlt -> None
      }
    """

    def __init__(self, jsonl_path: str, preprocess=None, return_mask: bool = True):
        self.jsonl_path = Path(jsonl_path)
        assert self.jsonl_path.exists(), f"JSONL not found: {self.jsonl_path}"
        self.items: List[Dict[str, Any]] = []
        with self.jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                # Pflichtfeld "image"
                assert "image" in rec, f"Missing 'image' field in {jsonl_path}"
                # Label optional
                if "label" not in rec:
                    rec["label"] = None
                # Mask optional + robust
                mp = rec.get("mask_path")
                if mp is None or not os.path.isfile(mp):
                    rec["mask_path"] = None
                self.items.append(rec)

        # Preprocessing (CLIP-Backbone-typisch)
        self.preprocess = preprocess or transforms.Compose([
            transforms.Resize((224, 224)),    # ggf. je nach Backbone anpassen
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                 std=[0.26862954, 0.26130258, 0.27577711]),
        ])
        self.return_mask = return_mask

    def __len__(self):
        return len(self.items)

    def _load_mask(self, mask_path: Optional[str], size_hw):
        if mask_path is None:
            return None
        try:
            m = Image.open(mask_path).convert("L")
            # Auf Bildgröße des Encoders bringen (oder Originalauflösung, je nach Repo)
            m = m.resize((size_hw[1], size_hw[0]), Image.NEAREST)
            return torch.from_numpy((torch.ByteTensor(torch.ByteStorage.from_buffer(m.tobytes()))
                                     .view(m.size[1], m.size[0]).numpy())).float() / 255.0
        except Exception:
            return None

    def __getitem__(self, idx):
        rec = self.items[idx]
        img = Image.open(rec["image"]).convert("RGB")
        img_t = self.preprocess(img)
        label = rec.get("label")
        mask = None
        if self.return_mask:
            # Größe der Vorverarbeitung (z.B. 224x224)
            size_hw = (img_t.shape[1], img_t.shape[2])
            mask = self._load_mask(rec.get("mask_path"), size_hw=size_hw)

        return {
            "image": img_t,                 # Tensor [3,H,W]
            "image_path": rec["image"],
            "label": label,                 # optional
            "mask": mask,                   # optional
            "caption": rec.get("caption", None),
        }
