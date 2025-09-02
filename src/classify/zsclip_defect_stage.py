# src/run_hybrid/zsclip_defect_stage.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
from PIL import Image, ImageOps

class ZSClipDefectStage:
    """
    Einfache Zero-Shot-Klassifikation:
      - Nur positive Prompt-Varianten je Klasse
      - Pro Klasse: Text-Prototyp = Mittelwert der Text-Embeddings
      - Bild-Embedding (mit optionalem TTA) -> Cosine-Sim zu allen Klassenprototypen
      - Softmax über Klassen mit Temperatur tau -> Wahrscheinlichkeiten
    """

    def __init__(
        self,
        model_name: str = "ViT-B-16",
        pretrained: str = "openai",
        device: str = "cuda",
        tta_scales: Tuple[float, ...] = (1.0, 0.9, 1.1),
        temperature: float = 0.01,
        topk: int = 3,
        defect_prompts: Optional[Dict[str, Any]] = None,  # {"class":[...]} ODER {"class":{"pos":[...]}}
    ):
        self.model_name = model_name
        self.pretrained = pretrained
        self.device_str = device
        self.tta_scales = tuple(tta_scales)
        self.tau = float(max(1e-4, temperature))
        self.topk = int(topk)
        self.prompts_raw = defect_prompts or {}

        self._lazy_imports()
        self.device = "cuda" if (self.torch.cuda.is_available() and self.device_str.startswith("cuda")) else "cpu"
        self.half = (self.device == "cuda")

        # OpenCLIP initialisieren
        self._load_openclip()
        # Klassen + Text-Prototypen bauen
        self.class_names: List[str] = []
        self.text_protos = None  # torch.Tensor [C, D]
        self._build_text_prototypes()

    # ---------- Lazy Imports ----------
    def _lazy_imports(self):
        import importlib
        import torch as _torch
        import torch.nn.functional as _F
        self.torch = _torch
        self.F = _F
        self.open_clip = importlib.import_module("open_clip")

    # ---------- OpenCLIP ----------
    def _map_openclip_name(self, name: str) -> str:
        n = name.replace("/", "-")
        if "ViT-L-14-336" in n: return "ViT-L-14-336"
        if "ViT-B-16" in n:     return "ViT-B-16"
        if "RN50" in n:         return "RN50"
        return "ViT-B-16"

    def _load_openclip(self):
        name = self._map_openclip_name(self.model_name)
        model, _, preprocess = self.open_clip.create_model_and_transforms(name, pretrained=self.pretrained)
        tok = self.open_clip.get_tokenizer(name)
        self.model = model.to("cpu").eval()  # erst CPU, dann ggf. Half setzen
        if self.torch.cuda.is_available() and self.device_str.startswith("cuda"):
            self.model = self.model.to("cuda").eval()
        self.preprocess = preprocess
        self.tokenizer = tok
        self.torch.set_grad_enabled(False)
        if self.device == "cuda":
            self.model = self.model.half()
            self.torch.backends.cudnn.benchmark = True

    # ---------- Prompts -> Text-Prototypen ----------
    def _extract_pos_list(self, v) -> List[str]:
        # akzeptiert:
        #  - Liste[str]
        #  - Dict mit key "pos": Liste[str]
        if isinstance(v, dict):
            pos = v.get("pos") or v.get("positive") or []
        else:
            pos = v
        if not isinstance(pos, (list, tuple)):
            return []
        # säubern
        out = [str(x).strip() for x in pos if str(x).strip()]
        return list(dict.fromkeys(out))  # de-dupe, Reihenfolge behalten

    def _encode_texts(self, texts: List[str]):
        toks = self.tokenizer(texts).to(self.device)
        with self.torch.cuda.amp.autocast(enabled=(self.device == "cuda")):
            t = self.model.encode_text(toks)  # [N,D]
        t = self.F.normalize(t.float(), dim=-1)
        return t  # [N,D]

    def _build_text_prototypes(self):
        names: List[str] = []
        protos = []

        for cname, v in self.prompts_raw.items():
            pos_list = self._extract_pos_list(v)
            if not pos_list:
                # Klasse ohne valide Pos-Prompts wird ignoriert
                continue
            t = self._encode_texts(pos_list)        # [N,D]
            proto = self.F.normalize(t.mean(dim=0, keepdim=True), dim=-1)  # [1,D]
            names.append(str(cname))
            protos.append(proto)

        if not names:
            raise ValueError("ZSClipDefectStage: keine gültigen Pos-Prompts gefunden.")

        self.class_names = names
        self.text_protos = self.torch.cat(protos, dim=0).to(self.device)  # [C,D]

    # ---------- Image Embedding (TTA) ----------
    def _img_feats(self, pil: Image.Image):
        img = ImageOps.exif_transpose(pil.convert("RGB"))
        W, H = img.size
        feats = []
        for s in self.tta_scales:
            w = int(max(8, W * s)); h = int(max(8, H * s))
            left = max((W - w)//2, 0); top = max((H - h)//2, 0)
            crop = img.crop((left, top, left + w, top + h)).resize((336, 336), Image.BICUBIC)
            x = self.preprocess(crop).unsqueeze(0).to(self.device)
            if self.half and self.device == "cuda":
                x = x.half()
            with self.torch.cuda.amp.autocast(enabled=(self.device == "cuda")):
                f = self.model.encode_image(x)  # [1,D]
            feats.append(self.F.normalize(f.float(), dim=-1))
        if len(feats) == 1:
            return feats[0]  # [1,D]
        return self.F.normalize(self.torch.cat(feats, dim=0).mean(dim=0, keepdim=True), dim=-1)  # [1,D]

    # ---------- Public: classify ----------
    def classify(self, pil: Image.Image) -> Dict[str, Any]:
        """
        Gibt:
          {
            "ok": True,
            "topk": [{"label": str, "score": float}, ...],  # score = Softmax-Prob
            "scores": {klasse: prob}
          }
        """
        if self.text_protos is None or not self.class_names:
            raise RuntimeError("Text-Prototypen nicht initialisiert.")

        f = self._img_feats(pil)  # [1,D]
        sims = (f @ self.text_protos.t()).squeeze(0)        # [C]
        logits = sims / self.tau
        probs = self.torch.softmax(logits, dim=-1)          # [C]

        probs_np = probs.detach().cpu().numpy().tolist()
        pairs = list(zip(self.class_names, probs_np))
        pairs.sort(key=lambda x: x[1], reverse=True)
        topk = [{"label": k, "score": float(v)} for k, v in pairs[: self.topk]]
        return {
            "ok": True,
            "topk": topk,
            "scores": {k: float(v) for k, v in pairs},
        }