# src/run_hybrid/zeroshot_clip.py
from __future__ import annotations
import math, os, json, time, traceback
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from PIL import Image, ImageOps

import torch
import torch.nn.functional as F

import open_clip as openclip


def _l2n(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, dim=-1)


def _to_np(x: torch.Tensor) -> np.ndarray:
    return x.detach().float().cpu().numpy()


def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


class ZeroShotClipStage:
    """
    Reiner Zero-Shot Defektklassifizierer:
      - OpenCLIP Image/Text-Encoder
      - Prompt-Ensembles je Klasse (DEFECT_PROMPTS)
      - Fenster-/Scale-basiertes Scoring (WinCLIP-Style), keine Support-Bilder
      - Score je Klasse = max über alle Fenster (Text-Ähnlichkeit)
      - Heatmap aus Fenster-Scores (für Top-1 Klasse)
    """

    def __init__(
        self,
        model_name: str = "ViT-L-14-336",
        pretrained: str = "openai",
        defect_prompts: Optional[Dict[str, List[str]]] = None,
        device: str = "cuda",
        temperature: float = 0.10,
        topk: int = 3,
        scales: Tuple[float, ...] = (0.70, 0.90, 1.00),
        overlap: float = 0.50,
        batch_size: int = 64,
        use_half: bool = True,
        cache_dir: Optional[str] = None,
        normal_prompts: Optional[List[str]] = None, rel_alpha: float = 1.0, topq: float = 0.02
    ):
        if defect_prompts is None or not isinstance(defect_prompts, dict) or len(defect_prompts) == 0:
            raise ValueError("ZeroShotClipStage: defect_prompts (dict) ist erforderlich.")
        self.class_names: List[str] = list(defect_prompts.keys())
        self.defect_prompts: Dict[str, List[str]] = defect_prompts

        self.device = "cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu"
        self.temperature = float(max(1e-3, temperature))
        self.topk = int(topk)
        self.scales = tuple(float(s) for s in scales)
        self.overlap = float(np.clip(overlap, 0.0, 0.95))
        self.batch_size = int(max(1, batch_size))
        self.use_half = bool(use_half) and (self.device == "cuda")
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir: _ensure_dir(self.cache_dir)
        self.normal_prompts = normal_prompts or []
        self.rel_alpha = float(rel_alpha)
        self.topq = float(topq)
        # OpenCLIP laden
        self.model, _, self.preprocess = openclip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.tokenizer = openclip.get_tokenizer(model_name)
        self.model = self.model.to(self.device).eval()
        if self.use_half:
            self.model.half()
            torch.backends.cudnn.benchmark = True
        torch.set_grad_enabled(False)

        # Text-Cache bauen
        self._text_cache: Dict[str, torch.Tensor] = {}
        self._build_text_cache()
        

    # --------- Text-Cache (Prompt-Ensembles je Klasse) -----------
    def _text_cache_path(self, cname: str) -> Optional[Path]:
        if not self.cache_dir: return None
        tag = f"{cname}-{len(self.defect_prompts.get(cname, []))}"
        return self.cache_dir / f"text_{tag}.pt"

    def _build_text_cache(self):
        self._text_cache.clear()
        dev = self.device
        for cname, variants in self.defect_prompts.items():
            if not variants: continue
            pt = self._text_cache_path(cname)
            if pt and pt.exists():
                try:
                    tfeat = torch.load(pt, map_location=dev)
                    if isinstance(tfeat, torch.Tensor) and tfeat.ndim == 2:
                        self._text_cache[cname] = _l2n(tfeat.float().to(dev))
                        continue
                except Exception:
                    pass
            toks = self.tokenizer(variants).to(dev)
            with torch.cuda.amp.autocast(enabled=(self.use_half and dev == "cuda")):
                tfeat = self.model.encode_text(toks)  # [V,D]
            tfeat = _l2n(tfeat.float())
            self._text_cache[cname] = tfeat
            if pt:
                try: torch.save(tfeat.cpu(), pt)
                except Exception: pass

        self._tfeat_clean = None
        if self.normal_prompts:
            toks = self.tokenizer(self.normal_prompts).to(self.device)
            with torch.cuda.amp.autocast(enabled=(self.use_half and self.device=="cuda")):
                tfeat = self.model.encode_text(toks)   # [V,D]
            self._tfeat_clean = _l2n(tfeat.float())    # [V,D]

    # --------- Sliding-Window Generator (pro Scale) -----------
    def _windows_for_scale(self, W: int, H: int, s: float) -> List[Tuple[int,int,int,int]]:
        # Fenstergröße anhand kleinerer Kante
        win = int(round(min(W, H) * s))
        win = max(16, min(win, min(W, H)))  # min/max Klammer
        if win >= min(W, H):
            # nur 1 Fenster (full crop)
            return [(0, 0, W, H)]
        stride = int(round(win * (1.0 - self.overlap)))
        stride = max(1, stride)
        boxes = []
        for y0 in range(0, H - win + 1, stride):
            for x0 in range(0, W - win + 1, stride):
                boxes.append((x0, y0, x0 + win, y0 + win))
        # Ränder abdecken
        if (H - win) % stride != 0:
            y0 = H - win
            for x0 in range(0, W - win + 1, stride):
                boxes.append((x0, y0, x0 + win, y0 + win))
        if (W - win) % stride != 0:
            x0 = W - win
            for y0 in range(0, H - win + 1, stride):
                boxes.append((x0, y0, x0 + win, y0 + win))
        # Ecke unten rechts
        boxes.append((W - win, H - win, W, H))
        # Duplikate raus
        boxes = list(dict.fromkeys(boxes))
        return boxes

    # --------- Batch-Encode der Fenster ----------
    def _encode_windows(self, img: Image.Image, boxes: List[Tuple[int,int,int,int]]) -> torch.Tensor:
        dev = self.device
        ims = []
        for (x0, y0, x1, y1) in boxes:
            crop = img.crop((x0, y0, x1, y1))
            ims.append(self.preprocess(crop))
        tens = torch.stack(ims).to(dev)  # [N,3,H,W]
        if self.use_half: tens = tens.half()
        feats = []
        with torch.cuda.amp.autocast(enabled=(self.use_half and dev == "cuda")):
            for i in range(0, tens.shape[0], self.batch_size):
                j = min(i + self.batch_size, tens.shape[0])
                f = self.model.encode_image(tens[i:j])  # [B,D]
                feats.append(f)
        feats = torch.cat(feats, dim=0)  # [N,D]
        return _l2n(feats.float())

    # --------- Scoring ----------
    def _class_scores_from_feats(self, img_feats: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        img_feats: [N,D] Fenster-Features
        returns dict: cname -> [N] Scores (pro Fenster)
        """
        tau = self.temperature
        out: Dict[str, torch.Tensor] = {}
        for cname, tfeat in self._text_cache.items():  # tfeat: [V,D]
            sims = img_feats @ tfeat.t()               # [N,V]
            # Softmax über Varianten, dann Mittel -> stabil
            probs = torch.softmax(sims / tau, dim=-1)  # [N,V]
            sc = probs.mean(dim=-1)                    # [N]
            out[cname] = sc
        return out

    def _class_scores_relative(self, img_feats: torch.Tensor) -> Dict[str, torch.Tensor]:
        """CLIP-AD-artig: pro Fenster Klassen-Score relativ zu 'clean'."""
        tau = self.temperature
        out = {}
        if self._tfeat_clean is None:
            # Fallback: absolut (wie vorher)
            return self._class_scores_from_feats(img_feats)
        # clean-Score pro Fenster (maximum über clean-Varianten)
        clean_sim = (img_feats @ self._tfeat_clean.t())             # [N,Vc]
        clean_max, _ = clean_sim.max(dim=1)                         # [N]
        for cname, tfeat in self._text_cache.items():
            sim = (img_feats @ tfeat.t()).max(dim=1).values         # [N] max über Varianten
            # relatives Logit: defekt - alpha*clean
            rel = sim - self.rel_alpha * clean_max
            # optional: Softmax gegen clean (2-Klassen-Logit)
            pair = torch.stack([rel, torch.zeros_like(rel)], dim=1) # [N,2], zweite Logit = 0 als Referenz
            prob = torch.softmax(pair / tau, dim=1)[:, 0]           # [N]
            out[cname] = prob
        return out

    def _heatmap_from_windows(self, W: int, H: int, boxes: List[Tuple[int,int,int,int]], win_scores: torch.Tensor) -> np.ndarray:
        """
        verteilt Fenster-Scores (für die gewählte Klasse) über Bildraster (H,W)
        einfache "add & count" Normalisierung -> [0..1]
        """
        heat = np.zeros((H, W), np.float32)
        cnt  = np.zeros((H, W), np.float32)
        s_np = _to_np(win_scores).astype(np.float32)   # [N]
        for (x0, y0, x1, y1), s in zip(boxes, s_np):
            heat[y0:y1, x0:x1] += s
            cnt[y0:y1, x0:x1]  += 1.0
        mask = cnt > 0
        heat[mask] = heat[mask] / cnt[mask]
        # Normierung 0..1 für Visualisierung
        if heat.max() > heat.min():
            heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-6)
        return heat

    # --------- Public API ----------
    def predict(self, pil_image: Image.Image) -> Dict:
        """
        Rückgabe:
          {
            "topk": [{"label": cname, "score": float}, ...],
            "heatmap": np.ndarray(H,W) in [0..1] (für beste Klasse),
            "best_class": "label"
          }
        """
        img = ImageOps.exif_transpose(pil_image.convert("RGB"))
        W, H = img.size

        # Fenster über alle Scales sammeln
        all_boxes: List[Tuple[int,int,int,int]] = []
        for s in self.scales:
            all_boxes += self._windows_for_scale(W, H, s)
        if not all_boxes:
            all_boxes = [(0, 0, W, H)]

        # Fenster batchweise encoden
        img_feats = self._encode_windows(img, all_boxes)  # [N,D]

        # Klassen-Scores pro Fenster
        per_class = self._class_scores_from_feats(img_feats)  # cname -> [N]

        # Bildscore je Klasse = max über Fenster
        scores = []
        for cname, vec in per_class.items():
            scores.append((cname, float(vec.max().item())))
        scores.sort(key=lambda x: x[1], reverse=True)
        topk = [{"label": k, "score": v} for k, v in scores[: self.topk]]

        # Heatmap für Top-1 Klasse
        best_class = scores[0][0] if scores else None
        heat = None
        if best_class is not None:
            win_scores = per_class[best_class]  # [N]
            heat = self._heatmap_from_windows(W, H, all_boxes, win_scores)

        return {"topk": topk, "heatmap": heat, "best_class": best_class}