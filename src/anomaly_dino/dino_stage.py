# src/run_hybrid/dino_stage.py
# DINOv2-kNN "AnomalyDINO"-Style Stage
# - train-free: nutzt DINOv2 Patch-Features aus timm
# - Bank (Support, nur "good") + kNN (max cosine) pro Patch
# - Bildscore: Mittel der Top-p% Patchscores
# - keine ROI/Calibration; pure Scoring-Stage

from __future__ import annotations
import os
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image
import glob
import timm
from timm.data import resolve_data_config, create_transform


def _l2n(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, dim=-1)


def _to_np(x: torch.Tensor) -> np.ndarray:
    return x.detach().float().cpu().numpy()


class DinoAnomalyStage:
    """
    DINOv2-basiertes, trainingsfreies AD:
      - DINOv2 ViT extrahiert Patch-Features
      - Feature-Bank (nur "good") je (sku_id, view)
      - Patchscore = 1 - max_cosine_sim(query_patch, bank)
      - Heatmap = Patchscore-Gitter -> auf Bildgröße
      - Bildscore = Mittel der Top-p% Patchscores

    Keine Kalibrierung, keine ROI—bewusst minimal und robust.
    """

    SUPPORTED_MODELS = {
        # erprobte timm DINOv2 Modellnamen (timm==1.0.19)
        "vit_small_patch14_dinov2.lvd142m",
        "vit_base_patch14_dinov2.lvd142m",
        "vit_large_patch14_dinov2.lvd142m",
        "vit_giant_patch14_dinov2.lvd142m",
    }

    def __init__(
        self,
        model_name: str = "vit_large_patch14_dinov2.lvd142m",
        device: str = "cuda",
        bank_dir: str = "banks/dino",
        max_bank_patches: int = 20000,
        top_p: float = 0.02,
        use_half: bool = True,
        sim_chunk_size: int = 32768,   # kNN in Stücken (N x chunk)
    ):
        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported model_name='{model_name}'. "
                f"Use one of: {sorted(self.SUPPORTED_MODELS)}"
            )
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.bank_dir = Path(bank_dir); self.bank_dir.mkdir(parents=True, exist_ok=True)
        self.max_bank = int(max_bank_patches)
        self.top_p = float(top_p)
        self.use_half = bool(use_half) and (self.device.type == "cuda")
        self.sim_chunk = int(sim_chunk_size)

        # ---- Modell & Transform laden (timm, DINOv2)
        self.model = timm.create_model(model_name, pretrained=True)
        self.model.eval().to(self.device)
        if self.use_half:
            self.model.half()
            torch.backends.cudnn.benchmark = True

        cfg = resolve_data_config({}, model=self.model)
        # create_transform liefert exakt das Pretrain-Preprocessing (Resize/Norm/CenterCrop etc.)
        self.tf = create_transform(**cfg, is_training=False)

        # Meta (Grid/ Patch)
        self.patch_size = getattr(self.model.patch_embed, "patch_size", (14, 14))
        self.grid_size = getattr(self.model.patch_embed, "grid_size", None)

    # ---------- Patch-Tokens extrahieren (ohne CLS) ----------
    @torch.no_grad()
    def _vit_patch_tokens(self, img: Image.Image) -> Tuple[torch.Tensor, Tuple[int,int], Tuple[int,int]]:
        """
        PIL Image -> (patch_feats [N,C] L2-normalisiert, (Gh,Gw), (H,W))
        """
        x = self.tf(img).unsqueeze(0).to(self.device)
        if self.use_half:
            x = x.half()

        m = self.model
        # Repliziere forward_features bis Token-Sequenz (timm VisionTransformer)
        # 1) Patch-Embedding
        x = m.patch_embed(x)  # [B, N, C]
        B, N, C = x.shape

        # 2) optional cls_token + pos_embed
        if hasattr(m, "cls_token") and m.cls_token is not None:
            cls = m.cls_token.expand(B, -1, -1)  # [B,1,C]
            x = torch.cat((cls, x), dim=1)       # [B,1+N,C]
        if hasattr(m, "pos_embed") and m.pos_embed is not None:
            x = x + m.pos_embed
        if hasattr(m, "pos_drop") and m.pos_drop is not None:
            x = m.pos_drop(x)

        # 3) Blocks
        for blk in m.blocks:
            x = blk(x)

        # 4) Norm
        if hasattr(m, "norm") and m.norm is not None:
            x = m.norm(x)

        # 5) Patch-Tokens (ohne CLS)
        if x.shape[1] == N + 1:
            x = x[:, 1:, :]
        else:
            x = x[:, :N, :]
        x = x.squeeze(0)          # [N, C]
        x = _l2n(x).float()       # float32 für stabile Cosine-Sim (auch wenn Model in FP16 läuft)

        # Grid (Gh, Gw)
        if self.grid_size is not None:
            Gh, Gw = int(self.grid_size[0]), int(self.grid_size[1])
        else:
            Gw = int(np.sqrt(N) + 1e-6)
            Gh = int(np.ceil(N / Gw))

        H, W = img.size[1], img.size[0]
        return x, (Gh, Gw), (H, W)

    # ---------- Bank-Dateipfad ----------
    def _bank_path(self, sku_id: str, view: str) -> Path:
        return self.bank_dir / f"{sku_id}__{view}.pt"

    # ---------- Bank Laden/Speichern ----------
    def _load_bank(self, sku_id: str, view: str) -> Optional[torch.Tensor]:
        p = self._bank_path(sku_id, view)
        if not p.exists():
            return None
        obj = torch.load(p, map_location="cpu")
        if isinstance(obj, dict) and "feats" in obj:
            feats = obj["feats"]
        elif isinstance(obj, torch.Tensor):
            feats = obj
        else:
            return None
        # auf Device; L2-normalisiert & float32
        feats = _l2n(feats.float()).to(self.device, non_blocking=True)
        return feats

    def _save_bank(self, sku_id: str, view: str, feats: torch.Tensor) -> None:
        self.bank_dir.mkdir(parents=True, exist_ok=True)
        # als float32 CPU speichern
        torch.save({"feats": feats.detach().float().cpu()}, self._bank_path(sku_id, view))

    # ---------- Enrollment: Good-Bilder -> Bank ----------
    @torch.no_grad()
    def enroll(self, sku_id: str, view: str, folder_path: str, shuffle: bool = True) -> Dict:

        image_paths = glob.glob(os.path.join(folder_path,"*.jpg")) + glob.glob(os.path.join(folder_path,"*.png")) + glob.glob(os.path.join(folder_path,"*.jpeg"))
        
        if not image_paths:
            return {"ok":False, "error":"no images found in folder"}
        
        feats_all: List[torch.Tensor] = []

        for p in image_paths:
            try:
                img = Image.open(p).convert("RGB")
            except Exception:
                continue
            f, _, _ = self._vit_patch_tokens(img)   # [N,C], float32
            feats_all.append(f)

        if not feats_all:
            return {"ok": False, "error": "no valid images"}

        bank = torch.cat(feats_all, dim=0)  # [M,C]
        # optional mischen + begrenzen (leichtes Coreset)
        if shuffle:
            idx = torch.randperm(bank.shape[0])
            bank = bank[idx]
        if bank.shape[0] > self.max_bank:
            bank = bank[: self.max_bank]

        bank = _l2n(bank)   # safety
        self._save_bank(sku_id, view, bank)
        return {"ok": True, "bank_size": int(bank.shape[0])}

    # ---------- Inferenz: Score + Heatmap ----------
    @torch.no_grad()
    def score_image(
        self,
        sku_id: str,
        view: str,
        img: Image.Image,
        require_bank: bool = True,
        return_heatmap: bool = True,
    ) -> Tuple[float, Optional[np.ndarray], Dict]:
        """
        Rückgabe:
          img_score, heatmap(np.float32 HxW | None), meta(dict)
        """
        bank = self._load_bank(sku_id, view)
        if bank is None:
            if require_bank:
                raise RuntimeError(f"No DINO bank for '{sku_id}::{view}'. Call enroll first.")
            else:
                # Degenerate: ohne Bank kein sinnvoller Score
                H, W = img.size[1], img.size[0]
                hm = np.zeros((H, W), np.float32) if return_heatmap else None
                return 0.0, hm, {"used_bank": 0}

        q, (Gh, Gw), (H, W) = self._vit_patch_tokens(img)  # q: [N,C], bank: [M,C]
        # Cosine-Sim per Dot-Product (Features sind L2-normalisiert)
        N, C = q.shape
        M = bank.shape[0]

        # Chunked Multiplikation: (q @ bank.T) ohne OOM
        max_sim = torch.empty(N, dtype=torch.float32, device=self.device)
        bt = bank.t().contiguous()  # [C, M]
        # Rechne in Blöcken über die Patches (q)
        bs = max(1, min(self.sim_chunk // max(1, M), N))  # einfache Heuristik
        for i in range(0, N, bs):
            j = min(i + bs, N)
            s = q[i:j].matmul(bt)            # [bs, M]
            # pro Patch die maximale Ähnlichkeit
            max_sim[i:j], _ = s.max(dim=1)

        patch_scores = (1.0 - max_sim.clamp(-1, 1)).float()  # [N]

        # Heatmap
        hm_np: Optional[np.ndarray] = None
        if return_heatmap:
            hm = patch_scores.view(Gh, Gw)                 # [Gh, Gw]
            hm_np = _to_np(hm)
            hm_np = cv2.resize(hm_np, (W, H), interpolation=cv2.INTER_CUBIC).astype(np.float32)

        # Bildscore: Mittel der Top-p% Patchscores
        n = patch_scores.numel()
        k = max(1, int(np.ceil(self.top_p * n)))
        topk_vals, _ = torch.topk(patch_scores, k, largest=True)
        img_score = float(topk_vals.mean().item())

        meta = {"used_bank": int(M), "Gh": Gh, "Gw": Gw, "num_patches": int(N)}
        return img_score, hm_np, meta
