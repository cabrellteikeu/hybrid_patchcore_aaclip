# utils/processor.py
from __future__ import annotations
import os, yaml, time, json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw
import cv2  # für Map-Resize bei Fusions-Overlay

from src.patchcore.patchcore_infer import PatchCoreStage
from src.classify.aaclip_wrapper import AAClipStage
from src.anomaly_dino.dino_stage import DinoAnomalyStage
from src.classify.zsclip_defect_stage import ZSClipDefectStage
from src.preprocess.grounded_sam import GroundedSAMPreprocessor
from src.defects.defect_prompts import DEFECT_PROMPTS

import io, yaml
from pathlib import Path

CFG = yaml.safe_load(Path("config/hybrid_config.yaml").read_text(encoding="utf-8"))

def _visualize_grid(img: Image.Image, anomaly_map: np.ndarray, threshold: float) -> Image.Image:
    """Raster-Overlay: anomaly_map kann grob- oder feinauflösend sein; wir zeichnen Zellrahmen > threshold."""
    out = img.copy()
    draw = ImageDraw.Draw(out)
    w, h = out.size
    H, W = anomaly_map.shape[:2]
    cell_w = max(1, w // W)
    cell_h = max(1, h // H)
    for yy in range(H):
        for xx in range(W):
            if float(anomaly_map[yy, xx]) > threshold:
                x0, y0 = xx * cell_w, yy * cell_h
                x1, y1 = min(w, (xx + 1) * cell_w), min(h, (yy + 1) * cell_h)
                draw.rectangle([x0, y0, x1, y1], outline="red", width=2)
    return out


def _top3_labels_from_topk(topk_obj) -> List[str]:
    """Extrahiert genau die 3 besten Labels (ohne Scores)."""
    labels = []
    for item in (topk_obj or [])[:3]:
        if isinstance(item, dict):
            labels.append(str(item.get("label") or item.get("name") or item.get("class") or item))
        elif isinstance(item, (list, tuple)) and item:
            labels.append(str(item[0]))
        else:
            labels.append(str(item))
    # Duplikate kompakt
    seen, uniq = set(), []
    for x in labels:
        if x not in seen:
            seen.add(x); uniq.append(x)
    # exakt 3 Einträge zurückgeben (ggf. auffüllen mit "")
    while len(uniq) < 3:
        uniq.append("")
    return uniq[:3]


class ImageProcessor:
    """
    Pipeline (per cfg + optional CLI-Override umschaltbar):
      - mode:
          "pc_only":   PatchCore -> (Anomalie) Klassifizierer
          "dino_only": DINOv2    -> (Anomalie) Klassifizierer
          "ensemble":  PatchCore + DINOv2 (Score/Map-Fusion) -> (Anomalie) Klassifizierer
      - classifier:
          "zsclip": Zero-Shot OpenCLIP
          "aaclip": dein AA-CLIP Wrapper
          "none":   keine Textklassifikation

    Schreibt nur Overlay nach CFG['processed_dir'].
    """
    def __init__(
        self,
        cfg_path: str = "config/hybrid_config.yaml",
        mode_override: Optional[str] = None,
        classifier_override: Optional[str] = None,
    ) -> None:
        self.cfg = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8"))

        # Defaults aus Config + optionale CLI-Overrides
        self.mode: str = (self.cfg.get("pipeline", {}) or {}).get("mode", "pc_only").lower()
        if mode_override:
            self.mode = mode_override.lower()

        self.classifier: str = (self.cfg.get("pipeline", {}) or {}).get("classifier", "zsclip").lower()
        if classifier_override:
            self.classifier = classifier_override.lower()

        self.explain_on_anomaly: bool = bool((self.cfg.get("pipeline", {}) or {}).get("explain_on_anomaly", True))

        # ---- Grounded-SAM
        self.pre = None
        pp = self.cfg.get("preprocess", {}) or {}
        if pp.get("enabled", False):
            try:
                self.pre = GroundedSAMPreprocessor(
                    dino_config_path=pp["dino"]["config"],
                    dino_checkpoint_path=pp["dino"]["checkpoint"],
                    sam_encoder=pp["sam"]["encoder"],
                    sam_checkpoint_path=pp["sam"]["checkpoint"],
                    device=pp.get("device", self.cfg.get("device", "cuda")),
                    classes=pp.get("classes", None),
                    box_threshold=float(pp["dino"].get("box_threshold", 0.25)),
                    text_threshold=float(pp["dino"].get("text_threshold", 0.25)),
                    nms_threshold=float(pp["dino"].get("nms_threshold", 0.5)),
                    target_size=tuple(pp.get("target_size", [1024, 1024])),
                    background_color=tuple(pp.get("background_bgr", [255, 255, 255])),
                )
            except Exception as e:
                try:
                    from utils.logger import logger
                    logger.warning(f"Preprocessor init failed: {e}")
                except Exception:
                    print(f"[WARN] Preprocessor init failed: {e}")
                self.pre = None

        # ---- PatchCore
        self.pc_tau: float = float(self.cfg["patchcore"]["threshold"])
        self.patch = PatchCoreStage(
            ckpt_path=self.cfg["patchcore"]["ckpt_path"],
            image_size=tuple(self.cfg["patchcore"]["image_size"]),
            device=self.cfg["device"],
            threshold=self.pc_tau,
        )

        # ---- DINOv2 (nur wenn benötigt)
        self.dino = None
        self.dino_tau: float = float((self.cfg.get("dino") or {}).get("threshold", 0.25))
        if self.mode in ("dino_only", "ensemble") and (self.cfg.get("dino", {}).get("enabled", True)):
            self.dino = DinoAnomalyStage(
                model_name=(self.cfg["dino"]["model_name"]),
                device=self.cfg["device"],
                bank_dir=self.cfg["dino"]["bank_dir"],
                max_bank_patches=int(self.cfg["dino"]["max_bank_patches"]),
                top_p=float(self.cfg["dino"]["top_p"]),
                use_half=bool(self.cfg["dino"]["use_half"]),
                sim_chunk_size=int(self.cfg["dino"]["sim_chunk_size"]),
            )

        # ---- Klassifizierer
        self.zsclip = None
        if self.classifier == "zsclip":
            zsc_cfg = (self.cfg.get("zsclip") or {})
            from src.defects.defect_prompts import DEFECT_PROMPTS as _PROMPTS
            self.zsclip = ZSClipDefectStage(
                model_name=zsc_cfg.get("model_name", "ViT-B-16"),
                pretrained=zsc_cfg.get("pretrained", "openai"),
                device=self.cfg.get("device", "cuda"),
                tta_scales=tuple(zsc_cfg.get("tta_scales", [1.0, 0.9, 1.1])),
                temperature=float(zsc_cfg.get("temperature", 0.01)),
                topk=int(zsc_cfg.get("topk", 3)),
                defect_prompts=_PROMPTS,
            )

        self.aac = None
        if self.classifier == "aaclip":
            self.aac = AAClipStage(
                repo_root=self.cfg["aaclip"]["repo_root"],
                data_root=self.cfg["aaclip"]["data_root"],
                model_name=self.cfg["aaclip"]["model_name"],
                shot=int(self.cfg["aaclip"]["shot"]),
                save_root=self.cfg["aaclip"]["save_root"],
                defect_prompts=DEFECT_PROMPTS,
                topk=int(self.cfg["aaclip"]["topk"]),
                timeout_sec=int(self.cfg["aaclip"]["timeout_sec"]),
                return_base64=self.cfg["output"]["return_base64"],
                enable_repo=bool(self.cfg["aaclip"].get("enable_repo", False)),
                temperature=float(self.cfg["aaclip"].get("temperature", 0.10)),
                text_weight=float(self.cfg["aaclip"].get("text_weight", 0.4)),
                tta_scales=tuple(self.cfg["aaclip"].get("tta_scales", [0.90, 1.00, 1.10])),
                cache_dir=self.cfg["aaclip"].get("cache_dir", "cache/openclip_vitl14_336"),
                support_dir=self.cfg["aaclip"].get("support_dir", None),
                max_support_per_class=int(self.cfg["aaclip"].get("max_support_per_class", 20)),
            )

        self.processed_dir = Path(CFG["processed_dir"]).resolve()
        self.processed_dir.mkdir(parents=True, exist_ok=True)



    # -------------- interner Helper: Fusion --------------
    def _fuse_maps(self, img: Image.Image, pc_map: np.ndarray, dino_map: np.ndarray) -> Tuple[np.ndarray, float]:
        """Bringt beide Maps auf Bildgröße, normalisiert jede Map durch ihren eigenen Threshold
        und nimmt elementweises Maximum. Rückgabe-Threshold ist dann fix 1.0."""
        H, W = img.size[1], img.size[0]

        def _resize_hw(m: np.ndarray) -> np.ndarray:
            if m is None:
                return np.zeros((H, W), np.float32)
            m = m.astype(np.float32)
            return cv2.resize(m, (W, H), interpolation=cv2.INTER_CUBIC) if m.shape != (H, W) else m

        pc_res = _resize_hw(pc_map) / max(self.pc_tau, 1e-6)
        dn_res = _resize_hw(dino_map) / max(self.dino_tau, 1e-6)
        fused_map = np.maximum(pc_res, dn_res)
        return fused_map, 1.0

    # -------------- interner Helper: Klassifikation -> 3 Labels --------------
    def _classify_top3(self, img: Image.Image) -> List[str]:
        if self.classifier == "none":
            return ["", "", ""]

        if self.classifier == "zsclip":
            if self.zsclip is None:
                # Safety: initialisiere on-demand (falls Config wechselt)
                zsc_cfg = (self.cfg.get("zsclip") or {})
                self.zsclip = ZSClipDefectStage(
                    model_name=zsc_cfg.get("model_name", "ViT-B-16"),
                    pretrained=zsc_cfg.get("pretrained", "openai"),
                    device=self.cfg.get("device", "cuda"),
                    tta_scales=tuple(zsc_cfg.get("tta_scales", [1.0, 0.9, 1.1])),
                    temperature=float(zsc_cfg.get("temperature", 0.01)),
                    topk=int(zsc_cfg.get("topk", 3)),
                    defect_prompts=DEFECT_PROMPTS,
                )
            z = self.zsclip.classify(img)
            return _top3_labels_from_topk(z.get("topk"))

        if self.classifier == "aaclip":
            if self.aac is None:
                # Safety: initialisiere on-demand
                self.aac = AAClipStage(
                    repo_root=self.cfg["aaclip"]["repo_root"],
                    data_root=self.cfg["aaclip"]["data_root"],
                    model_name=self.cfg["aaclip"]["model_name"],
                    shot=int(self.cfg["aaclip"]["shot"]),
                    save_root=self.cfg["aaclip"]["save_root"],
                    defect_prompts=DEFECT_PROMPTS,
                    topk=int(self.cfg["aaclip"]["topk"]),
                    timeout_sec=int(self.cfg["aaclip"]["timeout_sec"]),
                    return_base64=self.cfg["output"]["return_base64"],
                    enable_repo=bool(self.cfg["aaclip"].get("enable_repo", False)),
                    temperature=float(self.cfg["aaclip"].get("temperature", 0.10)),
                    text_weight=float(self.cfg["aaclip"].get("text_weight", 0.4)),
                    tta_scales=tuple(self.cfg["aaclip"].get("tta_scales", [0.90, 1.00, 1.10])),
                    cache_dir=self.cfg["aaclip"].get("cache_dir", "cache/openclip_vitl14_336"),
                    support_dir=self.cfg["aaclip"].get("support_dir", None),
                    max_support_per_class=int(self.cfg["aaclip"].get("max_support_per_class", 20)),
                )
            b = self.aac.infer(img)
            return _top3_labels_from_topk(b.get("topk"))

        # Fallback
        return ["", "", ""]

    # -------------- Hauptlogik: Einzelbild --------------
    def process(self, image_path: Path) -> Dict[str, Any]:
        p = Path(image_path)
        ts = datetime.now().isoformat()
        out: Dict[str, Any] = {
            "timestamp": ts,
            "image_path": str(p),
            "status": "pending",
            "anomaly_score": 0.0,
            "is_good": True,
            "overlay_path": None,
            "mode": self.mode,
            "classifier": self.classifier,
        }
        try:
            with Image.open(p) as im:
                img = im.convert("RGB")

            prep_used, prep_meta = False, {}
            if self.pre is not None:
                img, prep_meta = self.pre.prepare(img)
                prep_used = bool(prep_meta.get("detected", False)) or True

            # ---- Variante A: nur PatchCore
            if self.mode == "pc_only":
                a = self.patch.predict(img)  # {"score","is_anomalous","anomaly_map"}
                pc_score = float(a["score"])
                is_anom = bool(a["is_anomalous"])
                overlay_map = a["anomaly_map"]
                used_score = pc_score
                used_tau = self.pc_tau

                out.update({
                    "status": "success",
                    "anomaly_score": used_score,
                    "is_good": (not is_anom),
                    "scores": {"patchcore": pc_score},
                    "thresholds": {"patchcore": self.pc_tau},
                })

            # ---- Variante B: nur DINOv2
            elif self.mode == "dino_only":
                if self.dino is None:
                    raise RuntimeError("DINO stage not initialized (check config.dino.enabled).")
                dino_score, dino_map, dino_meta = self.dino.score_image(
                    sku_id="default", view="single", img=img, require_bank=False, return_heatmap=True
                )
                overlay_map = dino_map if dino_map is not None else np.zeros((img.size[1], img.size[0]), np.float32)
                used_score = float(dino_score)
                used_tau = self.dino_tau
                is_anom = bool(used_score > used_tau)

                out.update({
                    "status": "success",
                    "anomaly_score": used_score,
                    "is_good": (not is_anom),
                    "scores": {"dino": used_score},
                    "thresholds": {"dino": self.dino_tau},
                    "dino_meta": dino_meta,
                    "dino_bank_present": bool(isinstance(dino_meta, dict) and dino_meta.get("used_bank", 0) > 0),
                })

            # ---- Variante C: Ensemble (PatchCore + DINOv2)
            elif self.mode == "ensemble":
                if self.dino is None:
                    raise RuntimeError("DINO stage not initialized (check config.dino.enabled).")

                a = self.patch.predict(img)
                pc_score = float(a["score"])
                pc_map = a["anomaly_map"]

                dino_score, dino_map, dino_meta = self.dino.score_image(
                    sku_id="default", view="single", img=img, require_bank=False, return_heatmap=True
                )
                if dino_map is None:
                    dino_map = np.zeros((img.size[1], img.size[0]), np.float32)

                # ---- normalisierte Scores (gegen jeweilige Tauschschwelle)
                norm_pc = pc_score / max(self.pc_tau, 1e-6)
                norm_dn = float(dino_score) / max(self.dino_tau, 1e-6)
                fused_norm = max(norm_pc, norm_dn)

                # ---- normierte Heatmaps fusionieren (-> Threshold=1.0)
                fused_map, fused_tau = self._fuse_maps(img, pc_map, dino_map)

                # ---- Entscheidung auf normalisierter Skala
                review_margin = float((self.cfg.get("pipeline", {}) or {}).get("review_margin", 0.20))
                if fused_norm <= 1.0:
                    state = "OK"
                elif fused_norm >= 1.0 * (1.0 + review_margin):
                    state = "DEFECT"
                else:
                    state = "REVIEW"

                is_anom = (state != "OK")
                overlay_map = fused_map
                used_score = fused_norm     # normierter Score
                used_tau = fused_tau        # = 1.0

                out.update({
                    "status": "success",
                    "anomaly_score": used_score,
                    "is_good": (not is_anom),
                    "scores": {
                        "patchcore_raw": pc_score,
                        "dino_raw": float(dino_score),
                        "patchcore_norm": float(norm_pc),
                        "dino_norm": float(norm_dn),
                        "fused_norm": float(fused_norm),
                    },
                    "thresholds": {
                        "patchcore": self.pc_tau,
                        "dino": self.dino_tau,
                        "fused_norm": 1.0,
                        "review_margin": review_margin,
                    },
                    "dino_meta": dino_meta,
                    "dino_bank_present": bool(isinstance(dino_meta, dict) and dino_meta.get("used_bank", 0) > 0),
                })

            else:
                raise ValueError(f"Unknown pipeline.mode='{self.mode}'")

            out["preprocess_used"] = prep_used
            if prep_meta:
                out["preprocess_meta"] = prep_meta
                
            # ---- Overlay erzeugen (nur das landet im processed-Ordner)
            overlay = _visualize_grid(img, overlay_map, used_tau)
            overlay_path = self.processed_dir / f"{p.stem}_overlay.jpg"
            overlay.save(overlay_path)
            out["overlay_path"] = str(overlay_path)

            # ---- Nur bei Anomalie: 3 Defekt-Labels (je nach Klassifizierer)
            if (not out["is_good"]) and self.classifier in ("zsclip", "aaclip"):
                labels_top3 = self._classify_top3(img)
                out["defect_labels"] = labels_top3

        except Exception as e:
            out.update({"status": "error", "error": str(e), "is_good": False})
        return out

    # -------------- Paarweise Auswertung (2 Kameras) --------------
    def evaluate_bucket(self, cam1_path: Path, cam2_path: Path) -> Dict[str, Any]:
        cam1 = self.process(cam1_path)
        cam2 = self.process(cam2_path)
        good = bool(cam1.get("is_good")) and bool(cam2.get("is_good"))
        return {
            "status": "success" if cam1["status"] == "success" and cam2["status"] == "success" else "error",
            "timestamp": datetime.now().isoformat(),
            "bucket_id": str(int(time.time() * 1000))[-8:],
            "bucket_result": "gut" if good else "nicht_gut",
            "cam1": cam1,
            "cam2": cam2,
        }