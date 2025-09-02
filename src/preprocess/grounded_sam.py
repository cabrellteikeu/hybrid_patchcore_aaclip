# src/preprocess/grounded_sam.py
from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from PIL import Image
import torch
import torchvision
import cv2

class GroundedSAMPreprocessor:
    """
    GroundingDINO -> SAM -> Freistellen (weißer oder schwarzer Hintergrund) -> zentrierter Crop.
    - arbeitet komplett im Speicher (keine Disk-I/O)
    - robust: fällt bei Fehlern auf das Originalbild zurück
    """

    def __init__(
        self,
        dino_config_path: str,
        dino_checkpoint_path: str,
        sam_encoder: str,
        sam_checkpoint_path: str,
        device: str = "cuda",
        classes: Optional[List[str]] = None,
        box_threshold: float = 0.25,
        text_threshold: float = 0.25,
        nms_threshold: float = 0.5,
        target_size: Tuple[int, int] = (1024, 1024),
        background_color: Tuple[int, int, int] = (255, 255, 255),  # BGR für cv2
    ) -> None:
        self.device = torch.device(device if (device == "cuda" and torch.cuda.is_available()) else "cpu")
        self.classes = classes or ["bucket", "paint bucket", "plastic bucket"]
        self.box_threshold = float(box_threshold)
        self.text_threshold = float(text_threshold)
        self.nms_threshold = float(nms_threshold)
        self.target_size = (int(target_size[0]), int(target_size[1]))
        self.background_color = tuple(int(c) for c in background_color)

        # Lazy imports + load
        from groundingdino.util.inference import Model as GdinoModel
        from segment_anything import sam_model_registry, SamPredictor

        # GroundingDINO
        self._gdino = GdinoModel(
            model_config_path=dino_config_path,
            model_checkpoint_path=dino_checkpoint_path,
            device=torch.device(self.device),
        )

        # SAM
        sam = sam_model_registry[sam_encoder](checkpoint=sam_checkpoint_path)
        sam.to(self.device)
        self._sam_predictor = SamPredictor(sam)

    def _best_box_and_mask(self, image_bgr: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float]:
        """Liefert beste (xyxy)-Box, binäre Maske (H,W) und Detektionsscore. None, None wenn nichts gefunden."""
        # GroundingDINO erwartet BGR np.ndarray
        det = self._gdino.predict_with_classes(
            image=image_bgr,
            classes=self.classes,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
        )
        if det.xyxy is None or len(det.xyxy) == 0:
            return None, None, 0.0

        # NMS
        keep = torchvision.ops.nms(
            torch.from_numpy(det.xyxy),
            torch.from_numpy(det.confidence),
            self.nms_threshold,
        ).numpy().tolist()
        if not keep:
            return None, None, 0.0

        xyxy = det.xyxy[keep]
        conf = det.confidence[keep]

        # beste Box = höchste Confidence
        best_idx = int(np.argmax(conf))
        box = xyxy[best_idx].astype(np.float32)  # (x1,y1,x2,y2)
        score = float(conf[best_idx])

        # SAM Maske
        self._sam_predictor.set_image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        masks, scores, _ = self._sam_predictor.predict(box=box, multimask_output=True)
        if masks is None or len(masks) == 0:
            return box, None, score
        m = masks[int(np.argmax(scores))]  # (H,W) bool
        return box, m.astype(np.uint8), score

    def _crop_on_canvas(self, image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Wendet Maske an, croppt auf größte Kontur und zentriert auf Canvas."""
        H, W = mask.shape
        if (H, W) != (image_bgr.shape[0], image_bgr.shape[1]):
            mask = cv2.resize(mask, (image_bgr.shape[1], image_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)

        # größte Kontur
        cnts, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return image_bgr

        cnt = max(cnts, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(cnt)

        crop = image_bgr[y:y+h, x:x+w]
        mask_crop = (mask[y:y+h, x:x+w] > 0).astype(np.uint8) * 255
        obj = cv2.bitwise_and(crop, crop, mask=mask_crop)

        # auf Zielgröße (ohne übermäßiges Upscaling)
        th, tw = self.target_size[1], self.target_size[0]
        ch, cw = obj.shape[:2]
        scale = min(tw / max(cw, 1), th / max(ch, 1))
        new_w, new_h = max(1, int(cw * scale)), max(1, int(ch * scale))
        resized = cv2.resize(obj, (new_w, new_h), interpolation=cv2.INTER_AREA)

        canvas = np.ones((th, tw, 3), dtype=np.uint8)
        canvas[:] = self.background_color
        y_off = (th - new_h) // 2
        x_off = (tw - new_w) // 2
        canvas[y_off:y_off+new_h, x_off:x_off+new_w] = resized
        return canvas

    def prepare(self, pil_image: Image.Image) -> Tuple[Image.Image, Dict[str, Any]]:
        """
        Hauptmethode: nimmt PIL, gibt PIL (freigestellt, zentriert) + Meta zurück.
        Fallback: Originalbild, meta["detected"]=False falls nichts erkannt/segmentiert wurde.
        """
        meta: Dict[str, Any] = {"detected": False, "box": None, "score": 0.0, "used_sampler": "grounded_sam"}
        try:
            img_rgb = np.array(pil_image.convert("RGB"))
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

            box, mask, score = self._best_box_and_mask(img_bgr)
            meta["score"] = float(score)

            if mask is None:
                # Kein Segment → nichts tun
                return pil_image, meta

            prepped_bgr = self._crop_on_canvas(img_bgr, mask)
            out_pil = Image.fromarray(cv2.cvtColor(prepped_bgr, cv2.COLOR_BGR2RGB))
            meta["detected"] = True
            meta["box"] = box.tolist() if box is not None else None
            meta["target_size"] = {"w": self.target_size[0], "h": self.target_size[1]}
            meta["background_bgr"] = list(self.background_color)
            return out_pil, meta

        except Exception as e:
            meta["error"] = str(e)
            return pil_image, meta