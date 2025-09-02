import torch
from PIL import Image
from torchvision import transforms
from anomalib.models import Patchcore
import io, yaml
from pathlib import Path

CFG = yaml.safe_load(Path("config/hybrid_config.yaml").read_text(encoding="utf-8"))

class PatchCoreStage:
    def __init__(self, ckpt_path: str, image_size=(256,256), device=CFG["device"], threshold=CFG["patchcore"]["threshold"]):
        self.model = Patchcore.load_from_checkpoint(ckpt_path)
        self.model.eval()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.threshold = float(threshold)
        self.tf = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ])

    @torch.no_grad()
    def predict(self, pil_image: Image.Image):
        x = self.tf(pil_image).unsqueeze(0).to(self.device)
        out = self.model(x)

        # ---- Score
        if isinstance(out, dict):
            score = float(out.get("pred_scores", torch.tensor([0.0], device=self.device)).item())
            am = out.get("anomaly_maps", None)
        elif hasattr(out, 'anomaly_map'):
            score = float(out.pred_score.item())
            am = out.anomaly_map
        else:
            score = float(out.item())
            am = None

        # ---- Heatmap -> numpy [H, W]
        import numpy as np, cv2
        if am is None:
            H, W = pil_image.size[1], pil_image.size[0]
            anomaly_map = np.zeros((H, W), dtype=np.float32)
        else:
            if isinstance(am, torch.Tensor):
                # übliche Shapes: [1, H, W] oder [1,1,H,W]
                while am.ndim > 2:
                    am = am[0]
                am = am.detach().float().cpu().numpy()
            anomaly_map = am
            # auf Bildgröße bringen
            H, W = pil_image.size[1], pil_image.size[0]
            if anomaly_map.shape != (H, W):
                anomaly_map = cv2.resize(anomaly_map.astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)

        is_anom = score > self.threshold
        return {"score": score, "is_anomalous": is_anom, "anomaly_map": anomaly_map}
"""
    @torch.no_grad()
    def predict(self, pil_image: Image.Image, image_size=(256, 256)):
        x = self.tf(pil_image).unsqueeze(0).to(self.device)
        out = self.model(x)

        # Robust auslesen:
        if isinstance(out, dict):
            score = float(out.get("pred_scores", torch.tensor([0.0])).item())
            anomaly_map = out.get("anomaly_maps", torch.zeros(image_size))
            if anomaly_map is not None:
                anomaly_map = anomaly_map[0].detach().cpu().numpy()  # (H,W)
        elif hasattr(out, 'anomaly_map'):
            score = float(out.pred_score.item())
            anomaly_map = out.anomaly_map
        else:
            score = float(out.item())
            anomaly_map = torch.zeros(image_size)

        anomaly_map = anomaly_map.squeeze().cpu().numpy()
        
        is_anom = score > self.threshold
        return {"score": score, "is_anomalous": is_anom, "anomaly_map": anomaly_map}
"""

