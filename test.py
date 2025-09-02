# diag_groundingdino.py
import torch, os, importlib.util, sys
print("torch:", torch.__version__, "| cuda available:", torch.cuda.is_available())
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("gpu name:", torch.cuda.get_device_name(0))

try:
    import groundingdino
    print("groundingdino import: OK")
except Exception as e:
    print("groundingdino import FAILED:", e)

try:
    from groundingdino.models.GroundingDINO.ms_deform_attn import MSDeformAttn
    print("ms_deform_attn import: OK (GPU-kernel verfügbar)")
except Exception as e:
    print("ms_deform_attn import FAILED:", e)

# prüfe, dass deine Pfade existieren:
from pathlib import Path
DINO_CFG = r"C:\Users\Basti\hybrid_patchcore_aaClip\groundingdino\config\GroundingDINO_SwinT_OGC.py"
DINO_CKPT= r"C:\Users\Basti\hybrid_patchcore_aaClip\src\preprocess\extern\groundingdino_swint_ogc.pth"
print("cfg exists:", Path(DINO_CFG).exists(), DINO_CFG)
print("ckpt exists:", Path(DINO_CKPT).exists(), DINO_CKPT)