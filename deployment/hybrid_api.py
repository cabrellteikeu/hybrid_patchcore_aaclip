# deployment/hybrid_api.py
from fastapi import FastAPI, UploadFile, File, Body, Query
from fastapi.responses import JSONResponse, FileResponse
from PIL import Image
import os, json
import io, yaml
from pathlib import Path
from PIL import Image, ImageDraw
from src.patchcore.patchcore_infer import PatchCoreStage
from src.defects.defect_prompts import DEFECT_PROMPTS  # dict: {name: [variants]}
#from src.preprocess.runtime_cropper import RuntimeBucketCropper
#from src.preprocess.bucket_runtime_pipeline import BucketRuntimePipeline
import numpy as np, cv2
from src.anomaly_dino.dino_stage import DinoAnomalyStage
from src.classify.zeroshot_clip import ZeroShotClipStage
from src.classify.text_defect_stage import TextDefectStage
from src.classify.zsclip_defect_stage import ZSClipDefectStage
from src.preprocess.grounded_sam import GroundedSAMPreprocessor

app = FastAPI(title="Hybrid AD (DINO/PatchCore → ZSCLIP/AA-CLIP)")

CFG = yaml.safe_load(Path("config/hybrid_config.yaml").read_text(encoding="utf-8"))

ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

def ensure_dirs():
    Path("buffers/safe_ok").mkdir(parents=True, exist_ok=True)
    Path(CFG["patchcore"]["heatmap_dir"]).mkdir(parents=True, exist_ok=True)

ensure_dirs()

def _cfg(path, default=None):
    # kleiner Helper für verschachtelte Keys
    cur = CFG
    for k in path.split("."):
        if k not in cur:
            return default
        cur = cur[k]
    return cur

PATCH = PatchCoreStage(
    ckpt_path=CFG["patchcore"]["ckpt_path"],
    image_size=tuple(CFG["patchcore"]["image_size"]),
    device=CFG["device"],
    threshold=CFG["patchcore"]["threshold"],
)

DINO = DinoAnomalyStage(
    model_name=_cfg("dino.model_name", "vit_large_patch14_dinov2.lvd142m"),
    device=_cfg("device", "cuda"),
    bank_dir=_cfg("dino.bank_dir", "banks/dino"),
    max_bank_patches=int(_cfg("dino.max_bank_patches", 20000)),
    top_p=float(_cfg("dino.top_p", 0.02)),
    use_half=bool(_cfg("dino.use_half", True)),
    sim_chunk_size=int(_cfg("dino.sim_chunk_size", 32768)),
)
# optionaler DINO-Threshold (wenn nicht gesetzt, fallback = PatchCore-Threshold)
DINO_TAU = 0.25#float(_cfg("dino.threshold", _cfg("patchcore.threshold", 0.7)))

ZERO = ZeroShotClipStage(
    model_name=CFG["zeroshot"]["model_name"],
    pretrained=CFG["zeroshot"].get("pretrained", "openai"),
    defect_prompts=DEFECT_PROMPTS,
    device=CFG.get("device", "cuda"),
    temperature=float(CFG["zeroshot"].get("temperature", 0.10)),
    topk=int(CFG["zeroshot"].get("topk", 3)),
    scales=tuple(CFG["zeroshot"].get("scales", [0.70, 0.90, 1.00])),
    overlap=float(CFG["zeroshot"].get("overlap", 0.50)),
    batch_size=int(CFG["zeroshot"].get("batch_size", 64)),
    use_half=bool(CFG["zeroshot"].get("use_half", True)),
    cache_dir=CFG["zeroshot"].get("cache_dir", None),
    normal_prompts=CFG["zeroshot"].get("normal_prompts", []),
    rel_alpha=float(CFG["zeroshot"].get("rel_alpha", 1.0)),
    topq=float(CFG["zeroshot"].get("topq", 0.02)),
)
ZSHOT_HEAT_THR = float(CFG["zeroshot"].get("heatmap_threshold", 0.50))

ZSC_CFG = (CFG.get("zsclip") or {})
ZSCLIP = ZSClipDefectStage(
    model_name=ZSC_CFG.get("model_name", ""),
    pretrained=ZSC_CFG.get("pretrained", "openai"),
    device=CFG.get("device", "cuda"),
    tta_scales=tuple(ZSC_CFG.get("tta_scales", [1.0, 0.9, 1.1])),
    temperature=float(ZSC_CFG.get("temperature", 0.01)),
    topk=int(ZSC_CFG.get("topk", 3)),
    defect_prompts=DEFECT_PROMPTS,   # nutzt NUR die pos-Listen (oder pos-Feld)
)

TD = TextDefectStage(
    backend=CFG["textdefect"].get("backend", ""),
    model_name=CFG["textdefect"].get("model_name", ""),
    device=CFG["device"],
    tta_scales=tuple(CFG["textdefect"].get("tta_scales", [1.0])),
    temperature=float(CFG["textdefect"].get("temperature", 0.10)),
    text_weight=float(CFG["textdefect"].get("text_weight", 0.5)),
    topk=int(CFG["textdefect"].get("topk", 3)),
    return_base64=bool(CFG["output"].get("return_base64", False)),
    defect_prompts=DEFECT_PROMPTS,
    prompt_templates=CFG["textdefect"].get("prompt_templates", [
        "{x}", "a photo of {x}", "bucket with {x}", "industrial bucket: {x}"
    ]),
    synonyms_per_class=int(CFG["textdefect"].get("synonyms_per_class", 0)),
    support_dir=CFG["textdefect"].get("support_dir", None),
    max_support_per_class=int(CFG["textdefect"].get("max_support_per_class", 20)),
    cache_dir=CFG["textdefect"].get("cache_dir", "cache/textdefect"),
    save_root=CFG["textdefect"].get("save_root", "runs/textdefect"),
    enable_heatmap=bool(CFG["textdefect"].get("enable_heatmap", True)),
)

PRE = None
if CFG.get("preprocess", {}).get("enabled", False):
    try:
        PRE = GroundedSAMPreprocessor(
            dino_config_path=CFG["preprocess"]["dino"]["config"],
            dino_checkpoint_path=CFG["preprocess"]["dino"]["checkpoint"],
            sam_encoder=CFG["preprocess"]["sam"]["encoder"],
            sam_checkpoint_path=CFG["preprocess"]["sam"]["checkpoint"],
            device=CFG["preprocess"].get("device", CFG.get("device", "cpu")),
            classes=CFG["preprocess"].get("classes", None),
            box_threshold=float(CFG["preprocess"]["dino"].get("box_threshold", 0.25)),
            text_threshold=float(CFG["preprocess"]["dino"].get("text_threshold", 0.25)),
            nms_threshold=float(CFG["preprocess"]["dino"].get("nms_threshold", 0.5)),
            target_size=tuple(CFG["preprocess"].get("target_size", [1024, 1024])),
            background_color=tuple(CFG["preprocess"].get("background_bgr", [255, 255, 255])),
        )
    except Exception as e:
        print(f"[WARN] GroundedSAMPreprocessor init failed: {e}")
        PRE = None

def visualize_anomalies(image, anomaly_map, threshold):
    """Raster-Overlay wie bisher; anomaly_map darf jede Größe haben."""
    image = image.copy()
    draw = ImageDraw.Draw(image)
    W, H = image.size
    h, w = anomaly_map.shape[:2]
    cell_w = max(1, W // w)
    cell_h = max(1, H // h)
    for y in range(h):
        for x in range(w):
            if anomaly_map[y, x] > threshold:
                x0 = x * cell_w
                y0 = y * cell_h
                x1 = min(W, (x + 1) * cell_w)
                y1 = min(H, (y + 1) * cell_h)
                draw.rectangle([x0, y0, x1, y1], outline="red", width=2)
    return image

@app.get("/")
def health():
    return {"status": "ok", "device": CFG["device"]}

@app.post("/dino/enroll")
def dino_enroll(
    sku_id: str = Query("default"),
    view: str = Query("single"),
    folder_path: str = Body(..., embed=True),
):
    try:
        res = DINO.enroll(sku_id=sku_id, view=view, folder_path=folder_path)
        return JSONResponse(res)
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

@app.post("/dino/predict")
async def dino_predict(
    file: UploadFile = File(...),
    sku_id: str = Query("default"),
    view: str = Query("single"),
):
    try:
        raw = await file.read()
        img = Image.open(io.BytesIO(raw)).convert("RGB")

        score, heatmap, meta = DINO.score_image(sku_id, view, img, require_bank=True, return_heatmap=True)

        # Einfacher Tri-State nur auf DINO (ohne separate Kalibrierung)
        if score <= DINO_TAU: state = "OK"
        elif score >= (DINO_TAU * 1.2): state = "DEFECT"
        else: state = "REVIEW"

        # Overlay speichern
        out_dir = _cfg("patchcore.heatmap_dir", "results/inference")
        os.makedirs(out_dir, exist_ok=True)
        fname = os.path.splitext(file.filename or "upload")[0]
        out_path = os.path.join(out_dir, f"{fname}_dino_overlay.jpg")
        vis = visualize_anomalies(img, heatmap, threshold=DINO_TAU)
        vis.save(out_path)

        resp = {
            "state": state,
            "score": float(score),
            "dino_threshold": DINO_TAU,
            "meta": meta,
            "overlay_path": out_path
        }
        response = FileResponse(out_path, media_type="image/jpeg")
        response.headers["response"] = json.dumps(resp)
        return response

    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

@app.post("/zshot/predict")
async def zshot_predict(file: UploadFile = File(...)):
    try:
        # 0) Validierung
        if file is None:
            raise HTTPException(status_code=400, detail="No file provided.")
        fname = file.filename or ""
        ext = os.path.splitext(fname)[1].lower()
        if ext and ext not in ALLOWED_EXTS:
            ctype = (file.content_type or "").lower()
            if not ctype.startswith("image/"):
                raise HTTPException(status_code=415, detail=f"Unsupported file type: {ext or ctype}")

        # 1) Bild laden
        raw = await file.read()
        if not raw:
            raise HTTPException(status_code=400, detail="Empty file.")
        try:
            img = Image.open(io.BytesIO(raw)).convert("RGB")
        except UnidentifiedImageError:
            raise HTTPException(status_code=415, detail="Unable to decode image.")

        # 2) Zero-Shot Klassifikation (relatives Scoring + Top-q% passiert in ZERO.predict)
        out = ZERO.predict(img)
        topk = out.get("topk", [])
        heat = out.get("heatmap", None)   # np.ndarray [H,W] in [0..1] oder None
        best_cls = out.get("best_class", None)

        # 3) Overlay speichern
        out_dir = CFG["patchcore"].get("heatmap_dir", "results/inference")
        os.makedirs(out_dir, exist_ok=True)
        stamp = int(time.time() * 1000)
        base = os.path.splitext(fname or "upload")[0]
        overlay_path = os.path.join(out_dir, f"{base}_zshot_{stamp}.jpg")

        vis_img = visualize_anomalies(img, heat, threshold=ZSHOT_HEAT_THR) if heat is not None else img
        vis_img.save(overlay_path)

        # 4) Antwort (Bild als Body, JSON im Header)
        resp_payload = {
            "zshot_topk": topk,               # [{"label": "...", "score": float}, ...]
            "best_class": best_cls,           # str | None
            "overlay_path": overlay_path,
            "heatmap_threshold": float(ZSHOT_HEAT_THR),
        }

        response = FileResponse(overlay_path, media_type="image/jpeg")
        response.headers["response"] = json.dumps(resp_payload)
        return response

    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    prep: bool | None = Query(None, description="Preprocessor erzwingen (true/false) oder None=Config-Default"),
    engine: str = Query("patchcore", enum=["patchcore", "dino"]),
    classifier: str = Query("zsclip", enum=["zsclip", "textdefect"]),
    sku_id: str = Query("default"),
    view: str = Query("single"),
):
    try:
        raw = await file.read()
        img = Image.open(io.BytesIO(raw)).convert("RGB")

        use_prep = PRE is not None if prep is None else bool(prep)
        prep_meta = {}
        if use_prep and PRE is not None:
            img, prep_meta = PRE.prepare(img)

        # --- 1) Anomalie-Gate ---
        if engine == "patchcore":
            pa = PATCH.predict(img)
            anomaly_score = float(pa["score"])
            anomaly_map = pa["anomaly_map"]
            thr = float(CFG["patchcore"]["threshold"])
            is_anomalous = bool(pa["is_anomalous"])
        else:
            score, heatmap, meta = DINO.score_image(sku_id, view, img, require_bank=True, return_heatmap=True)
            anomaly_score = float(score)
            anomaly_map = heatmap
            thr = float(DINO_TAU)
            is_anomalous = bool(anomaly_score > thr)

        resp = {
            "engine": engine,
            "is_anomalous": is_anomalous,
            "anomaly_score": anomaly_score,
            "threshold": thr,
        }

        if not is_anomalous:
            resp["explanation"] = f"sample looks normal ({engine})"
            return JSONResponse(resp, status_code=200)

        # --- 2) Klassifikation (einfach & robust) ---
        if classifier == "zsclip":
            z = ZSCLIP.classify(img)
            resp["classifier"] = "zsclip"
            resp["defect_topk"] = z.get("topk", [])
            resp["defect_scores"] = z.get("scores", {})
        else :
            # Falls du deine alte TextDefectStage weiter behalten willst,
            # rufe hier TEXTDEFECT.infer_zero_shot(img) auf.
            b = TD.infer(img)
            if not b.get("ok", False) and not b.get("topk"):
                resp["explanation"] = f"TextDefectStage failed: {b.get('error')}"
                return JSONResponse(resp, status_code=200)
            
            resp["classifier"] = "textdefect(aa-clip-fallback)"
            resp["defect_topk"] = b.get("topk", [])
            #resp["defect_scores"] = b.get("scores", {})

        # --- 3) Overlay speichern & zurückgeben ---
        HEATMAP_DIR = Path(CFG["patchcore"]["heatmap_dir"]).resolve()
        HEATMAP_DIR.mkdir(parents=True, exist_ok=True)
        fname = os.path.splitext(file.filename or "upload")[0]
        suffix = "_overlay_dino" if engine == "dino" else "_overlay_pc"
        out_path = HEATMAP_DIR / f"{fname}{suffix}.jpg"
        overlay = visualize_anomalies(img, anomaly_map, thr)
        overlay.save(out_path)

        response = FileResponse(str(out_path), media_type="image/jpeg")
        response.headers["X-Response"] = json.dumps(resp, ensure_ascii=False)
        return response

    except FileNotFoundError as e:
        return JSONResponse({"error": f"{str(e)} — did you run /dino/enroll first?"}, status_code=400)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)