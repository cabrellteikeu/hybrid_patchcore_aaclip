# deployment/hybrid_api.py
from fastapi import FastAPI, UploadFile, File, Body, Query
from fastapi.responses import JSONResponse, FileResponse
from PIL import Image
import os, json
import io, yaml
from pathlib import Path
from PIL import Image, ImageDraw
from src.patchcore.patchcore_infer import PatchCoreStage
from src.classify.aaclip_wrapper import AAClipStage
from src.defects.defect_prompts import DEFECT_PROMPTS  # dict: {name: [variants]}
#from src.preprocess.runtime_cropper import RuntimeBucketCropper
#from src.preprocess.bucket_runtime_pipeline import BucketRuntimePipeline
import numpy as np, cv2
from src.anomaly_dino.dino_stage import DinoAnomalyStage
from src.classify.zeroshot_clip import ZeroShotClipStage
from src.classify.text_defect_stage import TextDefectStage
from src.classify.zsclip_defect_stage import ZSClipDefectStage
from src.preprocess.grounded_sam import GroundedSAMPreprocessor

app = FastAPI(title="Hybrid AD (PatchCore → AA-CLIP)")

CFG = yaml.safe_load(Path("config/hybrid_config.yaml").read_text(encoding="utf-8"))

NORMAL_KEY  = str(CFG["aaclip"].get("normal_key","normal"))
ABN_KEY     = str(CFG["aaclip"].get("abnormal_key","defective"))
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

AAC = AAClipStage(
    repo_root=CFG["aaclip"]["repo_root"],
    data_root=CFG["aaclip"]["data_root"],
    model_name=CFG["aaclip"]["model_name"],
    shot=int(CFG["aaclip"]["shot"]),
    save_root=CFG["aaclip"]["save_root"],
    defect_prompts=DEFECT_PROMPTS,
    topk=int(CFG["aaclip"]["topk"]),
    timeout_sec=int(CFG["aaclip"]["timeout_sec"]),
    return_base64=CFG["output"]["return_base64"],
    # Performance/Genauigkeit:
    enable_repo=bool(CFG["aaclip"].get("enable_repo", False)),
    temperature=float(CFG["aaclip"].get("temperature", 0.10)),
    text_weight=float(CFG["aaclip"].get("text_weight", 0.4)),
    tta_scales=tuple(CFG["aaclip"].get("tta_scales", [0.90, 1.00, 1.10])),
    cache_dir=CFG["aaclip"].get("cache_dir", "cache/openclip_vitl14_336"),
    support_dir=CFG["aaclip"].get("support_dir", None),
    max_support_per_class=int(CFG["aaclip"].get("max_support_per_class", 20)),
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

#RUNTIME = BucketRuntimePipeline()
#RUNTIME.load_models()
"""
def visualize_anomalies(image, anomaly_map, threshold=CFG["patchcore"]["threshold"]):
    image = image.copy()
    draw = ImageDraw.Draw(image)
    width, height = image.size

    for y in range(anomaly_map.shape[0]):
        for x in range(anomaly_map.shape[1]):
            if anomaly_map[y, x] > threshold:
                draw.rectangle([x * (width // anomaly_map.shape[1]), y * (height // anomaly_map.shape[0]),
                                (x + 1) * (width // anomaly_map.shape[1]), (y + 1) * (height // anomaly_map.shape[0])],
                               outline="red", width=2)
    return image
"""


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

"""
@app.post("/refresh")
def refresh_thresholds(
    sku_id: str = Query(CFG["calibration"]["default_sku"]),
    view: str   = Query(CFG["calibration"]["default_view"]),
):
    import glob, numpy as np
    buf_dir = Path(f"buffers/safe_ok/{sku_id}/{view}")
    paths = sorted(glob.glob(str(buf_dir / "*.jpg")))
    if len(paths) < 20:
        return {"ok": False, "error": "zu wenige safe-OK Bilder im Buffer (<20)", "count": len(paths)}
    scores = [float(PATCH.predict(Image.open(p).convert("RGB"))["score"]) for p in paths]
    q_a = float(CFG["calibration"]["q_accept"]); q_r = float(CFG["calibration"]["q_reject"])
    tau_a = float(np.quantile(np.array(scores, dtype=np.float32), q_a))
    tau_r = float(np.quantile(np.array(scores, dtype=np.float32), q_r))
    th = load_thresholds(TH_PATH); th[key(sku_id, view)] = {"tau_accept": tau_a, "tau_reject": tau_r}
    save_thresholds(TH_PATH, th)
    return {"ok": True, "sku_id": sku_id, "view": view, "tau_accept": tau_a, "tau_reject": tau_r, "n": len(scores)}

@app.post("/enroll")
def enroll(
    sku_id: str,
    good_dir_front: str = Body(...),
    good_dir_back: str  = Body(...),
    train_script: str = "src/models/train_patchcore.py",   # dein Fit-Skript
):
    import subprocess, sys
    # Idee: du übergibst dem Fit-Skript per ENV/Args die neuen good/ Ordner
    env = os.environ.copy()
    env["GOOD_FRONT_DIR"] = good_dir_front
    env["GOOD_BACK_DIR"]  = good_dir_back
    try:
        subprocess.run([sys.executable, train_script], check=True, env=env, timeout=3600)
        return {"ok": True, "msg": "Coreset neu aufgebaut (siehe neues ckpt). Bitte /calibrate aufrufen."}
    except subprocess.CalledProcessError as e:
        return {"ok": False, "error": f"train_script rc={e.returncode}"}
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "train_script timeout"}

@app.post("/defect-support")
def add_defect_support(class_name: str, roi_paths: list[str] = Body(..., embed=True)):
    # kopiere Bilder in CFG["aaclip"]["support_dir"]/class_name/
    sd = Path(AAC.support_dir or "support") / class_name
    sd.mkdir(parents=True, exist_ok=True)
    for p in roi_paths:
        pp = Path(p)
        (sd / pp.name).write_bytes(Path(p).read_bytes())
    # Cache neu bauen
    AAC._build_support_cache(sd.parent)
    return {"ok": True, "class": class_name, "count": len(roi_paths)}

@app.post("/calibrate")
def calibrate(
    sku_id: str = Query(CFG["calibration"]["default_sku"]),
    view: str   = Query(CFG["calibration"]["default_view"]),
    good_paths: list[str] = Body(..., embed=True),
):
    import numpy as np, cv2
    scores = []
    for p in good_paths:
        img = Image.open(p).convert("RGB")
        s = PATCH.predict(img)["score"]
        scores.append(float(s))
    import numpy as np
    q_a = float(CFG["calibration"]["q_accept"])
    q_r = float(CFG["calibration"]["q_reject"])
    tau_a = float(np.quantile(np.array(scores, dtype=np.float32), q_a))
    tau_r = float(np.quantile(np.array(scores, dtype=np.float32), q_r))

    th = load_thresholds(TH_PATH)
    th[key(sku_id, view)] = {"tau_accept": tau_a, "tau_reject": tau_r}
    save_thresholds(TH_PATH, th)
    return {"ok": True, "sku_id": sku_id, "view": view, "tau_accept": tau_a, "tau_reject": tau_r, "n": len(scores)}
"""

"""
@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    sku_id: str = Query(CFG["calibration"]["default_sku"]),
    view: str = Query(CFG["calibration"]["default_view"]),
):
    try:
        raw = await file.read()
        img_pil = Image.open(io.BytesIO(raw)).convert("RGB")

        a = PATCH.predict(img_pil)
        score = float(a["score"])
        hmap  = a["anomaly_map"]
        tau_a, tau_r = get_tau(sku_id, view, CFG["patchcore"]["threshold"])

        state_pc = decide(score, tau_a, tau_r, REVIEW_BAND)

        # ROI-Crop für AA-CLIP (nur bei REVIEW/DEFECT sinnvoll)
        img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        roi = heatmap_to_roi(img_bgr, hmap, q=0.98, min_size=48) if state_pc != "OK" else None
        crop_pil = img_pil if roi is None else Image.fromarray(
            cv2.cvtColor(img_bgr[roi[1]:roi[3], roi[0]:roi[2]], cv2.COLOR_BGR2RGB)
        )

        # Margin (abnormal - normal)
        margin = AAC.normal_abnormal_margin(crop_pil, normal_key=NORMAL_KEY, abnormal_key=ABN_KEY)

        # Gegenanker-Regeln
        state = state_pc
        if state_pc == "DEFECT" and margin < 0.0:
            state = "REVIEW"
        if state_pc == "REVIEW" and margin > 0.8:
            state = "DEFECT"

        # Safe-OK puffern (für späteres /refresh)
        if state == "OK":
            gap_ok = (tau_a - score) / max(tau_a, 1e-6)
            if gap_ok >= SAFETY_GAP and margin < -0.5:  # konservativ
                out_dir = Path(f"buffers/safe_ok/{sku_id}/{view}")
                out_dir.mkdir(parents=True, exist_ok=True)
                # speichere das Bild (oder nur Embedding/Score – einfach Bild reicht erstmal)
                out_path = out_dir / (Path(file.filename).stem + ".jpg")
                img_pil.save(out_path)

        # Visualisierung wie gehabt
        visualized_image = visualize_anomalies(img_pil, hmap, tau_a)
        filename = os.path.splitext(file.filename)[0]
        heatmap_path = os.path.join(CFG["patchcore"]["heatmap_dir"], f"{filename}_overlay.jpg")
        visualized_image.save(heatmap_path)

        resp = {
            "patchcore_score": score,
            "tau_accept": tau_a,
            "tau_reject": tau_r,
            "state_patchcore": state_pc,
            "state_final": state,                  # OK | REVIEW | DEFECT
            "clip_margin": float(margin),
            "roi": roi if roi else None,
        }

        response = FileResponse(heatmap_path, media_type="image/jpeg")
        response.headers["response"] = json.dumps(resp)
        return response
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    
"""   

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
"""    
@app.post("/predict")
async def predict(file: UploadFile = File(...),
                  sku_id: str = Query("default"),
                  view: str = Query("single"),
                  explain: int = Query(0)):
    try:
        raw = await file.read()
        img = Image.open(io.BytesIO(raw)).convert("RGB")

        # 1) PatchCore
        a = PATCH.predict(img)  # {"score", "is_anomalous", "anomaly_map"}
        pc_score = float(a["score"])
        pc_tau = float(_cfg("patchcore.threshold", 0.7))
        pc_map = a["anomaly_map"]  # np.ndarray (HxW oder kleiner)

        # 2) DINO (require_bank=False -> gibt 0er-Heatmap/Score, falls noch nicht enrolled)
        dino_score, dino_map, dino_meta = DINO.score_image(sku_id, view, img, require_bank=False, return_heatmap=True)
        if dino_map is None:
            # falls keine Bank: Null-Map in Bildgröße
            dino_map = np.zeros((img.size[1], img.size[0]), np.float32)

        # 3) Fused Score/Map (ohne Kalibrierung): max-Logik
        fused_tau = min(pc_tau, DINO_TAU)  # konservativer Schwellenwert
        # Heatmap-Größen angleichen
        H, W = img.size[1], img.size[0]
        if pc_map.shape != (H, W):
            pc_map = cv2.resize(pc_map.astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)
        if dino_map.shape != (H, W):
            dino_map = cv2.resize(dino_map.astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)

        fused_map = np.maximum(pc_map.astype(np.float32), dino_map.astype(np.float32))
        fused_score = max(pc_score, float(dino_score))

        # 4) Entscheidung (einfach, deterministisch)
        if fused_score <= fused_tau:
            state = "OK"
        elif fused_score >= fused_tau * 1.2:
            state = "DEFECT"
        else:
            state = "REVIEW"

        # 5) Nur wenn wirklich Anomalie vermutet → AA-CLIP Erklärung (optional)
        resp = {
            "state": state,
            "scores": {"patchcore": pc_score, "dino": float(dino_score), "fused": fused_score},
            "thresholds": {"patchcore": pc_tau, "dino": DINO_TAU, "fused": fused_tau},
        }

        if state != "OK" and explain:
            b = AAC.infer(img)
            if b.get("ok") and b.get("topk"):
                resp["defect_topk"] = b["topk"]
                if b.get("heatmap_base64"):
                    resp["heatmap_base64"] = b["heatmap_base64"]
                else:
                    resp["heatmap_path"] = b.get("heatmap_path")

        # 6) Overlay speichern (Fused Map)
        out_dir = _cfg("patchcore.heatmap_dir", "results/inference")
        os.makedirs(out_dir, exist_ok=True)
        fname = os.path.splitext(file.filename or "upload")[0]
        out_path = os.path.join(out_dir, f"{fname}_overlay.jpg")
        vis = visualize_anomalies(img, fused_map, threshold=fused_tau)
        vis.save(out_path)

        # FileResponse wie bei dir – JSON in Header
        response = FileResponse(out_path, media_type="image/jpeg")
        response.headers["response"] = json.dumps(resp)
        return response

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

""" 

""" 
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        raw = await file.read()
        img = Image.open(io.BytesIO(raw)).convert("RGB")

        #if CROP is not None:
        #    img = CROP.crop_pil(img)
        #paths = RUNTIME.process_pil(pil_image=img, image_name="api_input_123")  # schreibt die 3 Artefakte
# paths["cropped_image_path"]
        a = PATCH.predict(img)
        resp = {
            "is_anomalous": bool(a["is_anomalous"]),
            "anomaly_score": float(a["score"]),
        }
        anomaly_map = a["anomaly_map"]
        if not a["is_anomalous"]:
            resp["explanation"] = "sample looks normal (PatchCore)"
            return JSONResponse(resp)

        b = TD.infer(img)
        if not b.get("ok", False) and not b.get("topk"):
            resp["explanation"] = f"TextDefectStage failed: {b.get('error')}"
            return JSONResponse(resp, status_code=200)

        resp["defect_topk"] = b.get("topk", [])
        if b.get("heatmap_base64"):
            resp["heatmap_base64"] = b["heatmap_base64"]
        else:
            resp["heatmap_path"] = b.get("heatmap_path")

        visualized_image = visualize_anomalies(img, anomaly_map, CFG["patchcore"]["threshold"])

        # Speichern des visualisierten Bildes
        os.makedirs(CFG["patchcore"]["heatmap_dir"], exist_ok=True)
        filename = os.path.splitext(file.filename)[0]
        heatmap_path = os.path.join(CFG["patchcore"]["heatmap_dir"], f"{filename}_overlay.jpg")
        visualized_image.save(heatmap_path)

            # Bild als Antwort zurückgeben
        response = FileResponse(heatmap_path, media_type="image/jpeg")
        response.headers["response"] = str(resp)#str(resp) #str(score)
        return response

        #return JSONResponse(resp)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    
"""

"""
# best solution   
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        raw = await file.read()
        img = Image.open(io.BytesIO(raw)).convert("RGB")

        #if CROP is not None:
        #    img = CROP.crop_pil(img)
        #paths = RUNTIME.process_pil(pil_image=img, image_name="api_input_123")  # schreibt die 3 Artefakte
# paths["cropped_image_path"]
        a = PATCH.predict(img)
        resp = {
            "is_anomalous": bool(a["is_anomalous"]),
            "anomaly_score": float(a["score"]),
        }
        anomaly_map = a["anomaly_map"]
        if not a["is_anomalous"]:
            resp["explanation"] = "sample looks normal (PatchCore)"
            return JSONResponse(resp)

        b = AAC.infer(img)
        if not b.get("ok", False) and not b.get("topk"):
            resp["explanation"] = f"AA-CLIP failed: {b.get('error')}"
            return JSONResponse(resp, status_code=200)

        resp["defect_topk"] = b.get("topk", [])
        if b.get("heatmap_base64"):
            resp["heatmap_base64"] = b["heatmap_base64"]
        else:
            resp["heatmap_path"] = b.get("heatmap_path")

        visualized_image = visualize_anomalies(img, anomaly_map, CFG["patchcore"]["threshold"])

        # Speichern des visualisierten Bildes
        os.makedirs(CFG["patchcore"]["heatmap_dir"], exist_ok=True)
        filename = os.path.splitext(file.filename)[0]
        heatmap_path = os.path.join(CFG["patchcore"]["heatmap_dir"], f"{filename}_overlay.jpg")
        visualized_image.save(heatmap_path)

            # Bild als Antwort zurückgeben
        response = FileResponse(heatmap_path, media_type="image/jpeg")
        response.headers["response"] = str(resp)#str(resp) #str(score)
        return response

        #return JSONResponse(resp)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
"""

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    prep: bool | None = Query(None, description="Preprocessor erzwingen (true/false) oder None=Config-Default"),
    engine: str = Query("patchcore", enum=["patchcore", "dino"]),
    classifier: str = Query("zsclip", enum=["zsclip", "textdefect", "aaclip"]),
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
        elif classifier == "textdefect":
            # Falls du deine alte TextDefectStage weiter behalten willst,
            # rufe hier TEXTDEFECT.infer_zero_shot(img) auf.
            z = ZSCLIP.classify(img)  # Fallback: trotzdem ZSCLIP
            resp["classifier"] = "textdefect(zsclip-fallback)"
            resp["defect_topk"] = z.get("topk", [])
            resp["defect_scores"] = z.get("scores", {})
        else:  # "aaclip"
            b = AAC.infer(img)
            resp["classifier"] = "aaclip"
            resp["defect_topk"] = b.get("topk", [])
            # Normiere ggf. auf score-Key

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