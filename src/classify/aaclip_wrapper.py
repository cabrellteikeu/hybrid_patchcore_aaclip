# src/aaclip_wrapper.py
import os, json, subprocess, time, base64, hashlib, traceback
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
from PIL import Image, ImageOps

def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _to_b64(p: Optional[Path]) -> Optional[str]:
    if not p or not Path(p).exists():
        return None
    return base64.b64encode(Path(p).read_bytes()).decode("utf-8")

def _hash_str(s: str) -> str:
    import hashlib
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:10]

class AAClipStage:
    """
    AA‑CLIP Wrapper mit:
      • Lazy‑Import von torch/open_clip/grad‑cam (saubere Diagnose)
      • CPU‑Fallback wenn CUDA nicht verfügbar
      • Lokales OpenCLIP Zero-/Few‑Shot Ranking (schnell & stabil)
      • Optionaler Aufruf von AA‑CLIP/test.py
      • Heatmap via Grad‑CAM Rollout (optional)
    """

    # ---------------------- Konstruktor ----------------------
    def __init__(
        self,
        repo_root: str,
        data_root: Optional[str] = None,
        model_name: str = "ViT-L-14-336",
        shot: int = 2,
        save_root: Optional[str] = None,
        defect_prompts: Optional[Dict[str, List[str]]] = None,
        topk: int = 3,
        timeout_sec: int = 120,
        return_base64: bool = False,
        enable_repo: bool = False,
        temperature: float = 0.10,
        text_weight: float = 0.4,
        tta_scales: Tuple[float, ...] = (0.90, 1.00, 1.10),
        cache_dir: Optional[str] = "cache/openclip_vitl14_336",
        support_dir: Optional[str] = None,
        max_support_per_class: int = 20,
    ):
        self.repo_root = Path(repo_root).resolve()
        self.data_root = Path(data_root).resolve() if data_root else (self.repo_root / "data")
        self.bucket_root = self.data_root / "Bucket"
        self.test_api_dir = self.bucket_root / "test" / "api"

        self.model_name = model_name
        self.shot = int(shot)
        self.save_root = Path(save_root).resolve() if save_root else (self.repo_root / "runs/bucket_exp_fast")
        _ensure_dir(self.save_root)

        self.defect_prompts: Dict[str, List[str]] = defect_prompts or {}
        self.class_names: List[str] = list(self.defect_prompts.keys())
        self.topk = int(topk)
        self.timeout_sec = int(timeout_sec)
        self.return_base64 = bool(return_base64)

        self.enable_repo = bool(enable_repo)
        self.temperature = float(temperature)
        self.text_weight = float(text_weight)
        self.tta_scales = tuple(tta_scales)

        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            _ensure_dir(self.cache_dir)

        self.support_dir = Path(support_dir).resolve() if support_dir else None
        self.max_support_per_class = int(max_support_per_class)

        # Stabilität bei Windows/UVicorn
        self.env = os.environ.copy()
        self.env["AA_NUM_WORKERS"] = self.env.get("AA_NUM_WORKERS", "0")

        # Lazy-Import & Geräteauswahl
        self._lazy_imports()  # setzt self.torch, self.F, self.open_clip, self.Rollout, self.has_torch
        if not self.has_torch:
            raise RuntimeError(
                "AAClipStage: torch/open_clip konnten nicht importiert werden. "
                "Siehe Diagnosedetails oben im Log."
            )

        self._device = "cuda" if (self.torch.cuda.is_available()) else "cpu"
        self._half = (self._device == "cuda")

        # OpenCLIP vorbereiten
        self._oclip_loaded = False
        self._oclip_model = None
        self._oclip_preproc = None
        self._oclip_tokenizer = None
        self._text_cache: Dict[str, "torch.Tensor"] = {}
        self._support_cache: Dict[str, "torch.Tensor"] = {}

        self._load_openclip()
        self._build_text_cache()
        if self.support_dir:
            self._build_support_cache(self.support_dir)

    # ---------------------- Lazy Imports ----------------------
    def _lazy_imports(self):
        """Importe dynamisch & liefere saubere Diagnose bei Fehlern."""
        self.torch = None
        self.F = None
        self.open_clip = None
        self.Rollout = None
        self.has_torch = False

        # torch
        try:
            import torch as _torch
            import torch.nn.functional as _F
            self.torch = _torch
            self.F = _F
        except ImportError as e:
            print("[AAClipStage] ImportError: torch fehlt:", e)
            print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
            return
        except Exception as e:
            print("[AAClipStage] torch Import Exception:", repr(e))
            traceback.print_exc()
            return

        # open_clip
        try:
            import open_clip as _oc
            self.open_clip = _oc
        except ImportError as e:
            print("[AAClipStage] ImportError: open_clip fehlt:", e)
            print("  pip install open-clip-torch")
            return
        except Exception as e:
            print("[AAClipStage] open_clip Import Exception:", repr(e))
            traceback.print_exc()
            return

        # grad-cam (optional)
        try:
            from pytorch_grad_cam import ViTAttentionRollout as _Rollout
            self.Rollout = _Rollout
        except Exception as e:
            self.Rollout = None  # optional
            # leise… wer Heatmap braucht: pip install grad-cam

        self.has_torch = True

        # kurze Versionen-Info (hilft extrem beim Debuggen)
        try:
            print(f"[AAClipStage] torch {self.torch.__version__} | cuda? {self.torch.cuda.is_available()}")
            print(f"[AAClipStage] open_clip {self.open_clip.__version__}")
        except Exception:
            pass

    # ---------------------------- Public ----------------------------
    def infer(self, pil_image: Image.Image) -> Dict[str, Any]:
        """Schnelles Scoring + optionale Heatmap (lokal)."""
        img = ImageOps.exif_transpose(pil_image.convert("RGB"))

        ts = int(time.time() * 1000)
        _ensure_dir(self.test_api_dir)
        query_path = self.test_api_dir / f"query_{ts}.png"
        img.save(query_path)

        run_dir = self.save_root / f"fast_{ts}"
        _ensure_dir(run_dir)

        topk, heatmap_path, err = [], None, None

        # 1) Optional: AA‑CLIP/test.py (kann auf Windows/uvicorn schwerfällig sein)
        if self.enable_repo:
            ok, err = self._run_repo_test(save_path=run_dir)
            if ok:
                topk, heatmap_path = self._collect_outputs(run_dir)

        # 2) Lokale OpenCLIP-Pipeline
        if not topk:
            topk = self._rank_with_openclip(img)

        # 3) Heatmap (optional)
        if heatmap_path is None:
            heatmap_path = self._save_rollout_heatmap(img, out_dir=run_dir, tag=f"rollout_{ts}")

        heatmap_b64 = _to_b64(heatmap_path) if (self.return_base64 and heatmap_path) else None

        return {
            "ok": True if topk else False,
            "error": err,
            "topk": topk,
            "heatmap_path": str(heatmap_path) if heatmap_path else None,
            "heatmap_base64": heatmap_b64,
        }

    # ---------------------- AA‑CLIP test.py (optional) ----------------------
    def _run_repo_test(self, save_path: Path) -> Tuple[bool, Optional[str]]:  # pragma: no cover
        cmd = [
            self._python_exe(),
            str(self.repo_root / "test.py"),
            "--dataset", "Bucket",
            "--shot", str(self.shot),
            "--model_name", self.model_name,
            "--save_path", str(save_path),
            "--visualize",
        ]
        try:
            subprocess.run(cmd, cwd=str(self.repo_root), check=True, timeout=self.timeout_sec, env=self.env)
            return True, None
        except subprocess.TimeoutExpired:
            return False, "timeout"
        except subprocess.CalledProcessError as e:
            return False, f"returncode={e.returncode}"

    def _collect_outputs(self, run_dir: Path) -> Tuple[List[Dict[str, float]], Optional[Path]]:
        score_files = list(run_dir.rglob("scores*.json")) + list(run_dir.rglob("logits*.json")) + list(run_dir.rglob("results*.json"))
        heatmap_files = [p for p in run_dir.rglob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg")]

        topk_list: List[Dict[str, float]] = []
        if score_files:
            score_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            try:
                with open(score_files[0], "r", encoding="utf-8") as f:
                    scores = json.load(f)
                if isinstance(scores, dict):
                    items = sorted(scores.items(), key=lambda x: x[1], reverse=True)[: self.topk]
                    topk_list = [{"label": k, "score": float(v)} for k, v in items]
                elif isinstance(scores, list):
                    scored = sorted(scores, key=lambda d: d.get("score", 0.0), reverse=True)[: self.topk]
                    topk_list = [{"label": d.get("label", ""), "score": float(d.get("score", 0.0))} for d in scored]
            except Exception:
                topk_list = []

        heatmap_path = heatmap_files[0] if heatmap_files else None
        return topk_list, heatmap_path

    # ---------------------- OpenCLIP Pipeline ----------------------
    def _load_openclip(self):
        name = self._map_openclip_name(self.model_name)
        model, _, preprocess = self.open_clip.create_model_and_transforms(name, pretrained="openai")
        tokenizer = self.open_clip.get_tokenizer(name)
        model = model.to(self._device).eval()
        self.torch.set_grad_enabled(False)
        if self._device == "cuda":
            model = model.half()
            self.torch.backends.cudnn.benchmark = True
        self._oclip_model = model
        self._oclip_preproc = preprocess
        self._oclip_tokenizer = tokenizer
        self._oclip_loaded = True

    def _build_text_cache(self):
        self._text_cache.clear()
        if not self.class_names:
            return
        dev = self._device
        for cname, variants in self.defect_prompts.items():
            if not variants:
                continue
            toks = self._oclip_tokenizer(variants).to(dev)
            with self.torch.cuda.amp.autocast(enabled=(dev == "cuda")):
                tfeat = self._oclip_model.encode_text(toks)  # [V,D]
            tfeat = self.F.normalize(tfeat.float(), dim=-1)
            self._text_cache[cname] = tfeat

    def _support_cache_path(self, cname: str) -> Optional[Path]:
        if not self.cache_dir:
            return None
        mhash = _hash_str(self.model_name + "|" + cname)
        return self.cache_dir / f"support_{cname}_{mhash}.pt"

    def _build_support_cache(self, support_dir: Path):
        self._support_cache.clear()
        dev = self._device

        for cname in self.class_names:
            cls_dir = support_dir / cname
            if not cls_dir.exists():
                continue

            cache_pt = self._support_cache_path(cname)
            if cache_pt and cache_pt.exists():
                try:
                    data = self.torch.load(cache_pt, map_location=dev)
                    if isinstance(data, self.torch.Tensor) and data.ndim == 2:
                        self._support_cache[cname] = self.F.normalize(data.float(), dim=-1)
                        continue
                except Exception:
                    pass  # rebuild

            paths = [p for p in cls_dir.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
            paths = paths[: self.max_support_per_class]
            feats = []
            for p in paths:
                try:
                    img = Image.open(p).convert("RGB")
                    x = self._oclip_preproc(img).unsqueeze(0).to(dev)
                    if self._half and dev == "cuda":
                        x = x.half()
                    with self.torch.cuda.amp.autocast(enabled=(dev == "cuda")):
                        f = self._oclip_model.encode_image(x)  # [1,D]
                    f = self.F.normalize(f.float(), dim=-1)
                    feats.append(f)
                except Exception:
                    continue
            if feats:
                feats = self.torch.cat(feats, dim=0)  # [Ns, D]
                self._support_cache[cname] = feats
                if cache_pt:
                    try:
                        self.torch.save(feats.cpu(), cache_pt)
                    except Exception:
                        pass

    def _img_feat_tta(self, img: Image.Image):
        dev = self._device
        W, H = img.size
        crops = []
        for s in self.tta_scales:
            w = int(W * s); h = int(H * s)
            left = max((W - w) // 2, 0); top = max((H - h) // 2, 0)
            crop = img.crop((left, top, left + w, top + h)).resize((336, 336), Image.BICUBIC)
            crops.append(crop)
        tens = self.torch.stack([self._oclip_preproc(c) for c in crops]).to(dev)
        if self._half and dev == "cuda":
            tens = tens.half()
        with self.torch.cuda.amp.autocast(enabled=(dev == "cuda")):
            feats = self._oclip_model.encode_image(tens)  # [T,D]
        feats = self.F.normalize(feats.float(), dim=-1).mean(dim=0, keepdim=True)
        return feats  # [1,D]

    def _rank_with_openclip(self, img: Image.Image) -> List[Dict[str, float]]:
        if not self.class_names:
            return []
        img_feat = self._img_feat_tta(img)  # [1,D]
        tau = max(1e-3, self.temperature)
        tw = float(np.clip(self.text_weight, 0.0, 1.0))
        iw = 1.0 - tw

        results = []
        for cname in self.class_names:
            # Text‑Score
            text_score = 0.0
            if cname in self._text_cache:
                vfeats = self._text_cache[cname]             # [V,D]
                sims_t = (img_feat @ vfeats.t()).squeeze(0)  # [V]
                text_score = self.torch.softmax(sims_t / tau, dim=-1).mean().item()

            # Image‑Score (Support)
            image_score = 0.0
            if cname in self._support_cache:
                s = self._support_cache[cname]               # [Ns,D]
                sims_i = (img_feat @ s.t()).squeeze(0)       # [Ns]
                image_score = sims_i.max().item()

            fused = tw * text_score + iw * image_score
            results.append((cname, fused))

        results.sort(key=lambda x: x[1], reverse=True)
        return [{"label": k, "score": float(v)} for k, v in results[: self.topk]]

    # ---------------------- Heatmap (optional) ----------------------
    def _save_rollout_heatmap(self, img: Image.Image, out_dir: Path, tag: str) -> Optional[Path]:
        if self.Rollout is None or not self._oclip_loaded:
            return None
        try:
            model = self._oclip_model.visual  # ViT backbone
            dev = self._device
            x = self._oclip_preproc(img).unsqueeze(0).to(dev)
            if self._half and dev == "cuda":
                x = x.half()

            rollout = self.Rollout(model, head_fusion="mean", discard_ratio=0.0)
            mask = rollout(x)  # numpy [H,W] in [0,1]
            mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-6)

            m = Image.fromarray((mask * 255).astype(np.uint8)).resize(img.size, Image.BICUBIC).convert("L")
            heat_rgb = ImageOps.colorize(m, black="black", white="red")
            overlay = Image.blend(img, heat_rgb, alpha=0.45)

            out_path = out_dir / f"heatmap_{tag}.png"
            overlay.save(out_path)
            return out_path
        except Exception:
            traceback.print_exc()
            return None

    def normal_abnormal_margin(self, img: Image.Image, normal_key: str="normal",
                               abnormal_key: str="defective") -> float:
        """
        Liefert margin = score(abnormal) - score(normal).
        - Wenn 'defective' nicht existiert, nehmen wir den besten Defekt aus topk (ohne 'normal').
        - Ist 'normal' nicht vorhanden, setzen wir normal=0.0 (konservativ).
        """
        topk = self._rank_with_openclip(img)  # schnell + lokal
        # Score-Helper
        def _score(label: str) -> float:
            for d in topk:
                if d.get("label") == label:
                    return float(d.get("score", 0.0))
            return 0.0

        s_norm = _score(normal_key)
        s_abn  = _score(abnormal_key)

        if abnormal_key not in [d["label"] for d in topk]:
            # nimm beste Defektklasse (alles außer normal)
            cand = [float(d["score"]) for d in topk if d["label"] != normal_key]
            s_abn = max(cand) if cand else 0.0

        return float(s_abn - s_norm)
    
    # ---------------------- Utils ----------------------
    def _python_exe(self) -> str:
        return str(Path(os.sys.executable).resolve())

    def _map_openclip_name(self, name: str) -> str:
        n = name.replace("/", "-")
        if "ViT-L-14-336" in n:
            return "ViT-L-14-336"
        if "ViT-B-16" in n:
            return "ViT-B-16"
        if "RN50" in n:
            return "RN50"
        return "ViT-L-14-336"
