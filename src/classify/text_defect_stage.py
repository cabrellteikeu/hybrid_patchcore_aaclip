# src/run_hybrid/text_defect_stage.py
from __future__ import annotations
import os, time, json, base64, traceback
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


class TextDefectStage:
    """
    Text-geführte Defektklassifikation mit austauschbaren Backends:

      • backend='clip_ad'   : CLIP-AD-ähnlicher Zero/Few-Shot (Text+Support), leicht & Edge-tauglich
      • backend='aa_clip'   : vereinfachte AA-CLIP-Variante (ohne externes Repo), Text+Support Fusion
      • backend='promptad'  : PromptAD-ähnlich ohne Training (Prompt-Vorlagen + Textscore)

    Gemeinsame Features:
      • OpenCLIP-Encoder (Bild/Text), TTA, Temperatur-Softmax,
      • Support-Bank pro Klasse (Few-Shot, optional),
      • Top-K Ranking, optionale Heatmap (ViT Attention Rollout).

    Abhängigkeiten: torch, open_clip_torch (import as 'open_clip'), optional: pytorch_grad_cam
    """

    # ---------------------- Konstruktor ----------------------
    def __init__(
        self,
        backend: str ,                   # 'clip_ad' | 'aa_clip' | 'promptad'
        model_name: str ,               # bitte bei Raspberry klein halten (B/16 @ 224)
        device: str = "cuda",
        tta_scales: Tuple[float, ...] = (1.0,),
        temperature: float = 0.10,
        text_weight: float = 0.50,                  # Gewichtung Text vs. Support
        topk: int = 3,
        return_base64: bool = False,

        # Prompt-/Klassen-Setup
        defect_prompts: Optional[Dict[str, List[str]]] = None,  # {klasse: [prompt-varianten]}
        prompt_templates: Optional[List[str]] = None,           # extra Vorlagen, z.B. "a photo of {x}"
        synonyms_per_class: int = 0,                            # (promptad-like) automatische leichte Variation

        # Support-Bank (Few-Shot)
        support_dir: Optional[str] = None,          # Ordnerstruktur: support/<klasse>/*.jpg
        max_support_per_class: int = 20,
        cache_dir: Optional[str] = "cache/textdefect",

        # Ausgaben
        save_root: Optional[str] = "runs/textdefect",
        enable_heatmap: bool = True,                # benötigt pytorch_grad_cam ViTAttentionRollout
    ):
        self.backend = str(backend).lower().strip()
        assert self.backend in {"clip_ad", "aa_clip", "promptad"}

        self.model_name = model_name
        self.topk = int(topk)
        self.temperature = float(temperature)
        self.text_weight = float(np.clip(text_weight, 0.0, 1.0))
        self.tta_scales = tuple(tta_scales) if tta_scales else (1.0,)
        self.return_base64 = bool(return_base64)
        self.enable_heatmap = bool(enable_heatmap)

        self.save_root = Path(save_root) if save_root else None
        if self.save_root:
            _ensure_dir(self.save_root)

        self.defect_prompts: Dict[str, List[str]] = defect_prompts or {}
        self.class_names: List[str] = list(self.defect_prompts.keys())

        self.prompt_templates = list(prompt_templates or [])
        self.synonyms_per_class = int(max(0, synonyms_per_class))

        self.support_dir = Path(support_dir).resolve() if support_dir else None
        self.max_support_per_class = int(max_support_per_class)

        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            _ensure_dir(self.cache_dir)

        # Lazy-Imports & Device
        self._lazy_imports()
        if not self.has_torch or not self.has_openclip:
            raise RuntimeError("TextDefectStage: torch/open_clip konnten nicht importiert werden.")

        self._device = "cuda" if (device == "cuda" and self.torch.cuda.is_available()) else "cpu"
        self._half = (self._device == "cuda")

        # OpenCLIP vorbereiten
        self._oclip_loaded = False
        self._oclip_model = None
        self._oclip_preproc = None
        self._oclip_tokenizer = None
        self._text_cache: Dict[str, "self.torch.Tensor"] = {}
        self._support_cache: Dict[str, "self.torch.Tensor"] = {}

        self._load_openclip()
        self._build_text_cache()
        if self.support_dir:
            self._build_support_cache(self.support_dir)

    # ---------------------- Lazy Imports ----------------------
    def _lazy_imports(self):
        self.torch = None
        self.F = None
        self.open_clip = None
        self.Rollout = None
        self.has_torch = False
        self.has_openclip = False

        try:
            import torch as _torch
            import torch.nn.functional as _F
            self.torch = _torch
            self.F = _F
            self.has_torch = True
        except Exception as e:
            print("[TextDefectStage] torch import error:", repr(e))

        try:
            import open_clip as _oc
            self.open_clip = _oc
            self.has_openclip = True
        except Exception as e:
            print("[TextDefectStage] open_clip import error:", repr(e))

        # optional Heatmap
        try:
            from pytorch_grad_cam import ViTAttentionRollout as _Rollout
            self.Rollout = _Rollout
        except Exception:
            self.Rollout = None

        try:
            if self.has_torch:
                print(f"[TextDefectStage] torch {self.torch.__version__} | cuda? {self.torch.cuda.is_available()}")
            if self.has_openclip:
                print(f"[TextDefectStage] open_clip {self.open_clip.__version__}")
        except Exception:
            pass

    # ---------------------- OpenCLIP ----------------------
    def _load_openclip(self):
        # Hinweis: 'model_name' muss in eurer open_clip Version verfügbar sein.
        # Sichere Defaults: 'ViT-B-16', 'ViT-B-32', 'ViT-L-14', 'ViT-L-14-336'
        model, _, preprocess = self.open_clip.create_model_and_transforms(self.model_name, pretrained="openai")
        tokenizer = self.open_clip.get_tokenizer(self.model_name)
        model = model.to(self._device).eval()
        self.torch.set_grad_enabled(False)
        if self._device == "cuda":
            model = model.half()
            self.torch.backends.cudnn.benchmark = True
        self._oclip_model = model
        self._oclip_preproc = preprocess
        self._oclip_tokenizer = tokenizer
        self._oclip_loaded = True

    # ---------------------- Prompt Cache ----------------------
    def _prompt_aug(self, cname: str, variants: List[str]) -> List[str]:
        out = []
        # templating: "a photo of {x}", "{x} on a conveyor belt", ...
        if self.prompt_templates:
            for v in variants:
                for t in self.prompt_templates:
                    out.append(t.replace("{x}", v))
        else:
            out.extend(variants)

        # sehr leichte Synonym-Erweiterung (promptad-like, ohne externes Lexikon)
        if self.synonyms_per_class > 0:
            extras = []
            for v in variants[: self.synonyms_per_class]:
                # naive kleine Variationen (Artikel/Plural/Wortstellung)
                extras.extend([
                    f"a bucket with {v}",
                    f"{v} on bucket surface",
                    f"{v} defect on bucket",
                    f"industrial bucket, {v}",
                    f"bucket shows {v}"
                ])
            out.extend(extras)
        # Duplikate entfernen
        uniq = []
        seen = set()
        for s in out:
            s = s.strip()
            if s and s not in seen:
                seen.add(s); uniq.append(s)
        return uniq

    def _build_text_cache(self):
        self._text_cache.clear()
        if not self.class_names:
            return
        dev = self._device
        for cname, variants in self.defect_prompts.items():
            vs = self._prompt_aug(cname, list(variants or []))
            if not vs:
                continue
            toks = self._oclip_tokenizer(vs).to(dev)
            with self.torch.cuda.amp.autocast(enabled=(dev == "cuda")):
                tfeat = self._oclip_model.encode_text(toks)  # [V,D]
            tfeat = self.F.normalize(tfeat.float(), dim=-1)
            self._text_cache[cname] = tfeat

    # ---------------------- Support Cache ----------------------
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
                feats = self.torch.cat(feats, dim=0)  # [Ns,D]
                self._support_cache[cname] = feats
                if cache_pt:
                    try:
                        self.torch.save(feats.cpu(), cache_pt)
                    except Exception:
                        pass

    # ---------------------- Image Feature (mit TTA) ----------------------
    def _img_feat_tta(self, img: Image.Image):
        dev = self._device
        W, H = img.size
        crops = []
        for s in self.tta_scales:
            w = int(W * s); h = int(H * s)
            left = max((W - w) // 2, 0); top = max((H - h) // 2, 0)
            crop = img.crop((left, top, left + w, top + h)).resize((224, 224), Image.BICUBIC)
            crops.append(crop)
        tens = self.torch.stack([self._oclip_preproc(c) for c in crops]).to(dev)
        if self._half and dev == "cuda":
            tens = tens.half()
        with self.torch.cuda.amp.autocast(enabled=(dev == "cuda")):
            feats = self._oclip_model.encode_image(tens)  # [T,D]
        feats = self.F.normalize(feats.float(), dim=-1).mean(dim=0, keepdim=True)  # [1,D]
        return feats

    # ---------------------- Scoring-Kerne ----------------------
    def _text_score(self, img_feat, cname: str) -> float:
        tau = max(1e-3, self.temperature)
        if cname not in self._text_cache:
            return 0.0
        vfeats = self._text_cache[cname]                 # [V,D]
        sims = (img_feat @ vfeats.t()).squeeze(0)        # [V]
        return float(self.torch.softmax(sims / tau, dim=-1).mean().item())

    def _support_score(self, img_feat, cname: str) -> float:
        if cname not in self._support_cache:
            return 0.0
        s = self._support_cache[cname]                   # [Ns,D]
        sims = (img_feat @ s.t()).squeeze(0)             # [Ns]
        return float(sims.max().item())

    def _score_one_class(self, img_feat, cname: str) -> float:
        tw = float(np.clip(self.text_weight, 0.0, 1.0))
        iw = 1.0 - tw
        t_score = self._text_score(img_feat, cname)
        i_score = self._support_score(img_feat, cname)
        if self.backend == "clip_ad":
            # CLIP-AD-nahe Fusion: gewichtete Summe (Text + Support)
            return tw * t_score + iw * i_score
        elif self.backend == "aa_clip":
            # AA-CLIP-like (vereinfacht): leicht stärkerer Schub für Support (Nearest Neighbor)
            return (0.4 * t_score) + (0.6 * i_score)
        else:
            # promptad-like (ohne Training): nur Textscore, aber mit Template-Augmentierungen
            return t_score

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
            from PIL import ImageOps
            heat_rgb = ImageOps.colorize(m, black="black", white="red")
            overlay = Image.blend(img, heat_rgb, alpha=0.45)
            out_path = out_dir / f"heatmap_{tag}.png"
            overlay.save(out_path)
            return out_path
        except Exception:
            traceback.print_exc()
            return None

    # ---------------------- Public API ----------------------
    def infer(self, pil_image: Image.Image) -> Dict[str, Any]:
        """Gibt Top-K Defektlabels & optional Heatmap zurück (keine „Anomalie/Normal“-Entscheidung)."""
        img = ImageOps.exif_transpose(pil_image.convert("RGB"))

        ts = int(time.time() * 1000)
        run_dir = (self.save_root / f"textdefect_{ts}") if self.save_root else None
        if run_dir:
            _ensure_dir(run_dir)

        try:
            img_feat = self._img_feat_tta(img)  # [1,D]
            results = []
            for cname in self.class_names:
                sc = self._score_one_class(img_feat, cname)
                results.append((cname, float(sc)))

            results.sort(key=lambda x: x[1], reverse=True)
            topk = [{"label": k, "score": float(v)} for k, v in results[: self.topk]]

            heatmap_path = None
            if self.enable_heatmap and run_dir is not None:
                heatmap_path = self._save_rollout_heatmap(img, run_dir, tag=f"{ts}")

            heatmap_b64 = _to_b64(heatmap_path) if (self.return_base64 and heatmap_path) else None
            return {
                "ok": True,
                "backend": self.backend,
                "topk": topk,
                "heatmap_path": str(heatmap_path) if heatmap_path else None,
                "heatmap_base64": heatmap_b64,
            }
        except Exception as e:
            return {"ok": False, "error": str(e), "backend": self.backend}

    # ---------------------- Enrollment API (Few-Shot) ----------------------
    def enroll_from_dir(
        self,
        good_dir: str,
        class_name: str,
        recursive: bool = True,
        mode: str = "append",     # 'append' | 'replace'
    ) -> Dict[str, Any]:
        """
        Liest Support-Bilder für eine Klasse (Few-Shot) ein und aktualisiert den Support-Cache.
        """
        if not self.support_dir:
            return {"ok": False, "error": "support_dir is not set"}
        dev = self._device
        src = Path(good_dir).resolve()
        if not src.exists():
            return {"ok": False, "error": f"path not found: {src}"}

        # Zielordner
        cls_dir = self.support_dir / class_name
        _ensure_dir(cls_dir)

        # Sammle Bilder
        pats = ("*.jpg", "*.jpeg", "*.png")
        paths = []
        if recursive:
            for pat in pats:
                paths.extend(list(src.rglob(pat)))
        else:
            for pat in pats:
                paths.extend(list(src.glob(pat)))

        # Kopieren/Verlinken ist optional – hier nur Featurisierung
        feats = []
        for p in paths[: self.max_support_per_class]:
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

        if not feats:
            return {"ok": False, "error": "no valid images found"}

        feats = self.torch.cat(feats, dim=0)  # [N,D]
        if mode == "replace" or class_name not in self._support_cache:
            self._support_cache[class_name] = feats
        else:
            self._support_cache[class_name] = self.F.normalize(
                self.torch.cat([self._support_cache[class_name], feats], dim=0).float(), dim=-1
            )

        # Cache speichern
        cache_pt = self._support_cache_path(class_name)
        if cache_pt:
            try:
                self.torch.save(self._support_cache[class_name].cpu(), cache_pt)
            except Exception:
                pass

        # Klasse ggf. zu Prompts hinzufügen, falls unbekannt
        if class_name not in self.defect_prompts:
            self.defect_prompts[class_name] = [class_name]
            self.class_names.append(class_name)
            self._build_text_cache()

        return {"ok": True, "class_name": class_name, "num_support": int(self._support_cache[class_name].shape[0])}