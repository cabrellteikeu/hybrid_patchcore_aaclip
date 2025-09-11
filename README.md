# Hybrid Bucket QA – Raspberry Setup & Betrieb (CPU / Edge)

Diese Anleitung bringt die Eimer-QA-Pipeline auf einem **Raspberry** ins Laufen.  
Die Pipeline ist modular:

Diese Anleitung bringt die Eimer-QA-Pipeline auf einem **Raspberry** ins Laufen.  
Die Pipeline ist modular:

1) **(Optional) Preprocessing – GroundingDINO + SAM**  
   → Sucht den Bucket im Bild und erzeugt einen freigestellten, zentrierten Crop mit weißem Hintergrund.

2) **Anomalieerkennung (wähle 1)**
   - **PatchCore** *(anomalib 2.0.0)* – schnell & präzise pixelweise Heatmaps  
   - **DINOv2** – robust gegenüber neuen Varianten; ohne Bank sofort nutzbar *(optional: Bank verbessert Stabilität)*
   - **Ensemble** PatchCore + DINOv2 – normierte Score-/Heatmap-Fusion für mehr Robustheit

3) **Text-Erklärung (Zero-Shot, ohne Support-Bilder)**  
   - **ZS-CLIP/TextDefectStage**: liefert **Top-3 Defektbeschreibungen** aus den Prompts (z. B. `no_label`, `torn_label`, …)

Man kann die Pipeline als **FastAPI** (Swagger UI) oder als **Live-Watcher** (Ordnerüberwachung) betreiben.

## Verzeichnisstruktur

```
.
├─ config/
│  ├─ hybrid\_config.yaml          # zentrale Konfiguration (Edge-Profil unten)
│  └─ GroundingDINO_SwinT_OGC.py   # Konfigurationen für GroundingDINO
├─ deployment/
│  └─ hybrid\_api.py               # FastAPI: /predict, /dino/enroll, ...
├─ src/
│  ├─ anomaly\_dino_/
│     └─ dino\_stage_.py
│  ├─ classify/
│     ├─ text\_defect_stage_.py
│     ├─ zeroshot\_clip.py
│     └─ zsclip\_defect\_stage_.py
│  ├─ defects/
│     └─ defect\_prompts_.py
│  ├─ patchcore/
│     ├─ patchcore\_infer.py
│     └─ train\_patchcore_.py
│  ├─ preprocess/
│     └─ grounded\_sam_.py
│  ├─ viz/
│     └─ overlay.py         
├─ utils/
│  ├─ processor.py                # Pipeline-Logik (pc\_only / dino\_only / ensemble)
│  ├─ file\_monitor.py             # Ordner-Überwachung (2 Bilder = 1 Bucket)
│  └─ logger.py
├─ results/inference/             # Overlays/Heatmaps (Output)
├─ banks/dino/                    # (optional) DINO Patch-Bank Cache
├─ logs/weights/lightning/        # PatchCore .ckpt (lokal, NICHT im Repo)
├─ models/                        # 
│  ├─ groundingDINO\_SAM          # 
│  └─ patchcore
├─ main.py
└─ requirements.txt

````
> Große Dateien (.ckpt/.pt/.pth) **nicht** committen – lokal ablegen (siehe unten „Modelldateien“).

---

## 1) Systemvorbereitung

Auf einem **frischen Raspberry Pi OS (64-bit)**:

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3-venv python3-pip git build-essential \
                    libopenblas-dev libjpeg-dev zlib1g-dev \
                    libgl1 libglib2.0-0
# optional für schnellere Builds:
sudo apt install -y cmake ninja-build
````
**Swap erhöhen (optional, für schwere Builds/Modelle):**

```bash
# Datei editieren:
sudo nano /etc/dphys-swapfile
# CONF_SWAPSIZE=2048  (z.B. 2GB)
sudo dphys-swapfile setup
sudo dphys-swapfile swapon
```

## 2) Projekt klonen & Python-Umgebung

```bash
# Arbeitsverzeichnis (z.B.)
mkdir -p ~/apps && cd ~/apps

# Repo klonen
git clone <DEIN_REPO_URL> bucket-qa
cd bucket-qa

# Virtuelle Umgebung
python3 -m venv .venv
source .venv/bin/activate

# Pip aktualisieren
pip install --upgrade pip wheel
```

**PyTorch CPU-Wheels (ARM64, ohne CUDA):**

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

**Weitere Abhängigkeiten:**

```bash
pip install -r requirements.txt
```

> Falls `requirements.txt` GPUspezifisches enthält: auf dem Pi bleiben wir bei CPU-Wheels.

## 3) Modelle & große Dateien (lokal ablegen)

Lege folgende Dateien **lokal** ab (nicht ins Repo):

- **SAM Model** (nur bei Preprocessing):  
  `models/groundingDINO_SAM/sam_vit_h.pth`  
  [Download](https://huggingface.co/HCMUE-Research/SAM-vit-h/blob/main/sam_vit_h_4b8939.pth)

- **GroundingDINO** (nur bei Preprocessing):  
  `models/groundingDINO_SAM/groundingdino_swint_ogc.pth`  
  [Download](https://huggingface.co/pengxian/grounding-dino/blob/main/groundingdino_swint_ogc.pth)

- **PatchCore Checkpoint**:  
  `models/patchcore/model_resnet50.ckpt`



## 4) Konfiguration (Edge-Profil)

Bearbeite `config/hybrid_config.yaml`:

```yaml
device: "cpu"       # Für Raspberry Pi (CPU statt GPU)
# Preprocessing: deaktiviert
# PatchCore: Standard-Einstellungen
```



## 5) Sanity-Check (Importe)

```bash
python - <<'PY'
import torch, platform
print("torch:", torch.__version__, "| cuda:", torch.cuda.is_available(), "| arch:", platform.machine())
from PIL import Image
import numpy as np
print("Pillow/NumPy OK")
PY
```

Wenn das durchläuft, ist die Grundumgebung OK.

##  6) Startvarianten

### 6.1 FastAPI (Swagger UI)

```bash
source .venv/bin/activate
uvicorn deployment.hybrid_api:app --host 0.0.0.0 --port 8000 --workers 1
```

* Swagger UI: `http://<RASPI_IP>:8000/docs`
* **/predict** → Bild hochladen, optional Query `?detector=pc|dino|ensemble` (falls aktiviert), sonst `pipeline.mode`.
* **/dino/enroll** → DINO mit gutem Bilder-Ordner befüllen (*optional*).

**DINO-Enroll per cURL:**

```bash
curl -X POST "http://<RASPI_IP>:8000/dino/enroll?sku_id=default&view=single" \
     -H "Content-Type: application/json" \
     -d '{"folder_path": "/home/pi/data/good/default_single"}'
```

### 6.2 Live-Watcher (2 Bilder = 1 Eimer)

```bash
source .venv/bin/activate
python main.py --watch
# oder einmalige Verarbeitung:
# python main.py --once
```

Die Pfade für Eingangs- und Ausgabeverzeichnisse kommen aus `config/config.py` (LIVE\_IMAGES\_DIRS).
**Overlays/Heatmaps** liegen unter `results/inference/` bzw. `CFG["processed_dir"]`.

## 7) Optional: GroundingDINO + SAM aktivieren (schwer!)

Auf dem Pi **deutlich langsamer** – nutze das nur, wenn unbedingt nötig.

1. Installiere GroundingDINO **aus dem Git-Repo** (verhindert defekte C++-Ops Imports):

   ```bash
   source .venv/bin/activate
   pip install git+https://github.com/IDEA-Research/GroundingDINO.git
   pip install segment-anything==1.0 supervision==0.6.0  # falls nicht mitinstalliert
   ```
2. Lege Checkpoints ab:

   ```
   assets/checkpoints/groundingdino_swint_ogc.pth
   assets/checkpoints/sam_vit_h.pth
   ```
3. In `config/hybrid_config.yaml`:

   ```yaml
   preprocess:
     enabled: true
     # ggf. Thresholds etwas anheben (0.30–0.40), damit nur klare Bucketerkennung gecroppt wird
   ```
4. Start wie oben. Fallback: Wenn Preproc scheitert, verwendet die Pipeline automatisch das Originalbild.

## 8) Systemdienst (systemd) – Autostart auf dem Pi

### 8.1 Service für die **API**

**Datei erstellen:** `/etc/systemd/system/bucket-api.service`

```ini
[Unit]
Description=Bucket QA API (FastAPI, Raspberry)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/apps/bucket-qa
Environment="PYTHONUNBUFFERED=1"
Environment="PORT=8000"
ExecStart=/home/pi/apps/bucket-qa/.venv/bin/uvicorn deployment.hybrid_api:app --host 0.0.0.0 --port 8000 --workers 1
Restart=on-failure
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable bucket-api
sudo systemctl start bucket-api
# Logs ansehen:
journalctl -u bucket-api -f
```

### 8.2 Service für den **Watcher**

**Datei:** `/etc/systemd/system/bucket-watcher.service`

```ini
[Unit]
Description=Bucket QA Watcher (2 images = 1 bucket)
After=network-online.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/apps/bucket-qa
ExecStart=/home/pi/apps/bucket-qa/.venv/bin/python main.py --watch
Restart=always
RestartSec=3
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable bucket-watcher
sudo systemctl start bucket-watcher
journalctl -u bucket-watcher -f
```

> **API & Watcher getrennt** betreiben (verschiedene Services) ist praktisch:
> Man kann via API testen und parallel den Live-Watcher laufen lassen.

## 9) Wichtige Stellschrauben (Edge)

* **Pipeline-Modus**: `pipeline.mode` = `pc_only` *(empfohlen für Pi)*, `dino_only`, `ensemble`.
* **PatchCore**: `image_size` 224–256 (kleiner = schneller), `threshold ~0.7`.
* **DINOv2**: `use_half: false` (CPU), `sim_chunk_size` reduzieren, `max_bank_patches` klein halten.
* **TextDefect**: `model_name: ViT-B-16`, `temperature: 0.2`, `tta_scales: [1.0]`.
* **Preprocessing**: deaktiviert lassen, wenn die Laufzeit kritisch ist; sonst aktivieren und mit Thresholds arbeiten.

---

## 10) Troubleshooting (Raspberry)

**A) GroundingDINO: „Failed to load custom C++ ops / MSDeformAttn“**
→ Meist durch falsche Installation.

```bash
pip uninstall -y groundingdino groundingdino-py
pip install --no-cache-dir git+https://github.com/IDEA-Research/GroundingDINO.git
python -c "import groundingdino,inspect;print('GDINO @',groundingdino.__file__)"
```

Pfad sollte in dein **site-packages** zeigen (nicht in eine lokale Projektkopie).

**B) Sehr langsam / OOM**

* Preprocessing deaktivieren.
* Kleinere Bilder (`image_size: [224,224]`).
* DINO-Bank klein (`max_bank_patches: 4000–8000`).
* Nur **Top-3** Erklärungen und TTA aus.

**C) Kein Overlay / leere Heatmap**

* Prüfe, ob der Detector überhaupt triggert (Score vs. Threshold).
* `results/inference/` schreiben dürfen? Rechte / Pfade prüfen.

**D) API startet, aber 500 bei /predict**

* Fehlt ein Checkpoint (PatchCore `.ckpt`)? Pfad in YAML prüfen.
* Pillow/NumPy import ok? `python -c "import PIL, numpy; print('OK')"`

---

## 11) Kurztests

**API lokal testen (ohne Swagger):**

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/home/pi/sample.jpg"
```

**DINO-Enroll (Ordner mit guten Bildern):**

```bash
curl -X POST "http://localhost:8000/dino/enroll?sku_id=default&view=single" \
     -H "Content-Type: application/json" \
     -d '{"folder_path": "/home/pi/data/good/default_single"}'
```

---

## 12) Hinweise zur Wartung

* **Prompts** pflegst du in `src/defects/defect_prompts.py`.
* **Moduswechsel** & Tuning über `config/hybrid_config.yaml`.
* **Logs** via `journalctl -u bucket-api -f` bzw. `-u bucket-watcher -f`.
* **Updates**: Neue Version pullen, venv aktivieren, `pip install -r requirements.txt`, Services neu starten.

---

## 13) Zusammenfassung

* **Standard auf Pi**: `pc_only`, Preprocessing **aus**, `ViT-B-16` für TextDefect.
* **DINO** optional dazu (oder Ensemble) – gute Balance aus Robustheit/Tempo.
* **Preprocessing** (GroundingDINO+SAM) nur, wenn unbedingt erforderlich – dann Geduld einplanen.
