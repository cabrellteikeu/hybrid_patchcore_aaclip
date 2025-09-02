# main.py
from __future__ import annotations
import argparse
import signal
import sys
from utils.logger import logger
from utils.file_monitor import FolderMonitor
import io, yaml
from pathlib import Path

CFG = yaml.safe_load(Path("config/hybrid_config.yaml").read_text(encoding="utf-8"))

def main():
    parser = argparse.ArgumentParser(description="Hybrid AD: PatchCore/DINO + (optionaler) Textklassifizierer")
    parser.add_argument("--input", type=str, default=CFG["input_dir"], help="Eingangsordner (Default aus config)")
    parser.add_argument("--processed", type=str, default=CFG["processed_dir"], help="Output-Ordner für Overlays")
    parser.add_argument("--once", action="store_true", help="Nur bestehende Bilder paarweise verarbeiten und beenden")
    parser.add_argument("--watch", action="store_true", help="Ordner dauerhaft überwachen (paarweise Verarbeitung)")
    parser.add_argument("--mode", type=str, choices=["pc_only", "dino_only", "ensemble"], default=None,
                        help="Pipeline-Engine: PatchCore, DINOv2 oder Ensemble (überschreibt Config)")
    parser.add_argument("--classifier", type=str, choices=["zsclip", "none"], default=None,
                        help="Defekt-Klassifizierer: Zero-Shot CLIP, AA-CLIP oder keiner (überschreibt Config)")
    args = parser.parse_args()

    mon = FolderMonitor(
        input_dir=args.input,
        processed_dir=args.processed,
        mode=args.mode,
        classifier=args.classifier,
    )

    def _stop(*_):
        mon.stop()
        sys.exit(0)
    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    if args.once and not args.watch:
        mon.process_existing_pairs_once()
        return

    # Erst bestehende Bilder wegräumen, dann Live-Modus
    mon.process_existing_pairs_once()
    mon.start()

if __name__ == "__main__":
    main()