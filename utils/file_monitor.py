# utils/file_monitor.py
import time, threading
from pathlib import Path
from typing import List, Set, Optional
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileMovedEvent, FileCreatedEvent, FileModifiedEvent

from utils.logger import logger
from utils.processor import ImageProcessor
import io, yaml
from pathlib import Path

CFG = yaml.safe_load(Path("config/hybrid_config.yaml").read_text(encoding="utf-8"))
SUPPORTED = (".jpg", ".jpeg", ".png")

def _is_supported(p: Path) -> bool:
    return p.suffix.lower() in SUPPORTED


def _wait_until_stable(p: Path, timeout: float = 10.0, interval: float = 0.2) -> bool:
    """Warte bis Datei 'stabil' ist (Größe unverändert in zwei aufeinanderfolgenden Checks)."""
    end = time.time() + timeout
    try:
        prev = -1
        while time.time() < end:
            if not p.exists():
                time.sleep(interval)
                continue
            size = p.stat().st_size
            if size > 0 and size == prev:
                return True
            prev = size
            time.sleep(interval)
    except Exception:
        return False
    return False


class FolderMonitor:
    """Verarbeitet strikt PAARWEISE (2 Dateien = 1 Eimer). Originale werden nach Erfolg GELÖSCHT.
       In processed_images liegen nur Overlays.
       mode/classifier können via CLI übergeben werden und überschreiben die Config.
    """
    def __init__(
        self,
        input_dir: Optional[str] = None,
        processed_dir: Optional[str] = None,
        mode: Optional[str] = None,          # << neu
        classifier: Optional[str] = None,    # << neu
    ):
        self.input_dir = Path(input_dir or CFG["input_dir"]).resolve()
        self.processed_dir = Path(processed_dir or CFG["processed_dir"]).resolve()
        self.processor = ImageProcessor(
            mode_override=mode,
            classifier_override=classifier,
        )
        self._setup_dirs()

        # Pair-Queue strikt sequenziell
        self._queue: List[Path] = []
        self._seen: Set[str] = set()   # String-Pfade zur Duplikatvermeidung
        self._cv = threading.Condition()
        self._stop = threading.Event()
        self._worker = threading.Thread(target=self._run, daemon=True)

        logger.info(f"Pipeline mode      : {self.processor.mode}")
        logger.info(f"Classifier         : {self.processor.classifier}")

    def _setup_dirs(self):
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def start(self):
        logger.info(f"Watching input:    {self.input_dir}")
        logger.info(f"Writing overlays:  {self.processed_dir}")
        observer = Observer()
        observer.schedule(_NewImageHandler(self), str(self.input_dir), recursive=False)
        observer.start()
        self._worker.start()
        logger.info(f"Monitoring started on {self.input_dir}")
        try:
            while not self._stop.is_set():
                time.sleep(0.5)
        except KeyboardInterrupt:
            logger.info("Stopping monitor ...")
        finally:
            self._stop.set()
            with self._cv:
                self._cv.notify_all()
            observer.stop()
            observer.join()

    def stop(self):
        self._stop.set()
        with self._cv:
            self._cv.notify_all()

    def enqueue_if_ready(self, path: Path):
        """Datei, wenn unterstützt & stabil, in 2er-Queue aufnehmen (ohne Duplikate)."""
        path = path.resolve()
        if not _is_supported(path):
            return
        if not _wait_until_stable(path):
            #from utils.logger import logger
            logger.warning(f"Ignoriere instabile Datei: {path.name}")
            return
        with self._cv:
            key = str(path)
            if key in self._seen:
                return
            self._queue.append(path)
            self._seen.add(key)
            logger.info(f"Enqueued: {path.name} (queue={len(self._queue)})")
            self._cv.notify_all()

    def _run(self):
        while not self._stop.is_set():
            try:
                with self._cv:
                    self._cv.wait_for(lambda: self._stop.is_set() or len(self._queue) >= 2)
                    if self._stop.is_set():
                        return
                    if len(self._queue) < 2:
                        continue
                    p1 = self._queue.pop(0); self._seen.discard(str(p1))
                    p2 = self._queue.pop(0); self._seen.discard(str(p2))
                self._process_pair(p1, p2)
            except Exception as e:
                try:
                    logger.error(f"Worker-Loop error: {e}")
                except Exception:
                    pass

    def _process_pair(self, p1: Path, p2: Path):
        #from utils.logger import logger
        time.sleep(0.2)
        try:
            res = self.processor.evaluate_bucket(p1, p2)
            bid = res["bucket_id"]

            # Einzelbild-Logs
            c1, c2 = res["cam1"], res["cam2"]
            if c1["status"] == "success":
                m1 = f"[{bid}] {Path(c1['image_path']).name}: is_good={c1['is_good']} score={c1['anomaly_score']:.4f}"
                if not c1["is_good"]:
                    reasons = ", ".join(c1.get("defect_labels", []) or []) or "unbekannt"
                    m1 += f" | reasons: {reasons}"
                logger.info(m1)
            else:
                logger.error(f"[{bid}] Fehler bei {p1.name}: {c1.get('error')}")

            if c2["status"] == "success":
                m2 = f"[{bid}] {Path(c2['image_path']).name}: is_good={c2['is_good']} score={c2['anomaly_score']:.4f}"
                if not c2["is_good"]:
                    reasons = ", ".join(c2.get("defect_labels", []) or []) or "unbekannt"
                    m2 += f" | reasons: {reasons}"
                logger.info(m2)
            else:
                logger.error(f"[{bid}] Fehler bei {p2.name}: {c2.get('error')}")

            # Eimer-Entscheidung
            logger.info(f"[{bid}] Eimer-Entscheidung: {res['bucket_result']}")

            # Originale löschen
            for px in (p1, p2):
                try:
                    if px.exists():
                        px.unlink()
                except Exception as e:
                    logger.warning(f"[{bid}] Konnte {px.name} nicht löschen: {e}")

        except Exception as e:
            logger.error(f"Fehler bei Paar-Verarbeitung {p1.name} + {p2.name}: {e}")

    def process_existing_pairs_once(self):
        files = sorted(
            [p for p in self.input_dir.iterdir() if p.is_file() and _is_supported(p)],
            key=lambda p: p.stat().st_ctime
        )
        for i in range(0, len(files) - 1, 2):
            self._process_pair(files[i], files[i + 1])


class _NewImageHandler(FileSystemEventHandler):
    def __init__(self, monitor: FolderMonitor):
        self.monitor = monitor

    def on_created(self, event: FileCreatedEvent):
        if event.is_directory: return
        self.monitor.enqueue_if_ready(Path(event.src_path))

    def on_moved(self, event: FileMovedEvent):
        if event.is_directory: return
        self.monitor.enqueue_if_ready(Path(event.dest_path))

    def on_modified(self, event: FileModifiedEvent):
        if event.is_directory: return
        self.monitor.enqueue_if_ready(Path(event.src_path))