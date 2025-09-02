# src/models/train_patchcore.py
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
import torch
from anomalib.data import Folder
from anomalib.models import Patchcore
from anomalib.engine import Engine

from torchvision import transforms as T
import io, yaml
from pathlib import Path

CFG = yaml.safe_load(Path("config/hybrid_config.yaml").read_text(encoding="utf-8"))
def main():
    seed_everything(42)

    # Datenmodul entsprechend deiner config.yaml
    # Konsistente Transforms
    torch.set_float32_matmul_precision("high")  # nutzt Tensor Cores deiner RTX A2000

    # Nur bildgeometrische/photometrische Augs auf Tensor-Basis, KEIN ToTensor/Normalize hier!
    train_aug = T.Compose([
        T.Resize((384, 384)),
        T.ColorJitter(brightness=0.1, contrast=0.1),  # mild
    ])
    eval_aug = T.Compose([
        T.Resize((384, 384)),
    ])

    datamodule = Folder(
        name=CFG["patchcore"]["module_name"],
        root=CFG["patchcore"]["image_dir"],
        normal_dir=CFG["patchcore"]["good_folder"],
        train_batch_size=8,
        #eval_batch_size=32,
        num_workers=0,  # <- Workaround: vermeidet Multiprocessing auf Windows
        extensions=(".jpg", ".jpeg", ".png"),
        #train_augmentations=train_aug,
        #val_augmentations=eval_aug,
        #test_augmentations=eval_aug,
    )

    # PatchCore-Modell initialisieren
    pre_proc = Patchcore.configure_pre_processor(image_size=(384, 384))
    model = Patchcore(
        backbone=CFG["patchcore"]["backbone"], # wide_resnet50_2 resnet18
        layers=["layer2","layer3"],
        pre_trained=True,
        coreset_sampling_ratio=0.1,
        num_neighbors=9,
        #pre_processor=pre_proc,
    )

    # Engine verwenden (Referenz für Engine API in v2.0.0)
    engine = Engine()

    # Trainer-Setting aus config
    trainer = Trainer(
        max_epochs=1,
        accelerator="auto",
        devices=1 ,#if os.getenv("CUDA_VISIBLE_DEVICES") else None,
        log_every_n_steps=10,
        logger=TensorBoardLogger(save_dir="logs", name="patchcore"),
    )

    # Model trainieren
    #engine.train(datamodule=datamodule, model=model, trainer=trainer)
    engine.fit(model, datamodule)

if __name__ == "__main__":
    main()
