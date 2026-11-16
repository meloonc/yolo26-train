import os
import shutil
import yaml
from ultralytics import YOLO
from app.config import settings

class YOLO26Trainer:
    def __init__(self, dataset_dir="data/dataset", prod_dir="models/production"):
        self.dataset_dir = dataset_dir
        self.prod_dir = prod_dir
        os.makedirs(self.prod_dir, exist_ok=True)
        self.status = {"state": "idle", "mode": settings.mode, "message": f"Ready in {settings.mode.upper()} mode"}

    def run_training_pipeline(self, class_names: list, epochs: int = None, imgsz: int = None):
        epochs = epochs or settings.epochs
        imgsz = imgsz or settings.imgsz
        try:
            self.status = {
                "state": "training",
                "mode": settings.mode,
                "message": f"[{settings.mode.upper()}] 正在训练 {settings.yolo_base_model} (epochs={epochs}, imgsz={imgsz})..."
            }

            yaml_path = os.path.join(self.dataset_dir, "beverage.yaml")
            with open(yaml_path, "w", encoding="utf-8") as f:
                yaml.dump({
                    "path": os.path.abspath(self.dataset_dir),
                    "train": "images/train",
                    "val": "images/val",
                    "names": {i: name for i, name in enumerate(class_names)}
                }, f, allow_unicode=True)

            model = YOLO(settings.yolo_base_model)
            results = model.train(
                data=yaml_path,
                epochs=epochs,
                imgsz=imgsz,
                batch=settings.batch,
                device=settings.device,
                freeze=settings.freeze,
                cache="ram" if settings.mode == "cpu" else False,
                workers=settings.workers,
                optimizer="MuSGD",
                mosaic=0.5 if settings.mode == "cpu" else 1.0,
                copy_paste=0.3,
                project="runs/detect",
                name=f"exp_{settings.mode}",
                exist_ok=True
            )

            best_pt = os.path.join(results.save_dir, "weights", "best.pt")
            shutil.copy(best_pt, os.path.join(self.prod_dir, "best.pt"))

            prod_model = YOLO(best_pt)
            exported_onnx = prod_model.export(
                format="onnx",
                imgsz=imgsz,
                half=settings.half_export,
                simplify=True,
                nms=False,
                device=settings.device
            )
            shutil.copy(exported_onnx, os.path.join(self.prod_dir, "best.onnx"))

            self.status = {"state": "completed", "mode": settings.mode, "message": "训练完成，已发布 best.onnx 至生产目录！"}
        except Exception as e:
            self.status = {"state": "failed", "mode": settings.mode, "message": str(e)}
