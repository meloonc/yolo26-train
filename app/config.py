import os
import torch

class RuntimeConfig:
    def __init__(self):
        mode_env = os.getenv("COMPUTE_MODE", "auto").lower()
        if mode_env == "gpu" or (mode_env == "auto" and torch.cuda.is_available()):
            self.mode = "gpu"
        else:
            self.mode = "cpu"

        if self.mode == "gpu":
            self.device = "0"
            self.yolo_base_model = "yolo26m.pt"
            self.zero_shot_model = "yoloe-26s-seg.pt"
            self.embedder_type = "dinov2"
            self.imgsz = 1280
            self.batch = 16
            self.epochs = 100
            self.freeze = None
            self.workers = 4
            self.half_export = True
        else:
            self.device = "cpu"
            torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))
            self.yolo_base_model = "yolo26n.pt"
            self.zero_shot_model = "yoloe-26n-seg.pt"
            self.embedder_type = "mobilenet"
            self.imgsz = 640
            self.batch = 4
            self.epochs = 35
            self.freeze = 10
            self.workers = 2
            self.half_export = False

settings = RuntimeConfig()
