import os
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from PIL import Image
from ultralytics import YOLOE
from app.config import settings

class SKUPreAnnotator:
    def __init__(self, sku_dir="data/sku_standards"):
        self.torch_device = "cuda" if settings.mode == "gpu" else "cpu"
        self.detector = YOLOE(settings.zero_shot_model)
        self.detector.set_classes(["bottle", "can", "beverage carton"])

        if settings.embedder_type == "dinov2":
            self.embedder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(self.torch_device).eval()
            self.transform = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            weights = MobileNet_V3_Small_Weights.DEFAULT
            base = mobilenet_v3_small(weights=weights)
            self.embedder = nn.Sequential(base.features, base.avgpool, nn.Flatten()).to(self.torch_device).eval()
            self.transform = weights.transforms()

        self.sku_dir = sku_dir
        self.sku_gallery = {}
        self.reload_sku_gallery()

    @torch.no_grad()
    def _get_embedding(self, pil_img: Image.Image):
        tensor = self.transform(pil_img.convert("RGB")).unsqueeze(0).to(self.torch_device)
        feat = self.embedder(tensor)
        return feat / feat.norm(dim=-1, keepdim=True)

    def reload_sku_gallery(self):
        os.makedirs(self.sku_dir, exist_ok=True)
        self.sku_gallery.clear()
        for fname in os.listdir(self.sku_dir):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                sku_name = os.path.splitext(fname)[0]
                img = Image.open(os.path.join(self.sku_dir, fname))
                self.sku_gallery[sku_name] = self._get_embedding(img)

    def auto_annotate(self, image_path: str, conf_thres: float = 0.25):
        img = Image.open(image_path).convert("RGB")
        W, H = img.size
        results = self.detector.predict(
            img, imgsz=settings.imgsz, conf=conf_thres, device=settings.device, verbose=False
        )[0]

        predictions = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            crop_emb = self._get_embedding(img.crop((x1, y1, x2, y2)))

            best_sku, best_score = "unknown", 0.0
            for sku_name, sku_emb in self.sku_gallery.items():
                score = float((crop_emb @ sku_emb.T).item())
                if score > best_score:
                    best_score, best_sku = score, sku_name

            predictions.append({
                "from_name": "label",
                "to_name": "image",
                "type": "rectanglelabels",
                "score": round(best_score, 3),
                "value": {
                    "x": (x1 / W) * 100,
                    "y": (y1 / H) * 100,
                    "width": ((x2 - x1) / W) * 100,
                    "height": ((y2 - y1) / H) * 100,
                    "rectanglelabels": [best_sku]
                }
            })
        return predictions
