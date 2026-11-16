import os
import json
import random
import shutil
from urllib.parse import unquote
import urllib.request

class LabelStudioToYOLOConverter:
    def __init__(self, raw_img_dir="data/raw_images", dataset_dir="data/dataset"):
        self.raw_img_dir = raw_img_dir
        self.dataset_dir = dataset_dir

    def _reset_dirs(self):
        if os.path.exists(self.dataset_dir):
            shutil.rmtree(self.dataset_dir)
        for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
            os.makedirs(os.path.join(self.dataset_dir, sub), exist_ok=True)

    def _resolve_image_path(self, ls_image_url: str) -> str:
        decoded_url = unquote(ls_image_url)
        if "?d=" in decoded_url:
            rel_path = decoded_url.split("?d=")[-1]
            candidate = os.path.join("data", rel_path)
            if os.path.exists(candidate):
                return candidate

        filename = os.path.basename(decoded_url)
        raw_candidate = os.path.join(self.raw_img_dir, filename)
        if os.path.exists(raw_candidate):
            return raw_candidate

        if "-" in filename:
            clean_name = filename.split("-", 1)[-1]
            clean_candidate = os.path.join(self.raw_img_dir, clean_name)
            if os.path.exists(clean_candidate):
                return clean_candidate

        for root, _, files in os.walk("data"):
            if "dataset" in root:
                continue
            if filename in files:
                return os.path.join(root, filename)

        raise FileNotFoundError(f"无法定位原图文件: {ls_image_url}")

    def convert_and_split(self, ls_tasks: list, train_ratio: float = 0.8, seed: int = 42) -> dict:
        self._reset_dirs()
        valid_samples = []
        class_set = set()

        for task in ls_tasks:
            annotations = task.get("annotations", [])
            if not annotations:
                continue

            latest_ann = annotations[-1]
            if latest_ann.get("was_cancelled", False):
                continue

            img_url = task.get("data", {}).get("image", "")
            try:
                src_img_path = self._resolve_image_path(img_url)
            except FileNotFoundError:
                continue

            yolo_boxes = []
            for item in latest_ann.get("result", []):
                if item.get("type") != "rectanglelabels":
                    continue
                val = item.get("value", {})
                labels = val.get("rectanglelabels", [])
                if not labels:
                    continue

                sku_name = labels[0]
                class_set.add(sku_name)

                x_min_pct = float(val["x"])
                y_min_pct = float(val["y"])
                w_pct = float(val["width"])
                h_pct = float(val["height"])

                x_center = max(0.0, min(1.0, (x_min_pct + w_pct / 2.0) / 100.0))
                y_center = max(0.0, min(1.0, (y_min_pct + h_pct / 2.0) / 100.0))
                w_norm = max(0.0, min(1.0, w_pct / 100.0))
                h_norm = max(0.0, min(1.0, h_pct / 100.0))

                yolo_boxes.append((sku_name, x_center, y_center, w_norm, h_norm))

            if yolo_boxes:
                valid_samples.append({
                    "task_id": task.get("id", random.randint(1000, 9999)),
                    "img_path": src_img_path,
                    "boxes": yolo_boxes
                })

        if not valid_samples:
            raise ValueError("未找到有效的人工标注数据，请先在 Label Studio 中点击 Submit/Update 提交标注！")

        class_names = sorted(list(class_set))
        class_to_id = {name: idx for idx, name in enumerate(class_names)}

        random.seed(seed)
        random.shuffle(valid_samples)

        total = len(valid_samples)
        if total > 1:
            split_idx = max(1, min(total - 1, int(round(total * train_ratio))))
            train_samples = valid_samples[:split_idx]
            val_samples = valid_samples[split_idx:]
        else:
            train_samples = valid_samples
            val_samples = valid_samples

        def _write_subset(samples, subset_name):
            for sample in samples:
                src_img = sample["img_path"]
                ext = os.path.splitext(src_img)[1]
                base_name = f"task_{sample['task_id']}_{os.path.splitext(os.path.basename(src_img))[0]}"

                dst_img = os.path.join(self.dataset_dir, "images", subset_name, base_name + ext)
                dst_txt = os.path.join(self.dataset_dir, "labels", subset_name, base_name + ".txt")

                shutil.copy2(src_img, dst_img)
                with open(dst_txt, "w", encoding="utf-8") as f:
                    for sku_name, xc, yc, w, h in sample["boxes"]:
                        cls_id = class_to_id[sku_name]
                        f.write(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

        _write_subset(train_samples, "train")
        _write_subset(val_samples, "val")

        classes_file = os.path.join(self.dataset_dir, "classes.txt")
        with open(classes_file, "w", encoding="utf-8") as f:
            f.write("\n".join(class_names))

        return {
            "total_annotated_images": total,
            "train_count": len(train_samples),
            "val_count": len(val_samples),
            "class_names": class_names,
            "dataset_dir": os.path.abspath(self.dataset_dir)
        }

    def pull_and_convert_from_api(self, ls_url: str, api_token: str, project_id: int, train_ratio: float = 0.8) -> dict:
        export_url = f"{ls_url.rstrip('/')}/api/projects/{project_id}/export?exportType=JSON"
        req = urllib.request.Request(export_url, headers={"Authorization": f"Token {api_token}"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            ls_tasks = json.loads(resp.read().decode("utf-8"))
        return self.convert_and_split(ls_tasks, train_ratio=train_ratio)
