import os
import json
import shutil
from io import BytesIO
from collections import Counter
from typing import Optional
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from pydantic import BaseModel
from PIL import Image
from ultralytics import YOLO

from app.config import settings
from app.pre_annotator import SKUPreAnnotator
from app.converter import LabelStudioToYOLOConverter
from app.trainer import YOLO26Trainer

app = FastAPI(title=f"YOLO26 冰柜饮料识别线性平台 [{settings.mode.upper()} 模式]")

pre_annotator = SKUPreAnnotator()
converter = LabelStudioToYOLOConverter()
trainer = YOLO26Trainer()

prod_onnx_path = "models/production/best.onnx"
inference_model = YOLO(prod_onnx_path, task="detect") if os.path.exists(prod_onnx_path) else YOLO(settings.yolo_base_model)

@app.post("/step1/upload_sku_standard", tags=["Step 1: 数据上传"])
async def upload_sku_standard(sku_name: str, file: UploadFile = File(...)):
    save_path = os.path.join("data/sku_standards", f"{sku_name}.jpg")
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    pre_annotator.reload_sku_gallery()
    return {"msg": f"SKU [{sku_name}] 已入库，当前共 {len(pre_annotator.sku_gallery)} 个标准 SKU"}

@app.post("/step1/upload_cooler_image", tags=["Step 1: 数据上传"])
async def upload_cooler_image(file: UploadFile = File(...)):
    os.makedirs("data/raw_images", exist_ok=True)
    save_path = os.path.join("data/raw_images", file.filename)
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    pre_annotations = pre_annotator.auto_annotate(save_path)
    return {"image_path": save_path, "auto_annotations": pre_annotations}

class LabelStudioPredictRequest(BaseModel):
    tasks: list

@app.post("/step2/label_studio_webhook/predict", tags=["Step 2: 自动+手动标注"])
async def label_studio_predict(req: LabelStudioPredictRequest):
    results = []
    for task in req.tasks:
        img_path = task["data"]["image"]
        preds = pre_annotator.auto_annotate(img_path)
        results.append({"result": preds, "score": 0.90})
    return {"results": results}

@app.post("/step2_5/prepare_dataset_from_json", tags=["Step 2.5: 数据集 8:2 拆分转换"])
async def prepare_dataset_from_json(
    train_ratio: float = 0.8,
    file: UploadFile = File(..., description="从 Label Studio 导出的 JSON 标注文件")
):
    try:
        content = await file.read()
        ls_tasks = json.loads(content.decode("utf-8"))
        summary = converter.convert_and_split(ls_tasks, train_ratio=train_ratio)
        return {"msg": "数据集 8:2 拆分构建成功！可直接进入 Step 3 启动训练", "dataset_summary": summary}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"数据集转换失败: {str(e)}")

class LabelStudioSyncRequest(BaseModel):
    ls_url: str = "http://label-studio:8080"
    api_token: str
    project_id: int = 1
    train_ratio: float = 0.8

@app.post("/step2_5/sync_from_label_studio", tags=["Step 2.5: 数据集 8:2 拆分转换"])
async def sync_from_label_studio(req: LabelStudioSyncRequest):
    try:
        summary = converter.pull_and_convert_from_api(
            ls_url=req.ls_url,
            api_token=req.api_token,
            project_id=req.project_id,
            train_ratio=req.train_ratio
        )
        return {"msg": "已从 Label Studio 同步并完成 8:2 拆分！", "dataset_summary": summary}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"从 Label Studio 同步失败: {str(e)}")

class TrainRequest(BaseModel):
    class_names: Optional[list[str]] = None
    epochs: Optional[int] = None
    imgsz: Optional[int] = None

@app.post("/step3/trigger_train", tags=["Step 3: 模型训练平台"])
async def trigger_train(req: TrainRequest, background_tasks: BackgroundTasks):
    if trainer.status["state"] == "training":
        raise HTTPException(status_code=400, detail="当前已有训练任务正在运行中")

    class_names = req.class_names
    classes_txt_path = os.path.join("data/dataset", "classes.txt")
    if not class_names:
        if not os.path.exists(classes_txt_path):
            raise HTTPException(status_code=400, detail="未找到拆分后的数据集，请先调用 /step2_5 接口转换数据！")
        with open(classes_txt_path, "r", encoding="utf-8") as f:
            class_names = [line.strip() for line in f.readlines() if line.strip()]

    background_tasks.add_task(trainer.run_training_pipeline, class_names, req.epochs, req.imgsz)
    return {
        "msg": f"YOLO26 [{settings.mode.upper()}模式] 训练已在后台启动",
        "classes": class_names,
        "status_endpoint": "/step3/train_status"
    }

@app.get("/step3/train_status", tags=["Step 3: 模型训练平台"])
async def get_train_status():
    return trainer.status

@app.post("/step4/reload_production_model", tags=["Step 4: 在线检测 API"])
async def reload_model():
    global inference_model
    if not os.path.exists(prod_onnx_path):
        raise HTTPException(status_code=404, detail="未找到训练产出的 best.onnx 生产模型")
    inference_model = YOLO(prod_onnx_path, task="detect")
    return {"msg": f"线上生产模型已热更新为最新 ONNX 版本 ({settings.mode.upper()} 模式)！"}

@app.post("/step4/detect_cooler", tags=["Step 4: 在线检测 API"])
async def detect_cooler(file: UploadFile = File(...), conf: float = 0.4):
    img = Image.open(BytesIO(await file.read())).convert("RGB")
    res = inference_model.predict(
        source=img, imgsz=settings.imgsz, conf=conf, device=settings.device, nms=False, verbose=False
    )[0]

    detections, sku_counts = [], Counter()
    for box in res.boxes:
        cls_id = int(box.cls[0].item())
        sku_name = res.names[cls_id]
        x1, y1, x2, y2 = [round(float(v), 1) for v in box.xyxy[0].tolist()]
        detections.append({
            "sku_name": sku_name,
            "confidence": round(float(box.conf[0].item()), 4),
            "bbox": [x1, y1, x2, y2]
        })
        sku_counts[sku_name] += 1

    return {
        "compute_mode": settings.mode,
        "total_bottles": len(detections),
        "sku_facing_summary": dict(sku_counts),
        "detections": detections
    }
