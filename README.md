# YOLO26 冰柜饮料陈列识别 MLOps 线性流水线

集成了 **SKU 标准图特征库构建 ➔ 自动预标注 ➔ Label Studio 人工微调 ➔ 8:2 数据集拆分 ➔ YOLO26 训练 ➔ ONNX 端到端无 NMS 在线检测 API** 的全流程平台，支持通过 Docker Compose `--profile cpu` / `--profile gpu` 无缝切换算力模式。

## 快速启动

### 1. 纯 CPU 调研模式启动（默认推荐初期验证）
```bash
docker compose --profile cpu up --build -d
```

### 2. GPU 生产环境启动
```bash
docker compose --profile gpu up --build -d
```

## 服务端口说明
- **Streamlit 可视化线性控制台**: `http://localhost:8501`
- **FastAPI 后端与检测 API 文档**: `http://localhost:8000/docs`
- **Label Studio 标注平台**: `http://localhost:8080`
