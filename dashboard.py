import os
import time
from io import BytesIO
import requests
import streamlit as st
from PIL import Image, ImageDraw

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
LABEL_STUDIO_EXTERNAL_URL = os.getenv("LABEL_STUDIO_EXTERNAL_URL", "http://localhost:8080")
LABEL_STUDIO_INTERNAL_URL = os.getenv("LABEL_STUDIO_INTERNAL_URL", "http://label-studio:8080")

st.set_page_config(
    page_title="YOLO26 冰柜陈列识别流水线平台",
    page_icon="🥤",
    layout="wide"
)

PALETTE = [
    "#FF3838", "#2C99A8", "#FF701F", "#FFB21D", "#CFD231",
    "#48F90A", "#92CC17", "#3DDB86", "#1A9334", "#00D4BB",
    "#2C99A8", "#00C2FF", "#344593", "#6473FF", "#0018EC"
]

def get_sku_color(sku_name: str) -> str:
    return PALETTE[abs(hash(sku_name)) % len(PALETTE)]

def draw_ls_pre_annotations(pil_img: Image.Image, annotations: list) -> Image.Image:
    img = pil_img.copy()
    W, H = img.size
    draw = ImageDraw.Draw(img)
    for item in annotations:
        val = item.get("value", {})
        labels = val.get("rectanglelabels", ["unknown"])
        sku = labels[0] if labels else "unknown"
        score = item.get("score", 0.0)

        x1 = (val.get("x", 0) / 100.0) * W
        y1 = (val.get("y", 0) / 100.0) * H
        x2 = x1 + (val.get("width", 0) / 100.0) * W
        y2 = y1 + (val.get("height", 0) / 100.0) * H

        color = get_sku_color(sku)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        draw.text((x1 + 4, max(0, y1 - 16)), f"{sku} ({score:.2f})", fill=color)
    return img

def draw_yolo_detections(pil_img: Image.Image, detections: list) -> Image.Image:
    img = pil_img.copy()
    draw = ImageDraw.Draw(img)
    for det in detections:
        sku = det["sku_name"]
        conf = det["confidence"]
        x1, y1, x2, y2 = det["bbox"]
        color = get_sku_color(sku)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        draw.text((x1 + 4, max(0, y1 - 16)), f"{sku} {conf:.2f}", fill=color)
    return img

with st.sidebar:
    st.title("🥤 YOLO26 线性平台")
    st.caption("Upload ➔ Annotate ➔ Split ➔ Train ➔ API")

    try:
        resp = requests.get(f"{API_BASE_URL}/step3/train_status", timeout=3)
        if resp.status_code == 200:
            status_data = resp.json()
            mode = status_data.get("mode", "cpu").upper()
            state = status_data.get("state", "idle")
            if mode == "GPU":
                st.success("⚡ 当前运行模式: **GPU 满血模式**")
            else:
                st.info("💻 当前运行模式: **CPU 极速调研模式**")
            st.metric("训练引擎状态", state.upper(), status_data.get("message", ""))
        else:
            st.warning("后端服务返回异常")
    except Exception:
        st.error(f"无法连接后端 API ({API_BASE_URL})，请确认服务已启动")

    st.divider()
    st.markdown("### 线性流程导航")
    st.markdown("""
    1. **Step 1**: 上传 SKU 标准图与冰柜图
    2. **Step 2**: Label Studio 预标注微调
    3. **Step 2.5**: 一键 8:2 拆分构建数据集
    4. **Step 3**: 启动 YOLO26 训练与监控
    5. **Step 4**: 模型热加载与传图画框测试
    """)

st.header("冰柜饮料陈列识别 MLOps 全流程工作台")

tab1, tab2, tab2_5, tab3, tab4 = st.tabs([
    "① Step 1: 上传与预标注预览",
    "② Step 2: Label Studio 人工微调",
    "③ Step 2.5: 8:2 数据集拆分",
    "④ Step 3: YOLO26 训练监控",
    "⑤ Step 4: 在线 API 画框测试"
])

with tab1:
    col_sku, col_cooler = st.columns(2)

    with col_sku:
        st.subheader("1A. 入库饮料 SKU 标准图")
        st.caption("上传单瓶高清标准图，系统将自动提取视觉特征用于自动预标注。")
        sku_name_input = st.text_input("输入 SKU 英文/拼音标识 (将作为类别名)", placeholder="例如: coca_cola_330ml")
        sku_file = st.file_uploader("选择 SKU 标准图", type=["jpg", "jpeg", "png"], key="sku_uploader")

        if st.button("📥 上传并提取 SKU 特征", type="primary", use_container_width=True):
            if not sku_name_input or not sku_file:
                st.warning("请填写 SKU 名称并选择图片！")
            else:
                with st.spinner("正在提取特征并更新特征库..."):
                    files = {"file": (sku_file.name, sku_file.getvalue(), sku_file.type)}
                    r = requests.post(
                        f"{API_BASE_URL}/step1/upload_sku_standard",
                        params={"sku_name": sku_name_input.strip()},
                        files=files
                    )
                    if r.status_code == 200:
                        st.success(r.json()["msg"])
                    else:
                        st.error(f"上传失败: {r.text}")

    with col_cooler:
        st.subheader("1B. 批量上传冰柜实拍图 & 自动预标注")
        st.caption("上传后立即调用零样本检测 + SKU 特征比对，生成预标注框。")
        cooler_files = st.file_uploader(
            "选择冰柜陈列原图 (支持多选)",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            key="cooler_uploader"
        )

        if st.button("🚀 批量上传并执行自动预标注", type="primary", use_container_width=True):
            if not cooler_files:
                st.warning("请先选择至少一张冰柜图片！")
            else:
                progress = st.progress(0)
                last_img, last_anns = None, []
                for idx, c_file in enumerate(cooler_files):
                    files = {"file": (c_file.name, c_file.getvalue(), c_file.type)}
                    r = requests.post(f"{API_BASE_URL}/step1/upload_cooler_image", files=files)
                    if r.status_code == 200:
                        data = r.json()
                        last_img = Image.open(BytesIO(c_file.getvalue())).convert("RGB")
                        last_anns = data.get("auto_annotations", [])
                    progress.progress((idx + 1) / len(cooler_files))

                st.success(f"成功上传并预标注 {len(cooler_files)} 张冰柜图片！")
                if last_img is not None:
                    st.markdown(f"**最新一张预标注效果预览（共自动框出 `{len(last_anns)}` 瓶饮料）：**")
                    preview_img = draw_ls_pre_annotations(last_img, last_anns)
                    st.image(preview_img, use_container_width=True)

with tab2:
    st.subheader("在 Label Studio 中微调预标注框")
    col_ls1, col_ls2 = st.columns([2, 1])
    with col_ls1:
        st.markdown(f"""
        #### 操作指引：
        1. 点击右侧按钮打开 **Label Studio**（默认地址：`{LABEL_STUDIO_EXTERNAL_URL}`）。
        2. 在项目设置 **Settings ➔ Machine Learning** 中点击 **Add Model**，填入后端回调地址：
           `http://pipeline-cpu:8000/step2/label_studio_webhook`（GPU 模式则为 `http://pipeline-gpu:8000/step2/label_studio_webhook`）。
        3. 导入 `data/raw_images` 中的图片，人工只需**微调错框或漏框**，每张图确认无误后点击 **Submit / Update**。
        """)
    with col_ls2:
        st.link_button("🔗 打开 Label Studio 标注平台", LABEL_STUDIO_EXTERNAL_URL, type="primary", use_container_width=True)
        st.info("完成人工校验并点击 Submit 后，请直接进入下一个标签页 **Step 2.5**。")

with tab2_5:
    st.subheader("将标注结果按 8:2 自动拆分为 YOLO26 训练集与验证集")
    train_ratio = st.slider("训练集划分比例 (Train Ratio)", min_value=0.5, max_value=0.95, value=0.8, step=0.05)

    mode_col1, mode_col2 = st.columns(2)

    with mode_col1:
        st.markdown("#### 方式 A：上传 Label Studio 导出的 JSON 文件（推荐）")
        st.caption("在 Label Studio 中点击 Export -> JSON，将导出的文件拖入下方：")
        ls_json_file = st.file_uploader("上传 Label Studio JSON 文件", type=["json"])
        if st.button("📂 解析 JSON 并执行 8:2 拆分", type="primary", use_container_width=True):
            if not ls_json_file:
                st.warning("请先选择导出的 JSON 文件！")
            else:
                with st.spinner("正在转换坐标并构建 train/val 目录..."):
                    files = {"file": (ls_json_file.name, ls_json_file.getvalue(), "application/json")}
                    r = requests.post(
                        f"{API_BASE_URL}/step2_5/prepare_dataset_from_json",
                        params={"train_ratio": train_ratio},
                        files=files
                    )
                    if r.status_code == 200:
                        res = r.json()
                        st.success(res["msg"])
                        st.session_state["dataset_summary"] = res["dataset_summary"]
                    else:
                        st.error(f"拆分失败: {r.text}")

    with mode_col2:
        st.markdown("#### 方式 B：通过 API Token 直接从容器拉取同步")
        st.caption("无需手动下载文件，直接从 Label Studio 同步已提交的标注任务：")
        ls_token = st.text_input("Label Studio API Token (Account & Settings 中获取)", type="password")
        ls_proj_id = st.number_input("Project ID", min_value=1, value=1, step=1)
        if st.button("🔄 从 Label Studio 一键拉取并 8:2 拆分", use_container_width=True):
            if not ls_token:
                st.warning("请填写 Label Studio API Token！")
            else:
                with st.spinner("正在从 Label Studio 拉取并转换..."):
                    payload = {
                        "ls_url": LABEL_STUDIO_INTERNAL_URL,
                        "api_token": ls_token,
                        "project_id": int(ls_proj_id),
                        "train_ratio": train_ratio
                    }
                    r = requests.post(f"{API_BASE_URL}/step2_5/sync_from_label_studio", json=payload)
                    if r.status_code == 200:
                        res = r.json()
                        st.success(res["msg"])
                        st.session_state["dataset_summary"] = res["dataset_summary"]
                    else:
                        st.error(f"同步失败: {r.text}")

    if "dataset_summary" in st.session_state:
        st.divider()
        summary = st.session_state["dataset_summary"]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("有效标注总图片数", summary["total_annotated_images"])
        m2.metric("训练集 (Train) 数量", summary["train_count"])
        m3.metric("验证集 (Val) 数量", summary["val_count"])
        m4.metric("识别出的 SKU 类别数", len(summary["class_names"]))
        st.write("**已提取的 SKU 类别列表：**", summary["class_names"])

with tab3:
    st.subheader("YOLO26 模型训练与自动导出 ONNX")
    st.caption("系统将自动读取 Step 2.5 生成的 classes.txt 与数据集，并根据当前 CPU/GPU 模式自动配置冻结层与超参数。")

    c1, c2 = st.columns(2)
    custom_epochs = c1.number_input("自定义训练轮数 Epochs (留 0 则使用当前模式默认值: CPU=35 / GPU=100)", min_value=0, max_value=500, value=0)
    custom_imgsz = c2.selectbox("训练输入分辨率 (imgsz)", options=[0, 640, 960, 1280], index=0, format_func=lambda x: "自动 (CPU=640 / GPU=1280)" if x == 0 else str(x))

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("🔥 一键启动 YOLO26 训练任务", type="primary", use_container_width=True):
            payload = {}
            if custom_epochs > 0:
                payload["epochs"] = int(custom_epochs)
            if custom_imgsz > 0:
                payload["imgsz"] = int(custom_imgsz)
            r = requests.post(f"{API_BASE_URL}/step3/trigger_train", json=payload)
            if r.status_code == 200:
                st.success(r.json()["msg"])
            else:
                st.error(f"启动失败: {r.text}")

    with col_btn2:
        auto_poll = st.checkbox("⏱️ 开启训练状态自动轮询刷新 (每 3 秒)", value=False)

    status_box = st.empty()
    def render_train_status():
        r = requests.get(f"{API_BASE_URL}/step3/train_status")
        if r.status_code == 200:
            st_data = r.json()
            state = st_data.get("state")
            msg = st_data.get("message")
            mode = st_data.get("mode", "").upper()
            if state == "training":
                status_box.warning(f"⏳ **[{mode} 训练进行中]** {msg}")
            elif state == "completed":
                status_box.success(f"✅ **[{mode} 训练已完成]** {msg}")
            elif state == "failed":
                status_box.error(f"❌ **[{mode} 训练异常中断]** {msg}")
            else:
                status_box.info(f"💤 **[{mode} 空闲就绪]** {msg}")
            return state
        return "unknown"

    current_state = render_train_status()
    if auto_poll and current_state == "training":
        time.sleep(3)
        st.rerun()

with tab4:
    st.subheader("生产模型热加载 & 冰柜陈列识别 API 实测")

    top_col1, top_col2 = st.columns([1, 2])
    with top_col1:
        if st.button("♻️ 一键热加载最新训练的 ONNX 模型", use_container_width=True):
            r = requests.post(f"{API_BASE_URL}/step4/reload_production_model")
            if r.status_code == 200:
                st.success(r.json()["msg"])
            else:
                st.error(f"热加载失败: {r.text}")
    with top_col2:
        conf_slider = st.slider("检测置信度阈值 (Confidence Threshold)", min_value=0.1, max_value=0.95, value=0.4, step=0.05)

    st.divider()
    test_file = st.file_uploader("上传一张测试冰柜照片验证检测效果", type=["jpg", "jpeg", "png"], key="test_uploader")

    if test_file is not None:
        if st.button("🔍 调用 /step4/detect_cooler 执行检测", type="primary", use_container_width=True):
            with st.spinner("正在执行端到端无 NMS 推理..."):
                t0 = time.perf_counter()
                files = {"file": (test_file.name, test_file.getvalue(), test_file.type)}
                r = requests.post(
                    f"{API_BASE_URL}/step4/detect_cooler",
                    params={"conf": conf_slider},
                    files=files
                )
                elapsed_ms = (time.perf_counter() - t0) * 1000

                if r.status_code == 200:
                    res = r.json()
                    orig_img = Image.open(BytesIO(test_file.getvalue())).convert("RGB")
                    boxed_img = draw_yolo_detections(orig_img, res["detections"])

                    img_col, stat_col = st.columns([3, 2])
                    with img_col:
                        st.markdown(f"**检测结果可视化（耗时 `{elapsed_ms:.1f} ms` | 模式 `{res['compute_mode'].upper()}`）**")
                        st.image(boxed_img, use_container_width=True)

                    with stat_col:
                        st.markdown("**📊 冰柜排面陈列统计 (Facing Summary)**")
                        total_bottles = res["total_bottles"]
                        st.metric("检出饮料总瓶数 (Total Facings)", total_bottles)

                        sku_summary = res.get("sku_facing_summary", {})
                        if sku_summary:
                            table_rows = []
                            for sku, count in sorted(sku_summary.items(), key=lambda x: x[1], reverse=True):
                                share = f"{(count / total_bottles) * 100:.1f}%" if total_bottles > 0 else "0%"
                                table_rows.append({
                                    "SKU 名称": sku,
                                    "排面瓶数": count,
                                    "冰柜排面占比 (SOS)": share
                                })
                            st.dataframe(table_rows, use_container_width=True, hide_index=True)
                        else:
                            st.info("当前阈值下未检出任何饮料瓶，请尝试调低置信度阈值或先完成模型训练。")

                        with st.expander("查看 API 原始 JSON 响应报文"):
                            st.json(res)
                else:
                    st.error(f"检测接口调用失败: {r.text}")
