# illust2psd

> **开发状态：实验性 / Alpha**
> 本项目仍处于早期开发阶段，接口和输出格式可能随时变更。
> 欢迎试用和反馈，但不建议用于生产环境。

将动漫/游戏角色立绘自动转换为 Live2D Cubism Editor 可用的多图层 PSD 文件。

输入一张角色立绘 PNG/JPG，输出一个分好图层的 PSD，可以直接导入 Cubism 开始建模。

## 功能

- 自动去背景（ISNet 动漫专用模型）
- 身体部位语义分割（SAM2 / 启发式）
- 面部精细分割（眼睛、眉毛、鼻子、嘴巴独立图层）
- 遮挡区域自动修复（OpenCV inpainting）
- 输出 Cubism 兼容的 PSD（RGB 8-bit，图层命名规范）
- 支持 Apple Silicon MPS 加速

## 输出图层结构

从底到顶排列（对应 PSD 中从下到上）：

```
Hair_Back       后发
Body            身体
Arm_L_Back      左上臂
Arm_R_Back      右上臂
Leg_L           左腿
Leg_R           右腿
Neck            脖子
Face            脸部皮肤
Ear_L / Ear_R   耳朵
Eye_L / Eye_R   眼睛
Nose            鼻子
Mouth           嘴巴
Brow_L / Brow_R 眉毛
Arm_L_Front     左前臂/手
Arm_R_Front     右前臂/手
Hair_Front      前发/刘海
Accessory       饰品（如有）
```

## 安装

### 环境要求

- Python >= 3.10
- macOS (Apple Silicon) / Linux / Windows
- 推荐 16GB+ 内存（使用 SAM2 时）

### 安装步骤

```bash
# 创建虚拟环境
python3 -m venv .venv
source .venv/bin/activate

# 安装项目（基础依赖）
pip install -e .

# 安装 GPU/MPS 推理依赖（推荐）
pip install torch torchvision onnxruntime sam2 mediapipe

# 预下载模型权重
illust2psd download-models
```

### 模型说明

| 模型 | 大小 | 用途 | 必需 |
|------|------|------|------|
| ISNet (isnet_anime) | 176 MB | 动漫前景提取 | 推荐 |
| SAM2.1 Hiera Large | 898 MB | 身体部位分割 + 面部分割 | 可选（精细模式） |
| MediaPipe Pose | 约 30 MB | 姿态关键点检测 | 自动下载 |

模型会缓存到 `~/.cache/illust2psd/models/`，首次使用时自动下载。

## 使用方法

### 基本用法

```bash
# 最简单的用法 — 输入图片，输出同名 PSD
illust2psd convert character.png

# 指定输出路径
illust2psd convert character.png -o output.psd
```

### 推荐用法（Apple Silicon）

```bash
# 使用 SAM2 做精细分割（效果最好，约 17 秒）
illust2psd convert character.png -o output.psd \
  --device mps \
  --segmentation-backend sam2
```

### 快速模式（纯 CPU，不依赖 GPU 模型）

```bash
# 启发式分割，约 1 秒完成
illust2psd convert character.png \
  --device cpu \
  --segmentation-backend heuristic \
  --foreground-model grabcut \
  --no-inpaint
```

### 批量处理

```bash
# 处理文件夹下所有图片
illust2psd batch input_images/ -o psd_output/

# 带 JSON 报告
illust2psd batch input_images/ -o psd_output/ --report report.json

# 用 SAM2 批量精细处理
illust2psd batch input_images/ -o psd_output/ \
  --device mps --segmentation-backend sam2
```

### 调试 — 导出中间结果

```bash
# 保存分割 mask 和各图层 PNG
illust2psd convert character.png \
  --dump-masks debug/masks/ \
  --dump-layers debug/layers/ \
  -v
```

### 质量评估

```bash
# 批量评估并输出 PSNR/SSIM 报告
python scripts/evaluate.py test_images/ \
  -o eval_output/ \
  --device mps \
  --segmentation-backend sam2
```

## 全部 CLI 选项

```
illust2psd convert [OPTIONS] INPUT_IMAGE

选项:
  -o, --output TEXT                输出 PSD 路径（默认：输入文件同名.psd）
  --max-size INTEGER               处理时最大边长（默认 2048）
  --segmentation-backend [sam2|heuristic]  分割后端（默认 heuristic）
  --foreground-model [isnet|rembg|grabcut] 前景提取模型（默认 isnet）
  --inpaint-backend [lama|opencv|none]     修复后端（默认 opencv）
  --device [cuda|cpu|mps]          计算设备（默认 mps）
  --no-inpaint                     跳过修复（更快，但图层可能有洞）
  --dump-masks PATH                保存中间 mask 到目录
  --dump-layers PATH               保存各图层 PNG 到目录
  -v, --verbose                    详细日志
```

```
illust2psd batch [OPTIONS] INPUT_DIR

选项:
  -o, --output-dir TEXT            输出目录（默认 input_dir/psd_output）
  --report PATH                    保存 JSON 报告
  （其余选项同 convert）
```

```
illust2psd download-models         预下载所有模型权重
illust2psd list-models             查看模型缓存状态
```

## 处理流程

```
输入图片 (PNG/JPG/WEBP)
  │
  ├─ S1  预处理：加载、RGBA 转换、EXIF 旋转、尺寸校验
  ├─ S2  前景提取：ISNet / rembg / GrabCut（三级降级）
  ├─ S3  姿态估计：MediaPipe / 启发式身体比例
  ├─ S4  语义分割：SAM2 点提示 / 启发式区域（核心步骤）
  ├─ S5  面部精细分割：SAM2 子提示 / 椭圆估计
  ├─ S6  遮挡修复：OpenCV Telea inpainting
  ├─ S7  图层组装：裁剪、偏移、z-order 排序、PSNR/SSIM 验证
  └─ S8  PSD 导出：pytoshop 写入 Cubism 兼容格式
  │
  输出 .psd
```

## 输出 PSD 规格（符合 Cubism 要求）

- 格式：PSD（非 PSB）
- 色彩模式：RGB 8-bit
- 色彩配置：sRGB
- 每个部位 = 一个图层，Normal 混合模式，100% 不透明度
- 无图层蒙版、无调整图层、无智能对象
- 无重复图层名
- 画布尺寸 = 原始图片尺寸
- 底部包含一个隐藏的 Reference 图层（原图）

## MCP Server

可以作为 MCP 工具集成到 Claude Code 或 VS Code 中：

```bash
python -m illust2psd.mcp_server.server
```

提供的工具：
- `convert_to_psd` — 转换单张图片
- `preview_segmentation` — 预览分割结果
- `batch_convert` — 批量转换
- `list_models` — 查看模型状态

## 项目结构

```
illust2psd/
├── cli.py              CLI 入口
├── pipeline.py         流程编排
├── config.py           配置和图层定义
├── steps/              8 个处理步骤
│   ├── s1_preprocess.py
│   ├── s2_foreground.py
│   ├── s3_pose.py
│   ├── s4_segment.py
│   ├── s5_face.py
│   ├── s6_inpaint.py
│   ├── s7_compose.py
│   └── s8_export.py
├── models/             模型加载和缓存
│   ├── model_manager.py
│   ├── seg_model.py
│   ├── pose_model.py
│   └── inpaint_model.py
├── utils/              工具函数
│   ├── image_utils.py
│   ├── mask_utils.py
│   ├── psd_utils.py
│   └── download.py
└── mcp_server/         MCP 接口
    └── server.py
```

## 已知限制

- 目前主要支持正面/微侧面的站立姿势，坐姿等非标准姿势效果较差
- MediaPipe 对动漫人物的关键点检测效果有限，会自动 fallback 到启发式估计
- SAM2 对动漫头发细节的分割不够精细，头发区域通过减法策略处理
- 修复目前使用 OpenCV Telea 算法，对大面积遮挡效果一般
- 配饰（帽子、眼镜等）统一归为一个 Accessory 图层

## 开发

```bash
# 安装开发依赖
pip install -e ".[dev]"

# 运行测试
pytest tests/ -v

# 代码格式化
black illust2psd/
ruff check illust2psd/
```

## 参考

- [Live2D Cubism PSD 要求](https://docs.live2d.com/en/cubism-editor-manual/precautions-for-psd-data/)
- [Live2D 素材分离指南](https://docs.live2d.com/en/cubism-editor-tutorials/psd/)
- [anime-segmentation (SkyTNT)](https://github.com/SkyTNT/anime-segmentation)
- [SAM2 (Meta)](https://github.com/facebookresearch/segment-anything-2)
- [pytoshop](https://github.com/mdboom/pytoshop)
