# 标准打标签流程 / Standard Labeling Workflow

本文档提供 `DeepLearning` 项目的**中英文结合**标注流程，按 QGIS 实际按钮名称（English UI）编写，方便直接操作。

---

## 1) 标签配置 / Label Config

当前项目标签配置（见 `configs/pipeline.yaml`）：

- 正样本文件 / Positive label file: `labels/label_points.gpkg`
- 负样本文件 / Negative label file: `labels/label_points_0.gpkg`
- 图层名 / Layer name: `labels`
- 字段名 / Field name: `class`
- 森林 / Forest: `1`（建议写在正样本文件）
- 非森林 / Non-forest: `0`（建议写在负样本文件）

> 标准要求：实际标注时只使用 `1` 和 `0`。

---

## 2) 数据含义 / What is `input_prepared.tif`

- `work/input_prepared.tif` 是**预处理后的底图** (preprocessed base raster)，不是原始影像的直接拷贝。
- 它用于后续检查、样本切分、训练与融合流程。

如果文件不存在，先运行：

```bash
python -m forestseg.cli prepare-input --config configs/pipeline.yaml
```

---

## 3) QGIS 打开数据 / Open Data in QGIS

### 3.1 打开底图 / Open base raster

在 QGIS：

1. `Layer` → `Add Layer` → `Add Raster Layer...`
2. 选择 `work/input_prepared.tif`
3. 点击 `Add`

### 3.2 打开正样本图层 / Open positive label layer

1. `Layer` → `Add Layer` → `Add Vector Layer...`
2. 选择 `labels/label_points.gpkg`
3. 选择图层 `labels`
4. 点击 `Add`

### 3.3 打开负样本图层 / Open negative label layer

1. `Layer` → `Add Layer` → `Add Vector Layer...`
2. 选择 `labels/label_points_0.gpkg`
3. 选择图层 `labels`
4. 点击 `Add`

---

## 4) 开始标注 / Start Labeling

### 4.1 开启编辑 / Toggle editing

1. 在 `Layers Panel` 选中 `labels`
2. 点击工具栏 `Toggle Editing`（铅笔图标）

### 4.2 添加点 / Add point feature

1. 点击 `Add Point Feature`
2. 在影像上点击目标位置
3. 在属性窗口输入：
   - `class = 1`（森林 / Forest）
   - `class = 0`（非森林 / Non-forest）
4. 点击 `OK`

### 4.3 删错标签 / Delete wrong labels

如果点打错位置或 `class` 填错，可按下面操作：

#### 方法 A：删除整条错误点要素 / Delete wrong point feature

1. 确保图层处于编辑状态（`Toggle Editing` 已开启）
2. 点击 `Select Features by area or single click`
3. 单击选中错误点
4. 点击 `Delete Selected`（或键盘 `Delete`）
5. 点击 `Save Layer Edits`

#### 方法 B：只改属性不删点 / Edit attribute only

1. 保持编辑状态
2. 右键错误点 → `Open Attribute Table`（或使用 `Identify Features`）
3. 把 `class` 改为正确值（`1` 或 `0`）
4. 点击 `Save Layer Edits`

#### 快速回退 / Quick undo

- 刚操作错可以直接 `Undo`（`Ctrl+Z`）
- 结束编辑时点击 `Toggle Editing`，会弹窗确认是否保存本次修改

---

## 5) 标注标准 / Labeling Rules

### 5.1 森林点 (class = 1) / Forest points

只在你非常确定为森林的位置打点：

- 树冠连续 (continuous canopy)
- 纹理稳定 (stable forest texture)
- 尽量在区域内部中心 (interior center)

不要打在：

- 边界混合区 (boundary/mixed zone)
- 阴影重、模糊区 (heavy shadow / blurry area)

### 5.2 非森林点 (class = 0) / Non-forest points

包括但不限于：

- 建筑 (buildings)
- 道路 (roads)
- 水体 (water)
- 裸地 (bare land)
- 农田/草地 (farmland/grassland)

同样避免边界混合区。

---

## 6) 字段规则 / Field Rules (Must Follow)

`class` 字段只允许：

- `1`
- `0`

禁止：

- 空值 / empty
- `nan`, `inf`
- 文本（如 `forest`, `non_forest`）

---

## 7) 推荐数量 / Recommended Sample Size

### 快速试标 / Pilot round

- Forest: 20–30 points
- Non-forest: 20–30 points

### 正式训练前 / Before formal training

- Forest: 50–100+ points
- Non-forest: 50–100+ points
- 两类尽量平衡 / keep classes balanced

---

## 8) 质检命令 / QA Commands

### 8.0 批量修正 class（推荐先执行）/ Batch-fix class (recommended first)

当你已经完成打点，但属性里 `class` 可能有空值或填错时，先运行：

```bash
python scripts/fix_label_classes.py
```

脚本会强制写入：

- `label_points.gpkg` 全部 `class=1`
- `label_points_0.gpkg` 全部 `class=0`

### 8.1 标签检查 / Check labels

> 若你看到报错 `ModuleNotFoundError: No module named 'forestseg'`，先设置 `PYTHONPATH` 指向 `src`，再执行检查命令。
>
> Windows CMD：
>
> ```bat
> set PYTHONPATH=E:\CORONA\DeepLearning\src
> ```
>
> PowerShell：
>
> ```powershell
> $env:PYTHONPATH = "E:/CORONA/DeepLearning/src"
> ```
>
> Git Bash / bash：
>
> ```bash
> export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}"
> ```

```bash
python -m forestseg.cli check-label-points --config configs/pipeline.yaml
```

检查项包括：

- 文件/图层是否可读 (readability)
- 几何是否为 Point (geometry type)
- `class` 字段完整性 (field completeness)
- 标签值合法性 (value validity)
- 训练/验证切分可行性 (split feasibility)

输出报告：

- `work/label_check_report.json`

### 8.2 生成训练/验证样本 / Prepare train/val samples

```bash
python -m forestseg.cli prepare-label-points --config configs/pipeline.yaml
```

当前切分参数：

- `split_ratio: 0.7`
- `grid_size: 1000.0`

---

## 9) 推荐完整流程 / End-to-end Sequence

1. 生成预处理底图 / Prepare raster

```bash
python -m forestseg.cli prepare-input --config configs/pipeline.yaml
```

2. QGIS 打点标注 / Label points in QGIS
3. 运行标签检查 / Run label check

```bash
python -m forestseg.cli check-label-points --config configs/pipeline.yaml
```

4. 准备样本 / Prepare samples

```bash
python -m forestseg.cli prepare-label-points --config configs/pipeline.yaml
```

5. 跑总流程 / Run pipeline

```bash
python -m forestseg.cli preflight-check --config configs/pipeline.yaml
python -m forestseg.cli run-all --config configs/pipeline.yaml
```

---

## 10) 快速口令 / One-line Rule

- `1 = Forest`
- `0 = Non-forest`

---

## 11) 代码启动完整步骤（从 0 到运行）/ Complete Startup Steps

> 下面是“可直接复制执行”的完整启动流程：从环境准备到训练与导出。
> 所有命令都在项目根目录 `E:/CORONA/DeepLearning` 执行。

### 11.1 进入项目目录 / Enter project root

```bash
cd E:/CORONA/DeepLearning
```

### 11.2 创建并激活虚拟环境 / Create and activate virtual environment

创建（首次一次即可）：

```bash
python -m venv .venv
```

激活（Windows PowerShell）：

```bash
.\.venv\Scripts\Activate.ps1
```

激活（Git Bash / bash）：

```bash
source .venv/Scripts/activate
```

### 11.3 安装依赖 / Install dependencies

先安装基础依赖：

```bash
pip install -r requirements/base.txt
```

如需 CUDA 12.1（GPU）版本的 PyTorch，再执行：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

> 若仅 CPU 运行，可跳过上面的 CUDA 轮子安装步骤。

### 11.4 确认配置路径 / Confirm config path

默认配置文件：

- `configs/pipeline.yaml`

当前关键路径（已在配置中给出）：

- `project_root: E:/CORONA`
- `work_dir: E:/CORONA/DeepLearning/work`
- `output_dir: E:/CORONA/input/generated/export`

### 11.5 先做预检查 / Run preflight first

```bash
python -m forestseg.cli preflight-check --config configs/pipeline.yaml
```

若失败，优先修复：

- 标签文件路径
- `class` 字段值（必须是 `0/1`）
- 输入底图/中间产物路径

### 11.6 准备输入与样本 / Prepare input and samples

```bash
python -m forestseg.cli prepare-input --config configs/pipeline.yaml
python -m forestseg.cli prepare-label-points --config configs/pipeline.yaml
```

### 11.7 构建特征并训练模型 / Build features and train model

```bash
python -m forestseg.cli build-feature-stack --config configs/pipeline.yaml
python -m forestseg.cli train-or-load-dl --config configs/pipeline.yaml --force-train
```

### 11.8 执行闭环融合 / Run closed-loop fusion

```bash
python -m forestseg.cli run-closed-loop --config configs/pipeline.yaml
```

### 11.9 后处理并导出 / Postprocess and export

```bash
python -m forestseg.cli postprocess-export --config configs/pipeline.yaml
```

### 11.10 关键输出位置 / Key outputs

- 中间产物目录：`E:/CORONA/DeepLearning/work`
- 导出结果目录：`E:/CORONA/input/generated/export`

---

## 12) 已打完标签后：最短启动顺序 / Minimal Commands After Labeling

你现在已经完成打标签，如果环境已装好，可直接按最短顺序启动：

```bash
python -m forestseg.cli preflight-check --config configs/pipeline.yaml
python -m forestseg.cli prepare-label-points --config configs/pipeline.yaml
python -m forestseg.cli build-feature-stack --config configs/pipeline.yaml
python -m forestseg.cli train-or-load-dl --config configs/pipeline.yaml --force-train
python -m forestseg.cli run-closed-loop --config configs/pipeline.yaml
python -m forestseg.cli postprocess-export --config configs/pipeline.yaml
```

---

## 13) 一键全流程（可选）/ One-command pipeline (optional)

如果你希望从预检查到导出一次跑完：
python scripts/fix_label_classes.py

set PYTHONPATH=E:\CORONA\DeepLearning\src
```bash
python -m forestseg.cli run-all --config configs/pipeline.yaml --force-train
```

> 备注：`run-all` 内部会输出 `[progress] ...` 阶段日志，便于你看到当前执行到哪一步。
