# dp-1 / `forestseg`

> **KH-4 卫星森林监督分割流水线 — 重构与标准化版本。**
> 英文版：see [README.md](README.md)

`dp-1` 是 [`jyhuang201900/dp`](https://github.com/jyhuang201900/dp)
的标准化重构版本：算法与配置完全兼容，但增加了规范的 Python 打包、lint /
typing / pre-commit 工具链、CI、跨平台配置示例与结构化文档。

整个流水线把原始 KH-4 影像 + 少量人工点标签 + 一份 YAML 配置，转化为干净的
森林 / 非森林二值掩膜（`.tif`）与矢量斑块（`.gpkg`）。

---

## 流水线总览

```
原始影像 + scene.sh + 点标签 + pipeline.yaml
    │
    ▼
[preprocess]           归一化、nodata、分位裁剪
    │
    ▼
[光谱先验] + [纹理先验]   Otsu + 分位 + 局部统计 / LBP
    │
    ▼
[标签 QA]              几何 / CRS / 类别平衡检查
    │
    ▼
[网格切分]              空间分块 train/val，控制泄漏
    │
    ▼
[特征堆栈]              多波段对齐 TIFF（含可选 DEM）
    │
    ▼
[DL 训练 + 推理]        DeepLabV3+（默认 ResNet-34），MC dropout
    │
    ▼
[融合]                 DL ⊕ 光谱 ⊕ 纹理，网格 / RL 搜索
    │
    ▼
[后处理]                形态学、小斑块清除、阴影惩罚
    │
    ▼
[导出]                 最终掩膜 + 矢量斑块
```

设计细节：[`docs/pipeline.md`](docs/pipeline.md)、[`docs/architecture.md`](docs/architecture.md)；
标注 SOP：[`docs/labeling_workflow.md`](docs/labeling_workflow.md)。

---

## 依赖

- Python **3.10 – 3.12**
- C 编译器 + GDAL 系统库（Linux/macOS 上 `rasterio` / `fiona` 提供二进制
  wheel；Windows 推荐 conda 或 `osgeo4w`）
- 可选：CUDA ≥ 12.1 的 NVIDIA GPU 用于训练

---

## 安装

### CPU 快速开始

```bash
git clone https://github.com/jyhuang201900/dp-1.git
cd dp-1
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -e ".[torch,rl,test]" \
    --extra-index-url https://download.pytorch.org/whl/cpu
```

### GPU（CUDA 12.1）

```bash
pip install -e ".[torch,rl]" \
    --extra-index-url https://download.pytorch.org/whl/cu121
```

### 开发者（含 lint / format / 类型检查 / pre-commit）

```bash
pip install -e ".[dev,torch,rl]" --extra-index-url https://download.pytorch.org/whl/cpu
pre-commit install
```

`Makefile` 已经封装了常用命令：

```bash
make help          # 查看目标
make install-dev   # editable + dev + test + torch (CPU)
make lint          # ruff check
make format        # ruff fix + format
make typecheck     # mypy
make test          # pytest
make ci            # lint + format-check + test （与 GitHub CI 一致）
```

---

## 配置

流水线通过 [`configs/`](configs/) 下的三份 YAML 驱动：

| 文件 | 说明 |
|---|---|
| [`pipeline.example.yaml`](configs/pipeline.example.yaml) | 跨平台示例（相对路径） |
| [`pipeline.windows.yaml`](configs/pipeline.windows.yaml) | 原 `E:/CORONA/...` Windows 布局 |
| [`model.yaml`](configs/model.yaml) | DeepLabV3+ 结构与训练 / 推理参数 |
| [`rl.yaml`](configs/rl.yaml) | 融合参数 RL / 网格搜索阶段配置 |

字段含义已写在 YAML 行内注释中。建议把 `pipeline.example.yaml` 复制为
`configs/pipeline.local.yaml`，按你的路径修改后再运行。

---

## 运行流水线

CLI 提供每个阶段的子命令，以及一个 `run-all`：

```bash
# 分阶段运行
forestseg prepare-input        --config configs/pipeline.example.yaml
forestseg build-spec-tex       --config configs/pipeline.example.yaml
forestseg check-label-points   --config configs/pipeline.example.yaml
forestseg prepare-label-points --config configs/pipeline.example.yaml
forestseg build-feature-stack  --config configs/pipeline.example.yaml
forestseg train-or-load-dl     --config configs/pipeline.example.yaml
forestseg run-rl-fusion        --config configs/pipeline.example.yaml
forestseg postprocess-export   --config configs/pipeline.example.yaml

# RL 闭环（多轮反馈）
forestseg run-closed-loop      --config configs/pipeline.example.yaml

# 一键端到端
forestseg run-all              --config configs/pipeline.example.yaml
```

也可以用 `python -m forestseg`：

```bash
python -m forestseg run-all --config configs/pipeline.example.yaml
```

`scripts/` 目录提供了等价的 bash 包装脚本，方便集成到现有作业系统。

---

## 测试

```bash
make test
# 或
pytest -ra
```

与上游 `dp` 完全对齐：**338 通过，3 个已知失败**（`test_cli_preflight.py`
里三个 `test_cmd_run_rl_fusion_*` 依赖真实栅格 fixture，在上游同样失败，
不属于本次重构引入的回归）。

GitHub Actions CI 在 Python 3.10 / 3.11 / 3.12 上同时跑 lint + format-check
+ pytest。

---

## 贡献

详见 [CONTRIBUTING.md](CONTRIBUTING.md)。简要：

1. fork 后从 `main` 拉分支。
2. `make install-dev && pre-commit install`。
3. 提 PR 之前 `make ci` 必须全绿。
4. 行长 120、风格遵循 ruff 默认；行为变更需要带上测试。

---

## License

[MIT](LICENSE) © 2025 jyhuang201900。

本仓库是 [`jyhuang201900/dp`](https://github.com/jyhuang201900/dp) 的
clean-room 重构，所有算法、配置与测试均保留原语义。
