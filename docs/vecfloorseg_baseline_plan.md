# VecFloorSeg 接入方案（空间轮廓提取 Baseline）

> 状态：**已接入并已拿到实测数字** —— submodule 已就位（真实 git checkout，`2491d2d`）；
> 环境已就绪；适配层已实现（`src/spatial/contours/vecfloorseg.py` +
> `src/spatial/vecfloorseg/`，设计见 `docs/vecfloorseg_adapter_design.md`）；已训练出
> `best1.ckpt` 并跑完 39 张图的基准矩阵。
>
> ⚠️ **但实测结果很差，根因是权重欠训练**（`best` = epoch 1，每张图只出 ~1 条轮廓 →
> 覆盖率 0.013），见 §10.8。**这是上游模型的问题，不是适配器/文档的问题。**
> 本文档记录调研结论、网络/代理实测、Windows 可行性评估、训练方案与接入路线。

## 1. 目标

为「空间轮廓提取」增加第三个可插拔 baseline，与现有的 **CDT**、**RGP** 并列：

- 论文/项目：*VectorFloorSeg: Two-Stream Graph Attention Network for Vectorized Roughcast Floorplan Segmentation*
- 仓库：<https://github.com/DrZiji/VecFloorSeg>

## 2. 阻塞项与当前状态

| 阻塞项 | 证据 / 说明 |
|---|---|
| 直连不可达，但**本地代理可绕过** | 直连 `github.com:443` 超时（DNS 也有污染）；发现本机代理 `http://127.0.0.1:7897` 可用：经代理 `github.com` 200、`drive.google.com/drive/folders/...` 200 → 已通过代理完成**真正的 submodule clone**（`third_party/VecFloorSeg` @ `2491d2d`，`heads/master`） |
| 预训练权重在 Google Drive | README 指引从 Google Drive 下载处理后数据与权重，同样需要外网 |
| 重量级依赖 | 需要 PyTorch + PyG 2.0.4，且仓库**内联了改造版 `torch_geometric` / `graphgym`**，需用它的副本替换 PyG 对应目录；与本仓库 `cadruler` venv 不是一套环境 |
| 输入输出范式不同 | 它是「栅格图像 + 线框图 + 图数据」的学习型分割，输出**图节点级分割**，并非直接给房间多边形 |

## 3. 接口差距分析

### 本仓库侧（需要对接的契约）

- 抽象接口：`src/spatial/contracts.py` → `ISpatialContourExtractor`
  - 必需属性：`name`、`visualization_suffix`
  - 必须实现：`extract(*, walls, doors, windows, texts, components, context) -> list[SpatialContour]`
  - 可选：`get_visualization_data()`
- 输入（由 `src/topology/builder.py` 调用时传入）：
  - `walls`：清洗后的墙体 `LineString` / `MultiLineString`（Shapely）
  - `doors` / `windows`：门窗补片 `Polygon`
  - `texts`：`TextAnnotation(raw_text, point)`
  - `components`：`SpatialComponent(uid, category, specific_type)`
- 输出：`list[SpatialContour]`，坐标单位为**图纸单位（毫米）**
- 注册：`CONTOUR_EXTRACTORS.register("VecFloorSeg", aliases=(...))`（见 `src/spatial/registry.py`、`src/spatial/contours/cdt.py` / `rgp.py` 的写法）
- 工厂/配置：`src/spatial/factory.py` → `create_contour_extractor()`；配置项 `spatial.contour.algorithm`（`src/config/settings.yaml`）
- 可视化：`src/spatial/visualization.py` → `plot_floor_plan()`，产物名 `<图纸名>_<visualization_suffix>.png`

### VecFloorSeg 侧

- 输入：roughcast SVG → **栅格化图像** + **线框（wireframe）** + mmseg 格式标注 / PyG 图数据（`DataPreparation/`、`Replace_with_CubiCasa/`）
- 推理命令（README）：
  ```bash
  python graphgym/main.py --cfg graphgym/configs/CUBI.yaml --eval train.epoch_resume 1 \
      train.ckpt_prefix best val.extra_infos True seed 0
  ```
- 输出：图节点级分割（墙体/房间/图标等类别），需**后处理**才能得到矢量房间多边形
- 依赖准备：`models/resnet101-torch.pth`（backbone）、预训练权重、替换 PyG 的 `graphgym`/`torch_geometric`

## 4. 后续接入路线（建议分阶段）

- **Phase 0 · 获取仓库**（需联网）
  ```bash
  git submodule add https://github.com/DrZiji/VecFloorSeg.git third_party/VecFloorSeg
  git submodule update --init --recursive
  ```
- **Phase 1 · 独立运行环境**：为其单独建 conda/venv（torch + PyG 2.0.4 + 其内联 `torch_geometric`/`graphgym`），**不要**污染 `cadruler`。
- **Phase 2 · 输入桥接**：本仓库图纸 → roughcast SVG → 栅格图 + 线框 + 图数据（复用其 `DataPreparation/` 流程或等价实现）。
- **Phase 3 · 推理调用**：以**子进程 + 临时目录交换文件**方式调用（推荐），保证依赖隔离；本仓库不 `import torch`。
- **Phase 4 · 输出适配**：节点级分割 → 房间多边形 → `SpatialContour`，需处理：
  - 像素 ↔ 毫米的比例与原点对齐（栅格化时的缩放矩阵需可逆回算）
  - 坐标系方向（图像 y 轴向下 vs 图纸 y 轴向上）
  - 轮廓简化/闭合与 `is_exterior` 之类的过滤
- **Phase 5 · 注册与接线**：新增 `src/spatial/contours/vecfloorseg.py`，注册 `VecFloorSeg`，配置 `spatial.contour.algorithm: VecFloorSeg`，`visualization_suffix = "vecfloorseg"`。
- **Phase 6 · 评估**：在 `exp_analysis.html` 的轮廓展示与 `overall_report.html`
  的端到端矩阵（`e2e_VecFloorSeg_*`）上与 CDT/RGP 横向对比。

## 5. 待确认问题

1. VecFloorSeg 的 `--eval` 输出具体落盘格式与后处理入口（需读其 `eval` 后处理代码）。
2. 是否支持**单文件**推理（现有流程面向数据集批处理）？
3. 栅格分辨率与坐标回算方案（决定轮廓精度与单位换算）。
4. 权重与数据的可获取渠道（Google Drive 是否可替代为内网镜像）。
5. 许可证与引用要求。

## 6. 参考位置

- `src/spatial/contracts.py` — `ISpatialContourExtractor`
- `src/spatial/registry.py` — `CONTOUR_EXTRACTORS`
- `src/spatial/contours/cdt.py`、`rgp.py` — 现有实现范例
- `src/spatial/factory.py` — `create_contour_extractor()`
- `src/topology/builder.py` — 调用 `extract(...)` 的位置
- `src/spatial/visualization.py` — `plot_floor_plan()`
- `README.md` — 「Pluggable Spatial Algorithms」章节

## 7. Windows 可行性评估（实测）

> 结论：**安装层面可行**（所需 Windows 预编译轮子齐全）；主要风险在 **Ada 架构(sm_89) vs CUDA 11.7** 与 **cairosvg**。

### 7.1 实测证据

| 项目 | 结果 |
|---|---|
| 源站可达（TCP 443） | `pypi.org` / `files.pythonhosted.org` / `download.pytorch.org` / `repo.anaconda.com` / `conda.anaconda.org` / `data.pyg.org` 均可达 |
| PyTorch 轮子 | `torch-1.13.0+cu117-cp38-cp38-win_amd64.whl` 存在 |
| PyG 扩展轮子 | `torch_scatter-2.1.1+pt113cu117-cp38-cp38-win_amd64.whl`、`torch_sparse-0.6.17+pt113cu117-cp38-cp38-win_amd64.whl`、`torch_cluster-1.6.1+pt113cu117-cp38-cp38-win_amd64.whl`、`torch_spline_conv-1.2.2+pt113cu117-cp38-cp38-win_amd64.whl` 均存在 |
| CPU 回退轮子 | 同名 `+pt113cpu` 的 win_amd64 cp38 轮子也存在 |
| 本机 GPU | NVIDIA GeForce RTX 4070 Ti（Ada, **sm_89**），驱动 591.86，12 GB |
| 本机 Python | 3.13.3（**不能**跑 torch 1.13，需另装 3.8） |
| conda | 不在 PATH（需安装 Miniconda/Anaconda） |
| Unix 专用 API | 未发现 `fcntl` / `os.fork` / `signal.SIGKILL` 等；`device.py` 调 `nvidia-smi`（Windows 可用） |
| mmseg / mmcv | 代码中**未 import**（仅 README 提到）→ 非运行必需 |

### 7.2 建议环境配方（conda 环境名：vecfloorseg）

**必须新建独立 conda 环境，不要复用 CAD2Graph 的 `cadruler` venv**：

- `cadruler` 是 **Python 3.13**，而 torch 1.13 只有 cp38/cp39/cp310 轮子（本项目固定 **py3.8.15**）；
- VecFloorSeg 要求 **numpy<1.24**，与 CAD2Graph 侧依赖冲突；
- torch 体积大，且直接用 `PYTHONPATH` 跑它自带的 vendored `torch_geometric`/`graphgym`，隔离最省事。

```bash
conda create -n vecfloorseg python=3.8 -y
conda activate vecfloorseg

# 代理（本机 Clash 端口）——PowerShell 语法（注意：不是 cmd 的 `set`）
$env:HTTP_PROXY  = "http://127.0.0.1:7897"
$env:HTTPS_PROXY = "http://127.0.0.1:7897"
# 也可一次性写入 pip 配置： pip config set global.proxy http://127.0.0.1:7897
# 注：pypi / download.pytorch.org / data.pyg.org 实测直连可达，代理非必需

# 1) PyTorch 1.13.0 —— 建议先装 CPU 版把链路跑通，再按需换 CUDA
pip install torch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 --index-url https://download.pytorch.org/whl/cpu
#   CUDA 版（可选，注意下方 sm_89 风险）：
# pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0 --index-url https://download.pytorch.org/whl/cu117

# 2) PyG 预编译扩展（必需；vendored torch_geometric 直接 import 它们）
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-1.13.0+cpu.html
#   CUDA 版： -f https://data.pyg.org/whl/torch-1.13.0+cu117.html

# 3) PyG 本体：装【干净完整】的 2.6.1（⚠️ 不要用 README 说的 2.0.4）
pip install torch-geometric==2.6.1

# 4) 其余依赖
pip install "numpy==1.23.5" triangle scikit-image svgpathtools opencv-python networkx tqdm matplotlib pillow yacs

# 5) 【关键】定向覆盖：只覆盖仓库真正独有的两棵子树，**不要全量覆盖**！
$sp   = "$env:CONDA_PREFIX\Lib\site-packages\torch_geometric"
$repo = "D:\Dev\BIM\CAD2Graph\third_party\VecFloorSeg"
foreach ($sub in @("graphgym", "nn")) {
    Copy-Item -Recurse -Force "$repo\torch_geometric\$sub" "$sp\"
}

# 6) 额外依赖（他们环境里也有）
pip install --no-deps "ogb==1.3.6"   # --no-deps 防止 pip 升级 torch
pip install outdated
```

> ⚠️ **实测教训（重要）**：仓库自带的 `torch_geometric/` 是 **1.49 MB / 307 文件的跨版本混杂部分覆盖包**，
> **不能全量覆盖**到任何已发布 PyG：
> - 全量覆盖到 2.0.4 → `ModuleNotFoundError: No module named 'torch_geometric.loader'`
> - 全量覆盖到 2.6.1 → `ImportError: cannot import name 'DatasetAdapter'`
>
> 我逐一探测了 2.0.4 / 2.3.0 / 2.4.0 / 2.5.0 / 2.5.1 / 2.6.0 / 2.6.1 / 2.7.0 的 wheel：
> **没有任何已发布版本**把 `get_input_nodes` 放在 `neighbor_loader.py`（均在 `loader/utils.py`）
> → 其改造代码是针对 PyG **dev 版**写的。
>
> **可行做法（已实测通过）**：干净 PyG 2.6.1 + **只覆盖 `torch_geometric/graphgym/` 与 `torch_geometric/nn/`**。
> 因为 `custom_graphgym` 实际依赖的就是这两棵子树（外加由官方 PyG 提供的 `data`、`transforms`、`typing`、`data.datapipes.functional_transform`）。

**骨干权重**（代码在 import 时就会实例化 4 个 backbone，4 个文件缺一不可）：

```powershell
$m = "D:\Dev\BIM\CAD2Graph\third_party\VecFloorSeg\models"
New-Item -ItemType Directory -Force $m | Out-Null
curl.exe -L --ssl-no-revoke -o "$m\resnet34-torch.pth"  https://download.pytorch.org/models/resnet34-b627a593.pth
curl.exe -L --ssl-no-revoke -o "$m\resnet50-torch.pth"  https://download.pytorch.org/models/resnet50-0676ba61.pth
curl.exe -L --ssl-no-revoke -o "$m\resnet101-torch.pth" https://download.pytorch.org/models/resnet101-63fe2227.pth
curl.exe -L --ssl-no-revoke -o "$m\vgg16_bn-torch.pth"  https://download.pytorch.org/models/vgg16_bn-6c64b313.pth

# 让 models/ 只在本地被忽略，保持 submodule 干净（不改动上游被跟踪文件）
$root   = "D:\Dev\BIM\CAD2Graph\third_party\VecFloorSeg"
$gitdir = ((Get-Content "$root\.git" -Raw) -replace '^gitdir:\s*', '').Trim()
$gitdir = [System.IO.Path]::GetFullPath((Join-Path $root $gitdir))
Add-Content (Join-Path $gitdir "info\exclude") "models/"
```

> 权重共约 **880 MB**（其中 `vgg16_bn` 一个就 528 MB）。

**运行**（cwd 必须是仓库根，因为权重路径是相对的 `./models/...`）：

```powershell
cd D:\Dev\BIM\CAD2Graph\third_party\VecFloorSeg
python graphgym/main.py --cfg graphgym/configs/CUBI.yaml --help     # ✅ 已验证 exit 0
```

> `pytorch_lightning` 未安装只会产生 **warning**（可选依赖），不影响运行。

> **为什么建议 CPU 优先**：RTX 4070 Ti 是 Ada(sm_89)，而 CUDA 11.7 早于 Ada，`torch_scatter` / `torch_sparse` 的预编译 kernel 可能不含 sm_89（报 `no kernel image is available`）。先用 CPU 把「预处理 → 推理 → 输出」链路跑通，确认无误后再换 CUDA 版试；两者结果应一致。

### 7.3 关键风险

1. **Ada (sm_89) vs CUDA 11.7（最高风险）**：torch 1.13(+cu117) 早于 Ada，预编译 kernel 最高约 sm_86，GPU 运行可能报 `no kernel image is available for execution on the device`。建议先按 **CPU** 跑通流程，再试 GPU。
2. **cairosvg**：`Replace_with_CubiCasa/ImgRasterization.py` 用它做 SVG→PNG；Windows 下需额外 Cairo DLL，pip 装完常报 `no library called "cairo" was found`。属输入预处理环节，可改用其他栅格化方式绕过。
3. **numpy 版本**：vendored `torch_geometric/data/dataset.py` 使用已被移除的 `np.bool` → 必须 `numpy<1.24`。
4. **Python 3.8**：本机只有 3.13，必须单独装 3.8（conda 最省事）。
5. **权重与数据集**：仍需从 Google Drive 获取（README 链接），本仓库无法代下。
6. **`torch_geometric` 需「先装完整版、再覆盖改造文件」**：仓库自带的 `torch_geometric/` 是部分覆盖包（缺 `loader/` 等），仅靠 `PYTHONPATH` 会 `ModuleNotFoundError`。在**专用 conda 环境**里覆盖 site-packages 是安全的（隔离靠环境，不靠 site-packages）。

### 7.4 冒烟测试顺序（逐步验证）

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import torch_scatter, torch_sparse, torch_cluster, torch_spline_conv; print('pyg ext ok')"
$env:PYTHONPATH = "D:\Dev\BIM\CAD2Graph\third_party\VecFloorSeg"
python -c "import torch_geometric, graphgym; print(torch_geometric.__file__)"
python graphgym/main.py --cfg graphgym/configs/CUBI.yaml --help
```

> 若 GPU 报 `no kernel image`，把 `graphgym/configs/CUBI.yaml` 的 `device: cuda:0` 改为 `device: cpu` 先跑通流程。

### 7.5 `PYTHONPATH` 到底设在哪里

> 若采用 7.2 的**覆盖式安装**（默认），则**不需要** `PYTHONPATH`。本节仅适用于 7.2 的「替代方案（合并目录）」。

`PYTHONPATH` 是**进程级**环境变量，只对被“从同一进程/会话启动”的 python 生效：

| 场景 | 做法 |
|---|---|
| 在 **Anaconda PowerShell** 手动跑 | 在**该会话**里设：`$env:PYTHONPATH = "D:\...\third_party\VecFloorSeg"`（**PowerShell 用 `$env:`，不是 cmd 的 `set`**） |
| 在 **VS Code 集成终端**跑 | 集成终端是**独立进程，不会继承** Anaconda PowerShell 的设置；需在当前终端重新设，或改 `settings.json` 的 `terminal.integrated.env.windows` |
| 在 **VS Code 调试** | `launch.json` 的 `"env": { "PYTHONPATH": "D:\\...\\third_party\\VecFloorSeg" }` |
| **CAD2Graph 适配层调用（最终形态）** | 由适配层在 `subprocess.run(..., env={...})` 里显式传入，**无需任何全局设置**，最可复现 |

> 也记得在 VS Code 里把解释器切到该环境：`Python: Select Interpreter` → `vecfloorseg`。

## 8. 网络与代理（实测）

本机直连受限，但**存在可用本地代理** `http://127.0.0.1:7897`（Clash 混合端口，HTTP 代理方式验证通过）。

| 目标 | 直连 | 经代理 |
|---|---|---|
| `pypi.org` / `files.pythonhosted.org` | ✅ | ✅ |
| `download.pytorch.org` / `data.pyg.org` | ✅ | ✅ |
| `repo.anaconda.com` / `conda.anaconda.org` | ✅ | ✅ |
| `github.com` | ❌ TCP 超时 | ✅ 200（`git ls-remote` 正常） |
| `drive.google.com/drive/folders/...` | ❌（DNS 污染） | ✅ 200 |
| `drive.usercontent.google.com` | ❌ | ✅ 可达（无 file id 返回 404） |

### 常用命令

```bash
# git 走代理（局部，只影响本仓库；需要时用 git config --local --unset http.proxy 撤销）
git config --local http.proxy http://127.0.0.1:7897

# git 走代理（全局）
git config --global http.proxy http://127.0.0.1:7897

# pip 走代理
pip install --proxy http://127.0.0.1:7897 <pkg>
```

> 注意：Windows 自带 `curl` 用 schannel，访问 Google 会报 `CRYPT_E_REVOCATION_OFFLINE` / TLS handshake 失败；加 `--ssl-no-revoke` 或改用 Python(OpenSSL) 即可。

> 当前 submodule 已通过代理完成真实 clone：`third_party/VecFloorSeg` @ `2491d2d`（`heads/master`）。

## 9. 预训练 checkpoint 状态（实测结论）

**结论：作者没有发布训练好的 checkpoint，必须自己训练才能得到 `best1.ckpt`。**

复查证据：

| 位置 | 内容 |
|---|---|
| `third_party/VecFloorSeg/models/` | 仅 4 个**骨干**权重（`resnet34/50/101-torch.pth`、`vgg16_bn-torch.pth`，本机自行下载），**无 `.ckpt`** |
| 作者 Google Drive（文件夹 id `1Rye_6crjcuII2LVaIwh4iDNowFqLp1Q6`） | 共 5 项：`upload_to_google_drive/`、`CUBI_3.zip`、`CubiCasa-5k.rar`、`R2V.rar`、`VecFloorSeg.zip`。其中 `upload_to_google_drive/` 只有 `icon.png` + `train/val/test.txt` |
| `VecFloorSeg.zip`（328 MB / 344,121,439 B） | 解包共 1064 项：权重只有 `models/resnet34\|50\|101-torch.pth`；`results/CUBI/` 下**只有一个 `config.yaml`**，没有 `0/` 运行目录、没有 `ckpt/` |
| 磁盘全局搜索 `*.ckpt` | 无任何结果 |

README 的流程本身就是 `Train --seed 0` → `Eval ... train.epoch_resume 1 train.ckpt_prefix best`，即 checkpoint 由**训练过程产生**，作者只发布了代码 + 骨干权重 + 数据集 + 训练配置。

> ⚠️ **危险点**：`torch_geometric/graphgym/checkpoint.py::load_ckpt()` 在文件不存在时**静默 `return 0`**（不抛异常）。
> 后果：推理会用**随机初始化的权重**跑完全程，输出全是垃圾，且没有任何报错或警告。
> 接入前必须在自己的封装里**显式校验 checkpoint 文件存在**。

### 期望的 checkpoint 路径

按 `config.py` 的 `set_out_dir` / `set_run_dir` 与 `checkpoint.py::get_ckpt_path()` 推导：

```
<out_dir>/<cfg_name>/<seed>/ckpt/<prefix><epoch>.ckpt
  → results/CUBI/0/ckpt/best1.ckpt
```

### 本次调查的附带发现

1. 已从压缩包中提取作者**论文使用的完整训练配置** → `docs/vecfloorseg_official_run_config.yaml`
   （`seed 0`、`batch_size 8`、`max_epoch 200`、`sgd lr 0.01 + cos`、`focal_loss`、
   `gnn.imgEncoder: resnet50`、`dataset.name: CUBICASA_Merge_DUALIMG_1`、`dim_inner 256`、`layers_mp 6`、
   `dataset.dir: /home/ybc2021/Datasets/CUBI_3`）。可用于**精确复现**训练。
2. `gnn.imgEncoder: resnet50` → 该配置**只需要 resnet50 骨干**。（但 `ImgEncoderDict` 在 import 时
   会实例化全部骨干，所以 4 个 `.pth` 都留着更稳妥。）
3. 压缩包中的 `Replace_with_CubiCasa/`（`roughcast_data_generation.py`、`ImgRasterization.py`、
   `svg_loader.py`）就是**官方的 SVG → roughcast → PNG → pkl 输入链路**，Phase 2 输入桥接可直接复用。
4. 数据集 `D:\Dev\BIM\CUBI_3` 已**完整**：`ann_dir/`、`img_dir/`、
   `merge_{train,val,test}_phase_V10.pkl`、`{train,val,test}.txt`（4200 / 400 / 400 条），
   与 loader 契约一致；只需把 `graphgym/configs/CUBI.yaml` 的 `dir:` 指向它。

### 可行路径

| 方案 | 说明 |
|---|---|
| **A. 向作者索取 checkpoint** | 提 GitHub issue / 邮件。最快，但不可控 |
| **B. 自行训练（`--seed 0`）** | 唯一完全可控。**已定：在 Ubuntu + GTX 2080 Ti 训练，见 §10** |
| C. 随机权重跑通链路 | **仅用于调试 I/O 管道**，输出无任何意义，且必须清楚这一点 |

## 10. 训练方案（最终选择：Ubuntu GPU 服务器）

### 10.1 为什么放服务器而不在本机

| | 本机（Windows + RTX 4070 Ti） | 服务器（Ubuntu + GTX 2080 Ti ×2） |
|---|---|---|
| GPU 架构 | Ada **sm_89** | Turing **sm_75** |
| 可用 CUDA | 必须 ≥ 11.8（CUDA 11.7 早于 Ada） | **CUDA 11.7 原生支持** |
| torch | 被迫换 2.1.2（**偏离作者栈**） | **1.13.0 完全一致** |
| 环境搭建 | 已踩 4 个坑（PyG 覆盖 / SSL 信任库 / pip 镜像 / 扩展轮子） | 一条脚本 + 预置轮子 |

作者 `requirements.txt` 明确 pin（这就是保真目标）：

```
python=3.8.15
pytorch=1.13.0=py3.8_cuda11.7_cudnn8.5.0_0
torchvision=0.14.0=py38_cu117
torchaudio=0.13.0=py38_cu117
```

### 10.2 需要传到服务器的东西

| 项 | 来源 | 大小 | 说明 |
|---|---|---|---|
| 仓库 | `git clone` + `--recurse-submodules` | — | 服务器网络受限就直接打包整个目录 |
| CUDA 轮子 | 本机 `D:\Dev\BIM\_wheels_linux\` | **1.72 GB** | 6 个 whl，服务器可**离线**装 |
| 数据集 | `D:\Dev\BIM\CUBI_3` | **~365 MB** | **可放任意路径** |
| 骨干权重 | 脚本自动下载，或拷本机 `third_party/VecFloorSeg/models/` | ~880 MB | 4 个缺一不可 |

本机已预下载的 6 个 Linux 轮子（`pt113cu117` = torch 1.13 + cu117）：

```
torch-1.13.0+cu117-cp38-cp38-linux_x86_64.whl                 1723.1 MB
torchvision-0.14.0+cu117-cp38-cp38-linux_x86_64.whl             23.1 MB
torch_scatter-2.1.1+pt113cu117-cp38-cp38-linux_x86_64.whl        9.6 MB
torch_sparse-0.6.17+pt113cu117-cp38-cp38-linux_x86_64.whl        4.5 MB
torch_cluster-1.6.1+pt113cu117-cp38-cp38-linux_x86_64.whl        3.0 MB
torch_spline_conv-1.2.2+pt113cu117-cp38-cp38-linux_x86_64.whl    0.8 MB
```

> `torch_geometric==2.6.1` 与其余纯 Python 依赖在 PyPI 上都有，走清华镜像即可，不必预下。

**数据集里 loader 真正使用的文件**（`CUBIMergeDualImgDataset_1.py`）：

```
{ train, val, test }.txt                 # 4200 / 400 / 400 行，形如 /high_quality_architectural/333/
img_dir/{ train, val, test }/*.png       # 文件名 = 样本 id（如 333.png）
merge_{ train, val, test }_phase_V10.pkl # dict[str(imgName) -> 标注列表]
```

- `imgName = line.split('/')[-2]`，并排除 `excludeList` 里 10 个样本
- 图像处理：`ToTensor → Resize(256) → Normalize(ImageNet)`
- **`ann_dir/` loader 不使用**（仅供可视化/GT 对照）→ 可以不传
- ✅ **pkl 里不含任何绝对路径**（已逐字节扫描验证）→ 数据集**可任意搬迁**，只需 `dataset.dir` 指对

实测体积（本机 `D:\Dev\BIM\CUBI_3`）：

| 项 | 文件数 | 大小 |
|---|---|---|
| `img_dir/` | 4991 | 44.6 MB |
| `merge_{train,val,test}_phase_V10.pkl` | 3 | 320.3 MB（269.3 + 24.6 + 26.4） |
| `{train,val,test}.txt` | 3 | 0.14 MB |
| **必须传的合计** | | **≈ 365 MB** |
| `ann_dir/`（可不传） | 4991 | 12.6 MB |

### 10.3 服务器一键搭建

```bash
bash scripts/setup_vecfloorseg_server.sh --wheels-dir ~/vfs_wheels
# 或让它在服务器上自行拉取（需代理）：
# bash scripts/setup_vecfloorseg_server.sh --proxy http://<proxy>:<port>
```

脚本做的事（按执行顺序）：conda py3.8 → torch 1.13.0+cu117 + torchvision 0.14.0+cu117 → 4 个 PyG 扩展（`pt113cu117`）→ **运行时依赖**（numpy/pillow/tqdm/yacs/networkx/**fsspec**/ogb）→ 干净 `torch-geometric==2.6.1` → **定向覆盖 `torch_geometric/{graphgym,nn}`** → **PyG≥2.3 兼容补丁** → 4 个 backbone → 自检。

> ⚠️ **顺序有约束**：运行时依赖**必须**在覆盖/补丁之前装 —— 因为 PyG 2.6.1 在 `import` 时就会 `import fsspec`（`torch_geometric/io/txt_array.py`），而兼容补丁的自我校验要 import `torch_geometric`。全新环境里缺 `fsspec` 会让补丁的校验步骤挂掉（补丁本身已生效，但脚本会因为校验失败而中止）。

### 10.4 训练与推理命令

`dataset.dir` 用 CLI 覆盖，**不需要改 submodule**（保持 submodule 干净）：

```bash
# 冒烟测试（1 epoch，验证数据加载 + 显存）
# 注意：max_epoch 在 `optim` 下，不在 `train` 下（yacs 对不存在的 key 会直接 assert）
cd third_party/VecFloorSeg
python graphgym/main.py --cfg graphgym/configs/CUBI.yaml seed 0 \
    dataset.dir /data/CUBI_3 optim.max_epoch 1 train.iter_per_epoch 1

# 正式训练（论文配方）
python graphgym/main.py --cfg graphgym/configs/CUBI.yaml seed 0 \
    dataset.dir /data/CUBI_3

# 推理
python graphgym/main.py --cfg graphgym/configs/CUBI.yaml --eval \
    train.epoch_resume 1 train.ckpt_prefix best val.extra_infos True \
    seed 0 dataset.dir /data/CUBI_3
```

> ⚠️ **绝不要写 `--seed 0`**（作者 README 的 Train 命令就是这么写的，**在当前代码版本下是错的**）。
> `torch_geometric/graphgym/cmd_args.py` 里 `opts` 用的是 `nargs=argparse.REMAINDER`，它**只在遇到第一个不像选项的 token 时才开始收纳**；
> 所以 `--cfg X --seed 0` 会直接报错：
> ```
> main.py: error: unrecognized arguments: --seed
> ```
> 覆盖项必须写成 `key value` 对，且**第一个不能以 `--` 开头**（README 的 Eval 命令恰好满足，所以它能跑）。
> 覆盖通过 `config.py` 的 `cfg.merge_from_list(args.opts)` 生效。

### 10.4.1 推理产物（已从源码确认）

`--eval` 时 `torch_geometric/graphgym/train.py::eval()` 会把每个 split 的预测落盘：

```python
predWrite = eval_epoch(...)
pickle.dump(predWrite, os.path.join(cfg.run_dir, f'{split}_result.pkl'))
```

| 文件 | 内容 |
|---|---|
| **`<run_dir>/val_result.pkl` / `test_result.pkl`** | **真正的预测**。`model.has_aux: true` 时每个样本是 6 元组：`(filename, pred, gt, aux_pred, aux_gt, valid_edge)`；`pred` 是 **veno 图（房间面）的节点级类别** |
| `<run_dir>/val_result_extra.pkl` | 仅在 `val.extra_infos True` 时写。内容是**注意力权重**（`forward_test()` 返回 `weight_ji`）+ `veno` 图的 `edge_index`，用于论文的注意力可视化，**不是分割结果** |

> 注意 `*_result_extra.pkl` 容易让人误以为是结果 —— 实际上房间多边形要从 `*_result.pkl` 的节点预测 + 图的 `edge_index` 重建。

### 10.5 双卡与显存

- GraphGym 是**单卡**框架（默认 `device: cuda:0`），**不会自动用第二张卡**；如需指定：`device cuda:1`
- 2080 Ti = **11 GB**。官方 `train.batch_size: 8` + resnet50 backbone，可能吃紧。若 OOM：
  1. `train.batch_size 4`，并相应调小 `optim.base_lr 0.005`（偏离原文，但能跑）
  2. `dataset.augmentation False` 可再省一点显存
- `train.iter_per_epoch` 是**无效参数**（见下），`eval_period: 2`（每 2 epoch 验证一次）、`ckpt_period: 50`

### 10.5.1 实测速度与工期（Windows 4070 Ti 实测，bs 8，`num_workers: 0`）

| 阶段 | 规模 | 耗时 | 指标（1 epoch 时） |
|---|---|---|---|
| train | 524 iter | **9:26**（1.08 s/it） | loss 0.8796 |
| val | 400 张 | 46 s（8.65 it/s） | mIoU 0.3431 |
| test | 399 张 | 49 s | mIoU 0.3291 |

> ⚠️ **`train.iter_per_epoch` 不起作用**：它只被 `torch_geometric/graphgym/loader.py` 用作 sampler 的 `num_steps`，对 `train.sampler: full_batch` 完全没有效果（`train_epoch` 直接遍历整个 loader）。
> 作者 config 里的 `iter_per_epoch: 32` 只是 GraphGym 的历史默认值。
> 所以 **1 epoch = len(train_loader) = 524 个 iteration**（4190 张 / bs 8），**不是 32**。

**工期推算**（`optim.max_epoch: 200`，即论文配方）：

| 机器 | 1 epoch | 200 epoch |
|---|---|---|
| RTX 4070 Ti | ≈ 11 min | **≈ 34 h** |
| GTX 2080 Ti（估） | ≈ 15–28 min | **≈ 2–4 天** |

（2080 Ti 估算依据：FP32 13.4 vs 40 TFLOPS 偏低，但带宽 616 vs 504 GB/s 偏高；该负载是小图 + 256×256 图像，偏访存/调度，故给 1.5–3× 区间。）

**提速**：`cfg.num_workers` 默认 **0**（单进程加载：PNG 解码 + 变换 + 建图都在主进程）。

拆解实测（RTX 4070 Ti，bs 8，train split 前 15 个 batch）：

```
纯数据加载 (num_workers=0) : 0.589 s/batch
端到端训练步              : 1.080 s/it
=> 数据加载占 55%，模型前向+反向仅约 0.49 s
```

→ **`num_workers 4` 在 Linux 上有实打实的收益**（负载下降至 ~0.15 s 的话，端到端 ~0.64 s/it，约 **1.7×**）。

> ⚠️ **Windows 上不要用 `num_workers>0`**：Windows 用 spawn，每个 worker 会**重新 import `custom_graphgym`**，而该模块在 import 时就会实例化 `ImgEncoderDict`、把 4 个 backbone（约 880 MB）重新 `torch.load` 一遍；而且 DataLoader worker 里 `open(...,'w')` 之类的模块级副作用会被重复执行。Linux 用 fork，worker 直接继承父进程内存，没有这个问题。

**验证方法**：在服务器上先跑 `optim.max_epoch 1`，头 10 秒看进度条的 `s/it`，对比 `num_workers 0` 与 `4` 再决定。

### 10.5.2 两个 loader 入口的坑（已踩）

1. **`main.py` 里的 `create_loader()` 是它自己定义的**（遍历 list），与 `torch_geometric/graphgym/loader.py::create_loader()` **同名但不是一个函数**。后者假设只有一个 dataset，会执行 `dataset.data`，而我们的自定义 loader 返回 `[train, val, test]` 列表 →
   `AttributeError("'list' object has no attribute 'data'")`。
2. **`load_dataset()` 的分发逻辑**：先在 `register.loader_dict` 里按 `cfg.dataset.name` 找自定义 loader，找不到才 fallback 到 PyG 内置（`load_pyg` → `Planetoid`）。
   若忘记调用 `load_cfg()`，`cfg.dataset.name` 会停留在 GraphGym 默认值 **`'Cora'`** → 走进 `Planetoid` → 触发**下载** → fsspec → aiohttp → 在 Windows 上报一个极其迷惑的
   `SSLError(142, '[ASN1: NOT_ENOUGH_DATA]')`（因为该 conda 环境没有可用的默认信任库）。
   → 排障时先确认 `cfg.dataset.name` 与 `register.loader_dict`。

## 11. 服务器部署步骤（可直接复制的命令）

### 11.1 要传什么 / 不要传什么（实测体积）

| 传 | 路径 | 体积 | 说明 |
|---|---|---|---|
| ✅ | `third_party/VecFloorSeg`（**排除** `models/`、`results*/`） | **1.7 MB** | 383 个文件 |
| ✅ | `scripts/` | ~0 MB | 含 `setup_vecfloorseg_server.sh`、`patch_vecfloorseg_pyg23_compat.py` |
| ✅ | `D:\Dev\BIM\_wheels_linux` | **1764 MB** | 6 个 linux 轮子，**离线装 CUDA 栈的关键** |
| ⭕ | `third_party/VecFloorSeg/models/` | **879 MB** | 4 个 backbone；服务器能直连 `download.pytorch.org` 就不必传 |
| ✅ | `D:\Dev\BIM\CUBI_3`（**排除** `ann_dir/`） | **365 MB** | loader 不用 `ann_dir` |
| ❌ | `cadruler/` | 613 MB | Windows venv，Linux 上完全无用 |
| ❌ | `D:\Dev\BIM\_wheels` | 2615 MB | Windows cu118 轮子，服务器**不需要** |
| ❌ | `input_data/`、`output/` | 291 MB | 训练不需要 |

**合计约 3.0 GB**（不传 models 则约 2.1 GB）。

### 11.2 本机（Windows）打包

```powershell
New-Item -ItemType Directory -Force D:\Dev\BIM\_xfer | Out-Null

# 1) 代码 + 脚本（保持 third_party/ 与 scripts/ 的相对结构）
tar -czf D:\Dev\BIM\_xfer\vfs_code.tar.gz -C D:\Dev\BIM\CAD2Graph `
    --exclude=models --exclude=results --exclude=results_smoke --exclude=__pycache__ `
    third_party/VecFloorSeg scripts

# 2) CUDA 轮子（已是压缩包，不必再 gz）
tar -cf  D:\Dev\BIM\_xfer\vfs_wheels.tar -C D:\Dev\BIM _wheels_linux

# 3) 数据集（排除 loader 不用的 ann_dir）
tar -cf  D:\Dev\BIM\_xfer\cubi3.tar -C D:\Dev\BIM\CUBI_3 --exclude=ann_dir `
    img_dir merge_train_phase_V10.pkl merge_val_phase_V10.pkl merge_test_phase_V10.pkl `
    train.txt val.txt test.txt

# 4)（可选）backbone 权重
tar -cf  D:\Dev\BIM\_xfer\vfs_models.tar -C D:\Dev\BIM\CAD2Graph\third_party\VecFloorSeg models

# 5) 传
scp D:\Dev\BIM\_xfer\vfs_code.tar.gz  user@server:~/
scp D:\Dev\BIM\_xfer\vfs_wheels.tar   user@server:~/
scp D:\Dev\BIM\_xfer\cubi3.tar        user@server:~/
scp D:\Dev\BIM\_xfer\vfs_models.tar   user@server:~/   # 传了 models 才需要
```

### 11.3 服务器解包

```bash
# 0) 自检
nvidia-smi                       # 确认能看到 GPU
conda --version
df -h $HOME                      # 需要约 12 GB：wheels 1.7 + 数据 0.4 + 装完 torch ~4.5 + 产物 ~1.5
free -g                          # 建议 >= 16 GB（train pkl 269 MB + DataLoader workers）

# 1) 解包（注意目录结构必须是 ~/vfs/third_party/VecFloorSeg 与 ~/vfs/scripts）
mkdir -p ~/vfs && cd ~/vfs
tar -xzf ~/vfs_code.tar.gz                 # -> ./third_party/VecFloorSeg, ./scripts
tar -xf  ~/vfs_wheels.tar                  # -> ./_wheels_linux
sudo mkdir -p /data/CUBI_3 && sudo chown "$USER" /data/CUBI_3
tar -xf ~/cubi3.tar -C /data/CUBI_3        # -> img_dir/, *.pkl, *.txt
# 已传 models 的话（注意排除后代码包里没有 models/，解包即得）
tar -xf ~/vfs_models.tar -C ~/vfs/third_party/VecFloorSeg

# 2) 若 conda 拉不动包，先配国内镜像
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
conda config --set show_channel_urls yes
```

### 11.4 一键搭建

```bash
cd ~/vfs
# 已传 models -> 加 --skip-backbones
bash scripts/setup_vecfloorseg_server.sh --wheels-dir ~/vfs/_wheels_linux --skip-backbones

# 没传 models -> 去掉 --skip-backbones（脚本自己去 download.pytorch.org 下 880 MB）
# bash scripts/setup_vecfloorseg_server.sh --wheels-dir ~/vfs/_wheels_linux

# 受限网络再加：--proxy http://<proxy>:<port>  --mirror <国内 pypi 镜像>
```

脚本会（按执行顺序）：conda py3.8 → torch 1.13.0+cu117 & torchvision 0.14.0+cu117 → 4 个 PyG 扩展（`pt113cu117`）→ **运行时依赖**（含 `fsspec`）→ 干净 `torch-geometric==2.6.1` → **定向覆盖** `torch_geometric/{graphgym,nn}` → **PyG≥2.3 兼容补丁** → backbone → 自检。

> ⚠️ 依赖必须在覆盖之前装：PyG 2.6.1 `import` 时要 `fsspec`，而补丁的自我校验会 import `torch_geometric`。若在全新环境里先跑覆盖，补丁会报 `ModuleNotFoundError: No module named 'fsspec'`（补丁其实已生效，只是校验失败导致脚本中止）。
> 修法：先 `python -m pip install -i <mirror> fsspec` 再重跑脚本（幂等，环境会复用）。

### 11.5 冒烟测试（约 10–30 分钟，**必须用独立的 out_dir**）

```bash
conda activate vecfloorseg
cd ~/vfs/third_party/VecFloorSeg          # ★ cwd 必须是这里（models 是相对路径）

python graphgym/main.py --cfg graphgym/configs/CUBI.yaml seed 0 \
    dataset.dir /data/CUBI_3 optim.max_epoch 1 out_dir results_smoke

# 期望看到： Num parameters: 42541109 / Start from epoch 0 / train-val-test 三个指标行
rm -rf results_smoke                       # ★ 删掉，避免和正式产物混淆
```

### 11.6 正式训练（50 epoch，后台跑）

```bash
cd ~/vfs/third_party/VecFloorSeg
tmux new -s vfs                            # 推荐；或直接 nohup

python graphgym/main.py --cfg graphgym/configs/CUBI.yaml seed 0 \
    dataset.dir /data/CUBI_3 num_workers 4 optim.max_epoch 50 \
    2>&1 | tee ~/vfs_train50.log
```

监控：

```bash
grep -a "mIoU" ~/vfs_train50.log | tail -5   # 每 2 epoch 一行（eval_period=2）
nvidia-smi
ls -lh results/CUBI/0/ckpt/
```

看曲线决定是否续跑到论文的 200 epoch（`optim.max_epoch 200`）。

### 11.7 取回 checkpoint 并在本地用

```bash
# 服务器
cd ~/vfs/third_party/VecFloorSeg
ls -lh results/CUBI/0/ckpt/best1.ckpt        # ★ 先确认真的存在（~324 MB）
tar -cf ~/best1.tar -C results/CUBI/0/ckpt best1.ckpt

# 本地 Windows
scp user@server:~/best1.tar D:\Dev\BIM\
New-Item -ItemType Directory -Force D:\Dev\BIM\CAD2Graph\third_party\VecFloorSeg\results\CUBI\0\ckpt | Out-Null
tar -xf D:\Dev\BIM\best1.tar -C D:\Dev\BIM\CAD2Graph\third_party\VecFloorSeg\results\CUBI\0\ckpt

# 本地推理验证
cd D:\Dev\BIM\CAD2Graph\third_party\VecFloorSeg
C:\Users\admin\miniconda3\envs\vecfloorseg\python.exe graphgym\main.py --cfg graphgym/configs/CUBI.yaml --eval train.epoch_resume 1 train.ckpt_prefix best val.extra_infos True seed 0 dataset.dir D:/Dev/BIM/CUBI_3 device cuda:0
```

### 11.8 一定会踩的坑（务必先看）

1. **不要写 `--seed 0`** —— 报 `unrecognized arguments: --seed`。写 `seed 0`（key/value 对，且**第一个覆盖项不能以 `--` 开头**）。
2. **`max_epoch` 在 `optim` 下**，不在 `train` 下；yacs 对不存在的 key 会直接 assert。
3. **cwd 必须是 `third_party/VecFloorSeg`**，否则 `torch.load('./models/resnet34-torch.pth')` 报 FileNotFoundError。
4. **`train.iter_per_epoch` 是死参数** —— 1 epoch = 524 iter，不是 32。
5. **`num_workers`**：Linux 上设 4（数据加载占 55%，约 1.7× 收益）；**Windows 上不要开**。
6. **如果重新执行过覆盖步骤**，必须重跑 `scripts/patch_vecfloorseg_pyg23_compat.py`，否则会出现那个毫无信息量的 `ValueError`（`message_passing.__lift__`）。
7. **`load_ckpt()` 找不到文件时静默返回 0** —— 推理会用随机权重跑完且不报错。每次 eval 前先 `ls -lh` 确认 ckpt。
8. **冒烟测试务必用 `out_dir results_smoke`**，否则会在 `results/CUBI/0/ckpt/` 留下一个"假的" best1.ckpt。
9. 不要传 Windows 的 `cadruler/` venv 和 `_wheels/`（Windows 轮子）。

**推荐策略（已与用户确认）**：先 `optim.max_epoch 50`，利用 `eval_period: 2` 的 mIoU 曲线判断收敛情况，再决定是否续跑到 200。
续训方式：`train.auto_resume True` + `train.epoch_resume <已保存的 epoch>`（周期 ckpt 名为 `<epoch>.ckpt`，`ckpt_prefix` 为空）—— **需实测确认**后再写进正式流程。

### 10.6 产物与验收

```
results/CUBI/config.yaml         # 解析后的完整配置（作者压缩包里也有一份，见 docs/vecfloorseg_official_run_config.yaml）
results/CUBI/0/ckpt/best1.ckpt   # ★ 目标 checkpoint
results/CUBI/0/ckpt/last*.ckpt   # 周期性快照
<run_dir>/val_result_extra.pkl   # val.extra_infos True 时的预测（含文件名/extra_info/边）
```

验收: `results/CUBI/0/ckpt/best1.ckpt` 真实存在且 size > 0（别信 `load_ckpt()` 的静默返回 0）。

### 10.7 本地环境现状

为换 CUDA，本地 `vecfloorseg` conda 环境**已卸载 torch**（torch 1.13 CPU 被移除；cu118 轮子已下到 `D:\Dev\BIM\_wheels\`，但按你的要求停止了安装）。

- 若推理也放服务器 → 本地环境可以暂时不管；
- 若要在本地做推理 → 用已下载的 `_wheels\torch-2.1.2+cu118-...whl` 离线装回（还需补 4 个 `pt21cu118` 扩展轮子）。

### 10.8 实测结果（接入基准矩阵之后）

`VecFloorSeg` 已进入 `docs/benchmark_protocol.md` 的三层矩阵（39 张图、1916 个 GT 空间）：

| 口径 | 结果 |
|---|---|
| 任务 1 · 轮廓（`contour_VecFloorSeg`） | 65 个空间 · 数量比 0.034 · 覆盖率 **0.013** · 一对一率 0.000 · mIoU 0.074 · 面积 MAE 183.04 m² |
| 端到端最好的一格（`VecFloorSeg + LLMMultiStage`） | **0.030** |
| 参照：`CDT` / `RGP` 覆盖率 | 0.547 / 0.656 |

**根因是权重欠训练，不是适配层。** `best1.ckpt` 的 `best` 落在 **epoch 1**：1291 个
region 收敛成 1 条轮廓（标签分布 `0=1083, 3=2, 11=206`，房间类命中率 16.1%）。
用本地 `0.ckpt` 做对照（`0=1126, 2=165`，房间类命中率 0.0%）得到 **0 条轮廓** ——
两者行为一致地"跟着权重走"，证明**管线是通的，是权重欠训练**。

→ 要得到有意义的数字，必须**按论文配方重训到收敛**（`max_epoch 200`，见 §10.4），
而不是继续调适配层。
