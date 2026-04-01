# OSTrack 单目标追踪工作区

本仓库现在只保留 **OSTrack 微调与推理方案**。

目标是支持一条清晰的论文工作流：

1. 视频抽帧
2. 逐帧人工标注单目标框
3. 转成 GOT10K-like 数据
4. 微调 OSTrack
5. 用不同 checkpoint 跑视频并导出结果

---

## 1. 环境准备

先创建并激活 Conda 环境：

```bash
conda env create -f env.yml
conda activate transformer_tracker
```

再安装 OSTrack 额外依赖：

```bash
pip install -r ostrack/requirements.txt
```

MAE 预训练权重默认位于：

```bash
ostrack/vendor/OSTrack/pretrained_models/mae_pretrain_vit_base.pth
```

---

## 2. 仓库中保留的核心脚本

### 根目录

- `env.yml`：基础 Conda 环境
- `AGENTS.md`：给 coding agent 的仓库说明
- `readme.md`：当前使用文档
- `test.mp4`：示例视频

### `ostrack/`

- `ostrack/tools/extract_frames.py`：视频抽帧
- `ostrack/tools/label_sequence.py`：逐帧人工标注并导出 `groundtruth.txt`
- `ostrack/tools/prepare_dataset.py`：转成 GOT10K-like 数据布局
- `ostrack/train_ostrack.py`：启动微调训练
- `ostrack/run_video.py`：用单个 checkpoint 跑视频推理
- `ostrack/run_video_batch.py`：批量比较多个 checkpoint
- `ostrack/README.md`：OSTrack 子工作区说明
- `ostrack/vendor/OSTrack/`：vendored 官方 OSTrack 代码

---

## 3. 推荐使用顺序

### Step 1：视频抽帧

```bash
python ostrack/tools/extract_frames.py --video test.mp4 --output-root ostrack\data\labeled --sequence-name my_sequence
```

#### 参数说明

- `--video`：输入视频路径，必填
- `--output-root`：抽帧输出根目录，默认 `ostrack/data/labeled`
- `--sequence-name`：输出序列名，默认使用视频文件名
- `--start-frame`：从第几帧开始导出，默认 `0`
- `--max-frames`：最多导出多少帧，默认导出全部
- `--stride`：每隔多少帧导出一张，默认 `1`
- `--overwrite`：如果目标目录已有 JPG，先删除再重新抽帧

输出目录示例：

```bash
ostrack/data/labeled/my_sequence/frames/
```

---

### Step 2：逐帧标注目标框

```bash
python ostrack/tools/label_sequence.py --frames-dir ostrack\data\labeled\my_sequence\frames --output-root ostrack\data\labeled --sequence-name my_sequence
```

#### 参数说明

- `--frames-dir`：待标注帧目录，必填
- `--output-root`：标注结果根目录，默认 `ostrack/data/labeled`
- `--sequence-name`：输出序列名；如果 `frames-dir` 末尾目录名是 `frames`，默认用其父目录名
- `--overwrite`：先清空已有 `groundtruth.txt` 再重标

#### 标注快捷键

- 鼠标左键拖拽：画框
- `Space`：确认当前帧并进入下一帧
- `R`：清空当前框
- `B`：返回上一帧
- `Q` / `ESC`：保存并退出

输出文件：

```bash
ostrack/data/labeled/my_sequence/groundtruth.txt
```

格式：每行一个 bbox，内容为 `x,y,w,h`

---

### Step 3：转换为 GOT10K-like 训练数据

```bash
python ostrack/tools/prepare_dataset.py --sequence-dir ostrack\data\labeled\my_sequence\frames --gt-file ostrack\data\labeled\my_sequence\groundtruth.txt --output-root ostrack\data\processed --sequence-name my_sequence --val-ratio 0.2 --overwrite
```

#### 参数说明

- `--sequence-dir`：输入图片序列目录，必填
- `--gt-file`：对应的 `groundtruth.txt`，必填
- `--output-root`：处理后数据根目录，默认 `ostrack/data/processed`
- `--sequence-name`：序列名，默认使用输入目录名
- `--val-ratio`：验证集比例，默认 `0.2`
- `--overwrite`：清掉旧的同名 train/val 序列后重建

注意：

- 脚本要求至少 **20 帧**
- 输出帧会统一转成 `00000001.jpg` 这种命名

输出目录示例：

```bash
ostrack/data/processed/got10k/train/
ostrack/data/processed/got10k/val/
```

---

### Step 4：训练 OSTrack

```bash
python ostrack/train_ostrack.py --mode single --nproc-per-node 1 --use-wandb 0
```

#### 参数说明

- `--data-root`：处理后数据根目录，默认 `ostrack/data/processed`
- `--save-dir`：训练输出根目录，默认 `ostrack/outputs`
- `--config`：实验配置名，默认 `vitb_256_mae_ce_32x4_custom_ep50`
- `--mode`：训练模式，`single` 或 `multiple`
- `--nproc-per-node`：每节点 GPU 数，单卡时填 `1`
- `--use-wandb`：是否启用 wandb，建议 `0`
- `--pretrained-file`：MAE 预训练权重路径
- `--dry-run`：只检查路径并打印命令，不真正开训

#### 当前仓库已做的训练优化

- 关闭了频繁验证（避免第 5 个 epoch 后卡住）
- 每个 epoch 都会保存 checkpoint
- 当前配置更适合小数据、单卡、快速出论文结果

checkpoint 默认输出到：

```bash
ostrack/outputs/checkpoints/train/ostrack/vitb_256_mae_ce_32x4_custom_ep50/
```

---

### Step 5：用单个 checkpoint 跑视频

```bash
python ostrack/run_video.py --video test.mp4 --config vitb_256_mae_ce_32x4_custom_ep50 --checkpoint "ostrack/outputs/checkpoints/train/ostrack/vitb_256_mae_ce_32x4_custom_ep50/OSTrack_ep0005.pth.tar" --init-bbox 181 386 223 186
```

#### 参数说明

- `--video`：输入视频路径，必填
- `--config`：OSTrack 配置名
- `--checkpoint`：显式指定 checkpoint；不传则自动找最新 checkpoint
- `--output-dir`：推理结果输出目录，默认 `ostrack/outputs/video_results`
- `--init-bbox`：初始框 `x y w h`；不传则会弹出交互 ROI 选择框

输出内容：

- 带框结果视频
- 每帧 bbox 文本

---

### Step 6：批量比较多个 checkpoint

```bash
python ostrack/run_video_batch.py --video test.mp4 --config vitb_256_mae_ce_32x4_custom_ep50 --epochs 1 3 5 --init-bbox 181 386 223 186
```

#### 参数说明

- `--video`：输入视频路径，必填
- `--config`：OSTrack 配置名
- `--epochs`：按 epoch 编号批量选择 checkpoint，例如 `1 3 5 10`
- `--checkpoints`：直接传多个 checkpoint 路径；提供后会覆盖 `--epochs`
- `--output-dir`：结果输出目录，默认 `ostrack/outputs/video_results`
- `--init-bbox`：为所有 checkpoint 指定同一个初始框
- `--dry-run`：只打印将执行的命令

---

## 4. 论文实验建议

推荐优先比较这些 checkpoint：

- `ep0001`
- `ep0003`
- `ep0005`
- `ep0010`
- `ep0020`

最常见的论文工作流是：

1. 先用 `ep0005` 做第一次视频结果预览
2. 再比较 `ep0010` 和 `ep0020`
3. 选出最稳的一个做最终结果图和视频

---
