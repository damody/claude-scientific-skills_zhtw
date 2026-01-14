# 分散式訓練 - 完整指南

## 概述

PyTorch Lightning 提供多種策略，用於在多個 GPU、節點和機器上高效訓練大型模型。根據模型大小和硬體配置選擇正確的策略。

## 策略選擇指南

### 何時使用每種策略

**常規訓練（單一裝置）**
- 模型大小：任何適合單一 GPU 記憶體的大小
- 使用案例：原型開發、小型模型、除錯

**DDP（分散式資料平行）**
- 模型大小：<500M 參數（例如 ResNet50 約 80M 參數）
- 適用時機：權重、激活值、優化器狀態和梯度都可以放入 GPU 記憶體
- 目標：跨多個 GPU 擴展批次大小和速度
- 最適合：大多數標準深度學習模型

**FSDP（完全分片資料平行）**
- 模型大小：500M+ 參數（例如大型 transformers 如 BERT-Large、GPT）
- 適用時機：模型無法放入單一 GPU 記憶體
- 推薦對象：模型平行新手或從 DDP 遷移的使用者
- 功能：激活值檢查點、CPU 參數卸載

**DeepSpeed**
- 模型大小：500M+ 參數
- 適用時機：需要尖端功能或已熟悉 DeepSpeed
- 功能：CPU/磁碟參數卸載、分散式檢查點、細粒度控制
- 權衡：更複雜的配置

## DDP（分散式資料平行）

### 基本用法

```python
# 單一 GPU
trainer = L.Trainer(accelerator="gpu", devices=1)

# 單一節點上的多 GPU（自動 DDP）
trainer = L.Trainer(accelerator="gpu", devices=4)

# 明確的 DDP 策略
trainer = L.Trainer(strategy="ddp", accelerator="gpu", devices=4)
```

### 多節點 DDP

```python
# 在每個節點上執行：
trainer = L.Trainer(
    strategy="ddp",
    accelerator="gpu",
    devices=4,  # 每個節點的 GPU 數
    num_nodes=4  # 總節點數
)
```

### DDP 配置

```python
from lightning.pytorch.strategies import DDPStrategy

trainer = L.Trainer(
    strategy=DDPStrategy(
        process_group_backend="nccl",  # GPU 用 "nccl"，CPU 用 "gloo"
        find_unused_parameters=False,   # 如果模型有未使用的參數則設為 True
        gradient_as_bucket_view=True    # 更節省記憶體
    ),
    accelerator="gpu",
    devices=4
)
```

### DDP Spawn

當 `ddp` 導致問題時使用（較慢但更相容）：

```python
trainer = L.Trainer(strategy="ddp_spawn", accelerator="gpu", devices=4)
```

### DDP 最佳實務

1. **批次大小：** 乘以 GPU 數量
   ```python
   # 如果使用 4 個 GPU，有效批次大小 = batch_size * 4
   dm = MyDataModule(batch_size=32)  # 32 * 4 = 128 有效批次大小
   ```

2. **學習率：** 通常隨批次大小縮放
   ```python
   # 線性縮放規則
   base_lr = 0.001
   num_gpus = 4
   lr = base_lr * num_gpus
   ```

3. **同步：** 指標使用 `sync_dist=True`
   ```python
   self.log("val_loss", loss, sync_dist=True)
   ```

4. **特定排名操作：** 使用裝飾器僅在主程序執行
   ```python
   from lightning.pytorch.utilities import rank_zero_only

   @rank_zero_only
   def save_results(self):
       # 僅在主程序（rank 0）上執行
       torch.save(self.results, "results.pt")
   ```

## FSDP（完全分片資料平行）

### 基本用法

```python
trainer = L.Trainer(
    strategy="fsdp",
    accelerator="gpu",
    devices=4
)
```

### FSDP 配置

```python
from lightning.pytorch.strategies import FSDPStrategy
import torch.nn as nn

trainer = L.Trainer(
    strategy=FSDPStrategy(
        # 分片策略
        sharding_strategy="FULL_SHARD",  # 或 "SHARD_GRAD_OP"、"NO_SHARD"、"HYBRID_SHARD"

        # 激活值檢查點（節省記憶體）
        activation_checkpointing_policy={nn.TransformerEncoderLayer},

        # CPU 卸載（節省 GPU 記憶體，較慢）
        cpu_offload=False,

        # 混合精度
        mixed_precision=True,

        # 包裝策略（自動包裝層）
        auto_wrap_policy=None
    ),
    accelerator="gpu",
    devices=8,
    precision="bf16-mixed"
)
```

### 分片策略

**FULL_SHARD（預設）**
- 分片優化器狀態、梯度和參數
- 最大記憶體節省
- 更多通訊開銷

**SHARD_GRAD_OP**
- 僅分片優化器狀態和梯度
- 參數保留在所有裝置上
- 較少記憶體節省但更快

**NO_SHARD**
- 無分片（等同於 DDP）
- 用於比較或不需要分片時

**HYBRID_SHARD**
- 節點內使用 FULL_SHARD，節點間使用 NO_SHARD
- 適合多節點設定

### 激活值檢查點

以計算換取記憶體：

```python
from lightning.pytorch.strategies import FSDPStrategy
import torch.nn as nn

# 對特定層類型進行檢查點
trainer = L.Trainer(
    strategy=FSDPStrategy(
        activation_checkpointing_policy={
            nn.TransformerEncoderLayer,
            nn.TransformerDecoderLayer
        }
    )
)
```

### CPU 卸載

不使用時將參數卸載到 CPU：

```python
trainer = L.Trainer(
    strategy=FSDPStrategy(
        cpu_offload=True  # 較慢但節省 GPU 記憶體
    ),
    accelerator="gpu",
    devices=4
)
```

### FSDP 與大型模型

```python
from lightning.pytorch.strategies import FSDPStrategy
import torch.nn as nn

class LargeTransformer(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=4096, nhead=32),
            num_layers=48
        )

    def configure_sharded_model(self):
        # 由 FSDP 呼叫以包裝模型
        pass

# 訓練
trainer = L.Trainer(
    strategy=FSDPStrategy(
        activation_checkpointing_policy={nn.TransformerEncoderLayer},
        cpu_offload=False,
        sharding_strategy="FULL_SHARD"
    ),
    accelerator="gpu",
    devices=8,
    precision="bf16-mixed",
    max_epochs=10
)

model = LargeTransformer()
trainer.fit(model, datamodule=dm)
```

## DeepSpeed

### 安裝

```bash
pip install deepspeed
```

### 基本用法

```python
trainer = L.Trainer(
    strategy="deepspeed_stage_2",  # 或 "deepspeed_stage_3"
    accelerator="gpu",
    devices=4,
    precision="16-mixed"
)
```

### DeepSpeed 階段

**Stage 1：優化器狀態分片**
- 分片優化器狀態
- 中等記憶體節省

```python
trainer = L.Trainer(strategy="deepspeed_stage_1")
```

**Stage 2：優化器 + 梯度分片**
- 分片優化器狀態和梯度
- 良好的記憶體節省

```python
trainer = L.Trainer(strategy="deepspeed_stage_2")
```

**Stage 3：完全模型分片（ZeRO-3）**
- 分片優化器狀態、梯度和模型參數
- 最大記憶體節省
- 可以訓練非常大的模型

```python
trainer = L.Trainer(strategy="deepspeed_stage_3")
```

**帶卸載的 Stage 2**
- 卸載到 CPU 或 NVMe

```python
trainer = L.Trainer(strategy="deepspeed_stage_2_offload")
trainer = L.Trainer(strategy="deepspeed_stage_3_offload")
```

### DeepSpeed 配置檔案

用於細粒度控制：

```python
from lightning.pytorch.strategies import DeepSpeedStrategy

# 建立配置檔案：ds_config.json
config = {
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": True
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": True
        },
        "overlap_comm": True,
        "contiguous_gradients": True,
        "sub_group_size": 1e9,
        "reduce_bucket_size": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9
    },
    "fp16": {
        "enabled": True,
        "loss_scale": 0,
        "initial_scale_power": 16,
        "loss_scale_window": 1000,
        "hysteresis": 2,
        "min_loss_scale": 1
    },
    "gradient_clipping": 1.0,
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto"
}

trainer = L.Trainer(
    strategy=DeepSpeedStrategy(config=config),
    accelerator="gpu",
    devices=8,
    precision="16-mixed"
)
```

### DeepSpeed 最佳實務

1. **<10B 參數的模型使用 Stage 2**
2. **>10B 參數的模型使用 Stage 3**
3. **如果 GPU 記憶體不足，啟用卸載**
4. **調整 `reduce_bucket_size` 以提高通訊效率**

## 比較表

| 功能 | DDP | FSDP | DeepSpeed |
|---------|-----|------|-----------|
| 模型大小 | <500M 參數 | 500M+ 參數 | 500M+ 參數 |
| 記憶體效率 | 低 | 高 | 非常高 |
| 速度 | 最快 | 快 | 快 |
| 設定複雜度 | 簡單 | 中等 | 複雜 |
| 卸載 | 無 | CPU | CPU + 磁碟 |
| 最適合 | 標準模型 | 大型模型 | 超大型模型 |
| 配置 | 最少 | 中等 | 廣泛 |

## 混合精度訓練

使用混合精度加速訓練並節省記憶體：

```python
# FP16 混合精度
trainer = L.Trainer(precision="16-mixed")

# BFloat16 混合精度（A100、H100）
trainer = L.Trainer(precision="bf16-mixed")

# 全精度（預設）
trainer = L.Trainer(precision="32-true")

# 雙精度
trainer = L.Trainer(precision="64-true")
```

### 不同策略的混合精度

```python
# DDP + FP16
trainer = L.Trainer(
    strategy="ddp",
    accelerator="gpu",
    devices=4,
    precision="16-mixed"
)

# FSDP + BFloat16
trainer = L.Trainer(
    strategy="fsdp",
    accelerator="gpu",
    devices=8,
    precision="bf16-mixed"
)

# DeepSpeed + FP16
trainer = L.Trainer(
    strategy="deepspeed_stage_2",
    accelerator="gpu",
    devices=4,
    precision="16-mixed"
)
```

## 多節點訓練

### SLURM

```bash
#!/bin/bash
#SBATCH --nodes=4
#SBATCH --gpus-per-node=4
#SBATCH --time=24:00:00

srun python train.py
```

```python
# train.py
trainer = L.Trainer(
    strategy="ddp",
    accelerator="gpu",
    devices=4,
    num_nodes=4
)
```

### 手動多節點設定

節點 0（主節點）：
```bash
python train.py --num_nodes=2 --node_rank=0 --master_addr=192.168.1.1 --master_port=12345
```

節點 1：
```bash
python train.py --num_nodes=2 --node_rank=1 --master_addr=192.168.1.1 --master_port=12345
```

```python
# train.py
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--num_nodes", type=int, default=1)
parser.add_argument("--node_rank", type=int, default=0)
parser.add_argument("--master_addr", type=str, default="localhost")
parser.add_argument("--master_port", type=int, default=12345)
args = parser.parse_args()

trainer = L.Trainer(
    strategy="ddp",
    accelerator="gpu",
    devices=4,
    num_nodes=args.num_nodes
)
```

## 常見模式

### DDP 的梯度累積

```python
# 模擬更大的批次大小
trainer = L.Trainer(
    strategy="ddp",
    accelerator="gpu",
    devices=4,
    accumulate_grad_batches=4  # 有效批次大小 = batch_size * devices * 4
)
```

### 分散式訓練的模型檢查點

```python
from lightning.pytorch.callbacks import ModelCheckpoint

checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    save_top_k=3,
    mode="min"
)

trainer = L.Trainer(
    strategy="ddp",
    accelerator="gpu",
    devices=4,
    callbacks=[checkpoint_callback]
)
```

### 分散式訓練的可重現性

```python
import lightning as L

L.seed_everything(42, workers=True)

trainer = L.Trainer(
    strategy="ddp",
    accelerator="gpu",
    devices=4,
    deterministic=True
)
```

## 疑難排解

### NCCL 逾時

增加慢速網路的逾時時間：

```python
import os
os.environ["NCCL_TIMEOUT"] = "3600"  # 1 小時

trainer = L.Trainer(strategy="ddp", accelerator="gpu", devices=4)
```

### CUDA 記憶體不足

解決方案：
1. 啟用梯度檢查點
2. 減少批次大小
3. 使用 FSDP 或 DeepSpeed
4. 啟用 CPU 卸載
5. 使用混合精度

```python
# 選項 1：梯度檢查點
class MyModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = MyTransformer()
        self.model.gradient_checkpointing_enable()

# 選項 2：較小的批次大小
dm = MyDataModule(batch_size=16)  # 從 32 減少

# 選項 3：帶卸載的 FSDP
trainer = L.Trainer(
    strategy=FSDPStrategy(cpu_offload=True),
    precision="bf16-mixed"
)

# 選項 4：梯度累積
trainer = L.Trainer(accumulate_grad_batches=4)
```

### 分散式採樣器問題

Lightning 自動處理 DistributedSampler：

```python
# 不要這樣做
from torch.utils.data import DistributedSampler
sampler = DistributedSampler(dataset)  # Lightning 自動執行此操作

# 只需使用 shuffle
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
```

### 通訊開銷

透過較大的 `find_unused_parameters` 減少通訊：

```python
trainer = L.Trainer(
    strategy=DDPStrategy(find_unused_parameters=False),
    accelerator="gpu",
    devices=4
)
```

## 最佳實務

### 1. 從單一 GPU 開始
擴展前先在單一 GPU 上測試程式碼：

```python
# 在單一 GPU 上除錯
trainer = L.Trainer(accelerator="gpu", devices=1, fast_dev_run=True)

# 然後擴展到多 GPU
trainer = L.Trainer(accelerator="gpu", devices=4, strategy="ddp")
```

### 2. 使用適當的策略
- <500M 參數：使用 DDP
- 500M-10B 參數：使用 FSDP
- >10B 參數：使用 DeepSpeed Stage 3

### 3. 啟用混合精度
現代 GPU 始終使用混合精度：

```python
trainer = L.Trainer(precision="bf16-mixed")  # A100、H100
trainer = L.Trainer(precision="16-mixed")    # V100、T4
```

### 4. 縮放超參數
擴展時調整學習率和批次大小：

```python
# 線性縮放規則
lr = base_lr * num_gpus
```

### 5. 同步指標
在分散式訓練中始終同步指標：

```python
self.log("val_loss", loss, sync_dist=True)
```

### 6. 使用 Rank-Zero 操作
檔案 I/O 和昂貴操作僅在主程序執行：

```python
from lightning.pytorch.utilities import rank_zero_only

@rank_zero_only
def save_predictions(self):
    torch.save(self.predictions, "predictions.pt")
```

### 7. 定期保存檢查點
儲存檢查點以便從故障恢復：

```python
checkpoint_callback = ModelCheckpoint(
    save_top_k=3,
    save_last=True,  # 始終儲存最後一個以便恢復
    every_n_epochs=5
)
```
