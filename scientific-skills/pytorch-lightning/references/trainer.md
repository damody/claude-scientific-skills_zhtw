# Trainer - 完整指南

## 概述

Trainer 在將 PyTorch 程式碼組織到 LightningModule 後自動化訓練工作流程。它自動處理迴圈細節、裝置管理、回呼函數、梯度操作、檢查點和分散式訓練。

## 核心用途

Trainer 管理：
- 自動啟用/停用梯度
- 執行訓練、驗證和測試 dataloaders
- 在適當時機呼叫回呼函數
- 將批次放置到正確的裝置
- 協調分散式訓練
- 進度條和日誌記錄
- 檢查點和提前停止

## 主要方法

### `fit(model, train_dataloaders=None, val_dataloaders=None, datamodule=None)`
執行完整的訓練流程，包括可選的驗證。

**參數：**
- `model` - 要訓練的 LightningModule
- `train_dataloaders` - 訓練 DataLoader(s)
- `val_dataloaders` - 可選的驗證 DataLoader(s)
- `datamodule` - 可選的 LightningDataModule（取代 dataloaders）

**範例：**
```python
# 使用 DataLoaders
trainer = L.Trainer(max_epochs=10)
trainer.fit(model, train_loader, val_loader)

# 使用 DataModule
trainer.fit(model, datamodule=dm)

# 從檢查點繼續訓練
trainer.fit(model, train_loader, ckpt_path="checkpoint.ckpt")
```

### `validate(model=None, dataloaders=None, datamodule=None)`
執行驗證迴圈而不進行訓練。

**範例：**
```python
trainer = L.Trainer()
trainer.validate(model, val_loader)
```

### `test(model=None, dataloaders=None, datamodule=None)`
執行測試迴圈。僅在發布結果前使用。

**範例：**
```python
trainer = L.Trainer()
trainer.test(model, test_loader)
```

### `predict(model=None, dataloaders=None, datamodule=None)`
對資料執行推論並回傳預測。

**範例：**
```python
trainer = L.Trainer()
predictions = trainer.predict(model, predict_loader)
```

## 基本參數

### 訓練時長

#### `max_epochs`（int）
訓練的最大 epochs 數。預設：1000

```python
trainer = L.Trainer(max_epochs=100)
```

#### `min_epochs`（int）
訓練的最小 epochs 數。預設：None

```python
trainer = L.Trainer(min_epochs=10, max_epochs=100)
```

#### `max_steps`（int）
最大優化器步驟數。覆蓋 max_epochs。預設：-1（無限制）

```python
trainer = L.Trainer(max_steps=10000)
```

#### `max_time`（str 或 dict）
最大訓練時間。適用於有時間限制的叢集。

```python
# 字串格式
trainer = L.Trainer(max_time="00:12:00:00")  # 12 小時

# 字典格式
trainer = L.Trainer(max_time={"days": 1, "hours": 6})
```

### 硬體配置

#### `accelerator`（str 或 Accelerator）
要使用的硬體："cpu"、"gpu"、"tpu"、"ipu"、"hpu"、"mps" 或 "auto"。預設："auto"

```python
trainer = L.Trainer(accelerator="gpu")
trainer = L.Trainer(accelerator="auto")  # 自動偵測可用硬體
```

#### `devices`（int、list 或 str）
要使用的裝置數量或裝置索引列表。

```python
# 使用 2 個 GPU
trainer = L.Trainer(devices=2, accelerator="gpu")

# 使用特定 GPU
trainer = L.Trainer(devices=[0, 2], accelerator="gpu")

# 使用所有可用裝置
trainer = L.Trainer(devices="auto", accelerator="gpu")

# 使用 4 個 CPU 核心
trainer = L.Trainer(devices=4, accelerator="cpu")
```

#### `strategy`（str 或 Strategy）
分散式訓練策略："ddp"、"ddp_spawn"、"fsdp"、"deepspeed" 等。預設："auto"

```python
# 資料分散式平行
trainer = L.Trainer(strategy="ddp", accelerator="gpu", devices=4)

# 完全分片資料平行
trainer = L.Trainer(strategy="fsdp", accelerator="gpu", devices=4)

# DeepSpeed
trainer = L.Trainer(strategy="deepspeed_stage_2", accelerator="gpu", devices=4)
```

#### `precision`（str 或 int）
浮點精度："32-true"、"16-mixed"、"bf16-mixed"、"64-true" 等。

```python
# 混合精度（FP16）
trainer = L.Trainer(precision="16-mixed")

# BFloat16 混合精度
trainer = L.Trainer(precision="bf16-mixed")

# 全精度
trainer = L.Trainer(precision="32-true")
```

### 優化配置

#### `gradient_clip_val`（float）
梯度裁剪值。預設：None

```python
# 按範數裁剪梯度
trainer = L.Trainer(gradient_clip_val=0.5)
```

#### `gradient_clip_algorithm`（str）
梯度裁剪演算法："norm" 或 "value"。預設："norm"

```python
trainer = L.Trainer(gradient_clip_val=0.5, gradient_clip_algorithm="norm")
```

#### `accumulate_grad_batches`（int 或 dict）
在優化器步驟前累積 N 個批次的梯度。

```python
# 累積 4 個批次
trainer = L.Trainer(accumulate_grad_batches=4)

# 每個 epoch 不同的累積
trainer = L.Trainer(accumulate_grad_batches={0: 4, 5: 2, 10: 1})
```

### 驗證配置

#### `check_val_every_n_epoch`（int）
每 N 個 epochs 執行驗證。預設：1

```python
trainer = L.Trainer(check_val_every_n_epoch=10)
```

#### `val_check_interval`（int 或 float）
在一個訓練 epoch 內多久檢查一次驗證。

```python
# 每 0.25 個訓練 epoch 檢查驗證
trainer = L.Trainer(val_check_interval=0.25)

# 每 100 個訓練批次檢查驗證
trainer = L.Trainer(val_check_interval=100)
```

#### `limit_val_batches`（int 或 float）
限制驗證批次。

```python
# 僅使用 10% 的驗證資料
trainer = L.Trainer(limit_val_batches=0.1)

# 僅使用 50 個驗證批次
trainer = L.Trainer(limit_val_batches=50)

# 停用驗證
trainer = L.Trainer(limit_val_batches=0)
```

#### `num_sanity_val_steps`（int）
訓練開始前執行的驗證批次數。預設：2

```python
# 跳過健全性檢查
trainer = L.Trainer(num_sanity_val_steps=0)

# 執行 5 個健全性驗證步驟
trainer = L.Trainer(num_sanity_val_steps=5)
```

### 日誌記錄和進度

#### `logger`（Logger 或 list 或 bool）
用於實驗追蹤的日誌器。

```python
from lightning.pytorch import loggers as pl_loggers

# TensorBoard 日誌器
tb_logger = pl_loggers.TensorBoardLogger("logs/")
trainer = L.Trainer(logger=tb_logger)

# 多個日誌器
wandb_logger = pl_loggers.WandbLogger(project="my-project")
trainer = L.Trainer(logger=[tb_logger, wandb_logger])

# 停用日誌記錄
trainer = L.Trainer(logger=False)
```

#### `log_every_n_steps`（int）
在訓練步驟內多久記錄一次。預設：50

```python
trainer = L.Trainer(log_every_n_steps=10)
```

#### `enable_progress_bar`（bool）
顯示進度條。預設：True

```python
trainer = L.Trainer(enable_progress_bar=False)
```

### 回呼函數

#### `callbacks`（list）
訓練期間使用的回呼函數列表。

```python
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    save_top_k=3,
    mode="min"
)

early_stop_callback = EarlyStopping(
    monitor="val_loss",
    patience=5,
    mode="min"
)

trainer = L.Trainer(callbacks=[checkpoint_callback, early_stop_callback])
```

### 檢查點

#### `default_root_dir`（str）
日誌和檢查點的預設目錄。預設：當前工作目錄

```python
trainer = L.Trainer(default_root_dir="./experiments/")
```

#### `enable_checkpointing`（bool）
啟用自動檢查點。預設：True

```python
trainer = L.Trainer(enable_checkpointing=True)
```

### 除錯

#### `fast_dev_run`（bool 或 int）
透過訓練/驗證/測試執行單一批次（或 N 個批次）進行除錯。

```python
# 執行 1 個訓練/驗證/測試批次
trainer = L.Trainer(fast_dev_run=True)

# 執行 5 個訓練/驗證/測試批次
trainer = L.Trainer(fast_dev_run=5)
```

#### `limit_train_batches`（int 或 float）
限制訓練批次。

```python
# 僅使用 25% 的訓練資料
trainer = L.Trainer(limit_train_batches=0.25)

# 僅使用 100 個訓練批次
trainer = L.Trainer(limit_train_batches=100)
```

#### `limit_test_batches`（int 或 float）
限制測試批次。

```python
trainer = L.Trainer(limit_test_batches=0.5)
```

#### `overfit_batches`（int 或 float）
在資料子集上過擬合進行除錯。

```python
# 在 10 個批次上過擬合
trainer = L.Trainer(overfit_batches=10)

# 在 1% 的資料上過擬合
trainer = L.Trainer(overfit_batches=0.01)
```

#### `detect_anomaly`（bool）
啟用 PyTorch 異常檢測以除錯 NaN。預設：False

```python
trainer = L.Trainer(detect_anomaly=True)
```

### 可重現性

#### `deterministic`（bool 或 str）
控制確定性行為。預設：False

```python
import lightning as L

# 設定所有種子
L.seed_everything(42, workers=True)

# 完全確定性（可能影響效能）
trainer = L.Trainer(deterministic=True)

# 檢測到非確定性操作時警告
trainer = L.Trainer(deterministic="warn")
```

#### `benchmark`（bool）
啟用 cudnn 基準測試以提高效能。預設：False

```python
trainer = L.Trainer(benchmark=True)
```

### 其他

#### `enable_model_summary`（bool）
訓練前列印模型摘要。預設：True

```python
trainer = L.Trainer(enable_model_summary=False)
```

#### `inference_mode`（bool）
在驗證/測試中使用 torch.inference_mode() 而非 torch.no_grad()。預設：True

```python
trainer = L.Trainer(inference_mode=True)
```

#### `profiler`（str 或 Profiler）
分析程式碼以優化效能。選項："simple"、"advanced" 或自訂 Profiler。

```python
# 簡單分析器
trainer = L.Trainer(profiler="simple")

# 進階分析器
trainer = L.Trainer(profiler="advanced")
```

## 常見配置

### 基本訓練
```python
trainer = L.Trainer(
    max_epochs=100,
    accelerator="auto",
    devices="auto"
)
trainer.fit(model, train_loader, val_loader)
```

### 多 GPU 訓練
```python
trainer = L.Trainer(
    max_epochs=100,
    accelerator="gpu",
    devices=4,
    strategy="ddp",
    precision="16-mixed"
)
trainer.fit(model, datamodule=dm)
```

### 帶檢查點的生產訓練
```python
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

checkpoint_callback = ModelCheckpoint(
    dirpath="checkpoints/",
    filename="{epoch}-{val_loss:.2f}",
    monitor="val_loss",
    mode="min",
    save_top_k=3,
    save_last=True
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=10,
    mode="min"
)

lr_monitor = LearningRateMonitor(logging_interval="step")

trainer = L.Trainer(
    max_epochs=100,
    accelerator="gpu",
    devices=2,
    strategy="ddp",
    precision="16-mixed",
    callbacks=[checkpoint_callback, early_stop, lr_monitor],
    log_every_n_steps=10,
    gradient_clip_val=1.0
)

trainer.fit(model, datamodule=dm)
```

### 除錯配置
```python
trainer = L.Trainer(
    fast_dev_run=True,          # 執行 1 個批次
    accelerator="cpu",
    enable_progress_bar=True,
    log_every_n_steps=1,
    detect_anomaly=True
)
trainer.fit(model, train_loader, val_loader)
```

### 研究配置（可重現性）
```python
import lightning as L

L.seed_everything(42, workers=True)

trainer = L.Trainer(
    max_epochs=100,
    accelerator="gpu",
    devices=1,
    deterministic=True,
    benchmark=False,
    precision="32-true"
)
trainer.fit(model, datamodule=dm)
```

### 有時間限制的訓練（叢集）
```python
trainer = L.Trainer(
    max_time={"hours": 23, "minutes": 30},  # SLURM 時間限制
    max_epochs=1000,
    callbacks=[ModelCheckpoint(save_last=True)]
)
trainer.fit(model, datamodule=dm)

# 從檢查點恢復
trainer.fit(model, datamodule=dm, ckpt_path="last.ckpt")
```

### 大型模型訓練（FSDP）
```python
from lightning.pytorch.strategies import FSDPStrategy

trainer = L.Trainer(
    max_epochs=100,
    accelerator="gpu",
    devices=8,
    strategy=FSDPStrategy(
        activation_checkpointing_policy={nn.TransformerEncoderLayer},
        cpu_offload=False
    ),
    precision="bf16-mixed",
    accumulate_grad_batches=4
)
trainer.fit(model, datamodule=dm)
```

## 恢復訓練

### 從檢查點
```python
# 從特定檢查點恢復
trainer.fit(model, datamodule=dm, ckpt_path="epoch=10-val_loss=0.23.ckpt")

# 從最後一個檢查點恢復
trainer.fit(model, datamodule=dm, ckpt_path="last.ckpt")
```

### 找到最後一個檢查點
```python
from lightning.pytorch.callbacks import ModelCheckpoint

checkpoint_callback = ModelCheckpoint(save_last=True)
trainer = L.Trainer(callbacks=[checkpoint_callback])
trainer.fit(model, datamodule=dm)

# 取得最後一個檢查點的路徑
last_checkpoint = checkpoint_callback.last_model_path
```

## 從 LightningModule 存取 Trainer

在 LightningModule 內部，透過 `self.trainer` 存取 Trainer：

```python
class MyModel(L.LightningModule):
    def training_step(self, batch, batch_idx):
        # 存取 trainer 屬性
        current_epoch = self.trainer.current_epoch
        global_step = self.trainer.global_step
        max_epochs = self.trainer.max_epochs

        # 存取回呼函數
        for callback in self.trainer.callbacks:
            if isinstance(callback, ModelCheckpoint):
                print(f"Best model: {callback.best_model_path}")

        # 存取日誌器
        self.trainer.logger.log_metrics({"custom": value})
```

## Trainer 屬性

| 屬性 | 說明 |
|-----------|-------------|
| `trainer.current_epoch` | 當前 epoch（0 索引） |
| `trainer.global_step` | 總優化器步驟 |
| `trainer.max_epochs` | 配置的最大 epochs |
| `trainer.max_steps` | 配置的最大步驟 |
| `trainer.callbacks` | 回呼函數列表 |
| `trainer.logger` | 日誌器實例 |
| `trainer.strategy` | 訓練策略 |
| `trainer.estimated_stepping_batches` | 估計的訓練總步驟 |

## 最佳實務

### 1. 從 Fast Dev Run 開始
完整訓練前始終使用 `fast_dev_run=True` 測試：

```python
trainer = L.Trainer(fast_dev_run=True)
trainer.fit(model, datamodule=dm)
```

### 2. 使用梯度裁剪
防止梯度爆炸：

```python
trainer = L.Trainer(gradient_clip_val=1.0, gradient_clip_algorithm="norm")
```

### 3. 啟用混合精度
在現代 GPU 上加速訓練：

```python
trainer = L.Trainer(precision="16-mixed")  # 或 A100+ 使用 "bf16-mixed"
```

### 4. 正確儲存檢查點
始終儲存最後一個檢查點以便恢復：

```python
checkpoint_callback = ModelCheckpoint(
    save_top_k=3,
    save_last=True,
    monitor="val_loss"
)
```

### 5. 監控學習率
使用 LearningRateMonitor 追蹤 LR 變化：

```python
from lightning.pytorch.callbacks import LearningRateMonitor

trainer = L.Trainer(callbacks=[LearningRateMonitor(logging_interval="step")])
```

### 6. 使用 DataModule 以確保可重現性
將資料邏輯封裝在 DataModule 中：

```python
# 比直接傳遞 DataLoaders 更好
trainer.fit(model, datamodule=dm)
```

### 7. 為研究設定確定性
確保發表論文的可重現性：

```python
L.seed_everything(42, workers=True)
trainer = L.Trainer(deterministic=True)
```
