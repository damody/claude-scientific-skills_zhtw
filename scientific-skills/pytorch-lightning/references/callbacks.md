# 回呼函數 - 完整指南

## 概述

回呼函數（Callbacks）可以在訓練中新增任意獨立程式，而不會使 LightningModule 研究程式碼變得雜亂。它們在訓練生命週期的特定鉤子處執行自訂邏輯。

## 架構

Lightning 將訓練邏輯組織為三個元件：
- **Trainer** - 工程基礎設施
- **LightningModule** - 研究程式碼
- **Callbacks** - 非必要功能（監控、檢查點、自訂行為）

## 建立自訂回呼函數

基本結構：

```python
from lightning.pytorch.callbacks import Callback

class MyCustomCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        print("Training is starting!")

    def on_train_end(self, trainer, pl_module):
        print("Training is done!")

# 與 Trainer 一起使用
trainer = L.Trainer(callbacks=[MyCustomCallback()])
```

## 內建回呼函數

### ModelCheckpoint

根據監控的指標儲存模型。

**主要參數：**
- `dirpath` - 儲存檢查點的目錄
- `filename` - 檢查點檔名模式
- `monitor` - 要監控的指標
- `mode` - 監控指標的 "min" 或 "max"
- `save_top_k` - 保留的最佳模型數量
- `save_last` - 儲存最後一個 epoch 的檢查點
- `every_n_epochs` - 每 N 個 epochs 儲存
- `save_on_train_epoch_end` - 在訓練 epoch 結束時儲存 vs 驗證結束時

**範例：**
```python
from lightning.pytorch.callbacks import ModelCheckpoint

# 根據驗證損失儲存最佳的 3 個模型
checkpoint_callback = ModelCheckpoint(
    dirpath="checkpoints/",
    filename="model-{epoch:02d}-{val_loss:.2f}",
    monitor="val_loss",
    mode="min",
    save_top_k=3,
    save_last=True
)

# 每 10 個 epochs 儲存
checkpoint_callback = ModelCheckpoint(
    dirpath="checkpoints/",
    filename="model-{epoch:02d}",
    every_n_epochs=10,
    save_top_k=-1  # 全部儲存
)

# 根據準確率儲存最佳模型
checkpoint_callback = ModelCheckpoint(
    dirpath="checkpoints/",
    filename="best-model",
    monitor="val_acc",
    mode="max",
    save_top_k=1
)

trainer = L.Trainer(callbacks=[checkpoint_callback])
```

**存取已儲存的檢查點：**
```python
# 取得最佳模型路徑
best_model_path = checkpoint_callback.best_model_path

# 取得最後一個檢查點路徑
last_checkpoint = checkpoint_callback.last_model_path

# 取得所有檢查點路徑
all_checkpoints = checkpoint_callback.best_k_models
```

### EarlyStopping

當監控的指標停止改善時停止訓練。

**主要參數：**
- `monitor` - 要監控的指標
- `patience` - 沒有改善後停止訓練的 epochs 數
- `mode` - 監控指標的 "min" 或 "max"
- `min_delta` - 認定為改善的最小變化量
- `verbose` - 列印訊息
- `strict` - 如果找不到監控指標則崩潰

**範例：**
```python
from lightning.pytorch.callbacks import EarlyStopping

# 當驗證損失停止改善時停止
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=10,
    mode="min",
    verbose=True
)

# 當準確率停滯時停止
early_stop = EarlyStopping(
    monitor="val_acc",
    patience=5,
    mode="max",
    min_delta=0.001  # 必須至少改善 0.001
)

trainer = L.Trainer(callbacks=[early_stop])
```

### LearningRateMonitor

追蹤來自調度器的學習率變化。

**主要參數：**
- `logging_interval` - 何時記錄："step" 或 "epoch"
- `log_momentum` - 同時記錄動量值

**範例：**
```python
from lightning.pytorch.callbacks import LearningRateMonitor

lr_monitor = LearningRateMonitor(logging_interval="step")
trainer = L.Trainer(callbacks=[lr_monitor])

# 自動記錄學習率為 "lr-{optimizer_name}"
```

### DeviceStatsMonitor

記錄裝置效能指標（GPU/CPU/TPU）。

**主要參數：**
- `cpu_stats` - 記錄 CPU 統計資訊

**範例：**
```python
from lightning.pytorch.callbacks import DeviceStatsMonitor

device_stats = DeviceStatsMonitor(cpu_stats=True)
trainer = L.Trainer(callbacks=[device_stats])

# 記錄：gpu_utilization、gpu_memory_usage 等
```

### ModelSummary / RichModelSummary

顯示模型架構和參數數量。

**範例：**
```python
from lightning.pytorch.callbacks import ModelSummary, RichModelSummary

# 基本摘要
summary = ModelSummary(max_depth=2)

# 豐富格式的摘要（更漂亮）
rich_summary = RichModelSummary(max_depth=3)

trainer = L.Trainer(callbacks=[rich_summary])
```

### Timer

追蹤和限制訓練時間。

**主要參數：**
- `duration` - 最大訓練時間（timedelta 或 dict）
- `interval` - 檢查間隔："step"、"epoch" 或 "batch"

**範例：**
```python
from lightning.pytorch.callbacks import Timer
from datetime import timedelta

# 限制訓練為 1 小時
timer = Timer(duration=timedelta(hours=1))

# 或使用字典
timer = Timer(duration={"hours": 23, "minutes": 30})

trainer = L.Trainer(callbacks=[timer])
```

### BatchSizeFinder

自動找到最佳批次大小。

**範例：**
```python
from lightning.pytorch.callbacks import BatchSizeFinder

batch_finder = BatchSizeFinder(mode="power", steps_per_trial=3)

trainer = L.Trainer(callbacks=[batch_finder])
trainer.fit(model, datamodule=dm)

# 最佳批次大小會自動設定
```

### GradientAccumulationScheduler

動態排程梯度累積步驟。

**範例：**
```python
from lightning.pytorch.callbacks import GradientAccumulationScheduler

# 前 5 個 epochs 累積 4 個批次，然後 2 個批次
accumulator = GradientAccumulationScheduler(scheduling={0: 4, 5: 2})

trainer = L.Trainer(callbacks=[accumulator])
```

### StochasticWeightAveraging (SWA)

應用隨機權重平均以獲得更好的泛化。

**範例：**
```python
from lightning.pytorch.callbacks import StochasticWeightAveraging

swa = StochasticWeightAveraging(swa_lrs=1e-2, swa_epoch_start=0.8)

trainer = L.Trainer(callbacks=[swa])
```

## 自訂回呼函數範例

### 簡單日誌記錄回呼函數

```python
class MetricsLogger(Callback):
    def __init__(self):
        self.metrics = []

    def on_validation_end(self, trainer, pl_module):
        # 存取已記錄的指標
        metrics = trainer.callback_metrics
        self.metrics.append(dict(metrics))
        print(f"Validation metrics: {metrics}")
```

### 梯度監控回呼函數

```python
class GradientMonitor(Callback):
    def on_after_backward(self, trainer, pl_module):
        # 記錄梯度範數
        for name, param in pl_module.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                pl_module.log(f"grad_norm/{name}", grad_norm)
```

### 自訂檢查點回呼函數

```python
class CustomCheckpoint(Callback):
    def __init__(self, save_dir):
        self.save_dir = save_dir

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        if epoch % 5 == 0:  # 每 5 個 epochs 儲存
            filepath = f"{self.save_dir}/custom-{epoch}.ckpt"
            trainer.save_checkpoint(filepath)
            print(f"Saved checkpoint: {filepath}")
```

### 模型凍結回呼函數

```python
class FreezeUnfreeze(Callback):
    def __init__(self, freeze_until_epoch=10):
        self.freeze_until_epoch = freeze_until_epoch

    def on_train_epoch_start(self, trainer, pl_module):
        epoch = trainer.current_epoch

        if epoch < self.freeze_until_epoch:
            # 凍結主幹網路
            for param in pl_module.backbone.parameters():
                param.requires_grad = False
        else:
            # 解凍主幹網路
            for param in pl_module.backbone.parameters():
                param.requires_grad = True
```

### 學習率查找器回呼函數

```python
class LRFinder(Callback):
    def __init__(self, min_lr=1e-5, max_lr=1e-1, num_steps=100):
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.num_steps = num_steps
        self.lrs = []
        self.losses = []

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx >= self.num_steps:
            trainer.should_stop = True
            return

        # 指數學習率排程
        lr = self.min_lr * (self.max_lr / self.min_lr) ** (batch_idx / self.num_steps)
        optimizer = trainer.optimizers[0]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        self.lrs.append(lr)
        self.losses.append(outputs['loss'].item())

    def on_train_end(self, trainer, pl_module):
        # 繪製 LR vs Loss
        import matplotlib.pyplot as plt
        plt.plot(self.lrs, self.losses)
        plt.xscale('log')
        plt.xlabel('Learning Rate')
        plt.ylabel('Loss')
        plt.savefig('lr_finder.png')
```

### 預測儲存回呼函數

```python
class PredictionSaver(Callback):
    def __init__(self, save_path):
        self.save_path = save_path
        self.predictions = []

    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.predictions.append(outputs)

    def on_predict_end(self, trainer, pl_module):
        # 儲存所有預測
        torch.save(self.predictions, self.save_path)
        print(f"Predictions saved to {self.save_path}")
```

## 可用的鉤子

### 設定和清理
- `setup(trainer, pl_module, stage)` - 在 fit/test/predict 開始時呼叫
- `teardown(trainer, pl_module, stage)` - 在 fit/test/predict 結束時呼叫

### 訓練生命週期
- `on_fit_start(trainer, pl_module)` - 在 fit 開始時呼叫
- `on_fit_end(trainer, pl_module)` - 在 fit 結束時呼叫
- `on_train_start(trainer, pl_module)` - 在訓練開始時呼叫
- `on_train_end(trainer, pl_module)` - 在訓練結束時呼叫

### Epoch 邊界
- `on_train_epoch_start(trainer, pl_module)` - 在訓練 epoch 開始時呼叫
- `on_train_epoch_end(trainer, pl_module)` - 在訓練 epoch 結束時呼叫
- `on_validation_epoch_start(trainer, pl_module)` - 在驗證開始時呼叫
- `on_validation_epoch_end(trainer, pl_module)` - 在驗證結束時呼叫
- `on_test_epoch_start(trainer, pl_module)` - 在測試開始時呼叫
- `on_test_epoch_end(trainer, pl_module)` - 在測試結束時呼叫

### 批次邊界
- `on_train_batch_start(trainer, pl_module, batch, batch_idx)` - 訓練批次之前
- `on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)` - 訓練批次之後
- `on_validation_batch_start(trainer, pl_module, batch, batch_idx)` - 驗證批次之前
- `on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx)` - 驗證批次之後

### 梯度事件
- `on_before_backward(trainer, pl_module, loss)` - 在 loss.backward() 之前
- `on_after_backward(trainer, pl_module)` - 在 loss.backward() 之後
- `on_before_optimizer_step(trainer, pl_module, optimizer)` - 在 optimizer.step() 之前

### 檢查點事件
- `on_save_checkpoint(trainer, pl_module, checkpoint)` - 儲存檢查點時
- `on_load_checkpoint(trainer, pl_module, checkpoint)` - 載入檢查點時

### 異常處理
- `on_exception(trainer, pl_module, exception)` - 發生異常時

## 狀態管理

對於需要跨檢查點持久化的回呼函數：

```python
class StatefulCallback(Callback):
    def __init__(self):
        self.counter = 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.counter += 1

    def state_dict(self):
        return {"counter": self.counter}

    def load_state_dict(self, state_dict):
        self.counter = state_dict["counter"]

    @property
    def state_key(self):
        # 此回呼函數的唯一識別碼
        return "my_stateful_callback"
```

## 最佳實務

### 1. 保持回呼函數隔離
每個回呼函數應該是自包含和獨立的：

```python
# 好：自包含
class MyCallback(Callback):
    def __init__(self):
        self.data = []

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.data.append(outputs['loss'].item())

# 不好：依賴外部狀態
global_data = []

class BadCallback(Callback):
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        global_data.append(outputs['loss'].item())  # 外部依賴
```

### 2. 避免回呼函數間的依賴
回呼函數不應該依賴其他回呼函數：

```python
# 不好：Callback B 依賴 Callback A
class CallbackA(Callback):
    def __init__(self):
        self.value = 0

class CallbackB(Callback):
    def __init__(self, callback_a):
        self.callback_a = callback_a  # 緊密耦合

# 好：獨立的回呼函數
class CallbackA(Callback):
    def __init__(self):
        self.value = 0

class CallbackB(Callback):
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # 改為存取 trainer 狀態
        value = trainer.callback_metrics.get('metric')
```

### 3. 永遠不要手動呼叫回呼函數方法
讓 Lightning 自動呼叫回呼函數：

```python
# 不好：手動呼叫
callback = MyCallback()
callback.on_train_start(trainer, model)  # 不要這樣做

# 好：讓 Trainer 處理
trainer = L.Trainer(callbacks=[MyCallback()])
```

### 4. 設計為任意執行順序
回呼函數可能以任意順序執行，所以不要依賴特定順序：

```python
# 好：順序獨立
class GoodCallback(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        # 使用 trainer 狀態，而非其他回呼函數
        metrics = trainer.callback_metrics
        self.log_metrics(metrics)
```

### 5. 將回呼函數用於非必要邏輯
將核心研究程式碼保留在 LightningModule 中，使用回呼函數處理輔助功能：

```python
# 好的分離
class MyModel(L.LightningModule):
    # 核心研究邏輯在這裡
    def training_step(self, batch, batch_idx):
        return loss

# 非必要的監控在回呼函數中
class MonitorCallback(Callback):
    def on_validation_end(self, trainer, pl_module):
        # 監控邏輯
        pass
```

## 常見模式

### 組合多個回呼函數

```python
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
    DeviceStatsMonitor
)

callbacks = [
    ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=3),
    EarlyStopping(monitor="val_loss", patience=10, mode="min"),
    LearningRateMonitor(logging_interval="step"),
    DeviceStatsMonitor()
]

trainer = L.Trainer(callbacks=callbacks)
```

### 條件式回呼函數啟動

```python
class ConditionalCallback(Callback):
    def __init__(self, activate_after_epoch=10):
        self.activate_after_epoch = activate_after_epoch

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch >= self.activate_after_epoch:
            # 僅在指定 epoch 後啟動
            self.do_something(trainer, pl_module)
```

### 多階段訓練回呼函數

```python
class MultiStageTraining(Callback):
    def __init__(self, stage_epochs=[10, 20, 30]):
        self.stage_epochs = stage_epochs
        self.current_stage = 0

    def on_train_epoch_start(self, trainer, pl_module):
        epoch = trainer.current_epoch

        if epoch in self.stage_epochs:
            self.current_stage += 1
            print(f"Entering stage {self.current_stage}")

            # 為新階段調整學習率
            for optimizer in trainer.optimizers:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.1
```
