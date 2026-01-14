# 日誌記錄 - 完整指南

## 概述

PyTorch Lightning 支援多種日誌整合，用於實驗追蹤和視覺化。預設情況下，Lightning 使用 TensorBoard，但你可以輕鬆切換或組合多個日誌器。

## 支援的日誌器

### TensorBoardLogger（預設）

記錄到本地或遠端檔案系統，使用 TensorBoard 格式。

**安裝：**
```bash
pip install tensorboard
```

**用法：**
```python
from lightning.pytorch import loggers as pl_loggers

tb_logger = pl_loggers.TensorBoardLogger(
    save_dir="logs/",
    name="my_model",
    version="version_1",
    default_hp_metric=False
)

trainer = L.Trainer(logger=tb_logger)
```

**檢視日誌：**
```bash
tensorboard --logdir logs/
```

### WandbLogger

Weights & Biases 整合，用於雲端實驗追蹤。

**安裝：**
```bash
pip install wandb
```

**用法：**
```python
from lightning.pytorch import loggers as pl_loggers

wandb_logger = pl_loggers.WandbLogger(
    project="my-project",
    name="experiment-1",
    save_dir="logs/",
    log_model=True  # 將模型檢查點記錄到 W&B
)

trainer = L.Trainer(logger=wandb_logger)
```

**功能：**
- 雲端實驗追蹤
- 模型版本控制
- 工件管理
- 協作功能
- 超參數掃描

### MLFlowLogger

MLflow 追蹤整合。

**安裝：**
```bash
pip install mlflow
```

**用法：**
```python
from lightning.pytorch import loggers as pl_loggers

mlflow_logger = pl_loggers.MLFlowLogger(
    experiment_name="my_experiment",
    tracking_uri="http://localhost:5000",
    run_name="run_1"
)

trainer = L.Trainer(logger=mlflow_logger)
```

### CometLogger

Comet.ml 實驗追蹤。

**安裝：**
```bash
pip install comet-ml
```

**用法：**
```python
from lightning.pytorch import loggers as pl_loggers

comet_logger = pl_loggers.CometLogger(
    api_key="YOUR_API_KEY",
    project_name="my-project",
    experiment_name="experiment-1"
)

trainer = L.Trainer(logger=comet_logger)
```

### NeptuneLogger

Neptune.ai 整合。

**安裝：**
```bash
pip install neptune
```

**用法：**
```python
from lightning.pytorch import loggers as pl_loggers

neptune_logger = pl_loggers.NeptuneLogger(
    api_key="YOUR_API_KEY",
    project="username/project-name",
    name="experiment-1"
)

trainer = L.Trainer(logger=neptune_logger)
```

### CSVLogger

記錄到本地檔案系統，使用 YAML 和 CSV 格式。

**用法：**
```python
from lightning.pytorch import loggers as pl_loggers

csv_logger = pl_loggers.CSVLogger(
    save_dir="logs/",
    name="my_model",
    version="1"
)

trainer = L.Trainer(logger=csv_logger)
```

**輸出檔案：**
- `metrics.csv` - 所有記錄的指標
- `hparams.yaml` - 超參數

## 記錄指標

### 基本記錄

在 LightningModule 中使用 `self.log()`：

```python
class MyModel(L.LightningModule):
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)

        # 記錄指標
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)
        acc = (y_hat.argmax(dim=1) == y).float().mean()

        # 記錄多個指標
        self.log("val_loss", loss)
        self.log("val_acc", acc)
```

### 記錄參數

#### `on_step`（bool）
在當前步驟記錄。預設：training_step 中為 True，其他為 False。

```python
self.log("loss", loss, on_step=True)
```

#### `on_epoch`（bool）
累積並在 epoch 結束時記錄。預設：training_step 中為 False，其他為 True。

```python
self.log("loss", loss, on_epoch=True)
```

#### `prog_bar`（bool）
在進度條中顯示。預設：False。

```python
self.log("train_loss", loss, prog_bar=True)
```

#### `logger`（bool）
發送到日誌後端。預設：True。

```python
self.log("internal_metric", value, logger=False)  # 不記錄到外部日誌器
```

#### `reduce_fx`（str 或 callable）
縮減函數："mean"、"sum"、"max"、"min"。預設："mean"。

```python
self.log("batch_size", batch.size(0), reduce_fx="sum")
```

#### `sync_dist`（bool）
在分散式訓練中跨裝置同步指標。預設：False。

```python
self.log("loss", loss, sync_dist=True)
```

#### `rank_zero_only`（bool）
僅從 rank 0 程序記錄。預設：False。

```python
self.log("debug_metric", value, rank_zero_only=True)
```

### 完整範例

```python
def training_step(self, batch, batch_idx):
    loss = self.compute_loss(batch)

    # 記錄每步和每 epoch，在進度條中顯示
    self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

    return loss

def validation_step(self, batch, batch_idx):
    loss = self.compute_loss(batch)
    acc = self.compute_accuracy(batch)

    # 記錄 epoch 級別的指標
    self.log("val_loss", loss, on_epoch=True)
    self.log("val_acc", acc, on_epoch=True, prog_bar=True)
```

### 記錄多個指標

使用 `log_dict()` 同時記錄多個指標：

```python
def training_step(self, batch, batch_idx):
    loss, acc, f1 = self.compute_metrics(batch)

    metrics = {
        "train_loss": loss,
        "train_acc": acc,
        "train_f1": f1
    }

    self.log_dict(metrics, on_step=True, on_epoch=True)

    return loss
```

## 記錄超參數

### 自動超參數記錄

在模型中使用 `save_hyperparameters()`：

```python
class MyModel(L.LightningModule):
    def __init__(self, learning_rate, hidden_dim, dropout):
        super().__init__()
        # 自動儲存和記錄超參數
        self.save_hyperparameters()
```

### 手動超參數記錄

```python
# 在 LightningModule 中
class MyModel(L.LightningModule):
    def __init__(self, learning_rate):
        super().__init__()
        self.save_hyperparameters()

# 或使用日誌器手動記錄
trainer.logger.log_hyperparams({
    "learning_rate": 0.001,
    "batch_size": 32
})
```

## 記錄頻率

預設情況下，Lightning 每 50 個訓練步驟記錄一次。使用 `log_every_n_steps` 調整：

```python
trainer = L.Trainer(log_every_n_steps=10)
```

## 多個日誌器

同時使用多個日誌器：

```python
from lightning.pytorch import loggers as pl_loggers

tb_logger = pl_loggers.TensorBoardLogger("logs/")
wandb_logger = pl_loggers.WandbLogger(project="my-project")
csv_logger = pl_loggers.CSVLogger("logs/")

trainer = L.Trainer(logger=[tb_logger, wandb_logger, csv_logger])
```

## 進階記錄

### 記錄影像

```python
import torchvision

def validation_step(self, batch, batch_idx):
    x, y = batch
    y_hat = self.model(x)

    # 每個 epoch 記錄第一個批次的影像一次
    if batch_idx == 0:
        # 建立影像網格
        grid = torchvision.utils.make_grid(x[:8])

        # 記錄到 TensorBoard
        self.logger.experiment.add_image("val_images", grid, self.current_epoch)

        # 記錄到 Wandb
        if isinstance(self.logger, pl_loggers.WandbLogger):
            import wandb
            self.logger.experiment.log({
                "val_images": [wandb.Image(img) for img in x[:8]]
            })
```

### 記錄直方圖

```python
def on_train_epoch_end(self):
    # 記錄參數直方圖
    for name, param in self.named_parameters():
        self.logger.experiment.add_histogram(name, param, self.current_epoch)

        if param.grad is not None:
            self.logger.experiment.add_histogram(
                f"{name}_grad", param.grad, self.current_epoch
            )
```

### 記錄模型圖

```python
def on_train_start(self):
    # 記錄模型架構
    sample_input = torch.randn(1, 3, 224, 224).to(self.device)
    self.logger.experiment.add_graph(self.model, sample_input)
```

### 記錄自訂圖表

```python
import matplotlib.pyplot as plt

def on_validation_epoch_end(self):
    # 建立自訂圖表
    fig, ax = plt.subplots()
    ax.plot(self.validation_losses)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")

    # 記錄到 TensorBoard
    self.logger.experiment.add_figure("loss_curve", fig, self.current_epoch)

    plt.close(fig)
```

### 記錄文字

```python
def validation_step(self, batch, batch_idx):
    # 生成預測
    predictions = self.generate_text(batch)

    # 記錄到 TensorBoard
    self.logger.experiment.add_text(
        "predictions",
        f"Batch {batch_idx}: {predictions}",
        self.current_epoch
    )
```

### 記錄音訊

```python
def validation_step(self, batch, batch_idx):
    audio = self.generate_audio(batch)

    # 記錄到 TensorBoard（音訊是形狀為 [1, samples] 的張量）
    self.logger.experiment.add_audio(
        "generated_audio",
        audio,
        self.current_epoch,
        sample_rate=22050
    )
```

## 在 LightningModule 中存取日誌器

```python
class MyModel(L.LightningModule):
    def training_step(self, batch, batch_idx):
        # 存取日誌器實驗物件
        logger = self.logger.experiment

        # 對於 TensorBoard
        if isinstance(self.logger, pl_loggers.TensorBoardLogger):
            logger.add_scalar("custom_metric", value, self.global_step)

        # 對於 Wandb
        if isinstance(self.logger, pl_loggers.WandbLogger):
            logger.log({"custom_metric": value})

        # 對於 MLflow
        if isinstance(self.logger, pl_loggers.MLFlowLogger):
            logger.log_metric("custom_metric", value)
```

## 自訂日誌器

透過繼承 `Logger` 建立自訂日誌器：

```python
from lightning.pytorch.loggers import Logger
from lightning.pytorch.utilities import rank_zero_only

class MyCustomLogger(Logger):
    def __init__(self, save_dir):
        super().__init__()
        self.save_dir = save_dir
        self._name = "my_logger"
        self._version = "0.1"

    @property
    def name(self):
        return self._name

    @property
    def version(self):
        return self._version

    @rank_zero_only
    def log_metrics(self, metrics, step):
        # 將指標記錄到你的後端
        print(f"Step {step}: {metrics}")

    @rank_zero_only
    def log_hyperparams(self, params):
        # 記錄超參數
        print(f"Hyperparameters: {params}")

    @rank_zero_only
    def save(self):
        # 儲存日誌器狀態
        pass

    @rank_zero_only
    def finalize(self, status):
        # 訓練結束時清理
        pass

# 用法
custom_logger = MyCustomLogger(save_dir="logs/")
trainer = L.Trainer(logger=custom_logger)
```

## 最佳實務

### 1. 同時記錄步驟和 Epoch 指標

```python
# 好：追蹤細粒度和聚合指標
self.log("train_loss", loss, on_step=True, on_epoch=True)
```

### 2. 對關鍵指標使用進度條

```python
# 在進度條中顯示重要指標
self.log("val_acc", acc, prog_bar=True)
```

### 3. 在分散式訓練中同步指標

```python
# 確保跨 GPU 的正確聚合
self.log("val_loss", loss, sync_dist=True)
```

### 4. 記錄學習率

```python
from lightning.pytorch.callbacks import LearningRateMonitor

trainer = L.Trainer(callbacks=[LearningRateMonitor(logging_interval="step")])
```

### 5. 記錄梯度範數

```python
def on_after_backward(self):
    # 監控梯度流
    grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=float('inf'))
    self.log("grad_norm", grad_norm)
```

### 6. 使用描述性指標名稱

```python
# 好：清晰的命名慣例
self.log("train/loss", loss)
self.log("train/accuracy", acc)
self.log("val/loss", val_loss)
self.log("val/accuracy", val_acc)
```

### 7. 記錄超參數

```python
# 始終儲存超參數以確保可重現性
class MyModel(L.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
```

### 8. 不要記錄太頻繁

```python
# 避免每步記錄昂貴的操作
if batch_idx % 100 == 0:
    self.log_images(batch)
```

## 常見模式

### 結構化記錄

```python
def training_step(self, batch, batch_idx):
    loss, metrics = self.compute_loss_and_metrics(batch)

    # 使用前綴組織日誌
    self.log("train/loss", loss)
    self.log_dict({f"train/{k}": v for k, v in metrics.items()})

    return loss

def validation_step(self, batch, batch_idx):
    loss, metrics = self.compute_loss_and_metrics(batch)

    self.log("val/loss", loss)
    self.log_dict({f"val/{k}": v for k, v in metrics.items()})
```

### 條件式記錄

```python
def training_step(self, batch, batch_idx):
    loss = self.compute_loss(batch)

    # 較少頻率記錄昂貴的指標
    if self.global_step % 100 == 0:
        expensive_metric = self.compute_expensive_metric(batch)
        self.log("expensive_metric", expensive_metric)

    self.log("train_loss", loss)
    return loss
```

### 多任務記錄

```python
def training_step(self, batch, batch_idx):
    x, y_task1, y_task2 = batch

    loss_task1 = self.compute_task1_loss(x, y_task1)
    loss_task2 = self.compute_task2_loss(x, y_task2)
    total_loss = loss_task1 + loss_task2

    # 記錄每個任務的指標
    self.log_dict({
        "train/loss_task1": loss_task1,
        "train/loss_task2": loss_task2,
        "train/loss_total": total_loss
    })

    return total_loss
```

## 疑難排解

### 找不到指標錯誤

如果調度器出現「找不到指標」錯誤：

```python
# 確保指標使用 logger=True 記錄
self.log("val_loss", loss, logger=True)

# 並配置調度器監控它
def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters())
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": scheduler,
            "monitor": "val_loss"  # 必須與記錄的指標名稱匹配
        }
    }
```

### 分散式訓練中指標不同步

```python
# 啟用 sync_dist 以正確聚合
self.log("val_acc", acc, sync_dist=True)
```

### 日誌器不儲存

```python
# 確保日誌器有寫入權限
trainer = L.Trainer(
    logger=pl_loggers.TensorBoardLogger("logs/"),
    default_root_dir="outputs/"  # 確保目錄存在且可寫
)
```
