# LightningModule - 完整指南

## 概述

`LightningModule` 將 PyTorch 程式碼組織成六個邏輯區段，無需抽象。程式碼仍然是純 PyTorch，只是組織得更好。Trainer 處理裝置管理、分散式採樣和基礎設施，同時保持對模型的完全控制。

## 核心結構

```python
import lightning as L
import torch
import torch.nn.functional as F

class MyModel(L.LightningModule):
    def __init__(self, learning_rate=0.001):
        super().__init__()
        self.save_hyperparameters()  # 儲存初始化參數
        self.model = YourNeuralNetwork()

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = F.cross_entropy(logits, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("val_loss", loss)
        self.log("val_acc", acc)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("test_loss", loss)
        self.log("test_acc", acc)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min')
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }
```

## 基本方法

### 訓練管線方法

#### `training_step(batch, batch_idx)`
計算前向傳播並回傳損失。在自動優化模式下，Lightning 自動處理反向傳播和優化器更新。

**參數：**
- `batch` - 來自 DataLoader 的當前訓練批次
- `batch_idx` - 當前批次的索引

**回傳：** 損失張量（純量）或包含 'loss' 鍵的字典

**範例：**
```python
def training_step(self, batch, batch_idx):
    x, y = batch
    y_hat = self.model(x)
    loss = F.mse_loss(y_hat, y)

    # 記錄訓練指標
    self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
    self.log("learning_rate", self.optimizers().param_groups[0]['lr'])

    return loss
```

#### `validation_step(batch, batch_idx)`
在驗證資料上評估模型。自動在禁用梯度和模型評估模式下執行。

**參數：**
- `batch` - 當前驗證批次
- `batch_idx` - 當前批次的索引

**回傳：** 可選 - 損失或指標字典

**範例：**
```python
def validation_step(self, batch, batch_idx):
    x, y = batch
    y_hat = self.model(x)
    loss = F.mse_loss(y_hat, y)

    # Lightning 自動跨驗證批次聚合
    self.log("val_loss", loss, prog_bar=True)
    return loss
```

#### `test_step(batch, batch_idx)`
在測試資料上評估模型。僅在明確呼叫 `trainer.test()` 時執行。通常在發布前訓練完成後使用。

**參數：**
- `batch` - 當前測試批次
- `batch_idx` - 當前批次的索引

**回傳：** 可選 - 損失或指標字典

#### `predict_step(batch, batch_idx, dataloader_idx=0)`
對資料進行推論。使用 `trainer.predict()` 時呼叫。

**參數：**
- `batch` - 當前批次
- `batch_idx` - 當前批次的索引
- `dataloader_idx` - dataloader 的索引（如果有多個）

**回傳：** 預測結果（任何你需要的格式）

**範例：**
```python
def predict_step(self, batch, batch_idx):
    x, y = batch
    return self.model(x)  # 回傳原始預測
```

### 配置方法

#### `configure_optimizers()`
回傳優化器和可選的學習率調度器。

**回傳格式：**

1. **單一優化器：**
```python
def configure_optimizers(self):
    return torch.optim.Adam(self.parameters(), lr=0.001)
```

2. **優化器 + 調度器：**
```python
def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)
    return [optimizer], [scheduler]
```

3. **帶調度器監控的進階配置：**
```python
def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": scheduler,
            "monitor": "val_loss",  # 要監控的指標
            "interval": "epoch",     # 何時更新（epoch/step）
            "frequency": 1,          # 更新頻率
            "strict": True           # 找不到監控指標時崩潰
        }
    }
```

4. **多個優化器（用於 GAN 等）：**
```python
def configure_optimizers(self):
    opt_g = torch.optim.Adam(self.generator.parameters(), lr=0.0002)
    opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=0.0002)
    return [opt_g, opt_d]
```

#### `forward(*args, **kwargs)`
標準 PyTorch forward 方法。用於推論或作為 training_step 的一部分。

**範例：**
```python
def forward(self, x):
    return self.model(x)

def training_step(self, batch, batch_idx):
    x, y = batch
    y_hat = self(x)  # 使用 forward()
    return F.mse_loss(y_hat, y)
```

### 日誌記錄和指標

#### `log(name, value, **kwargs)`
記錄指標，自動跨裝置進行 epoch 級別的縮減。

**主要參數：**
- `name` - 指標名稱（字串）
- `value` - 指標值（張量或數字）
- `on_step` - 在當前步驟記錄（預設：training_step 中為 True，其他為 False）
- `on_epoch` - 在 epoch 結束時記錄（預設：training_step 中為 False，其他為 True）
- `prog_bar` - 在進度條中顯示（預設：False）
- `logger` - 發送到日誌後端（預設：True）
- `reduce_fx` - 縮減函數："mean"、"sum"、"max"、"min"（預設："mean"）
- `sync_dist` - 在分散式訓練中跨裝置同步（預設：False）

**範例：**
```python
# 簡單記錄
self.log("train_loss", loss)

# 在進度條中顯示
self.log("accuracy", acc, prog_bar=True)

# 記錄每步和每 epoch
self.log("loss", loss, on_step=True, on_epoch=True)

# 分散式訓練的自訂縮減
self.log("batch_size", batch.size(0), reduce_fx="sum", sync_dist=True)
```

#### `log_dict(dictionary, **kwargs)`
同時記錄多個指標。

**範例：**
```python
metrics = {"train_loss": loss, "train_acc": acc, "learning_rate": lr}
self.log_dict(metrics, on_step=True, on_epoch=True)
```

#### `save_hyperparameters(*args, **kwargs)`
儲存初始化參數以便重現性和檢查點恢復。在 `__init__()` 中呼叫。

**範例：**
```python
def __init__(self, learning_rate, hidden_dim, dropout):
    super().__init__()
    self.save_hyperparameters()  # 儲存所有初始化參數
    # 透過 self.hparams.learning_rate、self.hparams.hidden_dim 等存取
```

## 主要屬性

| 屬性 | 說明 |
|----------|-------------|
| `self.current_epoch` | 當前 epoch 編號（0 索引） |
| `self.global_step` | 所有 epochs 的總優化器步驟數 |
| `self.device` | 當前裝置（cuda:0、cpu 等） |
| `self.global_rank` | 分散式訓練中的程序排名（主程序為 0） |
| `self.local_rank` | 當前節點上的 GPU 排名 |
| `self.hparams` | 已儲存的超參數（透過 save_hyperparameters） |
| `self.trainer` | 父 Trainer 實例的參考 |
| `self.automatic_optimization` | 是否使用自動優化（預設：True） |

## 手動優化

對於進階用例（GAN、強化學習、多個優化器），停用自動優化：

```python
class GANModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.automatic_optimization = False
        self.generator = Generator()
        self.discriminator = Discriminator()

    def training_step(self, batch, batch_idx):
        opt_g, opt_d = self.optimizers()

        # 訓練生成器
        opt_g.zero_grad()
        g_loss = self.compute_generator_loss(batch)
        self.manual_backward(g_loss)
        opt_g.step()

        # 訓練判別器
        opt_d.zero_grad()
        d_loss = self.compute_discriminator_loss(batch)
        self.manual_backward(d_loss)
        opt_d.step()

        self.log_dict({"g_loss": g_loss, "d_loss": d_loss})

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=0.0002)
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=0.0002)
        return [opt_g, opt_d]
```

## 重要的生命週期鉤子

### 設定和清理

#### `setup(stage)`
在 fit、validate、test 或 predict 開始時呼叫。用於階段特定的設定。

**參數：**
- `stage` - 'fit'、'validate'、'test' 或 'predict'

**範例：**
```python
def setup(self, stage):
    if stage == 'fit':
        # 設定訓練特定的元件
        self.train_dataset = load_train_data()
    elif stage == 'test':
        # 設定測試特定的元件
        self.test_dataset = load_test_data()
```

#### `teardown(stage)`
在 fit、validate、test 或 predict 結束時呼叫。清理資源。

### Epoch 邊界

#### `on_train_epoch_start()` / `on_train_epoch_end()`
在每個訓練 epoch 的開始/結束時呼叫。

**範例：**
```python
def on_train_epoch_end(self):
    # 計算 epoch 級別的指標
    all_preds = torch.cat(self.training_step_outputs)
    epoch_metric = compute_custom_metric(all_preds)
    self.log("epoch_metric", epoch_metric)
    self.training_step_outputs.clear()  # 釋放記憶體
```

#### `on_validation_epoch_start()` / `on_validation_epoch_end()`
在驗證 epoch 的開始/結束時呼叫。

#### `on_test_epoch_start()` / `on_test_epoch_end()`
在測試 epoch 的開始/結束時呼叫。

### 梯度鉤子

#### `on_before_backward(loss)`
在 loss.backward() 之前呼叫。

#### `on_after_backward()`
在 loss.backward() 之後但優化器步驟之前呼叫。

**範例 - 梯度檢查：**
```python
def on_after_backward(self):
    # 記錄梯度範數
    grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
    self.log("grad_norm", grad_norm)
```

### 檢查點鉤子

#### `on_save_checkpoint(checkpoint)`
自訂檢查點儲存。新增額外狀態以儲存。

**範例：**
```python
def on_save_checkpoint(self, checkpoint):
    checkpoint['custom_state'] = self.custom_data
```

#### `on_load_checkpoint(checkpoint)`
自訂檢查點載入。還原額外狀態。

**範例：**
```python
def on_load_checkpoint(self, checkpoint):
    self.custom_data = checkpoint.get('custom_state', default_value)
```

## 最佳實務

### 1. 裝置無關性
永遠不要使用明確的 `.cuda()` 或 `.cpu()` 呼叫。Lightning 自動處理裝置放置。

**不好的做法：**
```python
x = x.cuda()
model = model.cuda()
```

**好的做法：**
```python
x = x.to(self.device)  # 在 LightningModule 內部
# 或讓 Lightning 自動處理
```

### 2. 分散式訓練安全
不要手動建立 `DistributedSampler`。Lightning 自動處理。

**不好的做法：**
```python
sampler = DistributedSampler(dataset)
DataLoader(dataset, sampler=sampler)
```

**好的做法：**
```python
DataLoader(dataset, shuffle=True)  # Lightning 轉換為 DistributedSampler
```

### 3. 指標聚合
使用 `self.log()` 進行自動跨裝置縮減，而非手動收集。

**不好的做法：**
```python
self.validation_outputs.append(loss)

def on_validation_epoch_end(self):
    avg_loss = torch.stack(self.validation_outputs).mean()
```

**好的做法：**
```python
self.log("val_loss", loss)  # 自動聚合
```

### 4. 超參數追蹤
始終使用 `self.save_hyperparameters()` 以便輕鬆重新載入模型。

**範例：**
```python
def __init__(self, learning_rate, hidden_dim):
    super().__init__()
    self.save_hyperparameters()

# 稍後：從檢查點載入
model = MyModel.load_from_checkpoint("checkpoint.ckpt")
print(model.hparams.learning_rate)
```

### 5. 驗證放置
在單一裝置上執行驗證以確保每個樣本只被評估一次。Lightning 透過適當的策略配置自動處理。

## 從檢查點載入

```python
# 載入帶有已儲存超參數的模型
model = MyModel.load_from_checkpoint("path/to/checkpoint.ckpt")

# 如果需要可以覆蓋超參數
model = MyModel.load_from_checkpoint(
    "path/to/checkpoint.ckpt",
    learning_rate=0.0001  # 覆蓋已儲存的值
)

# 用於推論
model.eval()
predictions = model(data)
```

## 常見模式

### 梯度累積
讓 Lightning 處理梯度累積：

```python
trainer = L.Trainer(accumulate_grad_batches=4)
```

### 梯度裁剪
在 Trainer 中配置：

```python
trainer = L.Trainer(gradient_clip_val=0.5, gradient_clip_algorithm="norm")
```

### 混合精度訓練
在 Trainer 中配置精度：

```python
trainer = L.Trainer(precision="16-mixed")  # 或 "bf16-mixed"、"32-true"
```

### 學習率預熱
在 configure_optimizers 中實作：

```python
def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
    scheduler = {
        "scheduler": torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=0.01,
            total_steps=self.trainer.estimated_stepping_batches
        ),
        "interval": "step"
    }
    return [optimizer], [scheduler]
```
