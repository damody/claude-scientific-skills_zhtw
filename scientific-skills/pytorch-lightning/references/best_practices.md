# 最佳實務 - PyTorch Lightning

## 程式碼組織

### 1. 分離研究與工程

**好的做法：**
```python
class MyModel(L.LightningModule):
    # 研究程式碼（模型做什麼）
    def training_step(self, batch, batch_idx):
        loss = self.compute_loss(batch)
        return loss

# 工程程式碼（如何訓練）- 在 Trainer 中
trainer = L.Trainer(
    max_epochs=100,
    accelerator="gpu",
    devices=4,
    strategy="ddp"
)
```

**不好的做法：**
```python
# 混合研究和工程邏輯
class MyModel(L.LightningModule):
    def training_step(self, batch, batch_idx):
        loss = self.compute_loss(batch)

        # 不要手動進行裝置管理
        loss = loss.cuda()

        # 不要手動進行優化器步驟（除非使用手動優化）
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss
```

### 2. 使用 LightningDataModule

**好的做法：**
```python
class MyDataModule(L.LightningDataModule):
    def __init__(self, data_dir, batch_size):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def prepare_data(self):
        # 下載資料一次
        download_data(self.data_dir)

    def setup(self, stage):
        # 每個程序載入資料
        self.train_dataset = MyDataset(self.data_dir, split='train')
        self.val_dataset = MyDataset(self.data_dir, split='val')

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

# 可重用和可分享
dm = MyDataModule("./data", batch_size=32)
trainer.fit(model, datamodule=dm)
```

**不好的做法：**
```python
# 分散的資料邏輯
train_dataset = load_data()
val_dataset = load_data()
train_loader = DataLoader(train_dataset, ...)
val_loader = DataLoader(val_dataset, ...)
trainer.fit(model, train_loader, val_loader)
```

### 3. 保持模型模組化

```python
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(...)

    def forward(self, x):
        return self.layers(x)

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(...)

    def forward(self, x):
        return self.layers(x)

class MyModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
```

## 裝置無關性

### 1. 永遠不要使用明確的 CUDA 呼叫

**不好的做法：**
```python
x = x.cuda()
model = model.cuda()
torch.cuda.set_device(0)
```

**好的做法：**
```python
# 在 LightningModule 內部
x = x.to(self.device)

# 或讓 Lightning 自動處理
def training_step(self, batch, batch_idx):
    x, y = batch  # 已經在正確的裝置上
    return loss
```

### 2. 使用 `self.device` 屬性

```python
class MyModel(L.LightningModule):
    def training_step(self, batch, batch_idx):
        # 在正確的裝置上建立張量
        noise = torch.randn(batch.size(0), 100).to(self.device)

        # 或使用 type_as
        noise = torch.randn(batch.size(0), 100).type_as(batch)
```

### 3. 為非參數註冊緩衝區

```python
class MyModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        # 註冊緩衝區（自動移動到正確的裝置）
        self.register_buffer("running_mean", torch.zeros(100))

    def forward(self, x):
        # self.running_mean 自動在正確的裝置上
        return x - self.running_mean
```

## 超參數管理

### 1. 始終使用 `save_hyperparameters()`

**好的做法：**
```python
class MyModel(L.LightningModule):
    def __init__(self, learning_rate, hidden_dim, dropout):
        super().__init__()
        self.save_hyperparameters()  # 儲存所有參數

        # 透過 self.hparams 存取
        self.model = nn.Linear(self.hparams.hidden_dim, 10)

# 從檢查點載入並保留已儲存的 hparams
model = MyModel.load_from_checkpoint("checkpoint.ckpt")
print(model.hparams.learning_rate)  # 原始值被保留
```

**不好的做法：**
```python
class MyModel(L.LightningModule):
    def __init__(self, learning_rate, hidden_dim, dropout):
        super().__init__()
        self.learning_rate = learning_rate  # 手動追蹤
        self.hidden_dim = hidden_dim
```

### 2. 忽略特定參數

```python
class MyModel(L.LightningModule):
    def __init__(self, lr, model, dataset):
        super().__init__()
        # 不儲存 'model' 和 'dataset'（不可序列化）
        self.save_hyperparameters(ignore=['model', 'dataset'])

        self.model = model
        self.dataset = dataset
```

### 3. 在 `configure_optimizers()` 中使用超參數

```python
def configure_optimizers(self):
    # 使用已儲存的超參數
    optimizer = torch.optim.Adam(
        self.parameters(),
        lr=self.hparams.learning_rate,
        weight_decay=self.hparams.weight_decay
    )
    return optimizer
```

## 日誌記錄最佳實務

### 1. 記錄步驟和 epoch 指標

```python
def training_step(self, batch, batch_idx):
    loss = self.compute_loss(batch)

    # 記錄每步以進行詳細監控
    # 記錄每 epoch 以進行聚合檢視
    self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

    return loss
```

### 2. 使用結構化日誌記錄

```python
def training_step(self, batch, batch_idx):
    # 使用前綴組織
    self.log("train/loss", loss)
    self.log("train/acc", acc)
    self.log("train/f1", f1)

def validation_step(self, batch, batch_idx):
    self.log("val/loss", loss)
    self.log("val/acc", acc)
    self.log("val/f1", f1)
```

### 3. 在分散式訓練中同步指標

```python
def validation_step(self, batch, batch_idx):
    loss = self.compute_loss(batch)

    # 重要：sync_dist=True 用於跨 GPU 的正確聚合
    self.log("val_loss", loss, sync_dist=True)
```

### 4. 監控學習率

```python
from lightning.pytorch.callbacks import LearningRateMonitor

trainer = L.Trainer(
    callbacks=[LearningRateMonitor(logging_interval="step")]
)
```

## 可重現性

### 1. 設定所有種子

```python
import lightning as L

# 設定種子以確保可重現性
L.seed_everything(42, workers=True)

trainer = L.Trainer(
    deterministic=True,  # 使用確定性演算法
    benchmark=False      # 停用 cudnn 基準測試
)
```

### 2. 避免非確定性操作

```python
# 不好：非確定性
torch.use_deterministic_algorithms(False)

# 好：確定性
torch.use_deterministic_algorithms(True)
```

### 3. 記錄隨機狀態

```python
def on_save_checkpoint(self, checkpoint):
    # 儲存隨機狀態
    checkpoint['rng_state'] = {
        'torch': torch.get_rng_state(),
        'numpy': np.random.get_state(),
        'python': random.getstate()
    }

def on_load_checkpoint(self, checkpoint):
    # 還原隨機狀態
    if 'rng_state' in checkpoint:
        torch.set_rng_state(checkpoint['rng_state']['torch'])
        np.random.set_state(checkpoint['rng_state']['numpy'])
        random.setstate(checkpoint['rng_state']['python'])
```

## 除錯

### 1. 使用 `fast_dev_run`

```python
# 在完整訓練前用 1 個批次測試
trainer = L.Trainer(fast_dev_run=True)
trainer.fit(model, datamodule=dm)
```

### 2. 限制訓練資料

```python
# 僅使用 10% 的資料進行快速迭代
trainer = L.Trainer(
    limit_train_batches=0.1,
    limit_val_batches=0.1
)
```

### 3. 啟用異常檢測

```python
# 在梯度中檢測 NaN/Inf
trainer = L.Trainer(detect_anomaly=True)
```

### 4. 在小批次上過擬合

```python
# 在 10 個批次上過擬合以驗證模型容量
trainer = L.Trainer(overfit_batches=10)
```

### 5. 效能分析

```python
# 找出效能瓶頸
trainer = L.Trainer(profiler="simple")  # 或 "advanced"
```

## 記憶體優化

### 1. 使用混合精度

```python
# FP16/BF16 混合精度用於節省記憶體和加速
trainer = L.Trainer(
    precision="16-mixed",   # V100, T4
    # 或
    precision="bf16-mixed"  # A100, H100
)
```

### 2. 梯度累積

```python
# 模擬較大批次大小而不增加記憶體
trainer = L.Trainer(
    accumulate_grad_batches=4  # 累積 4 個批次
)
```

### 3. 梯度檢查點

```python
class MyModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = transformers.AutoModel.from_pretrained("bert-base")

        # 啟用梯度檢查點
        self.model.gradient_checkpointing_enable()
```

### 4. 清除快取

```python
def on_train_epoch_end(self):
    # 清除收集的輸出以釋放記憶體
    self.training_step_outputs.clear()

    # 如果需要，清除 CUDA 快取
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
```

### 5. 使用高效的資料類型

```python
# 使用適當的精度
# FP32 用於穩定性，FP16/BF16 用於速度/記憶體

class MyModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        # 使用 bfloat16 以獲得比 fp16 更好的數值穩定性
        self.model = MyTransformer().to(torch.bfloat16)
```

## 訓練穩定性

### 1. 梯度裁剪

```python
# 防止梯度爆炸
trainer = L.Trainer(
    gradient_clip_val=1.0,
    gradient_clip_algorithm="norm"  # 或 "value"
)
```

### 2. 學習率預熱

```python
def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=1e-2,
        total_steps=self.trainer.estimated_stepping_batches,
        pct_start=0.1  # 10% 預熱
    )

    return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": scheduler,
            "interval": "step"
        }
    }
```

### 3. 監控梯度

```python
class MyModel(L.LightningModule):
    def on_after_backward(self):
        # 記錄梯度範數
        for name, param in self.named_parameters():
            if param.grad is not None:
                self.log(f"grad_norm/{name}", param.grad.norm())
```

### 4. 使用 EarlyStopping

```python
from lightning.pytorch.callbacks import EarlyStopping

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=10,
    mode="min",
    verbose=True
)

trainer = L.Trainer(callbacks=[early_stop])
```

## 檢查點

### 1. 儲存 Top-K 和最後一個

```python
from lightning.pytorch.callbacks import ModelCheckpoint

checkpoint_callback = ModelCheckpoint(
    dirpath="checkpoints/",
    filename="{epoch}-{val_loss:.2f}",
    monitor="val_loss",
    mode="min",
    save_top_k=3,    # 保留最佳的 3 個
    save_last=True   # 始終儲存最後一個以便恢復
)

trainer = L.Trainer(callbacks=[checkpoint_callback])
```

### 2. 恢復訓練

```python
# 從最後一個檢查點恢復
trainer.fit(model, datamodule=dm, ckpt_path="last.ckpt")

# 從特定檢查點恢復
trainer.fit(model, datamodule=dm, ckpt_path="epoch=10-val_loss=0.23.ckpt")
```

### 3. 自訂檢查點狀態

```python
def on_save_checkpoint(self, checkpoint):
    # 新增自訂狀態
    checkpoint['custom_data'] = self.custom_data
    checkpoint['epoch_metrics'] = self.metrics

def on_load_checkpoint(self, checkpoint):
    # 還原自訂狀態
    self.custom_data = checkpoint.get('custom_data', {})
    self.metrics = checkpoint.get('epoch_metrics', [])
```

## 測試

### 1. 分離訓練和測試

```python
# 訓練
trainer = L.Trainer(max_epochs=100)
trainer.fit(model, datamodule=dm)

# 僅在發布前測試一次
trainer.test(model, datamodule=dm)
```

### 2. 使用驗證進行模型選擇

```python
# 使用驗證進行超參數調整
checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode="min")
trainer = L.Trainer(callbacks=[checkpoint_callback])
trainer.fit(model, datamodule=dm)

# 載入最佳模型
best_model = MyModel.load_from_checkpoint(checkpoint_callback.best_model_path)

# 僅用最佳模型測試一次
trainer.test(best_model, datamodule=dm)
```

## 程式碼品質

### 1. 類型提示

```python
from typing import Any, Dict, Tuple
import torch
from torch import Tensor

class MyModel(L.LightningModule):
    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        x, y = batch
        loss = self.compute_loss(x, y)
        return loss

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = torch.optim.Adam(self.parameters())
        return {"optimizer": optimizer}
```

### 2. 文件字串

```python
class MyModel(L.LightningModule):
    """
    用於影像分類的優秀模型。

    Args:
        num_classes: 輸出類別數量
        learning_rate: 優化器的學習率
        hidden_dim: 隱藏維度大小
    """

    def __init__(self, num_classes: int, learning_rate: float, hidden_dim: int):
        super().__init__()
        self.save_hyperparameters()
```

### 3. 屬性方法

```python
class MyModel(L.LightningModule):
    @property
    def learning_rate(self) -> float:
        """目前學習率。"""
        return self.hparams.learning_rate

    @property
    def num_parameters(self) -> int:
        """參數總數。"""
        return sum(p.numel() for p in self.parameters())
```

## 常見陷阱

### 1. 忘記回傳損失

**不好的做法：**
```python
def training_step(self, batch, batch_idx):
    loss = self.compute_loss(batch)
    self.log("train_loss", loss)
    # 忘記回傳損失！
```

**好的做法：**
```python
def training_step(self, batch, batch_idx):
    loss = self.compute_loss(batch)
    self.log("train_loss", loss)
    return loss  # 必須回傳損失
```

### 2. 在 DDP 中不同步指標

**不好的做法：**
```python
def validation_step(self, batch, batch_idx):
    self.log("val_acc", acc)  # 多 GPU 時數值錯誤！
```

**好的做法：**
```python
def validation_step(self, batch, batch_idx):
    self.log("val_acc", acc, sync_dist=True)  # 正確聚合
```

### 3. 手動裝置管理

**不好的做法：**
```python
def training_step(self, batch, batch_idx):
    x = x.cuda()  # 不要這樣做
    y = y.cuda()
```

**好的做法：**
```python
def training_step(self, batch, batch_idx):
    # Lightning 處理裝置放置
    x, y = batch  # 已經在正確的裝置上
```

### 4. 不使用 `self.log()`

**不好的做法：**
```python
def training_step(self, batch, batch_idx):
    loss = self.compute_loss(batch)
    self.training_losses.append(loss)  # 手動追蹤
    return loss
```

**好的做法：**
```python
def training_step(self, batch, batch_idx):
    loss = self.compute_loss(batch)
    self.log("train_loss", loss)  # 自動記錄
    return loss
```

### 5. 原地修改批次

**不好的做法：**
```python
def training_step(self, batch, batch_idx):
    x, y = batch
    x[:] = self.augment(x)  # 原地修改可能導致問題
```

**好的做法：**
```python
def training_step(self, batch, batch_idx):
    x, y = batch
    x = self.augment(x)  # 建立新張量
```

## 效能技巧

### 1. 使用 DataLoader Workers

```python
def train_dataloader(self):
    return DataLoader(
        self.train_dataset,
        batch_size=32,
        num_workers=4,           # 使用多個 workers
        pin_memory=True,         # 更快的 GPU 傳輸
        persistent_workers=True  # 保持 workers 存活
    )
```

### 2. 啟用基準測試模式（如果輸入大小固定）

```python
trainer = L.Trainer(benchmark=True)
```

### 3. 使用自動批次大小查找

```python
from lightning.pytorch.tuner import Tuner

trainer = L.Trainer()
tuner = Tuner(trainer)

# 找到最佳批次大小
tuner.scale_batch_size(model, datamodule=dm, mode="power")

# 然後訓練
trainer.fit(model, datamodule=dm)
```

### 4. 優化資料載入

```python
# 使用更快的影像解碼
import torch
import torchvision.transforms as T

transforms = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 使用 PIL-SIMD 進行更快的影像載入
# pip install pillow-simd
```
