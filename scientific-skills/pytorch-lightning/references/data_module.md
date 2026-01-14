# LightningDataModule - 完整指南

## 概述

LightningDataModule 是一個可重用、可分享的類別，封裝了 PyTorch Lightning 中所有資料處理步驟。它透過標準化資料集的管理和跨專案分享方式，解決了分散的資料準備邏輯問題。

## 它解決的核心問題

在傳統的 PyTorch 工作流程中，資料處理分散在多個檔案中，使得以下問題難以回答：
- "你使用了什麼分割？"
- "應用了什麼轉換？"
- "資料是如何準備的？"

DataModules 將這些資訊集中化，以實現可重現性和可重用性。

## 五個處理步驟

DataModule 將資料處理組織成五個階段：

1. **下載/分詞/處理** - 初始資料獲取
2. **清理和儲存** - 將處理後的資料持久化到磁碟
3. **載入到 Dataset** - 建立 PyTorch Dataset 物件
4. **應用轉換** - 資料增強、正規化等
5. **包裝成 DataLoader** - 配置批次處理和載入

## 主要方法

### `prepare_data()`
下載和處理資料。僅在單一程序上執行一次（非分散式）。

**用途：**
- 下載資料集
- 分詞文字
- 將處理後的資料儲存到磁碟

**重要：** 不要在這裡設定狀態（例如 self.x = y）。狀態不會傳輸到其他程序。

**範例：**
```python
def prepare_data(self):
    # 下載資料（執行一次）
    download_dataset("http://example.com/data.zip", "data/")

    # 分詞並儲存（執行一次）
    tokenize_and_save("data/raw/", "data/processed/")
```

### `setup(stage)`
建立資料集並應用轉換。在分散式訓練中的每個程序上執行。

**參數：**
- `stage` - 'fit'、'validate'、'test' 或 'predict'

**用途：**
- 建立訓練/驗證/測試分割
- 建構 Dataset 物件
- 應用轉換
- 設定狀態（self.train_dataset = ...）

**範例：**
```python
def setup(self, stage):
    if stage == 'fit':
        full_dataset = MyDataset("data/processed/")
        self.train_dataset, self.val_dataset = random_split(
            full_dataset, [0.8, 0.2]
        )

    if stage == 'test':
        self.test_dataset = MyDataset("data/processed/test/")

    if stage == 'predict':
        self.predict_dataset = MyDataset("data/processed/predict/")
```

### `train_dataloader()`
回傳訓練 DataLoader。

**範例：**
```python
def train_dataloader(self):
    return DataLoader(
        self.train_dataset,
        batch_size=self.batch_size,
        shuffle=True,
        num_workers=self.num_workers,
        pin_memory=True
    )
```

### `val_dataloader()`
回傳驗證 DataLoader。

**範例：**
```python
def val_dataloader(self):
    return DataLoader(
        self.val_dataset,
        batch_size=self.batch_size,
        shuffle=False,
        num_workers=self.num_workers,
        pin_memory=True
    )
```

### `test_dataloader()`
回傳測試 DataLoader。

**範例：**
```python
def test_dataloader(self):
    return DataLoader(
        self.test_dataset,
        batch_size=self.batch_size,
        shuffle=False,
        num_workers=self.num_workers
    )
```

### `predict_dataloader()`
回傳預測 DataLoader。

**範例：**
```python
def predict_dataloader(self):
    return DataLoader(
        self.predict_dataset,
        batch_size=self.batch_size,
        shuffle=False,
        num_workers=self.num_workers
    )
```

## 完整範例

```python
import lightning as L
from torch.utils.data import DataLoader, Dataset, random_split
import torch

class MyDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transform
        self.data = self._load_data()

    def _load_data(self):
        # 在這裡載入你的資料
        return torch.randn(1000, 3, 224, 224)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample

class MyDataModule(L.LightningDataModule):
    def __init__(self, data_dir="./data", batch_size=32, num_workers=4):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        # 轉換
        self.train_transform = self._get_train_transforms()
        self.test_transform = self._get_test_transforms()

    def _get_train_transforms(self):
        # 定義訓練轉換
        return lambda x: x  # 佔位符

    def _get_test_transforms(self):
        # 定義測試/驗證轉換
        return lambda x: x  # 佔位符

    def prepare_data(self):
        # 下載資料（在單一程序上執行一次）
        # download_data(self.data_dir)
        pass

    def setup(self, stage=None):
        # 建立資料集（在每個程序上執行）
        if stage == 'fit' or stage is None:
            full_dataset = MyDataset(
                self.data_dir,
                transform=self.train_transform
            )
            train_size = int(0.8 * len(full_dataset))
            val_size = len(full_dataset) - train_size
            self.train_dataset, self.val_dataset = random_split(
                full_dataset, [train_size, val_size]
            )

        if stage == 'test' or stage is None:
            self.test_dataset = MyDataset(
                self.data_dir,
                transform=self.test_transform
            )

        if stage == 'predict':
            self.predict_dataset = MyDataset(
                self.data_dir,
                transform=self.test_transform
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
```

## 使用方法

```python
# 建立 DataModule
dm = MyDataModule(data_dir="./data", batch_size=64, num_workers=8)

# 與 Trainer 一起使用
trainer = L.Trainer(max_epochs=10)
trainer.fit(model, datamodule=dm)

# 測試
trainer.test(model, datamodule=dm)

# 預測
predictions = trainer.predict(model, datamodule=dm)

# 或在 PyTorch 中獨立使用
dm.prepare_data()
dm.setup(stage='fit')
train_loader = dm.train_dataloader()

for batch in train_loader:
    # 你的訓練程式碼
    pass
```

## 額外的鉤子

### `transfer_batch_to_device(batch, device, dataloader_idx)`
將批次移動到裝置的自訂邏輯。

**範例：**
```python
def transfer_batch_to_device(self, batch, device, dataloader_idx):
    # 自訂傳輸邏輯
    if isinstance(batch, dict):
        return {k: v.to(device) for k, v in batch.items()}
    return super().transfer_batch_to_device(batch, device, dataloader_idx)
```

### `on_before_batch_transfer(batch, dataloader_idx)`
在傳輸到裝置之前增強或修改批次（在 CPU 上執行）。

**範例：**
```python
def on_before_batch_transfer(self, batch, dataloader_idx):
    # 應用基於 CPU 的增強
    batch['image'] = apply_augmentation(batch['image'])
    return batch
```

### `on_after_batch_transfer(batch, dataloader_idx)`
在傳輸到裝置之後增強或修改批次（在 GPU 上執行）。

**範例：**
```python
def on_after_batch_transfer(self, batch, dataloader_idx):
    # 應用基於 GPU 的增強
    batch['image'] = gpu_augmentation(batch['image'])
    return batch
```

### `state_dict()` / `load_state_dict(state_dict)`
儲存和還原 DataModule 狀態以進行檢查點。

**範例：**
```python
def state_dict(self):
    return {"current_fold": self.current_fold}

def load_state_dict(self, state_dict):
    self.current_fold = state_dict["current_fold"]
```

### `teardown(stage)`
訓練/測試/預測後的清理操作。

**範例：**
```python
def teardown(self, stage):
    # 清理資源
    if stage == 'fit':
        self.train_dataset = None
        self.val_dataset = None
```

## 進階模式

### 多個驗證/測試 DataLoaders

回傳 DataLoaders 的列表或字典：

```python
def val_dataloader(self):
    return [
        DataLoader(self.val_dataset_1, batch_size=32),
        DataLoader(self.val_dataset_2, batch_size=32)
    ]

# 或使用名稱（用於日誌記錄）
def val_dataloader(self):
    return {
        "val_easy": DataLoader(self.val_easy, batch_size=32),
        "val_hard": DataLoader(self.val_hard, batch_size=32)
    }

# 在 LightningModule 中
def validation_step(self, batch, batch_idx, dataloader_idx=0):
    if dataloader_idx == 0:
        # 處理 val_dataset_1
        pass
    else:
        # 處理 val_dataset_2
        pass
```

### 交叉驗證

```python
class CrossValidationDataModule(L.LightningDataModule):
    def __init__(self, data_dir, batch_size, num_folds=5):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_folds = num_folds
        self.current_fold = 0

    def setup(self, stage=None):
        full_dataset = MyDataset(self.data_dir)
        fold_size = len(full_dataset) // self.num_folds

        # 建立 fold 索引
        indices = list(range(len(full_dataset)))
        val_start = self.current_fold * fold_size
        val_end = val_start + fold_size

        val_indices = indices[val_start:val_end]
        train_indices = indices[:val_start] + indices[val_end:]

        self.train_dataset = Subset(full_dataset, train_indices)
        self.val_dataset = Subset(full_dataset, val_indices)

    def set_fold(self, fold):
        self.current_fold = fold

    def state_dict(self):
        return {"current_fold": self.current_fold}

    def load_state_dict(self, state_dict):
        self.current_fold = state_dict["current_fold"]

# 使用方法
dm = CrossValidationDataModule("./data", batch_size=32, num_folds=5)

for fold in range(5):
    dm.set_fold(fold)
    trainer = L.Trainer(max_epochs=10)
    trainer.fit(model, datamodule=dm)
```

### 超參數儲存

```python
class MyDataModule(L.LightningDataModule):
    def __init__(self, data_dir, batch_size=32, num_workers=4):
        super().__init__()
        # 儲存超參數
        self.save_hyperparameters()

    def setup(self, stage=None):
        # 透過 self.hparams 存取
        print(f"Batch size: {self.hparams.batch_size}")
```

## 最佳實務

### 1. 分離 prepare_data 和 setup
- `prepare_data()` - 下載/處理（單一程序，無狀態）
- `setup()` - 建立資料集（每個程序，設定狀態）

### 2. 使用 stage 參數
在 `setup()` 中檢查 stage 以避免不必要的工作：

```python
def setup(self, stage):
    if stage == 'fit':
        # 僅在擬合時載入訓練/驗證資料
        self.train_dataset = ...
        self.val_dataset = ...
    elif stage == 'test':
        # 僅在測試時載入測試資料
        self.test_dataset = ...
```

### 3. 為 GPU 訓練固定記憶體
在 DataLoaders 中啟用 `pin_memory=True` 以加快 GPU 傳輸：

```python
def train_dataloader(self):
    return DataLoader(..., pin_memory=True)
```

### 4. 使用持久性 Workers
防止 epochs 之間的 worker 重新啟動：

```python
def train_dataloader(self):
    return DataLoader(
        ...,
        num_workers=4,
        persistent_workers=True
    )
```

### 5. 避免在驗證/測試中隨機打亂
永遠不要隨機打亂驗證或測試資料：

```python
def val_dataloader(self):
    return DataLoader(..., shuffle=False)  # 永遠不要設為 True
```

### 6. 使 DataModules 可重用
在 `__init__` 中接受配置參數：

```python
class MyDataModule(L.LightningDataModule):
    def __init__(self, data_dir, batch_size, num_workers, augment=True):
        super().__init__()
        self.save_hyperparameters()
```

### 7. 記錄資料結構
新增文件字串說明資料格式和預期：

```python
class MyDataModule(L.LightningDataModule):
    """
    XYZ 資料集的 DataModule。

    資料格式：(image, label) 元組
    - image: 形狀為 (C, H, W) 的 torch.Tensor
    - label: 範圍在 [0, num_classes) 的 int

    Args:
        data_dir: 資料目錄路徑
        batch_size: dataloaders 的批次大小
        num_workers: 資料載入 workers 數量
    """
```

## 常見陷阱

### 1. 在 prepare_data 中設定狀態
**錯誤：**
```python
def prepare_data(self):
    self.dataset = load_data()  # 狀態不會傳輸到其他程序！
```

**正確：**
```python
def prepare_data(self):
    download_data()  # 僅下載，無狀態

def setup(self, stage):
    self.dataset = load_data()  # 在這裡設定狀態
```

### 2. 不使用 stage 參數
**低效：**
```python
def setup(self, stage):
    self.train_dataset = load_train()
    self.val_dataset = load_val()
    self.test_dataset = load_test()  # 即使只是擬合也會載入
```

**高效：**
```python
def setup(self, stage):
    if stage == 'fit':
        self.train_dataset = load_train()
        self.val_dataset = load_val()
    elif stage == 'test':
        self.test_dataset = load_test()
```

### 3. 忘記回傳 DataLoaders
**錯誤：**
```python
def train_dataloader(self):
    DataLoader(self.train_dataset, ...)  # 忘記 return！
```

**正確：**
```python
def train_dataloader(self):
    return DataLoader(self.train_dataset, ...)
```

## 與 Trainer 整合

```python
# 初始化 DataModule
dm = MyDataModule(data_dir="./data", batch_size=64)

# 所有資料載入由 DataModule 處理
trainer = L.Trainer(max_epochs=10)
trainer.fit(model, datamodule=dm)

# DataModule 也處理驗證
trainer.validate(model, datamodule=dm)

# 以及測試
trainer.test(model, datamodule=dm)

# 以及預測
predictions = trainer.predict(model, datamodule=dm)
```
