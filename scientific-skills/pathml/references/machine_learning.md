# 機器學習

## 概述

PathML 為計算病理學提供全面的機器學習功能，包括用於細胞核檢測和分割的預建模型、PyTorch 整合的訓練工作流程、公開資料集存取，以及基於 ONNX 的推論部署。該框架無縫連接影像預處理與深度學習，實現端到端的病理機器學習管線。

## 預建模型

PathML 包含用於細胞核分析的最先進預訓練模型：

### HoVer-Net

**HoVer-Net**（水平和垂直網路）同時執行細胞核實例分割和分類。

**架構：**
- 具有三個預測分支的編碼器-解碼器結構：
  - **Nuclear Pixel (NP)** - 細胞核區域的二元分割
  - **Horizontal-Vertical (HV)** - 到細胞核質心的距離圖
  - **Classification (NC)** - 細胞核類型分類

**細胞核類型：**
1. 上皮細胞
2. 發炎細胞
3. 結締/軟組織細胞
4. 死亡/壞死細胞
5. 背景

**使用方法：**
```python
from pathml.ml import HoVerNet
import torch

# 載入預訓練模型
model = HoVerNet(
    num_types=5,  # 細胞核類型數量
    mode='fast',  # 'fast' 或 'original'
    pretrained=True  # 載入預訓練權重
)

# 如果可用則移到 GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# 在圖磚上推論
tile_image = torch.from_numpy(tile.image).permute(2, 0, 1).unsqueeze(0).float()
tile_image = tile_image.to(device)

with torch.no_grad():
    output = model(tile_image)

# 輸出包含：
# - output['np']：細胞核像素預測
# - output['hv']：水平-垂直圖
# - output['nc']：分類預測
```

**後處理：**
```python
from pathml.ml import hovernet_postprocess

# 將模型輸出轉換為實例分割
instance_map, type_map = hovernet_postprocess(
    np_pred=output['np'],
    hv_pred=output['hv'],
    nc_pred=output['nc']
)

# instance_map：每個細胞核具有唯一 ID
# type_map：每個細胞核被分配類型（1-5）
```

### HACTNet

**HACTNet**（階層式細胞類型網路）執行具有不確定性量化的階層式細胞核分類。

**特點：**
- 階層式分類（從粗略到細緻的類型）
- 預測的不確定性估計
- 在不平衡資料集上效能改善

```python
from pathml.ml import HACTNet

# 載入模型
model = HACTNet(
    num_classes_coarse=3,
    num_classes_fine=8,
    pretrained=True
)

# 推論
output = model(tile_image)
coarse_pred = output['coarse']  # 粗略類別
fine_pred = output['fine']  # 具體細胞類型
uncertainty = output['uncertainty']  # 預測信心度
```

## 訓練工作流程

### 資料集準備

PathML 提供與 PyTorch 相容的資料集類別：

**TileDataset：**
```python
from pathml.ml import TileDataset
from pathml.core import SlideDataset

# 從處理過的切片創建資料集
tile_dataset = TileDataset(
    slide_dataset,
    tile_size=256,
    transform=None  # 可選的增強轉換
)

# 存取圖磚
image, label = tile_dataset[0]
```

**DataModule 整合：**
```python
from pathml.ml import PathMLDataModule

# 創建訓練/驗證/測試分割
data_module = PathMLDataModule(
    train_dataset=train_tile_dataset,
    val_dataset=val_tile_dataset,
    test_dataset=test_tile_dataset,
    batch_size=32,
    num_workers=4
)

# 與 PyTorch Lightning 一起使用
trainer = pl.Trainer(max_epochs=100)
trainer.fit(model, data_module)
```

### 訓練 HoVer-Net

在自定義資料上訓練 HoVer-Net 的完整工作流程：

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathml.ml import HoVerNet
from pathml.ml.datasets import PanNukeDataModule

# 1. 準備資料
data_module = PanNukeDataModule(
    data_dir='path/to/pannuke',
    batch_size=8,
    num_workers=4,
    tissue_types=['Breast', 'Colon']  # 特定組織類型
)

# 2. 初始化模型
model = HoVerNet(
    num_types=5,
    mode='fast',
    pretrained=False  # 從頭訓練或使用 pretrained=True 進行微調
)

# 3. 定義損失函數
class HoVerNetLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, output, target):
        # 細胞核像素分支損失
        np_loss = self.bce_loss(output['np'], target['np'])

        # 水平-垂直分支損失
        hv_loss = self.mse_loss(output['hv'], target['hv'])

        # 分類分支損失
        nc_loss = self.ce_loss(output['nc'], target['nc'])

        # 組合損失
        total_loss = np_loss + hv_loss + 2.0 * nc_loss
        return total_loss, {'np': np_loss, 'hv': hv_loss, 'nc': nc_loss}

criterion = HoVerNetLoss()

# 4. 配置最佳化器
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-4,
    weight_decay=1e-5
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=10
)

# 5. 訓練迴圈
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0

    for batch in data_module.train_dataloader():
        images = batch['image'].to(device)
        targets = {
            'np': batch['np_map'].to(device),
            'hv': batch['hv_map'].to(device),
            'nc': batch['type_map'].to(device)
        }

        optimizer.zero_grad()
        outputs = model(images)
        loss, loss_dict = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # 驗證
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in data_module.val_dataloader():
            images = batch['image'].to(device)
            targets = {
                'np': batch['np_map'].to(device),
                'hv': batch['hv_map'].to(device),
                'nc': batch['type_map'].to(device)
            }
            outputs = model(images)
            loss, _ = criterion(outputs, targets)
            val_loss += loss.item()

    scheduler.step(val_loss)

    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"  訓練損失：{train_loss/len(data_module.train_dataloader()):.4f}")
    print(f"  驗證損失：{val_loss/len(data_module.val_dataloader()):.4f}")

    # 儲存檢查點
    if (epoch + 1) % 10 == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': val_loss,
        }, f'hovernet_checkpoint_epoch_{epoch+1}.pth')
```

### PyTorch Lightning 整合

PathML 模型與 PyTorch Lightning 整合以簡化訓練：

```python
import pytorch_lightning as pl
from pathml.ml import HoVerNet
from pathml.ml.datasets import PanNukeDataModule

class HoVerNetModule(pl.LightningModule):
    def __init__(self, num_types=5, lr=1e-4):
        super().__init__()
        self.model = HoVerNet(num_types=num_types, pretrained=True)
        self.lr = lr
        self.criterion = HoVerNetLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images = batch['image']
        targets = {
            'np': batch['np_map'],
            'hv': batch['hv_map'],
            'nc': batch['type_map']
        }
        outputs = self(images)
        loss, loss_dict = self.criterion(outputs, targets)

        # 記錄指標
        self.log('train_loss', loss, prog_bar=True)
        for key, val in loss_dict.items():
            self.log(f'train_{key}_loss', val)

        return loss

    def validation_step(self, batch, batch_idx):
        images = batch['image']
        targets = {
            'np': batch['np_map'],
            'hv': batch['hv_map'],
            'nc': batch['type_map']
        }
        outputs = self(images)
        loss, loss_dict = self.criterion(outputs, targets)

        self.log('val_loss', loss, prog_bar=True)
        for key, val in loss_dict.items():
            self.log(f'val_{key}_loss', val)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss'
            }
        }

# 使用 PyTorch Lightning 訓練
data_module = PanNukeDataModule(data_dir='path/to/pannuke', batch_size=8)
model = HoVerNetModule(num_types=5, lr=1e-4)

trainer = pl.Trainer(
    max_epochs=100,
    accelerator='gpu',
    devices=1,
    callbacks=[
        pl.callbacks.ModelCheckpoint(monitor='val_loss', mode='min'),
        pl.callbacks.EarlyStopping(monitor='val_loss', patience=20)
    ]
)

trainer.fit(model, data_module)
```

## 公開資料集

PathML 提供便捷的公開病理資料集存取：

### PanNuke 資料集

**PanNuke** 包含來自 19 種組織類型的 7,901 個組織學影像切片，具有 5 種細胞類型的細胞核註釋。

```python
from pathml.ml.datasets import PanNukeDataModule

# 載入 PanNuke 資料集
pannuke = PanNukeDataModule(
    data_dir='path/to/pannuke',
    batch_size=16,
    num_workers=4,
    tissue_types=None,  # 使用所有組織類型，或指定列表
    fold='all'  # 'fold1'、'fold2'、'fold3' 或 'all'
)

# 存取資料載入器
train_loader = pannuke.train_dataloader()
val_loader = pannuke.val_dataloader()
test_loader = pannuke.test_dataloader()

# 批次結構
for batch in train_loader:
    images = batch['image']  # 形狀：(B, 3, 256, 256)
    inst_map = batch['inst_map']  # 實例分割圖
    type_map = batch['type_map']  # 細胞類型圖
    np_map = batch['np_map']  # 細胞核像素圖
    hv_map = batch['hv_map']  # 水平-垂直距離圖
    tissue_type = batch['tissue_type']  # 組織類別
```

**可用組織類型：**
乳房、結腸、前列腺、肺、腎、胃、膀胱、食道、子宮頸、肝、甲狀腺、頭頸部、睪丸、腎上腺、胰臟、膽管、卵巢、皮膚、子宮

### TCGA 資料集

存取癌症基因組圖譜資料集：

```python
from pathml.ml.datasets import TCGADataModule

# 載入 TCGA 資料集
tcga = TCGADataModule(
    data_dir='path/to/tcga',
    cancer_type='BRCA',  # 乳癌
    batch_size=32,
    tile_size=224
)
```

### 自定義資料集整合

為 PathML 工作流程創建自定義資料集：

```python
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path

class CustomPathologyDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.image_paths = list(self.data_dir.glob('images/*.png'))
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 載入影像
        image_path = self.image_paths[idx]
        image = np.array(Image.open(image_path))

        # 載入對應註釋
        annot_path = self.data_dir / 'annotations' / f'{image_path.stem}.npy'
        annotation = np.load(annot_path)

        # 應用轉換
        if self.transform:
            image = self.transform(image)

        return {
            'image': torch.from_numpy(image).permute(2, 0, 1).float(),
            'annotation': torch.from_numpy(annotation).long(),
            'path': str(image_path)
        }

# 在 PathML 工作流程中使用
dataset = CustomPathologyDataset('path/to/data')
dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)
```

## 資料增強

應用增強以改善模型泛化：

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2

# 定義增強管線
train_transform = A.Compose([
    A.RandomRotate90(p=0.5),
    A.Flip(p=0.5),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
    A.GaussianBlur(blur_limit=(3, 7), p=0.3),
    A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.3),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

# 應用到資料集
train_dataset = TileDataset(slide_dataset, transform=train_transform)
val_dataset = TileDataset(val_slide_dataset, transform=val_transform)
```

## 模型評估

### 指標

使用病理專用指標評估模型效能：

```python
from pathml.ml.metrics import (
    dice_coefficient,
    aggregated_jaccard_index,
    panoptic_quality
)

# 分割的 Dice 係數
dice = dice_coefficient(pred_mask, true_mask)

# 實例分割的聚合 Jaccard 指數（AJI）
aji = aggregated_jaccard_index(pred_inst, true_inst)

# 聯合分割和分類的 Panoptic Quality（PQ）
pq, sq, rq = panoptic_quality(pred_inst, true_inst, pred_types, true_types)

print(f"Dice：{dice:.4f}")
print(f"AJI：{aji:.4f}")
print(f"PQ：{pq:.4f}，SQ：{sq:.4f}，RQ：{rq:.4f}")
```

### 評估迴圈

```python
from pathml.ml.metrics import evaluate_hovernet

# 全面的 HoVer-Net 評估
model.eval()
all_preds = []
all_targets = []

with torch.no_grad():
    for batch in test_loader:
        images = batch['image'].to(device)
        outputs = model(images)

        # 後處理預測
        for i in range(len(images)):
            inst_pred, type_pred = hovernet_postprocess(
                outputs['np'][i],
                outputs['hv'][i],
                outputs['nc'][i]
            )
            all_preds.append({'inst': inst_pred, 'type': type_pred})
            all_targets.append({
                'inst': batch['inst_map'][i],
                'type': batch['type_map'][i]
            })

# 計算指標
results = evaluate_hovernet(all_preds, all_targets)

print(f"檢測 F1：{results['detection_f1']:.4f}")
print(f"分類準確率：{results['classification_acc']:.4f}")
print(f"Panoptic Quality：{results['pq']:.4f}")
```

## ONNX 推論

使用 ONNX 部署模型以進行生產推論：

### 匯出到 ONNX

```python
import torch
from pathml.ml import HoVerNet

# 載入訓練過的模型
model = HoVerNet(num_types=5, pretrained=True)
model.eval()

# 創建虛擬輸入
dummy_input = torch.randn(1, 3, 256, 256)

# 匯出到 ONNX
torch.onnx.export(
    model,
    dummy_input,
    'hovernet_model.onnx',
    export_params=True,
    opset_version=11,
    input_names=['input'],
    output_names=['np_output', 'hv_output', 'nc_output'],
    dynamic_axes={
        'input': {0: 'batch_size'},
        'np_output': {0: 'batch_size'},
        'hv_output': {0: 'batch_size'},
        'nc_output': {0: 'batch_size'}
    }
)
```

### ONNX Runtime 推論

```python
import onnxruntime as ort
import numpy as np

# 載入 ONNX 模型
session = ort.InferenceSession('hovernet_model.onnx')

# 準備輸入
input_name = session.get_inputs()[0].name
tile_image = preprocess_tile(tile)  # 正規化，轉置為 (1, 3, H, W)

# 執行推論
outputs = session.run(None, {input_name: tile_image})
np_output, hv_output, nc_output = outputs

# 後處理
inst_map, type_map = hovernet_postprocess(np_output, hv_output, nc_output)
```

### 批次推論管線

```python
from pathml.core import SlideData
from pathml.preprocessing import Pipeline
import onnxruntime as ort

def run_onnx_inference_pipeline(slide_path, onnx_model_path):
    # 載入切片
    wsi = SlideData.from_slide(slide_path)
    wsi.generate_tiles(level=1, tile_size=256, stride=256)

    # 載入 ONNX 模型
    session = ort.InferenceSession(onnx_model_path)
    input_name = session.get_inputs()[0].name

    # 對所有圖磚推論
    results = []
    for tile in wsi.tiles:
        # 預處理
        tile_array = preprocess_tile(tile.image)

        # 推論
        outputs = session.run(None, {input_name: tile_array})

        # 後處理
        inst_map, type_map = hovernet_postprocess(*outputs)

        results.append({
            'coords': tile.coords,
            'instance_map': inst_map,
            'type_map': type_map
        })

    return results

# 在切片上執行
results = run_onnx_inference_pipeline('slide.svs', 'hovernet_model.onnx')
```

## 遷移學習

在自定義資料集上微調預訓練模型：

```python
from pathml.ml import HoVerNet

# 載入預訓練模型
model = HoVerNet(num_types=5, pretrained=True)

# 凍結編碼器層進行初始訓練
for name, param in model.named_parameters():
    if 'encoder' in name:
        param.requires_grad = False

# 僅微調解碼器和分類頭
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-4
)

# 訓練幾個週期
train_for_n_epochs(model, train_loader, optimizer, num_epochs=10)

# 解凍所有層進行完整微調
for param in model.parameters():
    param.requires_grad = True

# 以較低學習率繼續訓練
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
train_for_n_epochs(model, train_loader, optimizer, num_epochs=50)
```

## 最佳實踐

1. **可用時使用預訓練模型：**
   - 從 pretrained=True 開始以獲得更好的初始化
   - 在領域特定資料上微調

2. **應用適當的資料增強：**
   - 旋轉、翻轉以實現方向不變性
   - 顏色抖動以處理染色變異
   - 彈性變形以適應生物變異性

3. **監控多個指標：**
   - 分別追蹤檢測、分割和分類
   - 使用領域特定指標（AJI、PQ）而非僅標準準確率

4. **處理類別不平衡：**
   - 對稀有細胞類型使用加權損失函數
   - 過採樣少數類別
   - 對困難樣本使用 Focal loss

5. **在多種組織類型上驗證：**
   - 確保在不同組織間的泛化
   - 在保留的解剖部位上測試

6. **最佳化推論：**
   - 匯出到 ONNX 以加速部署
   - 批次處理圖磚以高效利用 GPU
   - 可能時使用混合精度（FP16）

7. **定期儲存檢查點：**
   - 根據驗證指標保留最佳模型
   - 儲存最佳化器狀態以便恢復訓練

## 常見問題與解決方案

**問題：細胞核邊界分割效果差**
- 使用 HV 圖（水平-垂直）分離接觸的細胞核
- 增加 HV 損失項的權重
- 應用形態學後處理

**問題：相似細胞類型的誤分類**
- 增加分類損失權重
- 添加階層式分類（HACTNet）
- 對混淆類別增強訓練資料

**問題：訓練不穩定或不收斂**
- 降低學習率
- 使用梯度裁剪：`torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`
- 檢查資料預處理問題

**問題：訓練時記憶體不足**
- 減少批次大小
- 使用梯度累積
- 啟用混合精度訓練：`torch.cuda.amp`

**問題：模型過擬合訓練資料**
- 增加資料增強
- 添加 dropout 層
- 減少模型容量
- 根據驗證損失使用早停

## 其他資源

- **PathML ML API：** https://pathml.readthedocs.io/en/latest/api_ml_reference.html
- **HoVer-Net 論文：** Graham 等人，"HoVer-Net: Simultaneous Segmentation and Classification of Nuclei in Multi-Tissue Histology Images"，Medical Image Analysis，2019
- **PanNuke 資料集：** https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke
- **PyTorch Lightning：** https://www.pytorchlightning.ai/
- **ONNX Runtime：** https://onnxruntime.ai/
