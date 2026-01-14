# 分子性質預測

## 概述

分子性質預測涉及從分子結構預測化學、物理或生物性質。TorchDrug 為分子圖上的分類和迴歸任務提供全面支援。

## 可用資料集

### 藥物發現資料集

**分類任務：**
- **BACE**（1,513 分子）：β-分泌酶抑制的二元分類
- **BBBP**（2,039 分子）：血腦屏障穿透預測
- **HIV**（41,127 分子）：抑制 HIV 複製的能力
- **Tox21**（7,831 分子）：12 個標靶的毒性預測
- **ToxCast**（8,576 分子）：毒理學篩選
- **ClinTox**（1,478 分子）：臨床試驗毒性
- **SIDER**（1,427 分子）：藥物副作用（27 個系統器官類別）
- **MUV**（93,087 分子）：虛擬篩選的最大無偏驗證

**迴歸任務：**
- **ESOL**（1,128 分子）：水溶性預測
- **FreeSolv**（642 分子）：水合自由能
- **Lipophilicity**（4,200 分子）：辛醇/水分配係數
- **SAMPL**（643 分子）：溶劑化自由能

### 大規模資料集

- **QM7**（7,165 分子）：量子力學性質
- **QM8**（21,786 分子）：電子光譜和激發態性質
- **QM9**（133,885 分子）：幾何、能量、電子和熱力學性質
- **PCQM4M**（3,803,453 分子）：大規模量子化學資料集
- **ZINC250k/2M**（250k/2M 分子）：用於生成模型的類藥化合物

## 任務類型

### PropertyPrediction

支援分類和迴歸的圖級性質預測標準任務。

**關鍵參數：**
- `model`：圖表示模型（GNN）
- `task`：「node」、「edge」或「graph」級預測
- `criterion`：損失函數（「mse」、「bce」、「ce」）
- `metric`：評估指標（「mae」、「rmse」、「auroc」、「auprc」）
- `num_mlp_layer`：讀出的 MLP 層數

**範例工作流程：**
```python
import torch
from torchdrug import core, models, tasks, datasets

# 載入資料集
dataset = datasets.BBBP("~/molecule-datasets/")

# 定義模型
model = models.GIN(input_dim=dataset.node_feature_dim,
                   hidden_dims=[256, 256, 256, 256],
                   edge_input_dim=dataset.edge_feature_dim,
                   batch_norm=True, readout="mean")

# 定義任務
task = tasks.PropertyPrediction(model, task=dataset.tasks,
                                 criterion="bce",
                                 metric=("auprc", "auroc"))
```

### MultipleBinaryClassification

專門用於多標籤場景的任務，其中每個分子可以有多個二元標籤（如 Tox21、SIDER）。

**關鍵特徵：**
- 優雅處理缺失標籤
- 按標籤和平均計算指標
- 支援不平衡資料集的加權損失

## 模型選擇

### 按任務推薦的模型

**小型分子（< 1000 分子）：**
- GIN（圖同構網路）
- SchNet（用於 3D 結構）

**中型資料集（1k-100k 分子）：**
- GCN、GAT 或 GIN
- NFP（神經指紋）
- MPNN（訊息傳遞神經網路）

**大型資料集（> 100k 分子）：**
- 帶微調的預訓練模型
- InfoGraph 或 MultiviewContrast 用於自監督預訓練
- 更深架構的 GIN

**3D 結構可用時：**
- SchNet（連續濾波卷積）
- GearNet（幾何感知關係圖）

## 特徵工程

### 節點特徵

TorchDrug 自動提取原子特徵：
- 原子類型
- 形式電荷
- 顯式/隱式氫
- 雜化
- 芳香性
- 手性

### 邊特徵

鍵特徵包括：
- 鍵類型（單鍵、雙鍵、三鍵、芳香）
- 立體化學
- 共軛
- 環成員資格

### 自訂特徵

使用轉換添加自訂節點/邊特徵：
```python
from torchdrug import data, transforms

# 添加自訂特徵
transform = transforms.VirtualNode()  # 添加虛擬節點
dataset = datasets.BBBP("~/molecule-datasets/",
                        transform=transform)
```

## 訓練工作流程

### 基本管線

1. **載入資料集**：選擇適當的資料集
2. **分割資料**：藥物發現使用骨架分割
3. **定義模型**：選擇 GNN 架構
4. **建立任務**：配置損失和指標
5. **設定最佳化器**：Adam 通常效果好
6. **訓練**：使用 PyTorch Lightning 或自訂循環

### 資料分割策略

**隨機分割**：標準訓練/驗證/測試分割
**骨架分割**：按 Bemis-Murcko 骨架分組分子（藥物發現推薦）
**分層分割**：跨分割維持標籤分佈

### 最佳實踐

- 使用骨架分割進行現實的藥物發現評估
- 對小型資料集應用資料增強（虛擬節點、邊）
- 監控多個指標（分類用 AUROC、AUPRC；迴歸用 MAE、RMSE）
- 基於驗證效能使用早停
- 對關鍵應用考慮集成方法
- 在微調小型資料集前在大型資料集上預訓練

## 常見問題和解決方案

**問題：不平衡資料集效能不佳**
- 解決方案：使用加權損失、focal loss 或過/欠採樣

**問題：小型資料集過擬合**
- 解決方案：增加正則化、使用更簡單模型、應用資料增強，或在更大資料集上預訓練

**問題：大量記憶體消耗**
- 解決方案：減少批次大小、使用梯度累積，或實作圖採樣

**問題：訓練緩慢**
- 解決方案：使用 GPU 加速、使用多個工作程序最佳化資料載入，或使用混合精度訓練
