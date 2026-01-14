# 核心概念和技術細節

## 概述

本參考文件涵蓋 TorchDrug 的基本架構、設計原則和技術實作細節。

## 架構理念

### 模組化設計

TorchDrug 將關注點分離為不同的模組：

1. **表示模型**（models.py）：將圖編碼為嵌入向量
2. **任務定義**（tasks.py）：定義學習目標和評估
3. **資料處理**（data.py、datasets.py）：圖結構和資料集
4. **核心組件**（core.py）：基礎類別和工具

**優點：**
- 跨任務重用表示
- 混合搭配組件
- 易於實驗和原型開發
- 清晰的關注點分離

### 可配置系統

所有組件繼承自 `core.Configurable`：
- 序列化為配置字典
- 從配置重建
- 儲存和載入完整管線
- 可重現的實驗

## 核心組件

### core.Configurable

所有 TorchDrug 組件的基礎類別。

**關鍵方法：**
- `config_dict()`：序列化為字典
- `load_config_dict(config)`：從字典載入
- `save(file)`：儲存到檔案
- `load(file)`：從檔案載入

**範例：**
```python
from torchdrug import core, models

model = models.GIN(input_dim=10, hidden_dims=[256, 256])

# 儲存配置
config = model.config_dict()
# {'class': 'GIN', 'input_dim': 10, 'hidden_dims': [256, 256], ...}

# 重建模型
model2 = core.Configurable.load_config_dict(config)
```

### core.Registry

用於註冊模型、任務和資料集的裝飾器。

**用法：**
```python
from torchdrug import core as core_td

@core_td.register("models.CustomModel")
class CustomModel(nn.Module, core_td.Configurable):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)

    def forward(self, graph, input, all_loss, metric):
        # 模型實作
        pass
```

**優點：**
- 模型自動可序列化
- 基於字串的模型指定
- 易於模型查找和實例化

## 資料結構

### Graph

表示分子或蛋白質圖的核心資料結構。

**屬性：**
- `num_node`：節點數量
- `num_edge`：邊數量
- `node_feature`：節點特徵張量 [num_node, feature_dim]
- `edge_feature`：邊特徵張量 [num_edge, feature_dim]
- `edge_list`：邊連接 [num_edge, 2 or 3]
- `num_relation`：邊類型數量（用於多關係）

**方法：**
- `node_mask(mask)`：選擇節點子集
- `edge_mask(mask)`：選擇邊子集
- `undirected()`：使圖無向
- `directed()`：使圖有向

**批次處理：**
- 圖被批次處理成單一斷開的圖
- DataLoader 中自動批次處理
- 保留每個圖的節點/邊索引

### Molecule（擴展 Graph）

分子的專用圖。

**額外屬性：**
- `atom_type`：原子序數
- `bond_type`：鍵類型（單鍵、雙鍵、三鍵、芳香）
- `formal_charge`：原子形式電荷
- `explicit_hs`：顯式氫計數

**方法：**
- `from_smiles(smiles)`：從 SMILES 字串建立
- `from_molecule(mol)`：從 RDKit 分子建立
- `to_smiles()`：轉換為 SMILES
- `to_molecule()`：轉換為 RDKit 分子
- `ion_to_molecule()`：中和電荷

**範例：**
```python
from torchdrug import data

# 從 SMILES
mol = data.Molecule.from_smiles("CCO")

# 原子特徵
print(mol.atom_type)  # [6, 6, 8] (C, C, O)
print(mol.bond_type)  # [1, 1] (單鍵)
```

### Protein（擴展 Graph）

蛋白質的專用圖。

**額外屬性：**
- `residue_type`：胺基酸類型
- `atom_name`：原子名稱（CA、CB 等）
- `atom_type`：原子序數
- `residue_number`：殘基編號
- `chain_id`：鏈標識符

**方法：**
- `from_pdb(pdb_file)`：從 PDB 檔案載入
- `from_sequence(sequence)`：從序列建立
- `to_pdb(pdb_file)`：儲存為 PDB 檔案

**圖構建：**
- 節點通常表示殘基（非原子）
- 邊可以是序列性、空間性（KNN）或基於接觸
- 可配置的邊構建策略

**範例：**
```python
from torchdrug import data

# 載入蛋白質
protein = data.Protein.from_pdb("1a3x.pdb")

# 建構具有多種邊類型的圖
graph = protein.residue_graph(
    node_position="ca",  # 使用 Cα 位置
    edge_types=["sequential", "radius"]  # 序列 + 空間邊
)
```

### PackedGraph

用於異質圖的高效批次處理結構。

**目的：**
- 批次處理不同大小的圖
- 單一 GPU 記憶體配置
- 高效並行處理

**屬性：**
- `num_nodes`：每個圖的節點計數列表
- `num_edges`：每個圖的邊計數列表
- `graph_ind`：每個節點的圖索引

**使用案例：**
- DataLoader 中自動使用
- 自訂批次處理策略
- 多圖操作

## 模型介面

### Forward 函數簽名

所有 TorchDrug 模型遵循標準化介面：

```python
def forward(self, graph, input, all_loss=None, metric=None):
    """
    參數：
        graph (Graph)：圖的批次
        input (Tensor)：節點輸入特徵
        all_loss (Tensor, optional)：損失累加器
        metric (dict, optional)：指標字典

    返回：
        dict：包含表示鍵的輸出字典
    """
    # 模型計算
    output = self.layers(graph, input)

    return {
        "node_feature": output,
        "graph_feature": graph_pooling(output)
    }
```

**關鍵點：**
- `graph`：批次圖結構
- `input`：節點特徵 [num_node, input_dim]
- `all_loss`：累積損失（用於多任務）
- `metric`：共享指標字典
- 返回包含表示類型的字典

### 必要屬性

**所有模型必須定義：**
- `input_dim`：預期輸入特徵維度
- `output_dim`：輸出表示維度

**目的：**
- 自動維度檢查
- 在管線中組合模型
- 錯誤檢查和驗證

**範例：**
```python
class CustomModel(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = hidden_dim
        # ... 層 ...
```

## 任務介面

### 核心任務方法

所有任務實作這些方法：

```python
class CustomTask(tasks.Task):
    def preprocess(self, train_set, valid_set, test_set):
        """資料集特定預處理（可選）"""
        pass

    def predict(self, batch):
        """為批次生成預測"""
        graph, label = batch
        output = self.model(graph, graph.node_feature)
        pred = self.mlp(output["graph_feature"])
        return pred

    def target(self, batch):
        """提取真實標籤"""
        graph, label = batch
        return label

    def forward(self, batch):
        """計算訓練損失"""
        pred = self.predict(batch)
        target = self.target(batch)
        loss = self.criterion(pred, target)
        return loss

    def evaluate(self, pred, target):
        """計算評估指標"""
        metrics = {}
        metrics["auroc"] = compute_auroc(pred, target)
        metrics["auprc"] = compute_auprc(pred, target)
        return metrics
```

### 任務組件

**典型任務結構：**
1. **表示模型**：將圖編碼為嵌入向量
2. **讀出/預測頭**：將嵌入向量映射到預測
3. **損失函數**：訓練目標
4. **指標**：評估度量

**範例：**
```python
from torchdrug import tasks, models

# 表示模型
model = models.GIN(input_dim=10, hidden_dims=[256, 256])

# 任務用預測頭包裝模型
task = tasks.PropertyPrediction(
    model=model,
    task=["task1", "task2"],  # 多任務
    criterion="bce",
    metric=["auroc", "auprc"],
    num_mlp_layer=2
)
```

## 訓練工作流程

### 標準訓練循環

```python
import torch
from torch.utils.data import DataLoader
from torchdrug import core, models, tasks, datasets

# 1. 載入資料集
dataset = datasets.BBBP("~/datasets/")
train_set, valid_set, test_set = dataset.split()

# 2. 建立資料載入器
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_set, batch_size=32)

# 3. 定義模型和任務
model = models.GIN(input_dim=dataset.node_feature_dim,
                   hidden_dims=[256, 256, 256])
task = tasks.PropertyPrediction(model, task=dataset.tasks,
                                 criterion="bce", metric=["auroc", "auprc"])

# 4. 設定最佳化器
optimizer = torch.optim.Adam(task.parameters(), lr=1e-3)

# 5. 訓練循環
for epoch in range(100):
    # 訓練
    task.train()
    for batch in train_loader:
        loss = task(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 驗證
    task.eval()
    preds, targets = [], []
    for batch in valid_loader:
        pred = task.predict(batch)
        target = task.target(batch)
        preds.append(pred)
        targets.append(target)

    preds = torch.cat(preds)
    targets = torch.cat(targets)
    metrics = task.evaluate(preds, targets)
    print(f"Epoch {epoch}: {metrics}")
```

### PyTorch Lightning 整合

TorchDrug 任務與 PyTorch Lightning 相容：

```python
import pytorch_lightning as pl

class LightningWrapper(pl.LightningModule):
    def __init__(self, task):
        super().__init__()
        self.task = task

    def training_step(self, batch, batch_idx):
        loss = self.task(batch)
        return loss

    def validation_step(self, batch, batch_idx):
        pred = self.task.predict(batch)
        target = self.task.target(batch)
        return {"pred": pred, "target": target}

    def validation_epoch_end(self, outputs):
        preds = torch.cat([o["pred"] for o in outputs])
        targets = torch.cat([o["target"] for o in outputs])
        metrics = self.task.evaluate(preds, targets)
        self.log_dict(metrics)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
```

## 損失函數

### 內建準則

**分類：**
- `"bce"`：二元交叉熵
- `"ce"`：交叉熵（多類）

**迴歸：**
- `"mse"`：均方誤差
- `"mae"`：平均絕對誤差

**知識圖譜：**
- `"bce"`：三元組的二元分類
- `"ce"`：交叉熵排名損失
- `"margin"`：基於邊界的排名

### 自訂損失

```python
class CustomTask(tasks.Task):
    def forward(self, batch):
        pred = self.predict(batch)
        target = self.target(batch)

        # 自訂損失計算
        loss = custom_loss_function(pred, target)

        return loss
```

## 指標

### 常見指標

**分類：**
- **AUROC**：ROC 曲線下面積
- **AUPRC**：精確度-召回率曲線下面積
- **Accuracy**：整體準確度
- **F1**：精確度和召回率的調和平均

**迴歸：**
- **MAE**：平均絕對誤差
- **RMSE**：均方根誤差
- **R**：決定係數
- **Pearson**：皮爾森相關係數

**排名（知識圖譜）：**
- **MR**：平均排名
- **MRR**：平均倒數排名
- **Hits@K**：前 K 名中的百分比

### 多任務指標

對於多標籤或多任務：
- 每個任務計算指標
- 跨任務的宏觀平均
- 可按任務重要性加權

## 資料轉換

### 分子轉換

```python
from torchdrug import transforms

# 添加連接到所有原子的虛擬節點
transform1 = transforms.VirtualNode()

# 添加虛擬邊
transform2 = transforms.VirtualEdge()

# 組合轉換
transform = transforms.Compose([transform1, transform2])

dataset = datasets.BBBP("~/datasets/", transform=transform)
```

### 蛋白質轉換

```python
# 基於空間鄰近性添加邊
transform = transforms.TruncateProtein(max_length=500)

dataset = datasets.Fold("~/datasets/", transform=transform)
```

## 最佳實踐

### 記憶體效率

1. **梯度累積**：用於大型模型
2. **混合精度**：FP16 訓練
3. **批次大小調整**：平衡速度和記憶體
4. **資料載入**：多個工作程序用於 I/O

### 可重現性

1. **設定種子**：PyTorch、NumPy、Python 隨機
2. **確定性操作**：`torch.use_deterministic_algorithms(True)`
3. **儲存配置**：使用 `core.Configurable`
4. **版本控制**：追蹤 TorchDrug 版本

### 除錯

1. **檢查維度**：驗證 `input_dim` 和 `output_dim`
2. **驗證批次處理**：列印批次統計
3. **監控梯度**：注意梯度消失/爆炸
4. **過擬合小批次**：確保模型容量

### 效能最佳化

1. **GPU 利用率**：使用 `nvidia-smi` 監控
2. **剖析程式碼**：使用 PyTorch 分析器
3. **最佳化資料載入**：預取、固定記憶體
4. **編譯模型**：如果可能使用 TorchScript

## 進階主題

### 多任務學習

在多個相關任務上訓練單一模型：
```python
task = tasks.PropertyPrediction(
    model,
    task=["task1", "task2", "task3"],
    criterion="bce",
    metric=["auroc"],
    task_weight=[1.0, 1.0, 2.0]  # 更重視任務 3
)
```

### 遷移學習

1. 在大型資料集上預訓練
2. 在目標資料集上微調
3. 可選凍結早期層

### 自監督預訓練

使用預訓練任務：
- `AttributeMasking`：遮罩節點特徵
- `EdgePrediction`：預測邊存在
- `ContextPrediction`：對比學習

### 自訂層

使用自訂 GNN 層擴展 TorchDrug：
```python
from torchdrug import layers

class CustomConv(layers.MessagePassingBase):
    def message(self, graph, input):
        # 自訂訊息函數
        pass

    def aggregate(self, graph, message):
        # 自訂聚合
        pass

    def combine(self, input, update):
        # 自訂組合
        pass
```

## 常見陷阱

1. **忘記 `input_dim` 和 `output_dim`**：模型將無法組合
2. **未正確批次處理**：對可變大小的圖使用 PackedGraph
3. **資料洩漏**：注意骨架分割和預訓練
4. **忽略邊特徵**：鍵/空間資訊可能很關鍵
5. **錯誤的評估指標**：將指標與任務匹配（對不平衡使用 AUROC）
6. **正則化不足**：使用 dropout、權重衰減、早停
7. **未驗證化學**：生成的分子必須有效
8. **小型資料集過擬合**：使用預訓練或更簡單的模型
