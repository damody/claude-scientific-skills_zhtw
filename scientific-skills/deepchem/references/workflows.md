# DeepChem 工作流程

本文件提供常見 DeepChem 使用案例的詳細工作流程。

## 工作流程 1：從 SMILES 進行分子屬性預測

**目標**：從 SMILES 字串預測分子屬性（例如溶解度、毒性、活性）。

### 逐步流程

#### 1. 準備資料
資料應為 CSV 格式，至少包含：
- 包含 SMILES 字串的欄位
- 一個或多個包含屬性值（目標）的欄位

CSV 結構範例：
```csv
smiles,solubility,toxicity
CCO,-0.77,0
CC(=O)OC1=CC=CC=C1C(=O)O,-1.19,1
```

#### 2. 選擇特徵化器
決策樹：
- **小型資料集（<1K）**：使用 `CircularFingerprint` 或 `RDKitDescriptors`
- **中型資料集（1K-100K）**：使用 `CircularFingerprint` 或 `MolGraphConvFeaturizer`
- **大型資料集（>100K）**：使用基於圖的特徵化器（`MolGraphConvFeaturizer`、`DMPNNFeaturizer`）
- **遷移學習**：使用預訓練模型特徵化器（`GroverFeaturizer`）

#### 3. 載入並特徵化資料
```python
import deepchem as dc

# 用於基於指紋的
featurizer = dc.feat.CircularFingerprint(radius=2, size=2048)
# 或用於基於圖的
featurizer = dc.feat.MolGraphConvFeaturizer()

loader = dc.data.CSVLoader(
    tasks=['solubility', 'toxicity'],  # 要預測的欄位名稱
    feature_field='smiles',             # 包含 SMILES 的欄位
    featurizer=featurizer
)
dataset = loader.create_dataset('data.csv')
```

#### 4. 分割資料
**關鍵**：藥物發現使用 `ScaffoldSplitter` 以防止資料洩漏。

```python
splitter = dc.splits.ScaffoldSplitter()
train, valid, test = splitter.train_valid_test_split(
    dataset,
    frac_train=0.8,
    frac_valid=0.1,
    frac_test=0.1
)
```

#### 5. 轉換資料（可選但建議）
```python
transformers = [
    dc.trans.NormalizationTransformer(
        transform_y=True,
        dataset=train
    )
]

for transformer in transformers:
    train = transformer.transform(train)
    valid = transformer.transform(valid)
    test = transformer.transform(test)
```

#### 6. 選擇並訓練模型
```python
# 用於指紋
model = dc.models.MultitaskRegressor(
    n_tasks=2,                    # 要預測的屬性數量
    n_features=2048,              # 指紋大小
    layer_sizes=[1000, 500],      # 隱藏層大小
    dropouts=0.25,
    learning_rate=0.001
)

# 或用於圖
model = dc.models.GCNModel(
    n_tasks=2,
    mode='regression',
    batch_size=128,
    learning_rate=0.001
)

# 訓練
model.fit(train, nb_epoch=50)
```

#### 7. 評估
```python
metric = dc.metrics.Metric(dc.metrics.r2_score)
train_score = model.evaluate(train, [metric])
valid_score = model.evaluate(valid, [metric])
test_score = model.evaluate(test, [metric])

print(f"訓練 R²：{train_score}")
print(f"驗證 R²：{valid_score}")
print(f"測試 R²：{test_score}")
```

#### 8. 進行預測
```python
# 對新分子進行預測
new_smiles = ['CCO', 'CC(C)O', 'c1ccccc1']
new_featurizer = dc.feat.CircularFingerprint(radius=2, size=2048)
new_features = new_featurizer.featurize(new_smiles)
new_dataset = dc.data.NumpyDataset(X=new_features)

# 應用相同的轉換
for transformer in transformers:
    new_dataset = transformer.transform(new_dataset)

predictions = model.predict(new_dataset)
```

---

## 工作流程 2：使用 MoleculeNet 基準資料集

**目標**：在標準基準上快速訓練和評估模型。

### 快速開始
```python
import deepchem as dc

# 載入基準資料集
tasks, datasets, transformers = dc.molnet.load_tox21(
    featurizer='GraphConv',
    splitter='scaffold'
)
train, valid, test = datasets

# 訓練模型
model = dc.models.GCNModel(
    n_tasks=len(tasks),
    mode='classification'
)
model.fit(train, nb_epoch=50)

# 評估
metric = dc.metrics.Metric(dc.metrics.roc_auc_score)
test_score = model.evaluate(test, [metric])
print(f"測試 ROC-AUC：{test_score}")
```

### 可用特徵化器選項
呼叫 `load_*()` 函數時：
- `'ECFP'`：擴展連接指紋（環形指紋）
- `'GraphConv'`：圖卷積特徵
- `'Weave'`：Weave 特徵
- `'Raw'`：原始 SMILES 字串
- `'smiles2img'`：2D 分子影像

### 可用分割器選項
- `'scaffold'`：基於骨架的分割（藥物發現建議使用）
- `'random'`：隨機分割
- `'stratified'`：分層分割（保留類別分佈）
- `'butina'`：基於 Butina 聚類的分割

---

## 工作流程 3：超參數優化

**目標**：系統性地找到最佳模型超參數。

### 使用 GridHyperparamOpt
```python
import deepchem as dc
import numpy as np

# 載入資料
tasks, datasets, transformers = dc.molnet.load_bbbp(
    featurizer='ECFP',
    splitter='scaffold'
)
train, valid, test = datasets

# 定義參數網格
params_dict = {
    'layer_sizes': [[1000], [1000, 500], [1000, 1000]],
    'dropouts': [0.0, 0.25, 0.5],
    'learning_rate': [0.001, 0.0001]
}

# 定義模型建構函數
def model_builder(model_params, model_dir):
    return dc.models.MultitaskClassifier(
        n_tasks=len(tasks),
        n_features=1024,
        **model_params
    )

# 設定優化器
metric = dc.metrics.Metric(dc.metrics.roc_auc_score)
optimizer = dc.hyper.GridHyperparamOpt(model_builder)

# 執行優化
best_model, best_params, all_results = optimizer.hyperparam_search(
    params_dict,
    train,
    valid,
    metric,
    transformers=transformers
)

print(f"最佳參數：{best_params}")
print(f"最佳驗證分數：{all_results['best_validation_score']}")
```

---

## 工作流程 4：使用預訓練模型的遷移學習

**目標**：利用預訓練模型改善小型資料集的效能。

### 使用 ChemBERTa
```python
import deepchem as dc
from transformers import AutoTokenizer

# 載入資料
loader = dc.data.CSVLoader(
    tasks=['activity'],
    feature_field='smiles',
    featurizer=dc.feat.DummyFeaturizer()  # ChemBERTa 處理特徵化
)
dataset = loader.create_dataset('data.csv')

# 分割資料
splitter = dc.splits.ScaffoldSplitter()
train, test = splitter.train_test_split(dataset)

# 載入預訓練 ChemBERTa
model = dc.models.HuggingFaceModel(
    model='seyonec/ChemBERTa-zinc-base-v1',
    task='regression',
    n_tasks=1
)

# 微調
model.fit(train, nb_epoch=10)

# 評估
predictions = model.predict(test)
```

### 使用 GROVER
```python
# GROVER：在分子圖上預訓練
model = dc.models.GroverModel(
    task='classification',
    n_tasks=1,
    model_dir='./grover_model'
)

# 在您的資料上微調
model.fit(train_dataset, nb_epoch=20)
```

---

## 工作流程 5：使用 GAN 的分子生成

**目標**：生成具有所需屬性的新穎分子。

### 基本 MolGAN
```python
import deepchem as dc

# 載入訓練資料（供生成器學習的分子）
tasks, datasets, _ = dc.molnet.load_qm9(
    featurizer='GraphConv',
    splitter='random'
)
train, _, _ = datasets

# 建立並訓練 MolGAN
gan = dc.models.BasicMolGANModel(
    learning_rate=0.001,
    vertices=9,  # 分子中最大原子數
    edges=5,     # 最大鍵數
    nodes=[128, 256, 512]
)

# 訓練
gan.fit_gan(
    train,
    nb_epoch=100,
    generator_steps=0.2,
    checkpoint_interval=10
)

# 生成新分子
generated_molecules = gan.predict_gan_generator(1000)
```

### 條件生成
```python
# 用於屬性導向生成
from deepchem.models.optimizers import ExponentialDecay

gan = dc.models.BasicMolGANModel(
    learning_rate=ExponentialDecay(0.001, 0.9, 1000),
    conditional=True  # 啟用條件生成
)

# 使用屬性訓練
gan.fit_gan(train, nb_epoch=100)

# 生成具有目標屬性的分子
target_properties = np.array([[5.0, 300.0]])  # 例如 [logP, MW]
molecules = gan.predict_gan_generator(
    1000,
    conditional_inputs=target_properties
)
```

---

## 工作流程 6：材料屬性預測

**目標**：預測晶態材料的屬性。

### 使用晶體圖卷積網路
```python
import deepchem as dc

# 載入材料資料（CIF 格式的結構檔案）
loader = dc.data.CIFLoader()
dataset = loader.create_dataset('materials.csv')

# 分割資料
splitter = dc.splits.RandomSplitter()
train, test = splitter.train_test_split(dataset)

# 建立 CGCNN 模型
model = dc.models.CGCNNModel(
    n_tasks=1,
    mode='regression',
    batch_size=32,
    learning_rate=0.001
)

# 訓練
model.fit(train, nb_epoch=100)

# 評估
metric = dc.metrics.Metric(dc.metrics.mae_score)
test_score = model.evaluate(test, [metric])
```

---

## 工作流程 7：蛋白質序列分析

**目標**：從序列預測蛋白質屬性。

### 使用 ProtBERT
```python
import deepchem as dc

# 載入蛋白質序列資料
loader = dc.data.FASTALoader()
dataset = loader.create_dataset('proteins.fasta')

# 使用 ProtBERT
model = dc.models.HuggingFaceModel(
    model='Rostlab/prot_bert',
    task='classification',
    n_tasks=1
)

# 分割並訓練
splitter = dc.splits.RandomSplitter()
train, test = splitter.train_test_split(dataset)
model.fit(train, nb_epoch=5)

# 預測
predictions = model.predict(test)
```

---

## 工作流程 8：自訂模型整合

**目標**：使用您自己的 PyTorch/scikit-learn 模型與 DeepChem。

### 包裝 Scikit-Learn 模型
```python
from sklearn.ensemble import RandomForestRegressor
import deepchem as dc

# 建立 scikit-learn 模型
sklearn_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=42
)

# 包裝在 DeepChem 中
model = dc.models.SklearnModel(model=sklearn_model)

# 與 DeepChem 資料集一起使用
model.fit(train)
predictions = model.predict(test)

# 評估
metric = dc.metrics.Metric(dc.metrics.r2_score)
score = model.evaluate(test, [metric])
```

### 建立自訂 PyTorch 模型
```python
import torch
import torch.nn as nn
import deepchem as dc

class CustomNetwork(nn.Module):
    def __init__(self, n_features, n_tasks):
        super().__init__()
        self.fc1 = nn.Linear(n_features, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, n_tasks)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)

# 包裝在 DeepChem TorchModel 中
model = dc.models.TorchModel(
    model=CustomNetwork(n_features=2048, n_tasks=1),
    loss=nn.MSELoss(),
    output_types=['prediction']
)

# 訓練
model.fit(train, nb_epoch=50)
```

---

## 常見陷阱和解決方案

### 問題 1：藥物發現中的資料洩漏
**問題**：使用隨機分割允許訓練和測試集中存在相似分子。
**解決方案**：始終對分子資料集使用 `ScaffoldSplitter`。

### 問題 2：不平衡分類
**問題**：少數類別效能差。
**解決方案**：使用 `BalancingTransformer` 或加權指標。
```python
transformer = dc.trans.BalancingTransformer(dataset=train)
train = transformer.transform(train)
```

### 問題 3：大型資料集的記憶體問題
**問題**：資料集無法載入記憶體。
**解決方案**：使用 `DiskDataset` 而非 `NumpyDataset`。
```python
dataset = dc.data.DiskDataset.from_numpy(X, y, w, ids)
```

### 問題 4：小型資料集上的過擬合
**問題**：模型記住訓練資料。
**解決方案**：
1. 使用更強的正則化（增加 dropout）
2. 使用更簡單的模型（Random Forest、Ridge）
3. 應用遷移學習（預訓練模型）
4. 收集更多資料

### 問題 5：圖神經網路效能差
**問題**：GNN 效能比指紋差。
**解決方案**：
1. 檢查資料集是否足夠大（GNN 通常需要 >10K 樣本）
2. 增加訓練輪數
3. 嘗試不同的 GNN 架構（AttentiveFP、DMPNN）
4. 使用預訓練模型（GROVER）
