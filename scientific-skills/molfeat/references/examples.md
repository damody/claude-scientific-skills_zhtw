# Molfeat 使用範例

本文件提供常見 molfeat 使用案例的實用範例。

## 安裝

```bash
# 推薦：使用 conda/mamba
mamba install -c conda-forge molfeat

# 替代方案：使用 pip
pip install molfeat

# 包含所有可選依賴
pip install "molfeat[all]"

# 包含特定依賴
pip install "molfeat[dgl]"          # 用於 GNN 模型
pip install "molfeat[graphormer]"   # 用於 Graphormer
pip install "molfeat[transformer]"  # 用於 ChemBERTa、ChemGPT
```

---

## 快速入門

### 基本特徵化工作流程

```python
import datamol as dm
from molfeat.calc import FPCalculator
from molfeat.trans import MoleculeTransformer

# 載入範例資料
data = dm.data.freesolv().sample(100).smiles.values

# 單分子特徵化
calc = FPCalculator("ecfp")
features_single = calc(data[0])
print(f"Single molecule features shape: {features_single.shape}")
# 輸出：(2048,)

# 帶平行化的批次特徵化
transformer = MoleculeTransformer(calc, n_jobs=-1)
features_batch = transformer(data)
print(f"Batch features shape: {features_batch.shape}")
# 輸出：(100, 2048)
```

---

## 計算器範例

### 指紋計算器

```python
from molfeat.calc import FPCalculator

# ECFP（擴展連接指紋）
ecfp = FPCalculator("ecfp", radius=3, fpSize=2048)
fp = ecfp("CCO")  # 乙醇
print(f"ECFP shape: {fp.shape}")  # (2048,)

# MACCS 鍵
maccs = FPCalculator("maccs")
fp = maccs("c1ccccc1")  # 苯
print(f"MACCS shape: {fp.shape}")  # (167,)

# 計數指紋
ecfp_count = FPCalculator("ecfp-count", radius=3)
fp_count = ecfp_count("CC(C)CC(C)C")  # 非二元計數

# MAP4 指紋
map4 = FPCalculator("map4")
fp = map4("CC(=O)Oc1ccccc1C(=O)O")  # 阿斯匹靈
```

### 描述符計算器

```python
from molfeat.calc import RDKitDescriptors2D, MordredDescriptors

# RDKit 2D 描述符（200+ 屬性）
desc2d = RDKitDescriptors2D()
descriptors = desc2d("CCO")
print(f"Number of 2D descriptors: {len(descriptors)}")

# 取得描述符名稱
names = desc2d.columns
print(f"First 5 descriptors: {names[:5]}")

# Mordred 描述符（1800+ 屬性）
mordred = MordredDescriptors()
descriptors = mordred("c1ccccc1O")  # 苯酚
print(f"Mordred descriptors: {len(descriptors)}")
```

### 藥效團計算器

```python
from molfeat.calc import CATSCalculator

# 2D CATS 描述符
cats = CATSCalculator(mode="2D", scale="raw")
descriptors = cats("CC(C)Cc1ccc(C)cc1C")  # 傘花烴
print(f"CATS descriptors: {descriptors.shape}")  # (21,)

# 3D CATS 描述符（需要構象）
cats3d = CATSCalculator(mode="3D", scale="num")
```

---

## 轉換器範例

### 基本轉換器使用

```python
from molfeat.trans import MoleculeTransformer
from molfeat.calc import FPCalculator
import datamol as dm

# 準備資料
smiles_list = [
    "CCO",
    "CC(=O)O",
    "c1ccccc1",
    "CC(C)O",
    "CCCC"
]

# 建立轉換器
calc = FPCalculator("ecfp")
transformer = MoleculeTransformer(calc, n_jobs=-1)

# 轉換分子
features = transformer(smiles_list)
print(f"Features shape: {features.shape}")  # (5, 2048)
```

### 錯誤處理

```python
# 優雅處理無效 SMILES
smiles_with_errors = [
    "CCO",           # 有效
    "invalid",       # 無效
    "CC(=O)O",       # 有效
    "xyz123",        # 無效
]

transformer = MoleculeTransformer(
    FPCalculator("ecfp"),
    n_jobs=-1,
    verbose=True,           # 記錄錯誤
    ignore_errors=True      # 失敗時繼續
)

features = transformer(smiles_with_errors)
# 返回：對失敗分子為 None 的陣列
print(features)  # [array(...), None, array(...), None]
```

### 連接多個特徵提取器

```python
from molfeat.trans import FeatConcat, MoleculeTransformer
from molfeat.calc import FPCalculator

# 組合 MACCS（167）+ ECFP（2048）= 2215 維
concat_calc = FeatConcat([
    FPCalculator("maccs"),
    FPCalculator("ecfp", radius=3, fpSize=2048)
])

transformer = MoleculeTransformer(concat_calc, n_jobs=-1)
features = transformer(smiles_list)
print(f"Combined features shape: {features.shape}")  # (n, 2215)

# 三重組合
triple_concat = FeatConcat([
    FPCalculator("maccs"),
    FPCalculator("ecfp"),
    FPCalculator("rdkit")
])
```

### 儲存和載入配置

```python
from molfeat.trans import MoleculeTransformer
from molfeat.calc import FPCalculator

# 建立和儲存轉換器
transformer = MoleculeTransformer(
    FPCalculator("ecfp", radius=3, fpSize=2048),
    n_jobs=-1
)

# 儲存為 YAML
transformer.to_state_yaml_file("my_featurizer.yml")

# 儲存為 JSON
transformer.to_state_json_file("my_featurizer.json")

# 從儲存的狀態載入
loaded_transformer = MoleculeTransformer.from_state_yaml_file("my_featurizer.yml")

# 使用載入的轉換器
features = loaded_transformer(smiles_list)
```

---

## 預訓練模型範例

### 使用 ModelStore

```python
from molfeat.store.modelstore import ModelStore

# 初始化模型商店
store = ModelStore()

# 列出所有可用模型
print(f"Total available models: {len(store.available_models)}")

# 搜索特定模型
chemberta_models = store.search(name="ChemBERTa")
for model in chemberta_models:
    print(f"- {model.name}: {model.description}")

# 取得模型資訊
model_card = store.search(name="ChemBERTa-77M-MLM")[0]
print(f"Model: {model_card.name}")
print(f"Version: {model_card.version}")
print(f"Authors: {model_card.authors}")

# 檢視使用說明
model_card.usage()

# 直接載入模型
transformer = store.load("ChemBERTa-77M-MLM")
```

### ChemBERTa 嵌入

```python
from molfeat.trans.pretrained import PretrainedMolTransformer

# 載入 ChemBERTa 模型
chemberta = PretrainedMolTransformer("ChemBERTa-77M-MLM", n_jobs=-1)

# 生成嵌入
smiles = ["CCO", "CC(=O)O", "c1ccccc1"]
embeddings = chemberta(smiles)
print(f"ChemBERTa embeddings shape: {embeddings.shape}")
# 輸出：(3, 768) - 768 維嵌入

# 在 ML 管線中使用
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    embeddings, labels, test_size=0.2
)

clf = RandomForestClassifier()
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
```

### ChemGPT 模型

```python
# 小型模型（470 萬參數）
chemgpt_small = PretrainedMolTransformer("ChemGPT-4.7M", n_jobs=-1)

# 中型模型（1900 萬參數）
chemgpt_medium = PretrainedMolTransformer("ChemGPT-19M", n_jobs=-1)

# 大型模型（12 億參數）
chemgpt_large = PretrainedMolTransformer("ChemGPT-1.2B", n_jobs=-1)

# 生成嵌入
embeddings = chemgpt_small(smiles)
```

### 圖神經網路模型

```python
# 具有不同預訓練目標的 GIN 模型
gin_masking = PretrainedMolTransformer("gin-supervised-masking", n_jobs=-1)
gin_infomax = PretrainedMolTransformer("gin-supervised-infomax", n_jobs=-1)
gin_edgepred = PretrainedMolTransformer("gin-supervised-edgepred", n_jobs=-1)

# 生成圖嵌入
embeddings = gin_masking(smiles)
print(f"GIN embeddings shape: {embeddings.shape}")

# Graphormer（用於量子化學）
graphormer = PretrainedMolTransformer("Graphormer-pcqm4mv2", n_jobs=-1)
embeddings = graphormer(smiles)
```

---

## 機器學習整合

### Scikit-learn 管線

```python
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from molfeat.trans import MoleculeTransformer
from molfeat.calc import FPCalculator

# 建立 ML 管線
pipeline = Pipeline([
    ('featurizer', MoleculeTransformer(FPCalculator("ecfp"), n_jobs=-1)),
    ('classifier', RandomForestClassifier(n_estimators=100))
])

# 訓練和評估
pipeline.fit(smiles_train, y_train)
predictions = pipeline.predict(smiles_test)

# 交叉驗證
scores = cross_val_score(pipeline, smiles_all, y_all, cv=5)
print(f"CV scores: {scores.mean():.3f} (+/- {scores.std():.3f})")
```

### 超參數調優的網格搜索

```python
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

# 定義管線
pipeline = Pipeline([
    ('featurizer', MoleculeTransformer(FPCalculator("ecfp"), n_jobs=-1)),
    ('classifier', SVC())
])

# 定義參數網格
param_grid = {
    'classifier__C': [0.1, 1, 10],
    'classifier__kernel': ['rbf', 'linear'],
    'classifier__gamma': ['scale', 'auto']
}

# 網格搜索
grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1)
grid_search.fit(smiles_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_:.3f}")
```

### 多特徵提取器比較

```python
from sklearn.metrics import roc_auc_score

# 測試不同的特徵提取器
featurizers = {
    'ECFP': FPCalculator("ecfp"),
    'MACCS': FPCalculator("maccs"),
    'RDKit': FPCalculator("rdkit"),
    'Descriptors': RDKitDescriptors2D(),
    'Combined': FeatConcat([
        FPCalculator("maccs"),
        FPCalculator("ecfp")
    ])
}

results = {}
for name, calc in featurizers.items():
    transformer = MoleculeTransformer(calc, n_jobs=-1)
    X_train = transformer(smiles_train)
    X_test = transformer(smiles_test)

    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train, y_train)

    y_pred = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred)
    results[name] = auc

    print(f"{name}: AUC = {auc:.3f}")
```

### PyTorch 深度學習

```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from molfeat.trans import MoleculeTransformer
from molfeat.calc import FPCalculator

# 自訂資料集
class MoleculeDataset(Dataset):
    def __init__(self, smiles, labels, transformer):
        self.features = transformer(smiles)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.features[idx], dtype=torch.float32),
            self.labels[idx]
        )

# 準備資料
transformer = MoleculeTransformer(FPCalculator("ecfp"), n_jobs=-1)
train_dataset = MoleculeDataset(smiles_train, y_train, transformer)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 簡單神經網路
class MoleculeClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)

# 訓練模型
model = MoleculeClassifier(input_dim=2048)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()

for epoch in range(10):
    for batch_features, batch_labels in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_features).squeeze()
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()
```

---

## 進階使用模式

### 自訂預處理

```python
from molfeat.trans import MoleculeTransformer
import datamol as dm

class CustomTransformer(MoleculeTransformer):
    def preprocess(self, mol):
        """自訂預處理：標準化分子"""
        if isinstance(mol, str):
            mol = dm.to_mol(mol)

        # 標準化
        mol = dm.standardize_mol(mol)

        # 移除鹽
        mol = dm.remove_salts(mol)

        return mol

# 使用自訂轉換器
transformer = CustomTransformer(FPCalculator("ecfp"), n_jobs=-1)
features = transformer(smiles_list)
```

### 帶構象的特徵化

```python
import datamol as dm
from molfeat.calc import RDKitDescriptors3D

# 生成構象
def prepare_3d_mol(smiles):
    mol = dm.to_mol(smiles)
    mol = dm.add_hs(mol)
    mol = dm.conform.generate_conformers(mol, n_confs=1)
    return mol

# 3D 描述符
calc_3d = RDKitDescriptors3D()

smiles = "CC(C)Cc1ccc(C)cc1C"
mol_3d = prepare_3d_mol(smiles)
descriptors_3d = calc_3d(mol_3d)
```

### 平行批次處理

```python
from molfeat.trans import MoleculeTransformer
from molfeat.calc import FPCalculator
import time

# 大型資料集
smiles_large = load_large_dataset()  # 例如，100,000 個分子

# 測試不同的平行化等級
for n_jobs in [1, 2, 4, -1]:
    transformer = MoleculeTransformer(
        FPCalculator("ecfp"),
        n_jobs=n_jobs
    )

    start = time.time()
    features = transformer(smiles_large)
    elapsed = time.time() - start

    print(f"n_jobs={n_jobs}: {elapsed:.2f}s")
```

### 昂貴操作的快取

```python
from molfeat.trans.pretrained import PretrainedMolTransformer
import pickle

# 載入昂貴的預訓練模型
transformer = PretrainedMolTransformer("ChemBERTa-77M-MLM", n_jobs=-1)

# 快取嵌入以供重用
cache_file = "embeddings_cache.pkl"

try:
    # 嘗試載入快取的嵌入
    with open(cache_file, "rb") as f:
        embeddings = pickle.load(f)
    print("Loaded cached embeddings")
except FileNotFoundError:
    # 計算並快取
    embeddings = transformer(smiles_list)
    with open(cache_file, "wb") as f:
        pickle.dump(embeddings, f)
    print("Computed and cached embeddings")
```

---

## 常見工作流程

### 虛擬篩選工作流程

```python
from molfeat.calc import FPCalculator
from sklearn.ensemble import RandomForestClassifier
import datamol as dm

# 1. 準備訓練資料（已知的活性/非活性化合物）
train_smiles = load_training_data()
train_labels = load_training_labels()  # 1=活性，0=非活性

# 2. 特徵化訓練集
transformer = MoleculeTransformer(FPCalculator("ecfp"), n_jobs=-1)
X_train = transformer(train_smiles)

# 3. 訓練分類器
clf = RandomForestClassifier(n_estimators=500, n_jobs=-1)
clf.fit(X_train, train_labels)

# 4. 特徵化篩選資料庫
screening_smiles = load_screening_library()  # 例如，100 萬化合物
X_screen = transformer(screening_smiles)

# 5. 預測和排序
predictions = clf.predict_proba(X_screen)[:, 1]
ranked_indices = predictions.argsort()[::-1]

# 6. 取得頂級命中
top_n = 1000
top_hits = [screening_smiles[i] for i in ranked_indices[:top_n]]
```

### QSAR 模型建構

```python
from molfeat.calc import RDKitDescriptors2D
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import numpy as np

# 載入 QSAR 資料集
smiles = load_molecules()
y = load_activity_values()  # 例如，IC50、logP

# 使用可解釋描述符特徵化
transformer = MoleculeTransformer(RDKitDescriptors2D(), n_jobs=-1)
X = transformer(smiles)

# 標準化特徵
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 建構線性模型
model = Ridge(alpha=1.0)
scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2')
print(f"R² = {scores.mean():.3f} (+/- {scores.std():.3f})")

# 擬合最終模型
model.fit(X_scaled, y)

# 解釋特徵重要性
feature_names = transformer.featurizer.columns
importance = np.abs(model.coef_)
top_features_idx = importance.argsort()[-10:][::-1]

print("Top 10 important features:")
for idx in top_features_idx:
    print(f"  {feature_names[idx]}: {model.coef_[idx]:.3f}")
```

### 相似性搜索

```python
from molfeat.calc import FPCalculator
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# 查詢分子
query_smiles = "CC(=O)Oc1ccccc1C(=O)O"  # 阿斯匹靈

# 分子資料庫
database_smiles = load_molecule_database()  # 大型集合

# 計算指紋
calc = FPCalculator("ecfp")
query_fp = calc(query_smiles).reshape(1, -1)

transformer = MoleculeTransformer(calc, n_jobs=-1)
database_fps = transformer(database_smiles)

# 計算相似性
similarities = cosine_similarity(query_fp, database_fps)[0]

# 尋找最相似的
top_k = 10
top_indices = similarities.argsort()[-top_k:][::-1]

print(f"Top {top_k} similar molecules:")
for i, idx in enumerate(top_indices, 1):
    print(f"{i}. {database_smiles[idx]} (similarity: {similarities[idx]:.3f})")
```

---

## 疑難排解

### 處理無效分子

```python
# 使用 ignore_errors 跳過無效分子
transformer = MoleculeTransformer(
    FPCalculator("ecfp"),
    ignore_errors=True,
    verbose=True
)

# 轉換後過濾掉 None 值
features = transformer(smiles_list)
valid_mask = [f is not None for f in features]
valid_features = [f for f in features if f is not None]
valid_smiles = [s for s, m in zip(smiles_list, valid_mask) if m]
```

### 大型資料集的記憶體管理

```python
# 對非常大的資料集進行分塊處理
def featurize_in_chunks(smiles_list, transformer, chunk_size=10000):
    all_features = []

    for i in range(0, len(smiles_list), chunk_size):
        chunk = smiles_list[i:i+chunk_size]
        features = transformer(chunk)
        all_features.append(features)
        print(f"Processed {i+len(chunk)}/{len(smiles_list)}")

    return np.vstack(all_features)

# 用於大型資料集
features = featurize_in_chunks(large_smiles_list, transformer)
```

### 可重現性

```python
import random
import numpy as np
import torch

# 設定所有隨機種子
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

# 儲存確切配置
transformer.to_state_yaml_file("config.yml")

# 記錄版本
import molfeat
print(f"molfeat version: {molfeat.__version__}")
```
