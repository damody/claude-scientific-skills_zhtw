# Molfeat API 參考

## 核心模組

Molfeat 組織為幾個關鍵模組，提供分子特徵化的不同面向：

- **`molfeat.store`** - 管理模型載入、列表和註冊
- **`molfeat.calc`** - 提供單分子特徵化的計算器
- **`molfeat.trans`** - 提供 scikit-learn 相容的批次處理轉換器
- **`molfeat.utils`** - 資料處理的工具函數
- **`molfeat.viz`** - 分子特徵的視覺化工具

---

## molfeat.calc - 計算器

計算器（Calculators）是可呼叫物件，將單個分子轉換為特徵向量。它們接受 RDKit `Chem.Mol` 物件或 SMILES 字串作為輸入。

### SerializableCalculator（基礎類別）

所有計算器的基礎抽象類別。建立子類別時，必須實作：
- `__call__()` - 特徵化的必要方法
- `__len__()` - 可選，返回輸出長度
- `columns` - 可選屬性，返回特徵名稱
- `batch_compute()` - 可選，用於高效批次處理

**狀態管理方法：**
- `to_state_json()` - 將計算器狀態儲存為 JSON
- `to_state_yaml()` - 將計算器狀態儲存為 YAML
- `from_state_dict()` - 從狀態字典載入計算器
- `to_state_dict()` - 將計算器狀態匯出為字典

### FPCalculator

計算分子指紋。支援 15 種以上的指紋方法。

**支援的指紋類型：**

**結構指紋：**
- `ecfp` - 擴展連接指紋（Extended-connectivity fingerprints，圓形）
- `fcfp` - 功能類別指紋（Functional-class fingerprints）
- `rdkit` - RDKit 拓撲指紋
- `maccs` - MACCS 鍵（166 位元結構鍵）
- `avalon` - Avalon 指紋
- `pattern` - 模式指紋
- `layered` - 分層指紋

**原子指紋：**
- `atompair` - 原子對指紋
- `atompair-count` - 計數原子對
- `topological` - 拓撲扭轉指紋
- `topological-count` - 計數拓撲扭轉

**專門指紋：**
- `map4` - 最小雜湊原子對指紋（最多 4 個鍵）
- `secfp` - SMILES 擴展連接指紋
- `erg` - 擴展簡化圖
- `estate` - 電拓撲狀態指數

**參數：**
- `method` (str) - 指紋類型名稱
- `radius` (int) - 圓形指紋的半徑（預設：3）
- `fpSize` (int) - 指紋大小（預設：2048）
- `includeChirality` (bool) - 包含手性資訊
- `counting` (bool) - 使用計數向量而非二元

**使用方式：**
```python
from molfeat.calc import FPCalculator

# 建立指紋計算器
calc = FPCalculator("ecfp", radius=3, fpSize=2048)

# 計算單個分子的指紋
fp = calc("CCO")  # 返回 numpy 陣列

# 取得指紋長度
length = len(calc)  # 2048

# 取得特徵名稱
names = calc.columns
```

**常見指紋維度：**
- MACCS：167 維
- ECFP（預設）：2048 維
- MAP4（預設）：1024 維

### 描述符計算器

**RDKitDescriptors2D**
使用 RDKit 計算 2D 分子描述符。

```python
from molfeat.calc import RDKitDescriptors2D

calc = RDKitDescriptors2D()
descriptors = calc("CCO")  # 返回 200+ 描述符
```

**RDKitDescriptors3D**
計算 3D 分子描述符（需要構象生成）。

**MordredDescriptors**
使用 Mordred 計算超過 1800 個分子描述符。

```python
from molfeat.calc import MordredDescriptors

calc = MordredDescriptors()
descriptors = calc("CCO")
```

### 藥效團計算器

**Pharmacophore2D**
RDKit 的 2D 藥效團指紋生成。

**Pharmacophore3D**
來自多個構象的共識藥效團指紋。

**CATSCalculator**
計算化學高級模板搜索（Chemically Advanced Template Search，CATS）描述符 - 藥效團點對分佈。

**參數：**
- `mode` - "2D" 或 "3D" 距離計算
- `dist_bins` - 對分佈的距離區間
- `scale` - 縮放模式："raw"、"num" 或 "count"

```python
from molfeat.calc import CATSCalculator

calc = CATSCalculator(mode="2D", scale="raw")
cats = calc("CCO")  # 預設返回 21 個描述符
```

### 形狀描述符

**USRDescriptors**
超快形狀識別（Ultrafast shape recognition）描述符（多種變體）。

**ElectroShapeDescriptors**
結合形狀、手性和靜電的電形狀描述符。

### 圖形計算器

**ScaffoldKeyCalculator**
計算 40+ 種基於骨架的分子屬性。

**AtomCalculator**
用於圖神經網路的原子級特徵化。

**BondCalculator**
用於圖神經網路的鍵級特徵化。

### 工具函數

**get_calculator()**
按名稱實例化計算器的工廠函數。

```python
from molfeat.calc import get_calculator

# 按名稱實例化任何計算器
calc = get_calculator("ecfp", radius=3)
calc = get_calculator("maccs")
calc = get_calculator("desc2D")
```

對不支援的特徵提取器引發 `ValueError`。

---

## molfeat.trans - 轉換器

轉換器（Transformers）將計算器包裝成完整的批次處理特徵化管線。

### MoleculeTransformer

用於批次分子特徵化的 Scikit-learn 相容轉換器。

**關鍵參數：**
- `featurizer` - 要使用的計算器或特徵提取器
- `n_jobs` (int) - 平行任務數（-1 表示所有核心）
- `dtype` - 輸出資料類型（numpy float32/64、torch 張量）
- `verbose` (bool) - 啟用詳細記錄
- `ignore_errors` (bool) - 失敗時繼續（對失敗的分子返回 None）

**基本方法：**
- `transform(mols)` - 處理批次並返回表示
- `_transform(mol)` - 處理單個分子特徵化
- `__call__(mols)` - transform() 的便捷包裝器
- `preprocess(mol)` - 準備輸入分子（不自動應用）
- `to_state_yaml_file(path)` - 儲存轉換器配置
- `from_state_yaml_file(path)` - 載入轉換器配置

**使用方式：**
```python
from molfeat.calc import FPCalculator
from molfeat.trans import MoleculeTransformer
import datamol as dm

# 載入分子
smiles = dm.data.freesolv().sample(100).smiles.values

# 建立轉換器
calc = FPCalculator("ecfp")
transformer = MoleculeTransformer(calc, n_jobs=-1)

# 批次特徵化
features = transformer(smiles)  # 返回 numpy 陣列 (100, 2048)

# 儲存配置
transformer.to_state_yaml_file("ecfp_config.yml")

# 重新載入
transformer = MoleculeTransformer.from_state_yaml_file("ecfp_config.yml")
```

**效能：** 在 642 個分子上測試顯示，使用 4 個平行任務相比單執行緒處理有 3.4 倍的加速。

### FeatConcat

將多個特徵提取器連接成統一的表示。

```python
from molfeat.trans import FeatConcat
from molfeat.calc import FPCalculator

# 組合多個指紋
concat = FeatConcat([
    FPCalculator("maccs"),      # 167 維
    FPCalculator("ecfp")         # 2048 維
])

# 結果：2167 維特徵
transformer = MoleculeTransformer(concat, n_jobs=-1)
features = transformer(smiles)
```

### PretrainedMolTransformer

用於預訓練深度學習模型的 `MoleculeTransformer` 子類別。

**獨特功能：**
- `_embed()` - 神經網路的批次推論
- `_convert()` - 將 SMILES/分子轉換為模型相容格式
  - 語言模型的 SELFIES 字串
  - 圖神經網路的 DGL 圖
- 用於高效儲存的整合快取系統

**使用方式：**
```python
from molfeat.trans.pretrained import PretrainedMolTransformer

# 載入預訓練模型
transformer = PretrainedMolTransformer("ChemBERTa-77M-MLM", n_jobs=-1)

# 生成嵌入
embeddings = transformer(smiles)
```

### PrecomputedMolTransformer

用於快取/預計算特徵的轉換器。

---

## molfeat.store - 模型商店

管理特徵提取器的發現、載入和註冊。

### ModelStore

存取可用特徵提取器的中央中心。

**關鍵方法：**
- `available_models` - 列出所有可用特徵提取器的屬性
- `search(name=None, **kwargs)` - 搜索特定特徵提取器
- `load(name, **kwargs)` - 按名稱載入特徵提取器
- `register(name, card)` - 註冊自訂特徵提取器

**使用方式：**
```python
from molfeat.store.modelstore import ModelStore

# 初始化商店
store = ModelStore()

# 列出所有可用模型
all_models = store.available_models
print(f"Found {len(all_models)} featurizers")

# 搜索特定模型
results = store.search(name="ChemBERTa-77M-MLM")
if results:
    model_card = results[0]

    # 檢視使用資訊
    model_card.usage()

    # 載入模型
    transformer = model_card.load()

# 直接載入
transformer = store.load("ChemBERTa-77M-MLM")
```

**ModelCard 屬性：**
- `name` - 模型識別碼
- `description` - 模型描述
- `version` - 模型版本
- `authors` - 模型作者
- `tags` - 分類標籤
- `usage()` - 顯示使用範例
- `load(**kwargs)` - 載入模型

---

## 常見模式

### 錯誤處理

```python
# 啟用錯誤容忍
featurizer = MoleculeTransformer(
    calc,
    n_jobs=-1,
    verbose=True,
    ignore_errors=True
)

# 失敗的分子返回 None
features = featurizer(smiles_with_errors)
```

### 資料類型控制

```python
# NumPy float32（預設）
features = transformer(smiles, enforce_dtype=True)

# PyTorch 張量
import torch
transformer = MoleculeTransformer(calc, dtype=torch.float32)
features = transformer(smiles)
```

### 持久性和可重現性

```python
# 儲存轉換器狀態
transformer.to_state_yaml_file("config.yml")
transformer.to_state_json_file("config.json")

# 從儲存的狀態載入
transformer = MoleculeTransformer.from_state_yaml_file("config.yml")
transformer = MoleculeTransformer.from_state_json_file("config.json")
```

### 預處理

```python
# 手動預處理
mol = transformer.preprocess("CCO")

# 帶預處理的轉換
features = transformer.transform(smiles_list)
```

---

## 整合範例

### Scikit-learn 管線

```python
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from molfeat.trans import MoleculeTransformer
from molfeat.calc import FPCalculator

# 建立管線
pipeline = Pipeline([
    ('featurizer', MoleculeTransformer(FPCalculator("ecfp"))),
    ('classifier', RandomForestClassifier())
])

# 擬合和預測
pipeline.fit(smiles_train, y_train)
predictions = pipeline.predict(smiles_test)
```

### PyTorch 整合

```python
import torch
from torch.utils.data import Dataset, DataLoader
from molfeat.trans import MoleculeTransformer

class MoleculeDataset(Dataset):
    def __init__(self, smiles, labels, transformer):
        self.smiles = smiles
        self.labels = labels
        self.transformer = transformer

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        features = self.transformer(self.smiles[idx])
        return torch.tensor(features), torch.tensor(self.labels[idx])

# 建立資料集和資料載入器
transformer = MoleculeTransformer(FPCalculator("ecfp"))
dataset = MoleculeDataset(smiles, labels, transformer)
loader = DataLoader(dataset, batch_size=32)
```

---

## 效能提示

1. **平行化**：使用 `n_jobs=-1` 以利用所有 CPU 核心
2. **批次處理**：一次處理多個分子而不是使用迴圈
3. **快取**：利用預訓練模型的內建快取
4. **資料類型**：精度允許時使用 float32 而非 float64
5. **錯誤處理**：對可能包含無效分子的大型資料集設定 `ignore_errors=True`
