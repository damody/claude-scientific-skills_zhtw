# TDC 工具和資料函數

本文件提供 TDC 資料處理、評估和工具函數的全面文件。

## 概述

TDC 提供組織為四個主要類別的工具：
1. **資料集分割** - 訓練/驗證/測試分割策略
2. **模型評估** - 標準化效能指標
3. **資料處理** - 分子轉換、過濾和轉換
4. **實體擷取** - 資料庫查詢和轉換

## 1. 資料集分割

資料集分割對於評估模型泛化至關重要。TDC 提供多種專為治療性機器學習設計的分割策略。

### 基本分割用法

```python
from tdc.single_pred import ADME

data = ADME(name='Caco2_Wang')

# 使用預設參數取得分割
split = data.get_split()
# 回傳：{'train': DataFrame, 'valid': DataFrame, 'test': DataFrame}

# 自訂分割參數
split = data.get_split(
    method='scaffold',
    seed=42,
    frac=[0.7, 0.1, 0.2]
)
```

### 分割方法

#### 隨機分割
隨機打亂資料 - 適用於一般機器學習任務。

```python
split = data.get_split(method='random', seed=1)
```

**何時使用：**
- 基線模型評估
- 當化學/時間結構不重要時
- 快速原型開發

**不建議用於：**
- 真實的藥物發現場景
- 評估對新化學物質的泛化

#### Scaffold 分割
基於分子骨架（Bemis-Murcko 骨架）分割 - 確保測試分子與訓練結構不同。

```python
split = data.get_split(method='scaffold', seed=1)
```

**何時使用：**
- 大多數單一預測任務的預設
- 評估對新化學系列的泛化
- 真實的藥物發現場景

**運作方式：**
1. 從每個分子提取 Bemis-Murcko 骨架
2. 按骨架分組分子
3. 將骨架分配到訓練/驗證/測試集
4. 確保測試分子具有未見過的骨架

#### Cold 分割（DTI/DDI 任務）
對於多實例預測，cold 分割確保測試集包含未見過的藥物、標靶或兩者。

**Cold Drug 分割：**
```python
from tdc.multi_pred import DTI
data = DTI(name='BindingDB_Kd')
split = data.get_split(method='cold_drug', seed=1)
```
- 測試集包含訓練期間未見過的藥物
- 評估對新化合物的泛化

**Cold Target 分割：**
```python
split = data.get_split(method='cold_target', seed=1)
```
- 測試集包含訓練期間未見過的標靶
- 評估對新蛋白質的泛化

**Cold Drug-Target 分割：**
```python
split = data.get_split(method='cold_drug_target', seed=1)
```
- 測試集包含新穎的藥物-標靶對
- 最具挑戰性的評估場景

#### 時間分割
對於具有時間資訊的資料集 - 確保測試資料來自較晚的時間點。

```python
split = data.get_split(method='temporal', seed=1)
```

**何時使用：**
- 具有時間戳的資料集
- 模擬前瞻性預測
- 臨床試驗結果預測

### 自訂分割比例

```python
# 80% 訓練，10% 驗證，10% 測試
split = data.get_split(method='scaffold', frac=[0.8, 0.1, 0.1])

# 70% 訓練，15% 驗證，15% 測試
split = data.get_split(method='scaffold', frac=[0.7, 0.15, 0.15])
```

### 分層分割

對於具有不平衡標籤的分類任務：

```python
split = data.get_split(method='scaffold', stratified=True)
```

維持訓練/驗證/測試集之間的標籤分布。

## 2. 模型評估

TDC 提供不同任務類型的標準化評估指標。

### 基本評估器用法

```python
from tdc import Evaluator

# 初始化評估器
evaluator = Evaluator(name='ROC-AUC')

# 評估預測
score = evaluator(y_true, y_pred)
```

### 分類指標

#### ROC-AUC
接收者操作特徵 - 曲線下面積

```python
evaluator = Evaluator(name='ROC-AUC')
score = evaluator(y_true, y_pred_proba)
```

**最適合：**
- 二元分類
- 不平衡資料集
- 整體判別能力

**範圍：** 0-1（較高較好，0.5 為隨機）

#### PR-AUC
精確率-召回率曲線下面積

```python
evaluator = Evaluator(name='PR-AUC')
score = evaluator(y_true, y_pred_proba)
```

**最適合：**
- 高度不平衡資料集
- 當正類稀少時
- 補充 ROC-AUC

**範圍：** 0-1（較高較好）

#### F1 分數
精確率和召回率的調和平均數

```python
evaluator = Evaluator(name='F1')
score = evaluator(y_true, y_pred_binary)
```

**最適合：**
- 精確率和召回率之間的平衡
- 多類別分類

**範圍：** 0-1（較高較好）

#### 準確率
正確預測的比例

```python
evaluator = Evaluator(name='Accuracy')
score = evaluator(y_true, y_pred_binary)
```

**最適合：**
- 平衡資料集
- 簡單基線指標

**不建議用於：** 不平衡資料集

#### Cohen's Kappa
預測與真實值之間的一致性，考慮機率

```python
evaluator = Evaluator(name='Kappa')
score = evaluator(y_true, y_pred_binary)
```

**範圍：** -1 到 1（較高較好，0 為隨機）

### 迴歸指標

#### RMSE - 均方根誤差
```python
evaluator = Evaluator(name='RMSE')
score = evaluator(y_true, y_pred)
```

**最適合：**
- 連續預測
- 嚴重懲罰大誤差

**範圍：** 0-∞（較低較好）

#### MAE - 平均絕對誤差
```python
evaluator = Evaluator(name='MAE')
score = evaluator(y_true, y_pred)
```

**最適合：**
- 連續預測
- 比 RMSE 對異常值更穩健

**範圍：** 0-∞（較低較好）

#### R² - 決定係數
```python
evaluator = Evaluator(name='R2')
score = evaluator(y_true, y_pred)
```

**最適合：**
- 模型解釋的變異
- 比較不同模型

**範圍：** -∞ 到 1（較高較好，1 為完美）

#### MSE - 均方誤差
```python
evaluator = Evaluator(name='MSE')
score = evaluator(y_true, y_pred)
```

**範圍：** 0-∞（較低較好）

### 排序指標

#### Spearman 相關
等級相關係數

```python
evaluator = Evaluator(name='Spearman')
score = evaluator(y_true, y_pred)
```

**最適合：**
- 排序任務
- 非線性關係
- 序數資料

**範圍：** -1 到 1（較高較好）

#### Pearson 相關
線性相關係數

```python
evaluator = Evaluator(name='Pearson')
score = evaluator(y_true, y_pred)
```

**最適合：**
- 線性關係
- 連續資料

**範圍：** -1 到 1（較高較好）

### 多標籤分類

```python
evaluator = Evaluator(name='Micro-F1')
score = evaluator(y_true_multilabel, y_pred_multilabel)
```

可用：`Micro-F1`、`Macro-F1`、`Micro-AUPR`、`Macro-AUPR`

### 基準組評估

對於基準組，評估需要多個種子：

```python
from tdc.benchmark_group import admet_group

group = admet_group(path='data/')
benchmark = group.get('Caco2_Wang')

# 預測必須是以種子為鍵的字典
predictions = {}
for seed in [1, 2, 3, 4, 5]:
    # 訓練模型並預測
    predictions[seed] = model_predictions

# 使用跨種子的平均值和標準差進行評估
results = group.evaluate(predictions)
print(results)  # {'Caco2_Wang': [mean_score, std_score]}
```

## 3. 資料處理

TDC 提供 11 個全面的資料處理工具。

### 分子格式轉換

在約 15 種分子表示之間轉換。

```python
from tdc.chem_utils import MolConvert

# SMILES 轉 PyTorch Geometric
converter = MolConvert(src='SMILES', dst='PyG')
pyg_graph = converter('CC(C)Cc1ccc(cc1)C(C)C(O)=O')

# SMILES 轉 DGL
converter = MolConvert(src='SMILES', dst='DGL')
dgl_graph = converter('CC(C)Cc1ccc(cc1)C(C)C(O)=O')

# SMILES 轉 Morgan 指紋（ECFP）
converter = MolConvert(src='SMILES', dst='ECFP')
fingerprint = converter('CC(C)Cc1ccc(cc1)C(C)C(O)=O')
```

**可用格式：**
- **文字**：SMILES、SELFIES、InChI
- **指紋**：ECFP（Morgan）、MACCS、RDKit、AtomPair、TopologicalTorsion
- **圖**：PyG（PyTorch Geometric）、DGL（Deep Graph Library）
- **3D**：Graph3D、Coulomb Matrix、Distance Matrix

**批次轉換：**
```python
converter = MolConvert(src='SMILES', dst='PyG')
graphs = converter(['SMILES1', 'SMILES2', 'SMILES3'])
```

### 分子過濾器

使用策劃的化學規則移除非類藥分子。

```python
from tdc.chem_utils import MolFilter

# 使用規則初始化過濾器
mol_filter = MolFilter(
    rules=['PAINS', 'BMS'],  # 化學過濾規則
    property_filters_dict={
        'MW': (150, 500),      # 分子量範圍
        'LogP': (-0.4, 5.6),   # 親脂性範圍
        'HBD': (0, 5),         # 氫鍵供體
        'HBA': (0, 10)         # 氫鍵受體
    }
)

# 過濾分子
filtered_smiles = mol_filter(smiles_list)
```

**可用過濾規則：**
- `PAINS` - 泛測定干擾化合物
- `BMS` - Bristol-Myers Squibb HTS 庫過濾器
- `Glaxo` - GlaxoSmithKline 過濾器
- `Dundee` - Dundee 大學過濾器
- `Inpharmatica` - Inpharmatica 過濾器
- `LINT` - Pfizer LINT 過濾器

### 標籤分布視覺化

```python
# 視覺化標籤分布
data.label_distribution()

# 列印統計
data.print_stats()
```

顯示直方圖並計算連續標籤的平均值、中位數、標準差。

### 標籤二值化

使用閾值將連續標籤轉換為二元。

```python
from tdc.utils import binarize

# 使用閾值二值化
binary_labels = binarize(y_continuous, threshold=5.0, order='ascending')
# order='ascending'：值 >= 閾值變為 1
# order='descending'：值 <= 閾值變為 1
```

### 標籤單位轉換

在測量單位之間轉換。

```python
from tdc.chem_utils import label_transform

# 將 nM 轉換為 pKd
y_pkd = label_transform(y_nM, from_unit='nM', to_unit='p')

# 將 μM 轉換為 nM
y_nM = label_transform(y_uM, from_unit='uM', to_unit='nM')
```

**可用轉換：**
- 結合親和力：nM、μM、pKd、pKi、pIC50
- 對數轉換
- 自然對數轉換

### 標籤意義

取得標籤的可解釋描述。

```python
# 取得標籤映射
label_map = data.get_label_map(name='DrugBank')
print(label_map)
# {0: 'No interaction', 1: 'Increased effect', 2: 'Decreased effect', ...}
```

### 資料平衡

透過過採樣/欠採樣處理類別不平衡。

```python
from tdc.utils import balance

# 過採樣少數類別
X_balanced, y_balanced = balance(X, y, method='oversample')

# 欠採樣多數類別
X_balanced, y_balanced = balance(X, y, method='undersample')
```

### 配對資料的圖轉換

將配對資料轉換為圖表示。

```python
from tdc.utils import create_graph_from_pairs

# 從藥物-藥物對建立圖
graph = create_graph_from_pairs(
    pairs=ddi_pairs,  # [(drug1, drug2, label), ...]
    format='edge_list'  # 或 'PyG'、'DGL'
)
```

### 負採樣

為二元任務生成負樣本。

```python
from tdc.utils import negative_sample

# 為 DTI 生成負樣本
negative_pairs = negative_sample(
    positive_pairs=known_interactions,
    all_drugs=drug_list,
    all_targets=target_list,
    ratio=1.0  # 負:正比例
)
```

**用例：**
- 藥物-標靶互動預測
- 藥物-藥物互動任務
- 建立平衡資料集

### 實體擷取

在資料庫識別碼之間轉換。

#### PubChem CID 轉 SMILES
```python
from tdc.utils import cid2smiles

smiles = cid2smiles(2244)  # 阿斯匹靈
# 回傳：'CC(=O)Oc1ccccc1C(=O)O'
```

#### UniProt ID 轉胺基酸序列
```python
from tdc.utils import uniprot2seq

sequence = uniprot2seq('P12345')
# 回傳：'MVKVYAPASS...'
```

#### 批次擷取
```python
# 多個 CID
smiles_list = [cid2smiles(cid) for cid in [2244, 5090, 6323]]

# 多個 UniProt ID
sequences = [uniprot2seq(uid) for uid in ['P12345', 'Q9Y5S9']]
```

## 4. 進階工具

### 擷取資料集名稱

```python
from tdc.utils import retrieve_dataset_names

# 取得任務的所有資料集
adme_datasets = retrieve_dataset_names('ADME')
dti_datasets = retrieve_dataset_names('DTI')
tox_datasets = retrieve_dataset_names('Tox')

print(f"ADME datasets: {adme_datasets}")
```

### 模糊搜尋

TDC 支援資料集名稱的模糊匹配：

```python
from tdc.single_pred import ADME

# 這些都有效（容錯拼寫）
data = ADME(name='Caco2_Wang')
data = ADME(name='caco2_wang')
data = ADME(name='Caco2')  # 部分匹配
```

### 資料格式選項

```python
# Pandas DataFrame（預設）
df = data.get_data(format='df')

# 字典
data_dict = data.get_data(format='dict')

# DeepPurpose 格式（用於 DeepPurpose 函式庫）
dp_format = data.get_data(format='DeepPurpose')

# PyG/DGL 圖（如適用）
graphs = data.get_data(format='PyG')
```

### 資料載入器工具

```python
from tdc.utils import create_fold

# 建立交叉驗證折疊
folds = create_fold(data, fold=5, seed=42)
# 回傳 (train_idx, test_idx) 元組列表

# 迭代折疊
for i, (train_idx, test_idx) in enumerate(folds):
    train_data = data.iloc[train_idx]
    test_data = data.iloc[test_idx]
    # 訓練和評估
```

## 常見工作流程

### 工作流程 1：完整資料流程

```python
from tdc.single_pred import ADME
from tdc import Evaluator
from tdc.chem_utils import MolConvert, MolFilter

# 1. 載入資料
data = ADME(name='Caco2_Wang')

# 2. 過濾分子
mol_filter = MolFilter(rules=['PAINS'])
filtered_data = data.get_data()
filtered_data = filtered_data[
    filtered_data['Drug'].apply(lambda x: mol_filter([x]))
]

# 3. 分割資料
split = data.get_split(method='scaffold', seed=42)
train, valid, test = split['train'], split['valid'], split['test']

# 4. 轉換為圖表示
converter = MolConvert(src='SMILES', dst='PyG')
train_graphs = converter(train['Drug'].tolist())

# 5. 訓練模型（使用者實作）
# model.fit(train_graphs, train['Y'])

# 6. 評估
evaluator = Evaluator(name='MAE')
# score = evaluator(test['Y'], predictions)
```

### 工作流程 2：多任務學習準備

```python
from tdc.benchmark_group import admet_group
from tdc.chem_utils import MolConvert

# 載入基準組
group = admet_group(path='data/')

# 取得多個資料集
datasets = ['Caco2_Wang', 'HIA_Hou', 'Bioavailability_Ma']
all_data = {}

for dataset_name in datasets:
    benchmark = group.get(dataset_name)
    all_data[dataset_name] = benchmark

# 準備多任務學習
converter = MolConvert(src='SMILES', dst='ECFP')
# 處理每個資料集...
```

### 工作流程 3：DTI Cold 分割評估

```python
from tdc.multi_pred import DTI
from tdc import Evaluator

# 載入 DTI 資料
data = DTI(name='BindingDB_Kd')

# Cold drug 分割
split = data.get_split(method='cold_drug', seed=42)
train, test = split['train'], split['test']

# 驗證沒有藥物重疊
train_drugs = set(train['Drug_ID'])
test_drugs = set(test['Drug_ID'])
assert len(train_drugs & test_drugs) == 0, "Drug leakage detected!"

# 訓練和評估
# model.fit(train)
evaluator = Evaluator(name='RMSE')
# score = evaluator(test['Y'], predictions)
```

## 最佳實務

1. **總是使用有意義的分割** - 使用 scaffold 或 cold 分割進行真實評估
2. **多個種子** - 使用多個種子執行實驗以獲得穩健結果
3. **適當的指標** - 選擇符合您任務和資料集特徵的指標
4. **資料過濾** - 訓練前移除 PAINS 和非類藥分子
5. **格式轉換** - 將分子轉換為適合您模型的格式
6. **批次處理** - 對大型資料集使用批次操作以提高效率

## 效能提示

- 以批次模式轉換分子以加快處理速度
- 快取轉換後的表示以避免重新計算
- 為您的框架使用適當的資料格式（PyG、DGL 等）
- 在流程早期過濾資料以減少計算

## 參考文獻

- TDC 文件：https://tdc.readthedocs.io
- 資料函數：https://tdcommons.ai/fct_overview/
- 評估指標：https://tdcommons.ai/functions/model_eval/
- 資料分割：https://tdcommons.ai/functions/data_split/
