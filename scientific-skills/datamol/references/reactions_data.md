# Datamol 反應與資料模組參考

## 反應模組（`datamol.reactions`）

反應模組能夠使用 SMARTS 反應模式程式化地應用化學轉換。

### 應用化學反應

#### `dm.reactions.apply_reaction(rxn, reactants, as_smiles=False, sanitize=True, single_product_group=True, rm_attach=True, product_index=0)`
將化學反應應用於反應物分子。
- **參數**：
  - `rxn`：反應物件（來自 SMARTS 模式）
  - `reactants`：反應物分子的元組
  - `as_smiles`：返回 SMILES 字串（True）或分子物件（False）
  - `sanitize`：清理產物分子
  - `single_product_group`：返回單一產物（True）或所有產物組（False）
  - `rm_attach`：移除連接點標記
  - `product_index`：從反應返回哪個產物
- **返回**：產物分子或 SMILES
- **範例**：
  ```python
  from rdkit import Chem

  # 定義反應：醇 + 羧酸 → 酯
  rxn = Chem.rdChemReactions.ReactionFromSmarts(
      '[C:1][OH:2].[C:3](=[O:4])[OH:5]>>[C:1][O:2][C:3](=[O:4])'
  )

  # 應用於反應物
  alcohol = dm.to_mol("CCO")
  acid = dm.to_mol("CC(=O)O")
  product = dm.reactions.apply_reaction(rxn, (alcohol, acid))
  ```

### 建立反應

反應通常使用 RDKit 從 SMARTS 模式建立：
```python
from rdkit.Chem import rdChemReactions

# 反應模式：[反應物1].[反應物2]>>[產物]
rxn = rdChemReactions.ReactionFromSmarts(
    '[1*][*:1].[1*][*:2]>>[*:1][*:2]'
)
```

### 驗證函數

模組包含以下函數：
- **檢查分子是否為反應物**：驗證分子是否符合反應物模式
- **驗證反應**：檢查反應是否在合成上合理
- **處理反應檔案**：從檔案或資料庫載入反應

### 常見反應模式

**醯胺形成**：
```python
# 胺 + 羧酸 → 醯胺
amide_rxn = rdChemReactions.ReactionFromSmarts(
    '[N:1].[C:2](=[O:3])[OH]>>[N:1][C:2](=[O:3])'
)
```

**Suzuki 偶聯**：
```python
# 芳基鹵化物 + 硼酸 → 聯芳基
suzuki_rxn = rdChemReactions.ReactionFromSmarts(
    '[c:1][Br].[c:2][B]([OH])[OH]>>[c:1][c:2]'
)
```

**官能基轉換**：
```python
# 醇 → 酯
esterification = rdChemReactions.ReactionFromSmarts(
    '[C:1][OH:2].[C:3](=[O:4])[Cl]>>[C:1][O:2][C:3](=[O:4])'
)
```

### 工作流程範例

```python
import datamol as dm
from rdkit.Chem import rdChemReactions

# 1. 定義反應
rxn_smarts = '[C:1](=[O:2])[OH:3]>>[C:1](=[O:2])[Cl:3]'  # 酸 → 醯氯
rxn = rdChemReactions.ReactionFromSmarts(rxn_smarts)

# 2. 應用於分子庫
acids = [dm.to_mol(smi) for smi in acid_smiles_list]
acid_chlorides = []

for acid in acids:
    try:
        product = dm.reactions.apply_reaction(
            rxn,
            (acid,),  # 單一反應物作為元組
            sanitize=True
        )
        acid_chlorides.append(product)
    except Exception as e:
        print(f"反應失敗：{e}")

# 3. 驗證產物
valid_products = [p for p in acid_chlorides if p is not None]
```

### 關鍵概念

- **SMARTS**：SMiles ARbitrary Target Specification - 反應的模式語言
- **原子對應**：像 [C:1] 這樣的編號在反應過程中保留原子身份
- **連接點**：[1*] 代表通用連接點
- **反應驗證**：並非所有 SMARTS 反應都在化學上合理

---

## 資料模組（`datamol.data`）

資料模組提供方便存取精選分子資料集以進行測試和學習。

### 可用資料集

#### `dm.data.cdk2(as_df=True, mol_column='mol')`
RDKit CDK2 資料集 - 激酶抑制劑資料。
- **參數**：
  - `as_df`：返回 DataFrame（True）或分子列表（False）
  - `mol_column`：分子欄的名稱
- **返回**：包含分子結構和活性資料的資料集
- **使用案例**：用於演算法測試的小型資料集
- **範例**：
  ```python
  cdk2_df = dm.data.cdk2(as_df=True)
  print(cdk2_df.shape)
  print(cdk2_df.columns)
  ```

#### `dm.data.freesolv()`
FreeSolv 資料集 - 實驗和計算的水合自由能。
- **內容**：642 個分子，包含：
  - IUPAC 名稱
  - SMILES 字串
  - 實驗水合自由能值
  - 計算值
- **警告**：「僅用作教學和測試目的的玩具資料集」
- **不適用於**：基準測試或生產模型訓練
- **範例**：
  ```python
  freesolv_df = dm.data.freesolv()
  # 欄位：iupac、smiles、expt（kcal/mol）、calc（kcal/mol）
  ```

#### `dm.data.solubility(as_df=True, mol_column='mol')`
RDKit 溶解度資料集，帶有訓練/測試分割。
- **內容**：水溶性資料，帶有預定義分割
- **欄位**：包含 'split' 欄，值為 'train' 或 'test'
- **使用案例**：使用適當的訓練/測試分離測試 ML 工作流程
- **範例**：
  ```python
  sol_df = dm.data.solubility(as_df=True)

  # 分割為訓練/測試
  train_df = sol_df[sol_df['split'] == 'train']
  test_df = sol_df[sol_df['split'] == 'test']

  # 用於模型開發
  X_train = dm.to_fp(train_df[mol_column])
  y_train = train_df['solubility']
  ```

### 使用指南

**用於測試和教學**：
```python
# 用於測試程式碼的快速資料集
df = dm.data.cdk2()
mols = df['mol'].tolist()

# 測試描述子計算
descriptors_df = dm.descriptors.batch_compute_many_descriptors(mols)

# 測試聚類
clusters = dm.cluster_mols(mols, cutoff=0.3)
```

**用於學習工作流程**：
```python
# 完整的 ML 管道範例
sol_df = dm.data.solubility()

# 預處理
train = sol_df[sol_df['split'] == 'train']
test = sol_df[sol_df['split'] == 'test']

# 特徵化
X_train = dm.to_fp(train['mol'])
X_test = dm.to_fp(test['mol'])

# 模型訓練（範例）
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
model.fit(X_train, train['solubility'])
predictions = model.predict(X_test)
```

### 重要說明

- **玩具資料集**：專為教學目的設計，非生產用途
- **小規模**：有限數量的化合物適合快速測試
- **已預處理**：資料已清理和格式化
- **引用**：若要發表，請檢查資料集文件以獲得適當的歸屬

### 最佳實踐

1. **僅用於開發**：不要從玩具資料集得出科學結論
2. **在真實資料上驗證**：始終在實際專案資料上測試生產程式碼
3. **適當歸屬**：若在出版物中使用，請引用原始資料來源
4. **了解限制**：了解每個資料集的範圍和品質
