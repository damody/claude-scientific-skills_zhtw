# Medchem API 參考

所有 medchem 模組和函數的完整參考。

## 模組：medchem.rules

### 類別：RuleFilters

基於多個藥物化學規則篩選分子。

**建構函數：**
```python
RuleFilters(rule_list: List[str])
```

**參數：**
- `rule_list`：要應用的規則名稱列表。請參閱下方的可用規則。

**方法：**

```python
__call__(mols: List[Chem.Mol], n_jobs: int = 1, progress: bool = False) -> Dict
```
- `mols`：RDKit 分子物件列表
- `n_jobs`：平行工作數（-1 使用所有核心）
- `progress`：顯示進度條
- **返回**：包含每個規則結果的字典

**範例：**
```python
rfilter = mc.rules.RuleFilters(rule_list=["rule_of_five", "rule_of_cns"])
results = rfilter(mols=mol_list, n_jobs=-1, progress=True)
```

### 模組：medchem.rules.basic_rules

可應用於單一分子的個別規則函數。

#### rule_of_five()

```python
rule_of_five(mol: Union[str, Chem.Mol]) -> bool
```

Lipinski 的五規則，用於口服生物利用度（oral bioavailability）。

**標準：**
- 分子量 ≤ 500 Da
- LogP ≤ 5
- 氫鍵供體 ≤ 5
- 氫鍵受體 ≤ 10

**參數：**
- `mol`：SMILES 字串或 RDKit 分子物件

**返回：** 如果分子通過所有標準則為 True

#### rule_of_three()

```python
rule_of_three(mol: Union[str, Chem.Mol]) -> bool
```

用於片段篩選庫的三規則。

**標準：**
- 分子量 ≤ 300 Da
- LogP ≤ 3
- 氫鍵供體 ≤ 3
- 氫鍵受體 ≤ 3
- 可旋轉鍵 ≤ 3
- 極性表面積 ≤ 60 Å²

#### rule_of_oprea()

```python
rule_of_oprea(mol: Union[str, Chem.Mol]) -> bool
```

Oprea 的先導化合物類標準，用於苗頭化合物到先導化合物（hit-to-lead）最佳化。

**標準：**
- 分子量：200-350 Da
- LogP：-2 到 4
- 可旋轉鍵 ≤ 7
- 環數 ≤ 4

#### rule_of_cns()

```python
rule_of_cns(mol: Union[str, Chem.Mol]) -> bool
```

CNS 類藥性規則。

**標準：**
- 分子量 ≤ 450 Da
- LogP：-1 到 5
- 氫鍵供體 ≤ 2
- TPSA ≤ 90 Å²

#### rule_of_leadlike_soft()

```python
rule_of_leadlike_soft(mol: Union[str, Chem.Mol]) -> bool
```

軟性先導化合物類標準（較寬鬆）。

**標準：**
- 分子量：250-450 Da
- LogP：-3 到 4
- 可旋轉鍵 ≤ 10

#### rule_of_leadlike_strict()

```python
rule_of_leadlike_strict(mol: Union[str, Chem.Mol]) -> bool
```

嚴格先導化合物類標準（較嚴格）。

**標準：**
- 分子量：200-350 Da
- LogP：-2 到 3.5
- 可旋轉鍵 ≤ 7
- 環數：1-3

#### rule_of_veber()

```python
rule_of_veber(mol: Union[str, Chem.Mol]) -> bool
```

Veber 的口服生物利用度規則。

**標準：**
- 可旋轉鍵 ≤ 10
- TPSA ≤ 140 Å²

#### rule_of_reos()

```python
rule_of_reos(mol: Union[str, Chem.Mol]) -> bool
```

Rapid Elimination Of Swill（REOS）篩選器。

**標準：**
- 分子量：200-500 Da
- LogP：-5 到 5
- 氫鍵供體：0-5
- 氫鍵受體：0-10

#### rule_of_drug()

```python
rule_of_drug(mol: Union[str, Chem.Mol]) -> bool
```

組合的類藥性標準。

**標準：**
- 通過五規則
- 通過 Veber 規則
- 無 PAINS 子結構

#### golden_triangle()

```python
golden_triangle(mol: Union[str, Chem.Mol]) -> bool
```

類藥性平衡的黃金三角形。

**標準：**
- 200 ≤ MW ≤ 50×LogP + 400
- LogP：-2 到 5

#### pains_filter()

```python
pains_filter(mol: Union[str, Chem.Mol]) -> bool
```

Pan Assay INterference compoundS（PAINS）篩選器。

**返回：** 如果分子不包含 PAINS 子結構則為 True

---

## 模組：medchem.structural

### 類別：CommonAlertsFilters

用於源自 ChEMBL 和文獻的常見結構警示的篩選器。

**建構函數：**
```python
CommonAlertsFilters()
```

**方法：**

```python
__call__(mols: List[Chem.Mol], n_jobs: int = 1, progress: bool = False) -> List[Dict]
```

將常見警示篩選器應用於分子列表。

**返回：** 包含以下鍵的字典列表：
- `has_alerts`：布林值，指示分子是否有警示
- `alert_details`：匹配警示模式的列表
- `num_alerts`：發現的警示數量

```python
check_mol(mol: Chem.Mol) -> Tuple[bool, List[str]]
```

檢查單一分子的結構警示。

**返回：** (has_alerts, 警示名稱列表) 的元組

### 類別：NIBRFilters

Novartis NIBR 藥物化學篩選器。

**建構函數：**
```python
NIBRFilters()
```

**方法：**

```python
__call__(mols: List[Chem.Mol], n_jobs: int = 1, progress: bool = False) -> List[bool]
```

將 NIBR 篩選器應用於分子。

**返回：** 布林值列表（True 表示分子通過）

### 類別：LillyDemeritsFilters

Eli Lilly 基於扣分的結構警示系統（275 條規則）。

**建構函數：**
```python
LillyDemeritsFilters()
```

**方法：**

```python
__call__(mols: List[Chem.Mol], n_jobs: int = 1, progress: bool = False) -> List[Dict]
```

計算分子的 Lilly 扣分。

**返回：** 包含以下鍵的字典列表：
- `demerits`：總扣分分數
- `passes`：布林值（如果扣分 ≤ 100 則為 True）
- `matched_patterns`：包含分數的匹配模式列表

---

## 模組：medchem.functional

用於常見操作的高階函數式 API。

### nibr_filter()

```python
nibr_filter(mols: List[Chem.Mol], n_jobs: int = 1) -> List[bool]
```

使用函數式 API 應用 NIBR 篩選器。

**參數：**
- `mols`：分子列表
- `n_jobs`：平行化級別

**返回：** 通過/失敗布林值列表

### common_alerts_filter()

```python
common_alerts_filter(mols: List[Chem.Mol], n_jobs: int = 1) -> List[Dict]
```

使用函數式 API 應用常見警示篩選器。

**返回：** 結果字典列表

### lilly_demerits_filter()

```python
lilly_demerits_filter(mols: List[Chem.Mol], n_jobs: int = 1) -> List[Dict]
```

使用函數式 API 計算 Lilly 扣分。

---

## 模組：medchem.groups

### 類別：ChemicalGroup

偵測分子中的特定化學基團。

**建構函數：**
```python
ChemicalGroup(groups: List[str], custom_smarts: Optional[Dict[str, str]] = None)
```

**參數：**
- `groups`：預定義基團名稱列表
- `custom_smarts`：將自訂基團名稱映射到 SMARTS 模式的字典

**預定義基團：**
- `"hinge_binders"`：激酶鉸鏈結合基序
- `"phosphate_binders"`：磷酸結合基團
- `"michael_acceptors"`：麥可受體親電子基
- `"reactive_groups"`：一般反應性功能

**方法：**

```python
has_match(mols: List[Chem.Mol]) -> List[bool]
```

檢查分子是否包含任何指定的基團。

```python
get_matches(mol: Chem.Mol) -> Dict[str, List[Tuple]]
```

取得單一分子的詳細匹配資訊。

**返回：** 將基團名稱映射到原子索引列表的字典

```python
get_all_matches(mols: List[Chem.Mol]) -> List[Dict]
```

取得所有分子的匹配資訊。

**範例：**
```python
group = mc.groups.ChemicalGroup(groups=["hinge_binders", "phosphate_binders"])
matches = group.get_all_matches(mol_list)
```

---

## 模組：medchem.catalogs

### 類別：NamedCatalogs

存取策展的化學目錄。

**可用目錄：**
- `"functional_groups"`：常見官能基
- `"protecting_groups"`：保護基結構
- `"reagents"`：常見試劑
- `"fragments"`：標準片段

**用法：**
```python
catalog = mc.catalogs.NamedCatalogs.get("functional_groups")
matches = catalog.get_matches(mol)
```

---

## 模組：medchem.complexity

計算分子複雜度指標。

### calculate_complexity()

```python
calculate_complexity(mol: Chem.Mol, method: str = "bertz") -> float
```

計算分子的複雜度分數。

**參數：**
- `mol`：RDKit 分子
- `method`：複雜度指標（"bertz"、"whitlock"、"barone"）

**返回：** 複雜度分數（越高 = 越複雜）

### 類別：ComplexityFilter

按複雜度閾值篩選分子。

**建構函數：**
```python
ComplexityFilter(max_complexity: float, method: str = "bertz")
```

**方法：**

```python
__call__(mols: List[Chem.Mol], n_jobs: int = 1) -> List[bool]
```

篩選超過複雜度閾值的分子。

---

## 模組：medchem.constraints

### 類別：Constraints

應用自訂的基於性質的約束。

**建構函數：**
```python
Constraints(
    mw_range: Optional[Tuple[float, float]] = None,
    logp_range: Optional[Tuple[float, float]] = None,
    tpsa_max: Optional[float] = None,
    tpsa_range: Optional[Tuple[float, float]] = None,
    hbd_max: Optional[int] = None,
    hba_max: Optional[int] = None,
    rotatable_bonds_max: Optional[int] = None,
    rings_range: Optional[Tuple[int, int]] = None,
    aromatic_rings_max: Optional[int] = None,
)
```

**參數：** 所有參數都是可選的。只需指定所需的約束。

**方法：**

```python
__call__(mols: List[Chem.Mol], n_jobs: int = 1) -> List[Dict]
```

將約束應用於分子。

**返回：** 包含以下鍵的字典列表：
- `passes`：布林值，指示是否通過所有約束
- `violations`：失敗的約束名稱列表

**範例：**
```python
constraints = mc.constraints.Constraints(
    mw_range=(200, 500),
    logp_range=(-2, 5),
    tpsa_max=140
)
results = constraints(mols=mol_list, n_jobs=-1)
```

---

## 模組：medchem.query

用於複雜篩選的查詢語言。

### parse()

```python
parse(query: str) -> Query
```

將 medchem 查詢字串解析為 Query 物件。

**查詢語法：**
- 運算子：`AND`、`OR`、`NOT`
- 比較：`<`、`>`、`<=`、`>=`、`==`、`!=`
- 性質：`complexity`、`lilly_demerits`、`mw`、`logp`、`tpsa`
- 規則：`rule_of_five`、`rule_of_cns` 等
- 篩選器：`common_alerts`、`nibr_filter`、`pains_filter`

**查詢範例：**
```python
"rule_of_five AND NOT common_alerts"
"rule_of_cns AND complexity < 400"
"mw > 200 AND mw < 500 AND logp < 5"
"(rule_of_five OR rule_of_oprea) AND NOT pains_filter"
```

### 類別：Query

**方法：**

```python
apply(mols: List[Chem.Mol], n_jobs: int = 1) -> List[bool]
```

將解析的查詢應用於分子。

**範例：**
```python
query = mc.query.parse("rule_of_five AND NOT common_alerts")
results = query.apply(mols=mol_list, n_jobs=-1)
passing_mols = [mol for mol, passes in zip(mol_list, results) if passes]
```

---

## 模組：medchem.utils

用於處理分子的工具函數。

### batch_process()

```python
batch_process(
    mols: List[Chem.Mol],
    func: Callable,
    n_jobs: int = 1,
    progress: bool = False,
    batch_size: Optional[int] = None
) -> List
```

以平行批次處理分子。

**參數：**
- `mols`：分子列表
- `func`：要應用於每個分子的函數
- `n_jobs`：平行工作者數量
- `progress`：顯示進度條
- `batch_size`：處理批次的大小

### standardize_mol()

```python
standardize_mol(mol: Chem.Mol) -> Chem.Mol
```

標準化分子表示（清理、中和電荷等）。

---

## 常見模式

### 模式：平行處理

所有篩選器支援平行化：

```python
# 使用所有 CPU 核心
results = filter_object(mols=mol_list, n_jobs=-1, progress=True)

# 使用特定數量的核心
results = filter_object(mols=mol_list, n_jobs=4, progress=True)
```

### 模式：組合多個篩選器

```python
import medchem as mc

# 應用多個篩選器
rule_filter = mc.rules.RuleFilters(rule_list=["rule_of_five"])
alert_filter = mc.structural.CommonAlertsFilters()
lilly_filter = mc.structural.LillyDemeritsFilters()

# 取得結果
rule_results = rule_filter(mols=mol_list, n_jobs=-1)
alert_results = alert_filter(mols=mol_list, n_jobs=-1)
lilly_results = lilly_filter(mols=mol_list, n_jobs=-1)

# 組合標準
passing_mols = [
    mol for i, mol in enumerate(mol_list)
    if rule_results[i]["passes"]
    and not alert_results[i]["has_alerts"]
    and lilly_results[i]["passes"]
]
```

### 模式：使用 DataFrame

```python
import pandas as pd
import datamol as dm
import medchem as mc

# 載入資料
df = pd.read_csv("molecules.csv")
df["mol"] = df["smiles"].apply(dm.to_mol)

# 應用篩選器
rfilter = mc.rules.RuleFilters(rule_list=["rule_of_five", "rule_of_cns"])
results = rfilter(mols=df["mol"].tolist(), n_jobs=-1)

# 將結果新增到 dataframe
df["passes_ro5"] = [r["rule_of_five"] for r in results]
df["passes_cns"] = [r["rule_of_cns"] for r in results]

# 篩選 dataframe
filtered_df = df[df["passes_ro5"] & df["passes_cns"]]
```
