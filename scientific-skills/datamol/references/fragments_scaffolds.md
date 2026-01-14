# Datamol 片段與骨架參考

## 骨架模組（`datamol.scaffold`）

骨架代表分子的核心結構，用於識別結構家族和分析構效關係（SAR）。

### Murcko 骨架

#### `dm.to_scaffold_murcko(mol)`
提取 Bemis-Murcko 骨架（分子框架）。
- **方法**：移除側鏈，保留環系統和連接子
- **返回**：代表骨架的分子物件
- **使用案例**：識別化合物系列的核心結構
- **範例**：
  ```python
  mol = dm.to_mol("c1ccc(cc1)CCN")  # 苯乙胺
  scaffold = dm.to_scaffold_murcko(mol)
  scaffold_smiles = dm.to_smiles(scaffold)
  # 返回：'c1ccccc1CC'（苯環 + 乙基連接子）
  ```

**骨架分析工作流程**：
```python
# 從化合物庫提取骨架
scaffolds = [dm.to_scaffold_murcko(mol) for mol in mols]
scaffold_smiles = [dm.to_smiles(s) for s in scaffolds]

# 計算骨架頻率
from collections import Counter
scaffold_counts = Counter(scaffold_smiles)
most_common = scaffold_counts.most_common(10)
```

### 模糊骨架

#### `dm.scaffold.fuzzy_scaffolding(mol, ...)`
生成帶有可強制執行群組（必須出現在核心中）的模糊骨架。
- **目的**：更靈活的骨架定義，允許指定的官能基
- **使用案例**：超越 Murcko 規則的自訂骨架定義

### 應用

**基於骨架的分割**（用於 ML 模型驗證）：
```python
# 按骨架分組化合物
scaffold_to_mols = {}
for mol, scaffold in zip(mols, scaffolds):
    smi = dm.to_smiles(scaffold)
    if smi not in scaffold_to_mols:
        scaffold_to_mols[smi] = []
    scaffold_to_mols[smi].append(mol)

# 確保訓練/測試集有不同的骨架
```

**SAR 分析**：
```python
# 按骨架分組並分析活性
for scaffold_smi, molecules in scaffold_to_mols.items():
    activities = [get_activity(mol) for mol in molecules]
    print(f"骨架：{scaffold_smi}，平均活性：{np.mean(activities)}")
```

---

## 片段模組（`datamol.fragment`）

分子片段化根據化學規則將分子分解成較小的片段，用於基於片段的藥物設計和子結構分析。

### BRICS 片段化

#### `dm.fragment.brics(mol, ...)`
使用 BRICS（逆合成相關化學子結構斷裂）片段化分子。
- **方法**：基於 16 種化學有意義的鍵類型進行解剖
- **考量**：考慮化學環境和周圍子結構
- **返回**：片段 SMILES 字串集合
- **使用案例**：逆合成分析、基於片段的設計
- **範例**：
  ```python
  mol = dm.to_mol("c1ccccc1CCN")
  fragments = dm.fragment.brics(mol)
  # 返回片段如：'[1*]CCN'、'[1*]c1ccccc1' 等
  # [1*] 代表連接點
  ```

### RECAP 片段化

#### `dm.fragment.recap(mol, ...)`
使用 RECAP（逆合成組合分析程序）片段化分子。
- **方法**：基於 11 種預定義鍵類型進行解剖
- **規則**：
  - 保持小於 5 個碳的烷基完整
  - 保留環狀鍵
- **返回**：片段 SMILES 字串集合
- **使用案例**：組合庫設計
- **範例**：
  ```python
  mol = dm.to_mol("CCCCCc1ccccc1")
  fragments = dm.fragment.recap(mol)
  ```

### MMPA 片段化

#### `dm.fragment.mmpa_frag(mol, ...)`
用於匹配分子對分析的片段化。
- **目的**：生成適合識別分子對的片段
- **使用案例**：分析小型結構變化如何影響屬性
- **範例**：
  ```python
  fragments = dm.fragment.mmpa_frag(mol)
  # 用於找出相差單一轉換的分子對
  ```

### 方法比較

| 方法 | 鍵類型 | 保留環 | 最適合 |
|------|--------|--------|--------|
| BRICS | 16 | 是 | 逆合成分析、片段重組 |
| RECAP | 11 | 是 | 組合庫設計 |
| MMPA | 可變 | 視情況而定 | 構效關係分析 |

### 片段化工作流程

```python
import datamol as dm

# 1. 片段化分子
mol = dm.to_mol("CC(=O)Oc1ccccc1C(=O)O")  # 阿斯匹靈
brics_frags = dm.fragment.brics(mol)
recap_frags = dm.fragment.recap(mol)

# 2. 分析庫中的片段頻率
all_fragments = []
for mol in molecule_library:
    frags = dm.fragment.brics(mol)
    all_fragments.extend(frags)

# 3. 識別常見片段
from collections import Counter
fragment_counts = Counter(all_fragments)
common_fragments = fragment_counts.most_common(20)

# 4. 將片段轉換回分子（移除連接點）
def clean_fragment(frag_smiles):
    # 移除 [1*]、[2*] 等連接點標記
    clean = frag_smiles.replace('[1*]', '[H]')
    return dm.to_mol(clean)
```

### 進階：基於片段的虛擬篩選

```python
# 從已知活性化合物建立片段庫
active_fragments = set()
for active_mol in active_compounds:
    frags = dm.fragment.brics(active_mol)
    active_fragments.update(frags)

# 篩選化合物以檢查是否存在活性片段
def score_by_fragments(mol, fragment_set):
    mol_frags = dm.fragment.brics(mol)
    overlap = mol_frags.intersection(fragment_set)
    return len(overlap) / len(mol_frags)

# 評分篩選庫
scores = [score_by_fragments(mol, active_fragments) for mol in screening_lib]
```

### 關鍵概念

- **連接點**：在片段 SMILES 中以 [1*]、[2*] 等標記
- **逆合成**：片段化模擬合成斷開
- **化學有意義**：在典型合成鍵處斷裂
- **重組**：片段理論上可以重組成有效分子
