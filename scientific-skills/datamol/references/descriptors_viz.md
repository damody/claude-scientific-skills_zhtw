# Datamol 描述子與視覺化參考

## 描述子模組（`datamol.descriptors`）

描述子模組提供計算分子屬性和描述子的工具。

### 專門描述子函數

#### `dm.descriptors.n_aromatic_atoms(mol)`
計算芳香原子的數量。
- **返回**：整數計數
- **使用案例**：芳香性分析

#### `dm.descriptors.n_aromatic_atoms_proportion(mol)`
計算芳香原子與總重原子的比例。
- **返回**：0 到 1 之間的浮點數
- **使用案例**：量化芳香特性

#### `dm.descriptors.n_charged_atoms(mol)`
計算具有非零形式電荷的原子數量。
- **返回**：整數計數
- **使用案例**：電荷分佈分析

#### `dm.descriptors.n_rigid_bonds(mol)`
計算不可旋轉鍵（既非單鍵也非環鍵）。
- **返回**：整數計數
- **使用案例**：分子柔性評估

#### `dm.descriptors.n_stereo_centers(mol)`
計算立體中心（手性中心）。
- **返回**：整數計數
- **使用案例**：立體化學分析

#### `dm.descriptors.n_stereo_centers_unspecified(mol)`
計算缺乏立體化學規格的立體中心。
- **返回**：整數計數
- **使用案例**：識別不完整的立體化學

### 批次描述子計算

#### `dm.descriptors.compute_many_descriptors(mol, properties_fn=None, add_properties=True)`
計算單一分子的多個分子屬性。
- **參數**：
  - `properties_fn`：描述子函數的自訂列表
  - `add_properties`：包含額外計算的屬性
- **返回**：描述子名稱 → 值對的字典
- **預設描述子包括**：
  - 分子量、LogP、氫鍵供體/受體數量
  - 芳香原子、立體中心、可旋轉鍵
  - TPSA（拓撲極性表面積）
  - 環數量、雜原子數量
- **範例**：
  ```python
  mol = dm.to_mol("CCO")
  descriptors = dm.descriptors.compute_many_descriptors(mol)
  # 返回：{'mw': 46.07, 'logp': -0.03, 'hbd': 1, 'hba': 1, ...}
  ```

#### `dm.descriptors.batch_compute_many_descriptors(mols, properties_fn=None, add_properties=True, n_jobs=1, batch_size=None, progress=False)`
平行計算多個分子的描述子。
- **參數**：
  - `mols`：分子列表
  - `n_jobs`：平行作業數（-1 表示所有核心）
  - `batch_size`：平行處理的區塊大小
  - `progress`：顯示進度條
- **返回**：每個分子一行的 Pandas DataFrame
- **範例**：
  ```python
  mols = [dm.to_mol(smi) for smi in smiles_list]
  df = dm.descriptors.batch_compute_many_descriptors(
      mols,
      n_jobs=-1,
      progress=True
  )
  ```

### RDKit 描述子存取

#### `dm.descriptors.any_rdkit_descriptor(name)`
按名稱從 RDKit 擷取任何描述子函數。
- **參數**：`name` - 描述子函數名稱（例如 'MolWt'、'TPSA'）
- **返回**：RDKit 描述子函數
- **可用描述子**：來自 `rdkit.Chem.Descriptors` 和 `rdkit.Chem.rdMolDescriptors`
- **範例**：
  ```python
  tpsa_fn = dm.descriptors.any_rdkit_descriptor('TPSA')
  tpsa_value = tpsa_fn(mol)
  ```

### 常見使用案例

**類藥性過濾（Lipinski 五規則）**：
```python
descriptors = dm.descriptors.compute_many_descriptors(mol)
is_druglike = (
    descriptors['mw'] <= 500 and
    descriptors['logp'] <= 5 and
    descriptors['hbd'] <= 5 and
    descriptors['hba'] <= 10
)
```

**ADME 屬性分析**：
```python
df = dm.descriptors.batch_compute_many_descriptors(compound_library)
# 按 TPSA 過濾以評估血腦屏障穿透性
bbb_candidates = df[df['tpsa'] < 90]
```

---

## 視覺化模組（`datamol.viz`）

viz 模組提供將分子和構象異構體渲染為圖像的工具。

### 主要視覺化函數

#### `dm.viz.to_image(mols, legends=None, n_cols=4, use_svg=False, mol_size=(200, 200), highlight_atom=None, highlight_bond=None, outfile=None, max_mols=None, copy=True, indices=False, ...)`
從分子生成圖像網格。
- **參數**：
  - `mols`：單一分子或分子列表
  - `legends`：作為標籤的字串或字串列表（每個分子一個）
  - `n_cols`：每行分子數（預設：4）
  - `use_svg`：輸出 SVG 格式（True）或 PNG（False，預設）
  - `mol_size`：元組（寬度，高度）或單一整數表示正方形圖像
  - `highlight_atom`：要突出顯示的原子索引（列表或字典）
  - `highlight_bond`：要突出顯示的鍵索引（列表或字典）
  - `outfile`：儲存路徑（本地或遠端，支援 fsspec）
  - `max_mols`：要顯示的最大分子數
  - `indices`：在結構上繪製原子索引（預設：False）
  - `align`：使用 MCS（最大公共子結構）對齊分子
- **返回**：圖像物件（可在 Jupyter 中顯示）或儲存到檔案
- **範例**：
  ```python
  # 基本網格
  dm.viz.to_image(mols[:10], legends=[dm.to_smiles(m) for m in mols[:10]])

  # 儲存到檔案
  dm.viz.to_image(mols, outfile="molecules.png", n_cols=5)

  # 突出顯示子結構
  dm.viz.to_image(mol, highlight_atom=[0, 1, 2], highlight_bond=[0, 1])

  # 對齊視覺化
  dm.viz.to_image(mols, align=True, legends=activity_labels)
  ```

### 構象異構體視覺化

#### `dm.viz.conformers(mol, n_confs=None, align_conf=True, n_cols=3, sync_views=True, remove_hs=True, ...)`
以網格佈局顯示多個構象異構體。
- **參數**：
  - `mol`：帶有嵌入構象異構體的分子
  - `n_confs`：要顯示的構象異構體數量或索引列表（None = 全部）
  - `align_conf`：對齊構象異構體以進行比較（預設：True）
  - `n_cols`：網格列數（預設：3）
  - `sync_views`：互動時同步 3D 視圖（預設：True）
  - `remove_hs`：移除氫原子以提高清晰度（預設：True）
- **返回**：構象異構體視覺化網格
- **使用案例**：比較構象多樣性
- **範例**：
  ```python
  mol_3d = dm.conformers.generate(mol, n_confs=20)
  dm.viz.conformers(mol_3d, n_confs=10, align_conf=True)
  ```

### 圓形網格視覺化

#### `dm.viz.circle_grid(center_mol, circle_mols, mol_size=200, circle_margin=50, act_mapper=None, ...)`
建立以中心分子為核心的同心環視覺化。
- **參數**：
  - `center_mol`：中心的分子
  - `circle_mols`：分子列表的列表（每個環一個列表）
  - `mol_size`：每個分子的圖像大小
  - `circle_margin`：環之間的間距（預設：50）
  - `act_mapper`：用於顏色編碼的活性對應字典
- **返回**：圓形網格圖像
- **使用案例**：視覺化分子鄰域、SAR 分析、相似性網路
- **範例**：
  ```python
  # 顯示被相似化合物包圍的參考分子
  dm.viz.circle_grid(
      center_mol=reference,
      circle_mols=[nearest_neighbors, second_tier]
  )
  ```

### 視覺化最佳實踐

1. **使用圖例以提高清晰度**：始終用 SMILES、ID 或活性值標記分子
2. **對齊相關分子**：在 `to_image()` 中使用 `align=True` 進行 SAR 分析
3. **調整網格大小**：根據分子數量和顯示寬度設定 `n_cols`
4. **出版物使用 SVG**：設定 `use_svg=True` 以獲得可縮放的向量圖形
5. **突出顯示子結構**：使用 `highlight_atom` 和 `highlight_bond` 強調特徵
6. **儲存大型網格**：使用 `outfile` 參數儲存而非在記憶體中顯示
