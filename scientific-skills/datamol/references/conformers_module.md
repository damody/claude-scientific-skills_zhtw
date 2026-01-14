# Datamol 構象異構體模組參考

`datamol.conformers` 模組提供用於生成和分析 3D 分子構象的工具。

## 構象異構體生成

### `dm.conformers.generate(mol, n_confs=None, rms_cutoff=None, minimize_energy=True, method='ETKDGv3', add_hs=True, ...)`
生成 3D 分子構象異構體。
- **參數**：
  - `mol`：輸入分子
  - `n_confs`：要生成的構象異構體數量（若為 None 則根據可旋轉鍵自動決定）
  - `rms_cutoff`：用於過濾相似構象異構體的 RMS 閾值（埃）（移除重複項）
  - `minimize_energy`：應用 UFF 能量最小化（預設：True）
  - `method`：嵌入方法 - 選項：
    - `'ETDG'` - 實驗扭轉距離幾何
    - `'ETKDG'` - 帶有額外基礎知識的 ETDG
    - `'ETKDGv2'` - 增強版本 2
    - `'ETKDGv3'` - 增強版本 3（預設，建議使用）
  - `add_hs`：嵌入前添加氫原子（預設：True，對品質至關重要）
  - `random_seed`：設定以確保可重現性
- **返回**：帶有嵌入構象異構體的分子
- **範例**：
  ```python
  mol = dm.to_mol("CCO")
  mol_3d = dm.conformers.generate(mol, n_confs=10, rms_cutoff=0.5)
  conformers = mol_3d.GetConformers()  # 存取所有構象異構體
  ```

## 構象異構體聚類

### `dm.conformers.cluster(mol, rms_cutoff=1.0, already_aligned=False, centroids=False)`
按 RMS 距離分組構象異構體。
- **參數**：
  - `rms_cutoff`：聚類閾值（埃）（預設：1.0）
  - `already_aligned`：構象異構體是否已預對齊
  - `centroids`：返回質心構象異構體（True）或聚類分組（False）
- **返回**：聚類資訊或質心構象異構體
- **使用案例**：識別不同的構象家族

### `dm.conformers.return_centroids(mol, conf_clusters, centroids=True)`
從聚類中提取代表性構象異構體。
- **參數**：
  - `conf_clusters`：來自 `cluster()` 的聚類索引序列
  - `centroids`：返回單一分子（True）或分子列表（False）
- **返回**：質心構象異構體

## 構象異構體分析

### `dm.conformers.rmsd(mol)`
計算所有構象異構體之間的成對 RMSD 矩陣。
- **需求**：最少 2 個構象異構體
- **返回**：NxN 的 RMSD 值矩陣
- **使用案例**：量化構象多樣性

### `dm.conformers.sasa(mol, n_jobs=1, ...)`
使用 FreeSASA 計算溶劑可及表面積（SASA）。
- **參數**：
  - `n_jobs`：多個構象異構體的平行化
- **返回**：SASA 值陣列（每個構象異構體一個）
- **儲存**：值以屬性 `'rdkit_free_sasa'` 儲存在每個構象異構體中
- **範例**：
  ```python
  sasa_values = dm.conformers.sasa(mol_3d)
  # 或從構象異構體屬性存取
  conf = mol_3d.GetConformer(0)
  sasa = conf.GetDoubleProp('rdkit_free_sasa')
  ```

## 低階構象異構體操作

### `dm.conformers.center_of_mass(mol, conf_id=-1, use_atoms=True, round_coord=None)`
計算分子中心。
- **參數**：
  - `conf_id`：構象異構體索引（-1 表示第一個構象異構體）
  - `use_atoms`：使用原子質量（True）或幾何中心（False）
  - `round_coord`：四捨五入的小數精度
- **返回**：中心的 3D 座標
- **使用案例**：將分子置中以進行視覺化或對齊

### `dm.conformers.get_coords(mol, conf_id=-1)`
從構象異構體擷取原子座標。
- **返回**：Nx3 的原子位置 numpy 陣列
- **範例**：
  ```python
  positions = dm.conformers.get_coords(mol_3d, conf_id=0)
  # positions.shape: (num_atoms, 3)
  ```

### `dm.conformers.translate(mol, conf_id=-1, transform_matrix=None)`
使用轉換矩陣重新定位構象異構體。
- **修改**：就地操作
- **使用案例**：對齊或重新定位分子

## 工作流程範例

```python
import datamol as dm

# 1. 建立分子並生成構象異構體
mol = dm.to_mol("CC(C)CCO")  # 異戊醇
mol_3d = dm.conformers.generate(
    mol,
    n_confs=50,           # 生成 50 個初始構象異構體
    rms_cutoff=0.5,       # 過濾相似構象異構體
    minimize_energy=True   # 最小化能量
)

# 2. 分析構象異構體
n_conformers = mol_3d.GetNumConformers()
print(f"生成了 {n_conformers} 個獨特的構象異構體")

# 3. 計算 SASA
sasa_values = dm.conformers.sasa(mol_3d)

# 4. 聚類構象異構體
clusters = dm.conformers.cluster(mol_3d, rms_cutoff=1.0, centroids=False)

# 5. 取得代表性構象異構體
centroids = dm.conformers.return_centroids(mol_3d, clusters)

# 6. 存取 3D 座標
coords = dm.conformers.get_coords(mol_3d, conf_id=0)
```

## 關鍵概念

- **距離幾何**：從連接資訊生成 3D 結構的方法
- **ETKDG**：使用實驗扭轉角偏好和額外的化學知識
- **RMS 截斷**：較低的值 = 更多獨特的構象異構體；較高的值 = 較少但更獨特的構象異構體
- **能量最小化**：將結構弛豫至最近的局部能量最小值
- **氫原子**：對於準確的 3D 幾何至關重要 - 嵌入時始終包含
