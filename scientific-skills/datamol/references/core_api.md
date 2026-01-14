# Datamol 核心 API 參考

本文件涵蓋 datamol 命名空間中可用的主要函數。

## 分子建立與轉換

### `to_mol(mol, ...)`
將 SMILES 字串或其他分子表示轉換為 RDKit 分子物件。
- **參數**：接受 SMILES 字串、InChI 或其他分子格式
- **返回**：`rdkit.Chem.Mol` 物件
- **常見用法**：`mol = dm.to_mol("CCO")`

### `from_inchi(inchi)`
將 InChI 字串轉換為分子物件。

### `from_smarts(smarts)`
將 SMARTS 模式轉換為分子物件。

### `from_selfies(selfies)`
將 SELFIES 字串轉換為分子物件。

### `copy_mol(mol)`
建立分子物件的副本以避免修改原始物件。

## 分子匯出

### `to_smiles(mol, ...)`
將分子物件轉換為 SMILES 字串。
- **常見參數**：`canonical=True`、`isomeric=True`

### `to_inchi(mol, ...)`
將分子轉換為 InChI 字串表示。

### `to_inchikey(mol)`
將分子轉換為 InChI 鍵（固定長度雜湊）。

### `to_smarts(mol)`
將分子轉換為 SMARTS 模式。

### `to_selfies(mol)`
將分子轉換為 SELFIES（自引用嵌入字串）格式。

## 清理與標準化

### `sanitize_mol(mol, ...)`
使用 mol→SMILES→mol 轉換和芳香族氮修復的 RDKit 清理操作增強版本。
- **目的**：修復常見的分子結構問題
- **返回**：清理後的分子，若清理失敗則返回 None

### `standardize_mol(mol, disconnect_metals=False, normalize=True, reionize=True, ...)`
應用全面的標準化程序，包括：
- 金屬斷開
- 正規化（電荷校正）
- 重新離子化
- 片段處理（最大片段選擇）

### `standardize_smiles(smiles, ...)`
直接對 SMILES 字串應用 SMILES 標準化程序。

### `fix_mol(mol)`
嘗試自動修復分子結構問題。

### `fix_valence(mol)`
校正分子結構中的價態錯誤。

## 分子屬性

### `reorder_atoms(mol, ...)`
確保同一分子的原子排序一致，無論原始 SMILES 表示如何。
- **目的**：維持可重現的特徵生成

### `remove_hs(mol, ...)`
從分子結構中移除氫原子。

### `add_hs(mol, ...)`
向分子結構添加顯式氫原子。

## 指紋與相似性

### `to_fp(mol, fp_type='ecfp', ...)`
生成用於相似性計算的分子指紋。
- **指紋類型**：
  - `'ecfp'` - 擴展連接指紋（Morgan）
  - `'fcfp'` - 功能連接指紋
  - `'maccs'` - MACCS 鍵
  - `'topological'` - 拓撲指紋
  - `'atompair'` - 原子對指紋
- **常見參數**：`n_bits`、`radius`
- **返回**：Numpy 陣列或 RDKit 指紋物件

### `pdist(mols, ...)`
計算列表中所有分子之間的成對 Tanimoto 距離。
- **支援**：通過 `n_jobs` 參數進行平行處理
- **返回**：距離矩陣

### `cdist(mols1, mols2, ...)`
計算兩組分子之間的 Tanimoto 距離。

## 聚類與多樣性

### `cluster_mols(mols, cutoff=0.2, feature_fn=None, n_jobs=1)`
使用 Butina 聚類演算法對分子進行聚類。
- **參數**：
  - `cutoff`：距離閾值（預設 0.2）
  - `feature_fn`：分子特徵的自訂函數
  - `n_jobs`：平行化（-1 表示所有核心）
- **重要**：建立完整距離矩陣 - 適用於約 1000 個結構，不適用於 10,000+ 個
- **返回**：聚類列表（每個聚類是分子索引的列表）

### `pick_diverse(mols, npick, ...)`
基於指紋多樣性選擇多樣化的分子子集。

### `pick_centroids(mols, npick, ...)`
選擇代表聚類的質心分子。

## 圖形操作

### `to_graph(mol)`
將分子轉換為圖形表示以進行基於圖形的分析。

### `get_all_path_between(mol, start, end)`
找出分子結構中兩個原子之間的所有路徑。

## DataFrame 整合

### `to_df(mols, smiles_column='smiles', mol_column='mol')`
將分子列表轉換為 pandas DataFrame。

### `from_df(df, smiles_column='smiles', mol_column='mol')`
將 pandas DataFrame 轉換為分子列表。
