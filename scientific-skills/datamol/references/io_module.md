# Datamol I/O 模組參考

`datamol.io` 模組提供跨多種格式的全面分子資料檔案處理。

## 讀取分子檔案

### `dm.read_sdf(filename, sanitize=True, remove_hs=True, as_df=True, mol_column='mol', ...)`
讀取結構資料檔案（SDF）格式。
- **參數**：
  - `filename`：SDF 檔案路徑（通過 fsspec 支援本地和遠端路徑）
  - `sanitize`：對分子應用清理
  - `remove_hs`：移除顯式氫原子
  - `as_df`：返回 DataFrame（True）或分子列表（False）
  - `mol_column`：DataFrame 中分子欄的名稱
  - `n_jobs`：啟用平行處理
- **返回**：DataFrame 或分子列表
- **範例**：`df = dm.read_sdf("compounds.sdf")`

### `dm.read_smi(filename, smiles_column='smiles', mol_column='mol', as_df=True, ...)`
讀取 SMILES 檔案（預設空格分隔）。
- **常見格式**：SMILES 後跟分子 ID/名稱
- **範例**：`df = dm.read_smi("molecules.smi")`

### `dm.read_csv(filename, smiles_column='smiles', mol_column=None, ...)`
讀取 CSV 檔案並可選擇自動進行 SMILES 到分子的轉換。
- **參數**：
  - `smiles_column`：包含 SMILES 字串的欄
  - `mol_column`：若指定，則從 SMILES 欄建立分子物件
- **範例**：`df = dm.read_csv("data.csv", smiles_column="SMILES", mol_column="mol")`

### `dm.read_excel(filename, sheet_name=0, smiles_column='smiles', mol_column=None, ...)`
讀取帶有分子處理的 Excel 檔案。
- **參數**：
  - `sheet_name`：要讀取的工作表（索引或名稱）
  - 其他參數與 `read_csv` 類似
- **範例**：`df = dm.read_excel("compounds.xlsx", sheet_name="Sheet1")`

### `dm.read_molblock(molblock, sanitize=True, remove_hs=True)`
解析 MOL 區塊字串（分子結構文字表示）。

### `dm.read_mol2file(filename, sanitize=True, remove_hs=True, cleanupSubstructures=True)`
讀取 Mol2 格式檔案。

### `dm.read_pdbfile(filename, sanitize=True, remove_hs=True, proximityBonding=True)`
讀取蛋白質資料庫（PDB）格式檔案。

### `dm.read_pdbblock(pdbblock, sanitize=True, remove_hs=True, proximityBonding=True)`
解析 PDB 區塊字串。

### `dm.open_df(filename, ...)`
通用 DataFrame 讀取器 - 自動偵測格式。
- **支援格式**：CSV、Excel、Parquet、JSON、SDF
- **範例**：`df = dm.open_df("data.csv")` 或 `df = dm.open_df("molecules.sdf")`

## 寫入分子檔案

### `dm.to_sdf(mols, filename, mol_column=None, ...)`
將分子寫入 SDF 檔案。
- **輸入類型**：
  - 分子列表
  - 帶有分子欄的 DataFrame
  - 分子序列
- **參數**：
  - `mol_column`：若輸入為 DataFrame 則為欄名稱
- **範例**：
  ```python
  dm.to_sdf(mols, "output.sdf")
  # 或從 DataFrame
  dm.to_sdf(df, "output.sdf", mol_column="mol")
  ```

### `dm.to_smi(mols, filename, mol_column=None, ...)`
將分子寫入 SMILES 檔案並進行可選驗證。
- **格式**：SMILES 字串以及可選的分子名稱/ID

### `dm.to_xlsx(df, filename, mol_columns=None, ...)`
將 DataFrame 匯出為 Excel 並渲染分子圖像。
- **參數**：
  - `mol_columns`：包含要渲染為圖像的分子欄
- **特殊功能**：在 Excel 儲存格中自動將分子渲染為圖像
- **範例**：`dm.to_xlsx(df, "molecules.xlsx", mol_columns=["mol"])`

### `dm.to_molblock(mol, ...)`
將分子轉換為 MOL 區塊字串。

### `dm.to_pdbblock(mol, ...)`
將分子轉換為 PDB 區塊字串。

### `dm.save_df(df, filename, ...)`
以多種格式儲存 DataFrame（CSV、Excel、Parquet、JSON）。

## 遠端檔案支援

所有 I/O 函數通過 fsspec 整合支援遠端檔案路徑：
- **支援協定**：S3（AWS）、GCS（Google Cloud）、Azure、HTTP/HTTPS
- **範例**：
  ```python
  dm.read_sdf("s3://bucket/compounds.sdf")
  dm.read_csv("https://example.com/data.csv")
  ```

## 跨函數的關鍵參數

- **`sanitize`**：應用分子清理（預設：True）
- **`remove_hs`**：移除顯式氫原子（預設：True）
- **`as_df`**：返回 DataFrame vs 列表（大多數函數預設：True）
- **`n_jobs`**：啟用平行處理（None = 所有核心，1 = 順序）
- **`mol_column`**：DataFrame 中分子欄的名稱
- **`smiles_column`**：DataFrame 中 SMILES 欄的名稱
