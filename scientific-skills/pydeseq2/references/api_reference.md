# PyDESeq2 API 參考

本文件提供 PyDESeq2 類別、方法和工具的完整 API 參考。

## 核心類別

### DeseqDataSet

差異表現分析的主要類別，處理從標準化到對數倍數變化擬合的資料處理。

**用途：** 實作 RNA-seq 計數資料的離散度和對數倍數變化（LFC）估計。

**初始化參數：**
- `counts`：形狀為（樣本 × 基因）的 pandas DataFrame，包含非負整數讀取計數
- `metadata`：形狀為（樣本 × 變數）的 pandas DataFrame，包含樣本註解
- `design`：str，Wilkinson 公式，指定統計模型（例如 "~condition"、"~group + condition"）
- `refit_cooks`：bool，是否在移除 Cook 距離離群值後重新擬合參數（預設：True）
- `n_cpus`：int，用於平行處理的 CPU 數量（可選）
- `quiet`：bool，抑制進度訊息（預設：False）

**主要方法：**

#### `deseq2()`
執行完整的 DESeq2 流程，進行標準化和離散度/LFC 擬合。

**執行步驟：**
1. 計算標準化因子（大小因子）
2. 擬合基因層級離散度
3. 擬合離散度趨勢曲線
4. 計算離散度先驗
5. 擬合 MAP（最大後驗）離散度
6. 擬合對數倍數變化
7. 計算 Cook 距離以偵測離群值
8. 若 `refit_cooks=True` 則選擇性重新擬合

**回傳：** None（就地修改物件）

#### `to_picklable_anndata()`
將 DeseqDataSet 轉換為可用 pickle 儲存的 AnnData 物件。

**回傳：** AnnData 物件，包含：
- `X`：計數資料矩陣
- `obs`：樣本層級詮釋資料（1D）
- `var`：基因層級詮釋資料（1D）
- `varm`：基因層級多維資料（例如 LFC 估計）

**使用範例：**
```python
import pickle
with open("result_adata.pkl", "wb") as f:
    pickle.dump(dds.to_picklable_anndata(), f)
```

**屬性（執行 deseq2() 後）：**
- `layers`：包含各種矩陣的字典（標準化計數等）
- `varm`：包含基因層級結果的字典（對數倍數變化、離散度等）
- `obsm`：包含樣本層級資訊的字典
- `uns`：包含全域參數的字典

---

### DeseqStats

執行統計檢定和計算差異表現 p 值的類別。

**用途：** 使用 Wald 檢定和可選的 LFC 收縮，促進 PyDESeq2 統計檢定。

**初始化參數：**
- `dds`：已用 `deseq2()` 處理過的 DeseqDataSet 物件
- `contrast`：list 或 None，指定檢定對比
  - 格式：`[變數, 檢定層級, 參考層級]`
  - 範例：`["condition", "treated", "control"]` 檢定處理組 vs 對照組
  - 若為 None，使用設計公式中的最後一個係數
- `alpha`：float，獨立過濾的顯著性閾值（預設：0.05）
- `cooks_filter`：bool，是否根據 Cook 距離過濾離群值（預設：True）
- `independent_filter`：bool，是否執行獨立過濾（預設：True）
- `n_cpus`：int，用於平行處理的 CPU 數量（可選）
- `quiet`：bool，抑制進度訊息（預設：False）

**主要方法：**

#### `summary()`
執行 Wald 檢定並計算 p 值和校正後 p 值。

**執行步驟：**
1. 對指定對比執行 Wald 統計檢定
2. 可選的 Cook 距離過濾
3. 可選的獨立過濾以移除低統計檢定力的檢定
4. 多重檢定校正（Benjamini-Hochberg 程序）

**回傳：** None（結果儲存在 `results_df` 屬性中）

**結果 DataFrame 欄位：**
- `baseMean`：所有樣本的平均標準化計數
- `log2FoldChange`：條件之間的 log2 倍數變化
- `lfcSE`：log2 倍數變化的標準誤
- `stat`：Wald 檢定統計量
- `pvalue`：原始 p 值
- `padj`：校正後 p 值（FDR 校正）

#### `lfc_shrink(coeff=None)`
使用 apeGLM 方法對對數倍數變化應用收縮。

**用途：** 減少 LFC 估計中的雜訊，以便更好地視覺化和排名，特別是對於低計數或高變異性的基因。

**參數：**
- `coeff`：str 或 None，要收縮的係數名稱（若為 None，使用對比中的係數）

**重要事項：** 收縮僅用於視覺化/排名目的。統計檢定結果（p 值、校正後 p 值）保持不變。

**回傳：** None（使用收縮後的 LFC 更新 `results_df`）

**屬性：**
- `results_df`：包含檢定結果的 pandas DataFrame（在 `summary()` 後可用）

---

## 工具函數

### `pydeseq2.utils.load_example_data(modality="single-factor")`

載入用於測試和教學的合成範例資料集。

**參數：**
- `modality`：str，"single-factor" 或 "multi-factor"

**回傳：** (counts_df, metadata_df) 元組
- `counts_df`：包含合成計數資料的 pandas DataFrame
- `metadata_df`：包含樣本註解的 pandas DataFrame

---

## 預處理模組

`pydeseq2.preprocessing` 模組提供資料準備的工具。

**常見操作：**
- 根據最小讀取計數過濾基因
- 根據詮釋資料標準過濾樣本
- 資料轉換和標準化

---

## 推論類別

### Inference
定義 DESeq2 相關推論方法介面的抽象基礎類別。

### DefaultInference
使用 scipy、sklearn 和 numpy 的推論方法預設實作。

**用途：** 提供以下數學實作：
- GLM（廣義線性模型）擬合
- 離散度估計
- 趨勢曲線擬合
- 統計檢定

---

## 資料結構要求

### 計數矩陣
- **形狀：**（樣本 × 基因）
- **類型：** pandas DataFrame
- **值：** 非負整數（原始讀取計數）
- **索引：** 樣本識別碼（必須與詮釋資料索引匹配）
- **欄位：** 基因識別碼

### 詮釋資料
- **形狀：**（樣本 × 變數）
- **類型：** pandas DataFrame
- **索引：** 樣本識別碼（必須與計數矩陣索引匹配）
- **欄位：** 實驗因子（例如 "condition"、"batch"、"group"）
- **值：** 設計公式中使用的類別型或連續型變數

### 重要注意事項
- 計數和詮釋資料之間的樣本順序必須匹配
- 分析前應處理詮釋資料中的缺失值
- 基因名稱應唯一
- 計數檔案通常需要轉置：`counts_df = counts_df.T`

---

## 常見工作流程模式

```python
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats

# 1. 初始化資料集
dds = DeseqDataSet(
    counts=counts_df,
    metadata=metadata,
    design="~condition",
    refit_cooks=True
)

# 2. 擬合離散度和 LFC
dds.deseq2()

# 3. 執行統計檢定
ds = DeseqStats(
    dds,
    contrast=["condition", "treated", "control"],
    alpha=0.05
)
ds.summary()

# 4. 可選：收縮 LFC 以進行視覺化
ds.lfc_shrink()

# 5. 存取結果
results = ds.results_df
```

---

## 版本相容性

PyDESeq2 旨在匹配 DESeq2 v1.34.0 的預設設定。由於這是在 Python 中從頭實作，可能存在一些差異。

**測試環境：**
- Python 3.10-3.11
- anndata 0.8.0+
- numpy 1.23.0+
- pandas 1.4.3+
- scikit-learn 1.1.1+
- scipy 1.11.0+
