# ChEMBL Web 服務 API 參考

## 概述

ChEMBL 是由歐洲生物資訊研究所（EBI）維護的人工策展具有類藥性質的生物活性分子資料庫。它包含化合物、靶標、測定法、生物活性資料和已核准藥物的資訊。

ChEMBL 資料庫包含：
- 超過 200 萬筆化合物記錄
- 超過 140 萬筆測定記錄
- 超過 1,900 萬個活性值
- 13,000 多個藥物靶標的資訊
- 16,000 多種已核准藥物和臨床候選藥物的資料

## Python 用戶端安裝

```bash
pip install chembl_webresource_client
```

## 主要資源和端點

ChEMBL 提供 30 多個專門端點的存取：

### 核心資料類型

- **molecule** - 化合物結構、性質和同義詞
- **target** - 蛋白質和非蛋白質生物靶標
- **activity** - 生物測定測量結果
- **assay** - 實驗測定詳細資訊
- **drug** - 已核准的藥品資訊
- **mechanism** - 藥物作用機制資料
- **document** - 文獻來源和參考資料
- **cell_line** - 細胞系資訊
- **tissue** - 組織類型
- **protein_class** - 蛋白質分類
- **target_component** - 靶標組件詳細資訊
- **compound_structural_alert** - 毒性結構警示

## 查詢模式和篩選

### 篩選運算子

API 支援 Django 風格的篩選運算子：

- `__exact` - 精確匹配
- `__iexact` - 不區分大小寫的精確匹配
- `__contains` - 包含子字串
- `__icontains` - 不區分大小寫的包含
- `__startswith` - 以前綴開頭
- `__endswith` - 以後綴結尾
- `__gt` - 大於
- `__gte` - 大於或等於
- `__lt` - 小於
- `__lte` - 小於或等於
- `__range` - 值在範圍內
- `__in` - 值在清單中
- `__isnull` - 為空/非空
- `__regex` - 正規表達式匹配
- `__search` - 全文搜尋

### 篩選查詢範例

**分子量篩選：**
```python
molecules.filter(molecule_properties__mw_freebase__lte=300)
```

**名稱模式匹配：**
```python
molecules.filter(pref_name__endswith='nib')
```

**多個條件：**
```python
molecules.filter(
    molecule_properties__mw_freebase__lte=300,
    pref_name__endswith='nib'
)
```

## 化學結構搜尋

### 子結構搜尋
使用 SMILES 搜尋包含特定子結構的化合物：

```python
from chembl_webresource_client.new_client import new_client
similarity = new_client.similarity
results = similarity.filter(smiles='CC(=O)Oc1ccccc1C(=O)O', similarity=70)
```

### 相似性搜尋
尋找與查詢結構相似的化合物：

```python
similarity = new_client.similarity
results = similarity.filter(smiles='CC(=O)Oc1ccccc1C(=O)O', similarity=85)
```

## 常見資料擷取模式

### 依 ChEMBL ID 取得分子
```python
molecule = new_client.molecule.get('CHEMBL25')
```

### 取得靶標資訊
```python
target = new_client.target.get('CHEMBL240')
```

### 取得活性資料
```python
activities = new_client.activity.filter(
    target_chembl_id='CHEMBL240',
    standard_type='IC50',
    standard_value__lte=100
)
```

### 取得藥物資訊
```python
drug = new_client.drug.get('CHEMBL1234')
```

## 回應格式

API 支援多種回應格式：
- JSON（預設）
- XML
- YAML

## 快取和效能

Python 用戶端自動在本地快取結果：
- **預設快取持續時間**：24 小時
- **快取位置**：本地檔案系統
- **延遲評估**：查詢僅在存取資料時執行

### 設定選項

```python
from chembl_webresource_client.settings import Settings

# 停用快取
Settings.Instance().CACHING = False

# 調整快取過期時間（秒）
Settings.Instance().CACHE_EXPIRE = 86400  # 24 小時

# 設定逾時
Settings.Instance().TIMEOUT = 30

# 設定重試次數
Settings.Instance().TOTAL_RETRIES = 3
```

## 分子性質

可用的常見分子性質：

- `mw_freebase` - 分子量
- `alogp` - 計算的 LogP
- `hba` - 氫鍵受體數
- `hbd` - 氫鍵供體數
- `psa` - 極性表面積
- `rtb` - 可旋轉鍵數
- `ro3_pass` - 三規則符合性
- `num_ro5_violations` - Lipinski 五規則違規數
- `cx_most_apka` - 最酸性 pKa
- `cx_most_bpka` - 最鹼性 pKa
- `molecular_species` - 分子種類
- `full_mwt` - 完整分子量

## 生物活性資料欄位

主要生物活性欄位：

- `standard_type` - 活性類型（IC50、Ki、Kd、EC50 等）
- `standard_value` - 數值活性值
- `standard_units` - 單位（nM、uM 等）
- `pchembl_value` - 標準化活性值（-log 刻度）
- `activity_comment` - 活性註釋
- `data_validity_comment` - 資料有效性標誌
- `potential_duplicate` - 重複標誌

## 靶標資訊欄位

靶標資料包含：

- `target_chembl_id` - ChEMBL 靶標識別碼
- `pref_name` - 首選靶標名稱
- `target_type` - 類型（PROTEIN、ORGANISM 等）
- `organism` - 靶標生物體
- `tax_id` - NCBI 分類 ID
- `target_components` - 組件詳細資訊

## 進階查詢範例

### 尋找激酶抑制劑
```python
# 取得激酶靶標
targets = new_client.target.filter(
    target_type='SINGLE PROTEIN',
    pref_name__icontains='kinase'
)

# 取得這些靶標的活性
activities = new_client.activity.filter(
    target_chembl_id__in=[t['target_chembl_id'] for t in targets],
    standard_type='IC50',
    standard_value__lte=100
)
```

### 擷取藥物作用機制
```python
mechanisms = new_client.mechanism.filter(
    molecule_chembl_id='CHEMBL25'
)
```

### 取得化合物生物活性
```python
activities = new_client.activity.filter(
    molecule_chembl_id='CHEMBL25',
    pchembl_value__isnull=False
)
```

## 圖像產生

ChEMBL 可以產生分子結構的 SVG 圖像：

```python
from chembl_webresource_client.new_client import new_client
image = new_client.image
svg = image.get('CHEMBL25')
```

## 分頁

結果會自動分頁。迭代所有結果：

```python
activities = new_client.activity.filter(target_chembl_id='CHEMBL240')
for activity in activities:
    print(activity)
```

## 錯誤處理

常見錯誤：
- **404**：找不到資源
- **503**：服務暫時無法使用
- **Timeout**：請求耗時過長

用戶端根據 `TOTAL_RETRIES` 設定自動重試失敗的請求。

## 速率限制

ChEMBL 有公平使用政策：
- 請尊重查詢頻率
- 使用快取以減少重複請求
- 對大型資料集考慮批量下載

## 其他資源

- 官方 API 文件：https://www.ebi.ac.uk/chembl/api/data/docs
- Python 用戶端 GitHub：https://github.com/chembl/chembl_webresource_client
- ChEMBL 介面文件：https://chembl.gitbook.io/chembl-interface-documentation/
- 範例筆記本：https://github.com/chembl/notebooks
