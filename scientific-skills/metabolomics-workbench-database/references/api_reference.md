# Metabolomics Workbench REST API 參考

## 基礎 URL

所有 API 請求使用以下基礎 URL：
```
https://www.metabolomicsworkbench.org/rest/
```

## API 結構

REST API 遵循一致的 URL 模式：
```
/context/input_item/input_value/output_item/output_format
```

- **context**：要存取的資源類型（study、compound、refmet、metstat、gene、protein、moverz）
- **input_item**：識別碼或搜尋參數的類型
- **input_value**：要搜尋的特定值
- **output_item**：要返回的資料（例如：all、name、summary）
- **output_format**：json 或 txt（省略時預設為 json）

## 輸出格式

- **json**：機器可讀的 JSON 格式（預設）
- **txt**：制表符分隔的文字格式，便於人類閱讀

## 情境 1：Compound（化合物）

檢索代謝物結構和識別資料。

### 輸入項目

| 輸入項目 | 描述 | 範例 |
|------------|-------------|---------|
| `regno` | Metabolomics Workbench 註冊編號 | 11 |
| `pubchem_cid` | PubChem 化合物 ID | 5281365 |
| `inchi_key` | 國際化學識別碼密鑰 | WQZGKKKJIJFFOK-GASJEMHNSA-N |
| `formula` | 分子式 | C6H12O6 |
| `lm_id` | LIPID MAPS ID | LM... |
| `hmdb_id` | 人類代謝體資料庫 ID | HMDB0000122 |
| `kegg_id` | KEGG 化合物 ID | C00031 |

### 輸出項目

| 輸出項目 | 描述 |
|-------------|-------------|
| `all` | 所有可用的化合物資料 |
| `classification` | 化合物分類 |
| `regno` | 註冊編號 |
| `formula` | 分子式 |
| `exactmass` | 精確質量 |
| `inchi_key` | InChI Key |
| `name` | 常用名稱 |
| `sys_name` | 系統名稱 |
| `smiles` | SMILES 表示法 |
| `lm_id` | LIPID MAPS ID |
| `pubchem_cid` | PubChem CID |
| `hmdb_id` | HMDB ID |
| `kegg_id` | KEGG ID |
| `chebi_id` | ChEBI ID |
| `metacyc_id` | MetaCyc ID |
| `molfile` | MOL 檔案結構 |
| `png` | 結構的 PNG 圖片 |

### 範例請求

```bash
# 透過 PubChem CID 取得所有化合物資料
curl "https://www.metabolomicsworkbench.org/rest/compound/pubchem_cid/5281365/all/json"

# 透過註冊編號取得化合物名稱
curl "https://www.metabolomicsworkbench.org/rest/compound/regno/11/name/json"

# 下載 PNG 格式的結構
curl "https://www.metabolomicsworkbench.org/rest/compound/regno/11/png" -o structure.png

# 透過 KEGG ID 取得化合物
curl "https://www.metabolomicsworkbench.org/rest/compound/kegg_id/C00031/all/json"

# 透過分子式取得化合物
curl "https://www.metabolomicsworkbench.org/rest/compound/formula/C6H12O6/all/json"
```

## 情境 2：Study（研究）

存取代謝體學研究元資料和實驗結果。

### 輸入項目

| 輸入項目 | 描述 | 範例 |
|------------|-------------|---------|
| `study_id` | 研究識別碼 | ST000001 |
| `analysis_id` | 分析識別碼 | AN000001 |
| `study_title` | 研究標題中的關鍵字 | diabetes |
| `institute` | 機構名稱 | UCSD |
| `last_name` | 研究者姓氏 | Smith |
| `metabolite_id` | 代謝物註冊編號 | 11 |
| `refmet_name` | RefMet 標準化名稱 | Glucose |
| `kegg_id` | KEGG 化合物 ID | C00031 |

### 輸出項目

| 輸出項目 | 描述 |
|-------------|-------------|
| `summary` | 研究概述和元資料 |
| `factors` | 實驗因素和設計 |
| `analysis` | 分析方法和參數 |
| `metabolites` | 測量的代謝物列表 |
| `data` | 完整的實驗資料 |
| `mwtab` | mwTab 格式的完整研究 |
| `number_of_metabolites` | 測量的代謝物計數 |
| `species` | 生物物種 |
| `disease` | 研究的疾病 |
| `source` | 樣本來源/組織類型 |
| `untarg_studies` | 非靶向研究資訊 |
| `untarg_factors` | 非靶向研究因素 |
| `untarg_data` | 非靶向實驗資料 |
| `datatable` | 格式化的資料表 |
| `available` | 列出可用的研究（使用 ST 作為 input_value） |

### 範例請求

```bash
# 列出所有公開可用的研究
curl "https://www.metabolomicsworkbench.org/rest/study/study_id/ST/available/json"

# 取得研究摘要
curl "https://www.metabolomicsworkbench.org/rest/study/study_id/ST000001/summary/json"

# 取得實驗資料
curl "https://www.metabolomicsworkbench.org/rest/study/study_id/ST000001/data/json"

# 取得研究因素
curl "https://www.metabolomicsworkbench.org/rest/study/study_id/ST000001/factors/json"

# 尋找包含特定代謝物的研究
curl "https://www.metabolomicsworkbench.org/rest/study/refmet_name/Tyrosine/summary/json"

# 按研究者搜尋研究
curl "https://www.metabolomicsworkbench.org/rest/study/last_name/Smith/summary/json"

# 下載 mwTab 格式的完整研究
curl "https://www.metabolomicsworkbench.org/rest/study/study_id/ST000001/mwtab/txt"
```

## 情境 3：RefMet

查詢具有層級分類的標準化代謝物命名法資料庫。

### 輸入項目

| 輸入項目 | 描述 | 範例 |
|------------|-------------|---------|
| `name` | 代謝物名稱 | glucose |
| `inchi_key` | InChI Key | WQZGKKKJIJFFOK-GASJEMHNSA-N |
| `pubchem_cid` | PubChem CID | 5793 |
| `exactmass` | 精確質量 | 180.0634 |
| `formula` | 分子式 | C6H12O6 |
| `super_class` | 超類名稱 | Organic compounds |
| `main_class` | 主類名稱 | Carbohydrates |
| `sub_class` | 子類名稱 | Monosaccharides |
| `match` | 名稱匹配/標準化 | citrate |
| `refmet_id` | RefMet 識別碼 | 12345 |
| `all` | 檢索所有 RefMet 條目 | （不需要值） |

### 輸出項目

| 輸出項目 | 描述 |
|-------------|-------------|
| `all` | 所有可用的 RefMet 資料 |
| `name` | 標準化的 RefMet 名稱 |
| `inchi_key` | InChI Key |
| `pubchem_cid` | PubChem CID |
| `exactmass` | 精確質量 |
| `formula` | 分子式 |
| `sys_name` | 系統名稱 |
| `super_class` | 超類分類 |
| `main_class` | 主類分類 |
| `sub_class` | 子類分類 |
| `refmet_id` | RefMet 識別碼 |

### 範例請求

```bash
# 標準化代謝物名稱
curl "https://www.metabolomicsworkbench.org/rest/refmet/match/citrate/name/json"

# 取得代謝物的所有 RefMet 資料
curl "https://www.metabolomicsworkbench.org/rest/refmet/name/Glucose/all/json"

# 按分子式查詢
curl "https://www.metabolomicsworkbench.org/rest/refmet/formula/C6H12O6/all/json"

# 取得主類中的所有代謝物
curl "https://www.metabolomicsworkbench.org/rest/refmet/main_class/Fatty%20Acids/all/json"

# 按精確質量查詢
curl "https://www.metabolomicsworkbench.org/rest/refmet/exactmass/180.0634/all/json"

# 下載完整的 RefMet 資料庫
curl "https://www.metabolomicsworkbench.org/rest/refmet/all/json"
```

### RefMet 分類層級

RefMet 提供四級結構解析度：

1. **超類（Super Class）**：最廣泛的分類（例如："Organic compounds"、"Lipids"）
2. **主類（Main Class）**：主要的生化類別（例如："Fatty Acids"、"Carbohydrates"）
3. **子類（Sub Class）**：更具體的分組（例如："Monosaccharides"、"Amino acids"）
4. **個別代謝物（Individual Metabolite）**：具有標準化名稱的特定化合物

## 情境 4：MetStat

使用分號分隔格式按分析和生物參數篩選研究。

### 格式

```
/metstat/ANALYSIS_TYPE;POLARITY;CHROMATOGRAPHY;SPECIES;SAMPLE_SOURCE;DISEASE;KEGG_ID;REFMET_NAME
```

### 參數

| 位置 | 參數 | 選項 |
|----------|-----------|---------|
| 1 | 分析類型 | LCMS、GCMS、NMR、MS、ICPMS |
| 2 | 極性 | POSITIVE、NEGATIVE |
| 3 | 層析法 | HILIC、RP（逆相）、GC、IC |
| 4 | 物種 | Human、Mouse、Rat 等 |
| 5 | 樣本來源 | Blood、Plasma、Serum、Urine、Liver 等 |
| 6 | 疾病 | Diabetes、Cancer、Alzheimer 等 |
| 7 | KEGG ID | C00031 等 |
| 8 | RefMet 名稱 | Glucose、Tyrosine 等 |

**注意**：使用空位置（連續分號）跳過參數。所有參數都是可選的。

### 範例請求

```bash
# 使用 LC-MS HILIC 正離子模式的人類血液糖尿病研究
curl "https://www.metabolomicsworkbench.org/rest/metstat/LCMS;POSITIVE;HILIC;Human;Blood;Diabetes/json"

# 所有包含酪胺酸的人類血液研究
curl "https://www.metabolomicsworkbench.org/rest/metstat/;;;Human;Blood;;;Tyrosine/json"

# 所有 GC-MS 研究，不論其他參數
curl "https://www.metabolomicsworkbench.org/rest/metstat/GCMS;;;;;;/json"

# 小鼠肝臟研究
curl "https://www.metabolomicsworkbench.org/rest/metstat/;;;Mouse;Liver;;/json"

# 所有測量葡萄糖的研究
curl "https://www.metabolomicsworkbench.org/rest/metstat/;;;;;;;Glucose/json"
```

## 情境 5：Moverz

透過 m/z 值執行質譜前驅離子搜尋。

### m/z 搜尋格式

```
/moverz/DATABASE/mass/adduct/tolerance/format
```

- **DATABASE**：MB（Metabolomics Workbench）、LIPIDS、REFMET
- **mass**：m/z 值（例如：635.52）
- **adduct**：離子加合物類型（見下表）
- **tolerance**：質量容許誤差，單位為道爾頓（例如：0.5）
- **format**：json 或 txt

### 精確質量計算格式

```
/moverz/exactmass/metabolite_name/adduct/format
```

### 離子加合物類型

#### 正離子模式加合物

| 加合物 | 描述 | 使用範例 |
|--------|-------------|-------------|
| `M+H` | 質子化分子 | 最常見的正離子 ESI |
| `M+Na` | 鈉加合物 | ESI 中常見 |
| `M+K` | 鉀加合物 | ESI 中較不常見 |
| `M+NH4` | 銨加合物 | 使用銨鹽時常見 |
| `M+2H` | 雙質子化 | 多電荷離子 |
| `M+H-H2O` | 脫水質子化 | 失去水 |
| `M+2Na-H` | 雙鈉減氫 | 多個鈉 |
| `M+CH3OH+H` | 甲醇加合物 | 流動相含甲醇 |
| `M+ACN+H` | 乙腈加合物 | 流動相含 ACN |
| `M+ACN+Na` | ACN + 鈉 | ACN 和鈉 |

#### 負離子模式加合物

| 加合物 | 描述 | 使用範例 |
|--------|-------------|-------------|
| `M-H` | 去質子化分子 | 最常見的負離子 ESI |
| `M+Cl` | 氯化物加合物 | 含氯流動相 |
| `M+FA-H` | 甲酸加合物 | 流動相含甲酸 |
| `M+HAc-H` | 乙酸加合物 | 流動相含乙酸 |
| `M-H-H2O` | 去質子化減水 | 失去水 |
| `M-2H` | 雙去質子化 | 多電荷離子 |
| `M+Na-2H` | 鈉減二質子 | 混合電荷狀態 |

#### 無電荷

| 加合物 | 描述 |
|--------|-------------|
| `M` | 無電荷分子 | 直接游離方法 |

### 範例請求

```bash
# 在 MB 資料庫中搜尋 m/z 635.52 (M+H) 的化合物
curl "https://www.metabolomicsworkbench.org/rest/moverz/MB/635.52/M+H/0.5/json"

# 在 RefMet 中使用負離子模式搜尋
curl "https://www.metabolomicsworkbench.org/rest/moverz/REFMET/200.15/M-H/0.3/json"

# 搜尋脂質資料庫
curl "https://www.metabolomicsworkbench.org/rest/moverz/LIPIDS/760.59/M+Na/0.5/json"

# 計算已知代謝物的精確質量
curl "https://www.metabolomicsworkbench.org/rest/moverz/exactmass/PC(34:1)/M+H/json"

# 高解析度質譜搜尋（緊密容許誤差）
curl "https://www.metabolomicsworkbench.org/rest/moverz/MB/180.0634/M+H/0.01/json"
```

## 情境 6：Gene（基因）

從 Metabolome Gene/Protein（MGP）資料庫存取基因資訊。

### 輸入項目

| 輸入項目 | 描述 | 範例 |
|------------|-------------|---------|
| `mgp_id` | MGP 資料庫 ID | MGP001 |
| `gene_id` | NCBI 基因 ID | 31 |
| `gene_name` | 完整基因名稱 | acetyl-CoA carboxylase |
| `gene_symbol` | 基因符號 | ACACA |
| `taxid` | 分類 ID | 9606（人類） |

### 輸出項目

| 輸出項目 | 描述 |
|-------------|-------------|
| `all` | 所有基因資訊 |
| `mgp_id` | MGP 識別碼 |
| `gene_id` | NCBI 基因 ID |
| `gene_name` | 完整基因名稱 |
| `gene_symbol` | 基因符號 |
| `gene_synonyms` | 替代名稱 |
| `alt_names` | 替代命名法 |
| `chromosome` | 染色體位置 |
| `map_location` | 遺傳圖譜位置 |
| `summary` | 基因描述 |
| `taxid` | 分類 ID |
| `species` | 物種簡稱 |
| `species_long` | 完整物種名稱 |

### 範例請求

```bash
# 透過符號取得基因資訊
curl "https://www.metabolomicsworkbench.org/rest/gene/gene_symbol/ACACA/all/json"

# 透過 NCBI 基因 ID 取得基因
curl "https://www.metabolomicsworkbench.org/rest/gene/gene_id/31/all/json"

# 按基因名稱搜尋
curl "https://www.metabolomicsworkbench.org/rest/gene/gene_name/carboxylase/summary/json"
```

## 情境 7：Protein（蛋白質）

檢索蛋白質序列和註釋資料。

### 輸入項目

| 輸入項目 | 描述 | 範例 |
|------------|-------------|---------|
| `mgp_id` | MGP 資料庫 ID | MGP001 |
| `gene_id` | NCBI 基因 ID | 31 |
| `gene_name` | 基因名稱 | acetyl-CoA carboxylase |
| `gene_symbol` | 基因符號 | ACACA |
| `taxid` | 分類 ID | 9606 |
| `mrna_id` | mRNA 識別碼 | NM_001093.3 |
| `refseq_id` | RefSeq ID | NP_001084 |
| `protein_gi` | GenInfo 識別碼 | 4557237 |
| `uniprot_id` | UniProt ID | Q13085 |
| `protein_entry` | 蛋白質條目名稱 | ACACA_HUMAN |
| `protein_name` | 蛋白質名稱 | Acetyl-CoA carboxylase |

### 輸出項目

| 輸出項目 | 描述 |
|-------------|-------------|
| `all` | 所有蛋白質資訊 |
| `mgp_id` | MGP 識別碼 |
| `gene_id` | NCBI 基因 ID |
| `gene_name` | 基因名稱 |
| `gene_symbol` | 基因符號 |
| `taxid` | 分類 ID |
| `species` | 物種簡稱 |
| `species_long` | 完整物種名稱 |
| `mrna_id` | mRNA 識別碼 |
| `refseq_id` | RefSeq 蛋白質 ID |
| `protein_gi` | GenInfo 識別碼 |
| `uniprot_id` | UniProt 登錄號 |
| `protein_entry` | 蛋白質條目名稱 |
| `protein_name` | 完整蛋白質名稱 |
| `seqlength` | 序列長度 |
| `seq` | 胺基酸序列 |
| `is_identical_to` | 相同序列 |

### 範例請求

```bash
# 透過 UniProt ID 取得蛋白質資訊
curl "https://www.metabolomicsworkbench.org/rest/protein/uniprot_id/Q13085/all/json"

# 透過基因符號取得蛋白質
curl "https://www.metabolomicsworkbench.org/rest/protein/gene_symbol/ACACA/all/json"

# 取得蛋白質序列
curl "https://www.metabolomicsworkbench.org/rest/protein/uniprot_id/Q13085/seq/json"

# 透過 RefSeq ID 搜尋
curl "https://www.metabolomicsworkbench.org/rest/protein/refseq_id/NP_001084/all/json"
```

## 錯誤處理

API 返回適當的 HTTP 狀態碼：

- **200 OK**：請求成功
- **400 Bad Request**：無效的參數或格式錯誤的請求
- **404 Not Found**：找不到資源
- **500 Internal Server Error**：伺服器端錯誤

當找不到結果時，API 通常返回空陣列或物件，而非錯誤碼。

## 速率限制

截至 2025 年，Metabolomics Workbench REST API 對合理使用不強制執行嚴格的速率限制。但是，最佳實踐包括：

- 在批量請求之間實施延遲
- 快取常用的參考資料
- 對大規模查詢使用適當的批次大小

## 額外資源

- **互動式 REST URL 建立器**：https://www.metabolomicsworkbench.org/tools/mw_rest.php
- **官方 API 規格**：https://www.metabolomicsworkbench.org/tools/MWRestAPIv1.1.pdf
- **Python 函式庫**：mwtab 套件供 Python 使用者使用
- **R 套件**：metabolomicsWorkbenchR（Bioconductor）
- **Julia 套件**：MetabolomicsWorkbenchAPI.jl

## Python 範例：完整工作流程

```python
import requests
import json

# 1. 使用 RefMet 標準化代謝物名稱
metabolite = "citrate"
response = requests.get(f'https://www.metabolomicsworkbench.org/rest/refmet/match/{metabolite}/name/json')
standardized_name = response.json()['name']

# 2. 搜尋包含此代謝物的研究
response = requests.get(f'https://www.metabolomicsworkbench.org/rest/study/refmet_name/{standardized_name}/summary/json')
studies = response.json()

# 3. 從特定研究取得詳細資料
study_id = studies[0]['study_id']
response = requests.get(f'https://www.metabolomicsworkbench.org/rest/study/study_id/{study_id}/data/json')
data = response.json()

# 4. 執行 m/z 搜尋進行化合物識別
mz_value = 180.06
response = requests.get(f'https://www.metabolomicsworkbench.org/rest/moverz/MB/{mz_value}/M+H/0.5/json')
matches = response.json()

# 5. 取得化合物結構
regno = matches[0]['regno']
response = requests.get(f'https://www.metabolomicsworkbench.org/rest/compound/regno/{regno}/png')
with open('structure.png', 'wb') as f:
    f.write(response.content)
```
