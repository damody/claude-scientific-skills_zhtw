# UniProt 查詢語法參考

建構複雜搜尋的 UniProt 搜尋查詢語法完整指南。

## 基本語法

### 簡單查詢
```
insulin
kinase
```

### 欄位特定搜尋
```
gene:BRCA1
accession:P12345
organism_name:human
protein_name:kinase
```

## 布林運算子

### AND（兩個術語都必須存在）
```
insulin AND diabetes
kinase AND human
gene:BRCA1 AND reviewed:true
```

### OR（任一術語可以存在）
```
diabetes OR insulin
(cancer OR tumor) AND human
```

### NOT（排除術語）
```
kinase NOT human
protein_name:kinase NOT organism_name:mouse
```

### 使用括號分組
```
(diabetes OR insulin) AND reviewed:true
(gene:BRCA1 OR gene:BRCA2) AND organism_id:9606
```

## 常見搜尋欄位

### 識別資訊
- `accession:P12345` - UniProt 登錄號
- `id:INSR_HUMAN` - 條目名稱
- `gene:BRCA1` - 基因名稱
- `gene_exact:BRCA1` - 精確基因名稱比對

### 生物體/分類學
- `organism_name:human` - 生物體名稱
- `organism_name:"Homo sapiens"` - 精確生物體名稱（多字詞使用引號）
- `organism_id:9606` - NCBI 分類學 ID
- `taxonomy_id:9606` - 與 organism_id 相同
- `taxonomy_name:"Homo sapiens"` - 分類學名稱

### 蛋白質資訊
- `protein_name:insulin` - 蛋白質名稱
- `protein_name:"insulin receptor"` - 精確蛋白質名稱
- `reviewed:true` - 僅 Swiss-Prot（已審核）條目
- `reviewed:false` - 僅 TrEMBL（未審核）條目

### 序列屬性
- `length:[100 TO 500]` - 序列長度範圍
- `mass:[50000 TO 100000]` - 分子量（道爾頓）
- `sequence:MVLSPADKTNVK` - 精確序列比對
- `fragment:false` - 排除片段序列

### 基因本體論（GO）
- `go:0005515` - GO 術語 ID（0005515 = 蛋白質結合）
- `go_f:* ` - 任何分子功能
- `go_p:*` - 任何生物過程
- `go_c:*` - 任何細胞組件

### 註解
- `annotation:(type:signal)` - 具有訊號肽註解
- `annotation:(type:transmem)` - 具有跨膜區域
- `cc_function:*` - 具有功能註解
- `cc_interaction:*` - 具有交互作用註解
- `ft_domain:*` - 具有結構域特徵

### 資料庫交叉參考
- `xref:pdb` - 具有 PDB 結構
- `xref:ensembl` - 具有 Ensembl 參考
- `database:pdb` - 與 xref 相同
- `database:(type:pdb)` - 替代語法

### 蛋白質家族和結構域
- `family:"protein kinase"` - 蛋白質家族
- `keyword:"Protein kinase"` - 關鍵字註解
- `cc_similarity:*` - 具有相似性註解

## 範圍查詢

### 數值範圍
```
length:[100 TO 500]          # 介於 100 和 500 之間
mass:[* TO 50000]            # 小於或等於 50000
created:[2023-01-01 TO *]   # 2023 年 1 月 1 日之後建立
```

### 日期範圍
```
created:[2023-01-01 TO 2023-12-31]
modified:[2024-01-01 TO *]
```

## 萬用字元

### 單一字元（?）
```
gene:BRCA?      # 比對 BRCA1、BRCA2 等
```

### 多個字元（*）
```
gene:BRCA*      # 比對 BRCA1、BRCA2、BRCA1P1 等
protein_name:kinase*
organism_name:Homo*
```

## 進階搜尋

### 存在性查詢
```
cc_function:*              # 具有任何功能註解
ft_domain:*                # 具有任何結構域特徵
xref:pdb                   # 具有 PDB 結構
```

### 組合複雜查詢
```
# 具有 PDB 結構的人類已審核激酶
(protein_name:kinase OR family:kinase) AND organism_id:9606 AND reviewed:true AND xref:pdb

# 排除小鼠的癌症相關蛋白質
(disease:cancer OR keyword:cancer) NOT organism_name:mouse

# 具有訊號肽的膜蛋白
annotation:(type:transmem) AND annotation:(type:signal) AND reviewed:true

# 最近更新的人類蛋白質
organism_id:9606 AND modified:[2024-01-01 TO *] AND reviewed:true
```

## 欄位特定範例

### 蛋白質名稱
```
protein_name:"insulin receptor"    # 精確詞組
protein_name:insulin*              # 以 insulin 開頭
recommended_name:insulin           # 僅推薦名稱
alternative_name:insulin           # 僅替代名稱
```

### 基因
```
gene:BRCA1                        # 基因符號
gene_exact:BRCA1                  # 精確基因比對
olnName:BRCA1                     # 有序位點名稱
orfName:BRCA1                     # ORF 名稱
```

### 生物體
```
organism_name:human               # 通用名稱
organism_name:"Homo sapiens"      # 學名
organism_id:9606                  # 分類學 ID
lineage:primates                  # 分類學譜系
```

### 特徵
```
ft_signal:*                       # 訊號肽
ft_transmem:*                     # 跨膜區域
ft_domain:"Protein kinase"        # 特定結構域
ft_binding:*                      # 結合位點
ft_site:*                         # 任何位點
```

### 註解（cc_）
```
cc_function:*                     # 功能描述
cc_catalytic_activity:*           # 催化活性
cc_pathway:*                      # 路徑參與
cc_interaction:*                  # 蛋白質交互作用
cc_subcellular_location:*         # 亞細胞定位
cc_tissue_specificity:*           # 組織特異性
cc_disease:cancer                 # 疾病關聯
```

## 技巧與最佳實務

1. **精確詞組使用引號**：`organism_name:"Homo sapiens"` 而非 `organism_name:Homo sapiens`

2. **依審核狀態篩選**：加入 `AND reviewed:true` 以取得高品質 Swiss-Prot 條目

3. **謹慎組合萬用字元**：`*kinase*` 可能過於廣泛；`kinase*` 更為特定

4. **複雜邏輯使用括號**：`(A OR B) AND (C OR D)` 比 `A OR B AND C OR D` 更清晰

5. **數值範圍是包含的**：`length:[100 TO 500]` 包含 100 和 500

6. **欄位前綴**：學習常見前綴：
   - `cc_` = 註解
   - `ft_` = 特徵
   - `go_` = 基因本體論
   - `xref_` = 交叉參考

7. **檢查欄位名稱**：使用 API 的 `/configure/uniprotkb/result-fields` 端點查看所有可用欄位

## 查詢驗證

使用以下方式測試查詢：
- **網頁介面**：https://www.uniprot.org/uniprotkb
- **API**：https://rest.uniprot.org/uniprotkb/search?query=YOUR_QUERY
- **API 文件**：https://www.uniprot.org/help/query-fields

## 常見模式

### 尋找特徵完備的蛋白質
```
reviewed:true AND xref:pdb AND cc_function:*
```

### 尋找疾病相關蛋白質
```
cc_disease:* AND organism_id:9606 AND reviewed:true
```

### 尋找具有實驗證據的蛋白質
```
existence:"Evidence at protein level" AND reviewed:true
```

### 尋找分泌蛋白質
```
cc_subcellular_location:secreted AND reviewed:true
```

### 尋找藥物靶點
```
keyword:"Pharmaceutical" OR keyword:"Drug target"
```

## 資源

- 完整查詢欄位參考：https://www.uniprot.org/help/query-fields
- API 查詢文件：https://www.uniprot.org/help/api_queries
- 文字搜尋文件：https://www.uniprot.org/help/text-search
