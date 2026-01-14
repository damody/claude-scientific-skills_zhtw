# GEO 資料庫參考文件

## 完整 E-utilities API 規格

### 概述

NCBI Entrez 程式設計公用程式（E-utilities）透過一組九個伺服器端程式提供對 GEO 元資料的程式化存取。所有 E-utilities 預設以 XML 格式返回結果。

### 基礎 URL

```
https://eutils.ncbi.nlm.nih.gov/entrez/eutils/
```

### 核心 E-utility 程式

#### eSearch - 文字查詢到 ID 列表

**用途：** 搜尋資料庫並返回符合查詢的 UID 列表。

**URL 模式：**
```
https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi
```

**參數：**
- `db`（必要）：要搜尋的資料庫（例如 "gds"、"geoprofiles"）
- `term`（必要）：搜尋查詢字串
- `retmax`：返回的最大 UID 數量（預設：20，最大：10000）
- `retstart`：結果集中的起始位置（用於分頁）
- `usehistory`：設為 "y" 將結果儲存到歷史伺服器
- `sort`：排序順序（例如 "relevance"、"pub_date"）
- `field`：限制搜尋到特定欄位
- `datetype`：限制的日期類型
- `reldate`：限制為今天起 N 天內的項目
- `mindate`、`maxdate`：日期範圍限制（YYYY/MM/DD）

**範例：**
```python
from Bio import Entrez
Entrez.email = "your@email.com"

# 基本搜尋
handle = Entrez.esearch(
    db="gds",
    term="breast cancer AND Homo sapiens",
    retmax=100,
    usehistory="y"
)
results = Entrez.read(handle)
handle.close()

# 結果包含：
# - Count：符合的總數量
# - RetMax：返回的 UID 數量
# - RetStart：起始位置
# - IdList：UID 列表
# - QueryKey：歷史伺服器的鍵值（如果 usehistory="y"）
# - WebEnv：網頁環境字串（如果 usehistory="y"）
```

#### eSummary - 文件摘要

**用途：** 取得 UID 列表的文件摘要。

**URL 模式：**
```
https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi
```

**參數：**
- `db`（必要）：資料庫
- `id`（必要）：逗號分隔的 UID 列表或 query_key+WebEnv
- `retmode`：返回格式（"xml" 或 "json"）
- `version`：摘要版本（建議使用 "2.0"）

**範例：**
```python
from Bio import Entrez
Entrez.email = "your@email.com"

# 取得多個 ID 的摘要
handle = Entrez.esummary(
    db="gds",
    id="200000001,200000002",
    retmode="xml",
    version="2.0"
)
summaries = Entrez.read(handle)
handle.close()

# GEO DataSets 的摘要欄位：
# - Accession：GDS 登錄號
# - title：資料集標題
# - summary：資料集描述
# - PDAT：發布日期
# - n_samples：樣本數量
# - Organism：來源生物
# - PubMedIds：關聯的 PubMed ID
```

#### eFetch - 完整記錄

**用途：** 取得 UID 列表的完整記錄。

**URL 模式：**
```
https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi
```

**參數：**
- `db`（必要）：資料庫
- `id`（必要）：逗號分隔的 UID 列表
- `retmode`：返回格式（"xml"、"text"）
- `rettype`：記錄類型（資料庫特定）

**範例：**
```python
from Bio import Entrez
Entrez.email = "your@email.com"

# 取得完整記錄
handle = Entrez.efetch(
    db="gds",
    id="200000001",
    retmode="xml"
)
records = Entrez.read(handle)
handle.close()
```

#### eLink - 跨資料庫連結

**用途：** 在相同或不同資料庫中查找相關記錄。

**URL 模式：**
```
https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi
```

**參數：**
- `dbfrom`（必要）：來源資料庫
- `db`（必要）：目標資料庫
- `id`（必要）：來源資料庫的 UID
- `cmd`：連結命令類型
  - "neighbor"：返回連結的 UID（預設）
  - "neighbor_score"：返回評分連結
  - "acheck"：檢查連結
  - "ncheck"：計算連結
  - "llinks"：返回 LinkOut 資源的 URL

**範例：**
```python
from Bio import Entrez
Entrez.email = "your@email.com"

# 查找與 GEO 資料集連結的 PubMed 文章
handle = Entrez.elink(
    dbfrom="gds",
    db="pubmed",
    id="200000001"
)
links = Entrez.read(handle)
handle.close()
```

#### ePost - 上傳 UID 列表

**用途：** 將 UID 列表上傳到歷史伺服器以供後續請求使用。

**URL 模式：**
```
https://eutils.ncbi.nlm.nih.gov/entrez/eutils/epost.fcgi
```

**參數：**
- `db`（必要）：資料庫
- `id`（必要）：逗號分隔的 UID 列表

**範例：**
```python
from Bio import Entrez
Entrez.email = "your@email.com"

# 發布大量 ID 列表
large_id_list = [str(i) for i in range(200000001, 200000101)]
handle = Entrez.epost(db="gds", id=",".join(large_id_list))
result = Entrez.read(handle)
handle.close()

# 在後續呼叫中使用返回的 QueryKey 和 WebEnv
query_key = result["QueryKey"]
webenv = result["WebEnv"]
```

#### eInfo - 資料庫資訊

**用途：** 取得可用資料庫及其欄位的資訊。

**URL 模式：**
```
https://eutils.ncbi.nlm.nih.gov/entrez/eutils/einfo.fcgi
```

**參數：**
- `db`：資料庫名稱（省略以取得所有資料庫列表）
- `version`：設為 "2.0" 以獲取詳細欄位資訊

**範例：**
```python
from Bio import Entrez
Entrez.email = "your@email.com"

# 取得 gds 資料庫的資訊
handle = Entrez.einfo(db="gds", version="2.0")
info = Entrez.read(handle)
handle.close()

# 返回：
# - 資料庫描述
# - 最後更新日期
# - 記錄數量
# - 可用的搜尋欄位
# - 連結資訊
```

### GEO 的搜尋欄位限定符

用於建構目標查詢的常見搜尋欄位：

**一般欄位：**
- `[Accession]`：GEO 登錄號
- `[Title]`：資料集標題
- `[Author]`：作者名稱
- `[Organism]`：來源生物
- `[Entry Type]`：條目類型（例如 "Expression profiling by array"）
- `[Platform]`：平台登錄號或名稱
- `[PubMed ID]`：關聯的 PubMed ID

**日期欄位：**
- `[Publication Date]`：發布日期（YYYY 或 YYYY/MM/DD）
- `[Submission Date]`：提交日期
- `[Modification Date]`：最後修改日期

**MeSH 詞彙：**
- `[MeSH Terms]`：醫學主題詞
- `[MeSH Major Topic]`：主要 MeSH 主題

**研究類型欄位：**
- `[DataSet Type]`：研究類型（例如 "RNA-seq"、"ChIP-seq"）
- `[Sample Type]`：樣本類型

**複雜查詢範例：**
```python
query = """
    (breast cancer[MeSH] OR breast neoplasms[Title]) AND
    Homo sapiens[Organism] AND
    expression profiling by array[Entry Type] AND
    2020:2024[Publication Date] AND
    GPL570[Platform]
"""
```

## SOFT 檔案格式規格

### 概述

SOFT（Simple Omnibus Format in Text，簡易綜合文字格式）是 GEO 的主要資料交換格式。檔案以鍵值對和資料表格結構化。

### 檔案類型

**Family SOFT 檔案：**
- 檔名：`GSExxxxx_family.soft.gz`
- 包含：具有所有樣本和平台的完整系列
- 大小：可能非常大（壓縮後數百 MB）
- 用途：完整資料擷取

**Series Matrix 檔案：**
- 檔名：`GSExxxxx_series_matrix.txt.gz`
- 包含：具有最少元資料的表達矩陣
- 大小：比 family 檔案小
- 用途：快速存取表達資料

**Platform SOFT 檔案：**
- 檔名：`GPLxxxxx.soft`
- 包含：平台註解和探針資訊
- 用途：將探針對應到基因

### SOFT 檔案結構

```
^DATABASE = GeoMiame
!Database_name = Gene Expression Omnibus (GEO)
!Database_institute = NCBI NLM NIH
!Database_web_link = http://www.ncbi.nlm.nih.gov/geo
!Database_email = geo@ncbi.nlm.nih.gov

^SERIES = GSExxxxx
!Series_title = Study Title Here
!Series_summary = Study description and background...
!Series_overall_design = Experimental design...
!Series_type = Expression profiling by array
!Series_pubmed_id = 12345678
!Series_submission_date = Jan 01 2024
!Series_last_update_date = Jan 15 2024
!Series_contributor = John,Doe
!Series_contributor = Jane,Smith
!Series_sample_id = GSMxxxxxx
!Series_sample_id = GSMxxxxxx

^PLATFORM = GPLxxxxx
!Platform_title = Platform Name
!Platform_distribution = commercial or custom
!Platform_organism = Homo sapiens
!Platform_manufacturer = Affymetrix
!Platform_technology = in situ oligonucleotide
!Platform_data_row_count = 54675
#ID = Probe ID
#GB_ACC = GenBank accession
#SPOT_ID = Spot identifier
#Gene Symbol = Gene symbol
#Gene Title = Gene title
!platform_table_begin
ID    GB_ACC    SPOT_ID    Gene Symbol    Gene Title
1007_s_at    U48705    -    DDR1    discoidin domain receptor...
1053_at    M87338    -    RFC2    replication factor C...
!platform_table_end

^SAMPLE = GSMxxxxxx
!Sample_title = Sample name
!Sample_source_name_ch1 = cell line XYZ
!Sample_organism_ch1 = Homo sapiens
!Sample_characteristics_ch1 = cell type: epithelial
!Sample_characteristics_ch1 = treatment: control
!Sample_molecule_ch1 = total RNA
!Sample_label_ch1 = biotin
!Sample_platform_id = GPLxxxxx
!Sample_data_processing = normalization method
#ID_REF = Probe identifier
#VALUE = Expression value
!sample_table_begin
ID_REF    VALUE
1007_s_at    8.456
1053_at    7.234
!sample_table_end
```

### 解析 SOFT 檔案

**使用 GEOparse：**
```python
import GEOparse

# 解析系列
gse = GEOparse.get_GEO(filepath="GSE123456_family.soft.gz")

# 存取元資料
metadata = gse.metadata
phenotype_data = gse.phenotype_data

# 存取樣本
for gsm_name, gsm in gse.gsms.items():
    sample_data = gsm.table
    sample_metadata = gsm.metadata

# 存取平台
for gpl_name, gpl in gse.gpls.items():
    platform_table = gpl.table
    platform_metadata = gpl.metadata
```

**手動解析：**
```python
import gzip

def parse_soft_file(filename):
    """基本 SOFT 檔案解析器"""
    sections = {}
    current_section = None
    current_metadata = {}
    current_table = []
    in_table = False

    with gzip.open(filename, 'rt') as f:
        for line in f:
            line = line.strip()

            # 新區段
            if line.startswith('^'):
                if current_section:
                    sections[current_section] = {
                        'metadata': current_metadata,
                        'table': current_table
                    }
                parts = line[1:].split(' = ')
                current_section = parts[1] if len(parts) > 1 else parts[0]
                current_metadata = {}
                current_table = []
                in_table = False

            # 元資料
            elif line.startswith('!'):
                if in_table:
                    in_table = False
                key_value = line[1:].split(' = ', 1)
                if len(key_value) == 2:
                    key, value = key_value
                    if key in current_metadata:
                        if isinstance(current_metadata[key], list):
                            current_metadata[key].append(value)
                        else:
                            current_metadata[key] = [current_metadata[key], value]
                    else:
                        current_metadata[key] = value

            # 表格資料
            elif line.startswith('#') or in_table:
                in_table = True
                current_table.append(line)

    return sections
```

## MINiML 檔案格式

### 概述

MINiML（MIAME Notation in Markup Language，MIAME 標記語言表示法）是 GEO 基於 XML 的資料交換格式。

### 檔案結構

```xml
<?xml version="1.0" encoding="UTF-8"?>
<MINiML xmlns="http://www.ncbi.nlm.nih.gov/geo/info/MINiML"
        xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <Series iid="GDS123">
    <Status>
      <Submission-Date>2024-01-01</Submission-Date>
      <Release-Date>2024-01-15</Release-Date>
      <Last-Update-Date>2024-01-15</Last-Update-Date>
    </Status>
    <Title>Study Title</Title>
    <Summary>Study description...</Summary>
    <Overall-Design>Experimental design...</Overall-Design>
    <Type>Expression profiling by array</Type>
    <Contributor>
      <Person>
        <First>John</First>
        <Last>Doe</Last>
      </Person>
    </Contributor>
  </Series>

  <Platform iid="GPL123">
    <Title>Platform Name</Title>
    <Distribution>commercial</Distribution>
    <Technology>in situ oligonucleotide</Technology>
    <Organism taxid="9606">Homo sapiens</Organism>
    <Data-Table>
      <Column position="1">
        <Name>ID</Name>
        <Description>Probe identifier</Description>
      </Column>
      <Data>
        <Row>
          <Cell column="1">1007_s_at</Cell>
          <Cell column="2">U48705</Cell>
        </Row>
      </Data>
    </Data-Table>
  </Platform>

  <Sample iid="GSM123">
    <Title>Sample name</Title>
    <Source>cell line XYZ</Source>
    <Organism taxid="9606">Homo sapiens</Organism>
    <Characteristics tag="cell type">epithelial</Characteristics>
    <Characteristics tag="treatment">control</Characteristics>
    <Platform-Ref ref="GPL123"/>
    <Data-Table>
      <Column position="1">
        <Name>ID_REF</Name>
      </Column>
      <Column position="2">
        <Name>VALUE</Name>
      </Column>
      <Data>
        <Row>
          <Cell column="1">1007_s_at</Cell>
          <Cell column="2">8.456</Cell>
        </Row>
      </Data>
    </Data-Table>
  </Sample>
</MINiML>
```

## FTP 目錄結構

### Series 檔案

**模式：**
```
ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE{nnn}nnn/GSE{xxxxx}/
```

其中 `{nnn}` 代表將最後 3 位數字替換為 "nnn"，`{xxxxx}` 是完整登錄號。

**範例：**
- GSE123456 → `/geo/series/GSE123nnn/GSE123456/`
- GSE1234 → `/geo/series/GSE1nnn/GSE1234/`
- GSE100001 → `/geo/series/GSE100nnn/GSE100001/`

**子目錄：**
- `/matrix/` - Series matrix 檔案
- `/soft/` - Family SOFT 檔案
- `/miniml/` - MINiML XML 檔案
- `/suppl/` - 補充檔案

**檔案類型：**
```
matrix/
  └── GSE123456_series_matrix.txt.gz

soft/
  └── GSE123456_family.soft.gz

miniml/
  └── GSE123456_family.xml.tgz

suppl/
  ├── GSE123456_RAW.tar
  ├── filelist.txt
  └── [各種補充檔案]
```

### Sample 檔案

**模式：**
```
ftp://ftp.ncbi.nlm.nih.gov/geo/samples/GSM{nnn}nnn/GSM{xxxxx}/
```

**子目錄：**
- `/suppl/` - 樣本特定的補充檔案

### Platform 檔案

**模式：**
```
ftp://ftp.ncbi.nlm.nih.gov/geo/platforms/GPL{nnn}nnn/GPL{xxxxx}/
```

**檔案類型：**
```
soft/
  └── GPL570.soft.gz

miniml/
  └── GPL570.xml

annot/
  └── GPL570.annot.gz  # 增強註解（如果可用）
```

## 進階 GEOparse 使用

### 自訂解析選項

```python
import GEOparse

# 使用自訂選項解析
gse = GEOparse.get_GEO(
    geo="GSE123456",
    destdir="./data",
    silent=False,  # 顯示進度
    how="full",  # 解析模式："full"、"quick"、"brief"
    annotate_gpl=True,  # 包含平台註解
    geotype="GSE"  # 明確類型
)

# 存取特定樣本
gsm = gse.gsms['GSM1234567']

# 取得特定探針的表達值
probe_id = "1007_s_at"
if hasattr(gsm, 'table'):
    probe_data = gsm.table[gsm.table['ID_REF'] == probe_id]

# 取得所有特徵
characteristics = {}
for key, values in gsm.metadata.items():
    if key.startswith('characteristics'):
        for value in (values if isinstance(values, list) else [values]):
            if ':' in value:
                char_key, char_value = value.split(':', 1)
                characteristics[char_key.strip()] = char_value.strip()
```

### 處理平台註解

```python
import GEOparse
import pandas as pd

gse = GEOparse.get_GEO(geo="GSE123456", destdir="./data")

# 取得平台
gpl = list(gse.gpls.values())[0]

# 擷取註解表格
if hasattr(gpl, 'table'):
    annotation = gpl.table

    # 常見註解欄位：
    # - ID：探針識別碼
    # - Gene Symbol：基因符號
    # - Gene Title：基因描述
    # - GB_ACC：GenBank 登錄號
    # - Gene ID：Entrez Gene ID
    # - RefSeq：RefSeq 登錄號
    # - UniGene：UniGene 叢集

    # 將探針對應到基因
    probe_to_gene = dict(zip(
        annotation['ID'],
        annotation['Gene Symbol']
    ))

    # 處理每個基因多個探針的情況
    gene_to_probes = {}
    for probe, gene in probe_to_gene.items():
        if gene and gene != '---':
            if gene not in gene_to_probes:
                gene_to_probes[gene] = []
            gene_to_probes[gene].append(probe)
```

### 處理大型資料集

```python
import GEOparse
import pandas as pd
import numpy as np

def process_large_gse(gse_id, chunk_size=1000):
    """分塊處理大型 GEO 系列"""
    gse = GEOparse.get_GEO(geo=gse_id, destdir="./data")

    # 取得樣本列表
    sample_list = list(gse.gsms.keys())

    # 分塊處理
    for i in range(0, len(sample_list), chunk_size):
        chunk_samples = sample_list[i:i+chunk_size]

        # 擷取分塊資料
        chunk_data = {}
        for gsm_id in chunk_samples:
            gsm = gse.gsms[gsm_id]
            if hasattr(gsm, 'table'):
                chunk_data[gsm_id] = gsm.table['VALUE']

        # 處理分塊
        chunk_df = pd.DataFrame(chunk_data)

        # 儲存分塊結果
        chunk_df.to_csv(f"chunk_{i//chunk_size}.csv")

        print(f"已處理 {i+len(chunk_samples)}/{len(sample_list)} 個樣本")
```

## 常見問題疑難排解

### 問題：GEOparse 下載失敗

**症狀：** 逾時錯誤、連線失敗

**解決方案：**
1. 檢查網路連線
2. 嘗試先透過 FTP 直接下載
3. 解析本地檔案：
```python
gse = GEOparse.get_GEO(filepath="./local/GSE123456_family.soft.gz")
```
4. 增加逾時（如需要可修改 GEOparse 原始碼）

### 問題：缺少表達資料

**症狀：** `pivot_samples()` 失敗或返回空值

**原因：** 並非所有系列都有 series matrix 檔案（較舊的提交）

**解決方案：** 解析個別樣本表格：
```python
expression_data = {}
for gsm_name, gsm in gse.gsms.items():
    if hasattr(gsm, 'table') and 'VALUE' in gsm.table.columns:
        expression_data[gsm_name] = gsm.table.set_index('ID_REF')['VALUE']

expression_df = pd.DataFrame(expression_data)
```

### 問題：探針 ID 不一致

**症狀：** 樣本間探針 ID 不匹配

**原因：** 不同平台版本或樣本處理方式

**解決方案：** 使用平台註解標準化：
```python
# 取得共同探針集
all_probes = set()
for gsm in gse.gsms.values():
    if hasattr(gsm, 'table'):
        all_probes.update(gsm.table['ID_REF'].values)

# 建立標準化矩陣
standardized_data = {}
for gsm_name, gsm in gse.gsms.items():
    if hasattr(gsm, 'table'):
        sample_data = gsm.table.set_index('ID_REF')['VALUE']
        standardized_data[gsm_name] = sample_data.reindex(all_probes)

expression_df = pd.DataFrame(standardized_data)
```

### 問題：E-utilities 速率限制

**症狀：** HTTP 429 錯誤、回應緩慢

**解決方案：**
1. 從 NCBI 取得 API 金鑰
2. 實作速率限制：
```python
import time
from functools import wraps

def rate_limit(calls_per_second=3):
    min_interval = 1.0 / calls_per_second

    def decorator(func):
        last_called = [0.0]

        @wraps(func)
        def wrapper(*args, **kwargs):
            elapsed = time.time() - last_called[0]
            wait_time = min_interval - elapsed
            if wait_time > 0:
                time.sleep(wait_time)
            result = func(*args, **kwargs)
            last_called[0] = time.time()
            return result
        return wrapper
    return decorator

@rate_limit(calls_per_second=3)
def safe_esearch(query):
    handle = Entrez.esearch(db="gds", term=query)
    results = Entrez.read(handle)
    handle.close()
    return results
```

### 問題：大型資料集記憶體錯誤

**症狀：** MemoryError、系統變慢

**解決方案：**
1. 分塊處理資料
2. 對表達資料使用稀疏矩陣
3. 僅載入必要欄位
4. 使用記憶體效率較高的資料類型：
```python
import pandas as pd

# 使用特定 dtype 讀取
expression_df = pd.read_csv(
    "expression_matrix.csv",
    dtype={'ID': str, 'GSM1': np.float32}  # 使用 float32 而非 float64
)

# 或對大部分為零的資料使用稀疏格式
import scipy.sparse as sp
sparse_matrix = sp.csr_matrix(expression_df.values)
```

## 平台特定考量

### Affymetrix 晶片

- 探針 ID 格式：`1007_s_at`、`1053_at`
- 每個基因多個探針集很常見
- 檢查 `_at`、`_s_at`、`_x_at` 後綴
- 可能需要 RMA 或 MAS5 正規化

### Illumina 晶片

- 探針 ID 格式：`ILMN_1234567`
- 注意重複探針
- 可能需要 BeadChip 特定處理

### RNA-seq

- 可能沒有傳統的「探針」
- 檢查基因 ID（Ensembl、Entrez）
- 計數值 vs. FPKM/TPM 值
- 可能需要單獨的計數檔案

### 雙通道晶片

- 在元資料中尋找 `_ch1` 和 `_ch2` 後綴
- VALUE_ch1、VALUE_ch2 欄位
- 可能需要比值或強度值
- 檢查染料交換實驗

## 最佳實踐摘要

1. **在使用 E-utilities 之前務必設定 Entrez.email**
2. **使用 API 金鑰**以獲得更好的速率限制
3. **在本地快取下載的檔案**
4. **在分析前檢查資料品質**
5. **驗證平台註解**是最新的
6. **記錄資料處理**步驟
7. **引用原始研究**當使用資料時
8. **檢查批次效應**在整合分析中
9. **使用獨立資料集驗證結果**
10. **遵循 NCBI 使用指南**
