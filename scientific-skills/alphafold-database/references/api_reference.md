# AlphaFold Database API 參考

本文件提供程式化存取 AlphaFold 蛋白質結構資料庫的完整技術文件。

## 目錄

1. [REST API 端點](#rest-api-端點)
2. [檔案存取模式](#檔案存取模式)
3. [資料架構](#資料架構)
4. [Google Cloud 存取](#google-cloud-存取)
5. [BigQuery 架構](#bigquery-架構)
6. [最佳實務](#最佳實務)
7. [錯誤處理](#錯誤處理)
8. [速率限制](#速率限制)

---

## REST API 端點

### 基礎 URL

```
https://alphafold.ebi.ac.uk/api/
```

### 1. 透過 UniProt 登錄號取得預測

**端點：** `/prediction/{uniprot_id}`

**方法：** GET

**描述：** 擷取指定 UniProt 登錄號的 AlphaFold 預測中繼資料。

**參數：**
- `uniprot_id`（必要）：UniProt 登錄號（例如「P00520」）

**請求範例：**
```bash
curl https://alphafold.ebi.ac.uk/api/prediction/P00520
```

**回應範例：**
```json
[
  {
    "entryId": "AF-P00520-F1",
    "gene": "ABL1",
    "uniprotAccession": "P00520",
    "uniprotId": "ABL1_HUMAN",
    "uniprotDescription": "Tyrosine-protein kinase ABL1",
    "taxId": 9606,
    "organismScientificName": "Homo sapiens",
    "uniprotStart": 1,
    "uniprotEnd": 1130,
    "uniprotSequence": "MLEICLKLVGCKSKKGLSSSSSCYLEEALQRPVASDFEPQGLSEAARWNSKENLLAGPSENDPNLFVALYDFVASGDNTLSITKGEKLRVLGYNHNGEWCEAQTKNGQGWVPSNYITPVNSLEKHSWYHGPVSRNAAEYLLSSGINGSFLVRESESSPGQRSISLRYEGRVYHYRINTASDGKLYVSSESRFNTLAELVHHHSTVADGLITTLHYPAPKRNKPTVYGVSPNYDKWEMERTDITMKHKLGGGQYGEVYEGVWKKYSLTVAVKTLKEDTMEVEEFLKEAAVMKEIKHPNLVQLLGVCTREPPFYIITEFMTYGNLLDYLRECNRQEVNAVVLLYMATQISSAMEYLEKKNFIHRDLAARNCLVGENHLVKVADFGLSRLMTGDTYTAHAGAKFPIKWTAPESLAYNKFSIKSDVWAFGVLLWEIATYGMSPYPGIDLSQVYELLEKDYRMERPEGCPEKVYELMRACWQWNPSDRPSFAEIHQAFETMFQESSISDEVEKELGKQGVRGAVSTLLQAPELPTKTRTSRRAAEHRDTTDVPEMPHSKGQGESDPLDHEPAVSPLLPRKERGPPEGGLNEDERLLPKDKKTNLFSALIKKKKKTAPTPPKRSSSFREMDGQPERRGAGEEEGRDISNGALAFTPLDTADPAKSPKPSNGAGVPNGALRESGGSGFRSPHLWKKSSTLTSSRLATGEEEGGGSSSKRFLRSCSASCVPHGAKDTEWRSVTLPRDLQSTGRQFDSSTFGGHKSEKPALPRKRAGENRSDQVTRGTVTPPPRLVKKNEEAADEVFKDIMESSPGSSPPNLTPKPLRRQVTVAPASGLPHKEEAGKGSALGTPAAAEPVTPTSKAGSGAPGGTSKGPAEESRVRRHKHSSESPGRDKGKLSRLKPAPPPPPAASAGKAGGKPSQSPSQEAAGEAVLGAKTKATSLVDAVNSDAAKPSQPGEGLKKPVLPATPKPQSAKPSGTPISPAPVPSTLPSASSALAGDQPSSTAFIPLISTRVSLRKTRQPPERIASGAITKGVVLDSTEALCLAISRNSEQMASHSAVLEAGKNLYTFCVSYVDSIQQMRNKFAFREAINKLENNLRELQICPATAGSGPAATQDFSKLLSSVKEISDIVQR",
    "modelCreatedDate": "2021-07-01",
    "latestVersion": 4,
    "allVersions": [1, 2, 3, 4],
    "cifUrl": "https://alphafold.ebi.ac.uk/files/AF-P00520-F1-model_v4.cif",
    "bcifUrl": "https://alphafold.ebi.ac.uk/files/AF-P00520-F1-model_v4.bcif",
    "pdbUrl": "https://alphafold.ebi.ac.uk/files/AF-P00520-F1-model_v4.pdb",
    "paeImageUrl": "https://alphafold.ebi.ac.uk/files/AF-P00520-F1-predicted_aligned_error_v4.png",
    "paeDocUrl": "https://alphafold.ebi.ac.uk/files/AF-P00520-F1-predicted_aligned_error_v4.json"
  }
]
```

**回應欄位：**
- `entryId`：AlphaFold 內部識別碼（格式：AF-{uniprot}-F{fragment}）
- `gene`：基因符號
- `uniprotAccession`：UniProt 登錄號
- `uniprotId`：UniProt 條目名稱
- `uniprotDescription`：蛋白質描述
- `taxId`：NCBI 分類識別碼
- `organismScientificName`：物種學名
- `uniprotStart/uniprotEnd`：涵蓋的殘基範圍
- `uniprotSequence`：完整蛋白質序列
- `modelCreatedDate`：初始預測日期
- `latestVersion`：當前模型版本號
- `allVersions`：可用版本列表
- `cifUrl/bcifUrl/pdbUrl`：結構檔案下載 URL
- `paeImageUrl`：PAE 視覺化圖片 URL
- `paeDocUrl`：PAE 資料 JSON URL

### 2. 3D-Beacons 整合

AlphaFold 已整合至 3D-Beacons 網路，用於聯合結構存取。

**端點：** `https://www.ebi.ac.uk/pdbe/pdbe-kb/3dbeacons/api/uniprot/summary/{uniprot_id}.json`

**範例：**
```python
import requests

uniprot_id = "P00520"
url = f"https://www.ebi.ac.uk/pdbe/pdbe-kb/3dbeacons/api/uniprot/summary/{uniprot_id}.json"
response = requests.get(url)
data = response.json()

# 篩選 AlphaFold 結構
alphafold_structures = [
    s for s in data['structures']
    if s['provider'] == 'AlphaFold DB'
]
```

---

## 檔案存取模式

### 直接檔案下載

所有 AlphaFold 檔案都可透過直接 URL 存取，無需驗證。

**URL 模式：**
```
https://alphafold.ebi.ac.uk/files/{alphafold_id}-{file_type}_{version}.{extension}
```

**組成部分：**
- `{alphafold_id}`：條目識別碼（例如「AF-P00520-F1」）
- `{file_type}`：檔案類型（見下方）
- `{version}`：資料庫版本（例如「v4」）
- `{extension}`：檔案格式副檔名

### 可用檔案類型

#### 1. 模型座標

**mmCIF 格式（推薦）：**
```
https://alphafold.ebi.ac.uk/files/AF-P00520-F1-model_v4.cif
```
- 標準晶體學格式
- 包含完整中繼資料
- 支援大型結構
- 檔案大小：可變（通常 100KB - 10MB）

**二進位 CIF 格式：**
```
https://alphafold.ebi.ac.uk/files/AF-P00520-F1-model_v4.bcif
```
- mmCIF 的壓縮二進位版本
- 較小的檔案大小（約減少 70%）
- 更快的解析速度
- 需要專門的解析器

**PDB 格式（舊版）：**
```
https://alphafold.ebi.ac.uk/files/AF-P00520-F1-model_v4.pdb
```
- 傳統 PDB 文字格式
- 限制為 99,999 個原子
- 舊版工具廣泛支援
- 檔案大小：與 mmCIF 類似

#### 2. 信心指標

**每殘基信心（JSON）：**
```
https://alphafold.ebi.ac.uk/files/AF-P00520-F1-confidence_v4.json
```

**結構：**
```json
{
  "confidenceScore": [87.5, 91.2, 93.8, ...],
  "confidenceCategory": ["high", "very_high", "very_high", ...]
}
```

**欄位：**
- `confidenceScore`：每個殘基的 pLDDT 值陣列（0-100）
- `confidenceCategory`：分類分類（very_low、low、high、very_high）

#### 3. 預測對齊誤差（JSON）

```
https://alphafold.ebi.ac.uk/files/AF-P00520-F1-predicted_aligned_error_v4.json
```

**結構：**
```json
{
  "distance": [[0, 2.3, 4.5, ...], [2.3, 0, 3.1, ...], ...],
  "max_predicted_aligned_error": 31.75
}
```

**欄位：**
- `distance`：N×N PAE 值矩陣（埃）
- `max_predicted_aligned_error`：矩陣中的最大 PAE 值

#### 4. PAE 視覺化（PNG）

```
https://alphafold.ebi.ac.uk/files/AF-P00520-F1-predicted_aligned_error_v4.png
```
- 預先渲染的 PAE 熱圖
- 用於快速視覺評估
- 解析度：根據蛋白質大小變化

### 批量下載策略

高效下載多個檔案時，使用具有適當錯誤處理和速率限制的並行下載，以尊重伺服器資源。

---

## 資料架構

### 座標檔案（mmCIF）架構

AlphaFold mmCIF 檔案包含：

**關鍵資料類別：**
- `_entry`：條目層級中繼資料
- `_struct`：結構標題和描述
- `_entity`：分子實體資訊
- `_atom_site`：原子座標和屬性
- `_pdbx_struct_assembly`：生物組裝資訊

**`_atom_site` 中的重要欄位：**
- `group_PDB`：所有記錄均為「ATOM」
- `id`：原子序號
- `label_atom_id`：原子名稱（例如「CA」、「N」、「C」）
- `label_comp_id`：殘基名稱（例如「ALA」、「GLY」）
- `label_seq_id`：殘基序列號
- `Cartn_x/y/z`：笛卡爾座標（埃）
- `B_iso_or_equiv`：B 因子（包含 pLDDT 分數）

**B 因子欄位中的 pLDDT：**
AlphaFold 將每殘基信心（pLDDT）儲存在 B 因子欄位中。這使標準結構檢視器可以自動按信心著色。

### 信心 JSON 架構

```json
{
  "confidenceScore": [
    87.5,   // 殘基 1 pLDDT
    91.2,   // 殘基 2 pLDDT
    93.8    // 殘基 3 pLDDT
    // ... 每個殘基一個值
  ],
  "confidenceCategory": [
    "high",      // 殘基 1 類別
    "very_high", // 殘基 2 類別
    "very_high"  // 殘基 3 類別
    // ... 每個殘基一個類別
  ]
}
```

**信心類別：**
- `very_high`：pLDDT > 90
- `high`：70 < pLDDT ≤ 90
- `low`：50 < pLDDT ≤ 70
- `very_low`：pLDDT ≤ 50

### PAE JSON 架構

```json
{
  "distance": [
    [0.0, 2.3, 4.5, ...],     // 從殘基 1 到所有殘基的 PAE
    [2.3, 0.0, 3.1, ...],     // 從殘基 2 到所有殘基的 PAE
    [4.5, 3.1, 0.0, ...]      // 從殘基 3 到所有殘基的 PAE
    // ... N 個殘基的 N×N 矩陣
  ],
  "max_predicted_aligned_error": 31.75
}
```

**詮釋：**
- `distance[i][j]`：如果預測結構和真實結構在殘基 i 上對齊，殘基 j 的預期位置誤差（埃）
- 較低的值表示相對定位更有信心
- 對角線始終為 0（殘基與自身對齊）
- 矩陣不對稱：distance[i][j] ≠ distance[j][i]

---

## Google Cloud 存取

AlphaFold DB 託管在 Google Cloud Platform 上以供批量存取。

### Cloud Storage 儲存桶

**儲存桶：** `gs://public-datasets-deepmind-alphafold-v4`

**目錄結構：**
```
gs://public-datasets-deepmind-alphafold-v4/
├── accession_ids.csv              # 所有條目索引（13.5 GB）
├── sequences.fasta                # 所有蛋白質序列（16.5 GB）
└── proteomes/                     # 按物種分組（100 萬+ 檔案）
```

### 安裝 gsutil

```bash
# 使用 pip
pip install gsutil

# 或安裝 Google Cloud SDK
curl https://sdk.cloud.google.com | bash
```

### 下載蛋白質體

**按分類 ID：**

```bash
# 下載某物種的所有檔案
TAX_ID=9606  # 人類
gsutil -m cp gs://public-datasets-deepmind-alphafold-v4/proteomes/proteome-tax_id-${TAX_ID}-*_v4.tar .
```

---

## BigQuery 架構

AlphaFold 中繼資料可在 BigQuery 中使用 SQL 查詢。

**資料集：** `bigquery-public-data.deepmind_alphafold`
**資料表：** `metadata`

### 關鍵欄位

| 欄位 | 類型 | 描述 |
|-------|------|-------------|
| `entryId` | STRING | AlphaFold 條目 ID |
| `uniprotAccession` | STRING | UniProt 登錄號 |
| `gene` | STRING | 基因符號 |
| `organismScientificName` | STRING | 物種學名 |
| `taxId` | INTEGER | NCBI 分類 ID |
| `globalMetricValue` | FLOAT | 整體品質指標 |
| `fractionPlddtVeryHigh` | FLOAT | pLDDT ≥ 90 的比例 |
| `isReviewed` | BOOLEAN | Swiss-Prot 審核狀態 |
| `sequenceLength` | INTEGER | 蛋白質序列長度 |

### 查詢範例

```sql
SELECT
  entryId,
  uniprotAccession,
  gene,
  fractionPlddtVeryHigh
FROM `bigquery-public-data.deepmind_alphafold.metadata`
WHERE
  taxId = 9606  -- Homo sapiens
  AND fractionPlddtVeryHigh > 0.8
  AND isReviewed = TRUE
ORDER BY fractionPlddtVeryHigh DESC
LIMIT 100;
```

---

## 最佳實務

### 1. 快取策略

始終在本地快取已下載的檔案，以避免重複下載。

### 2. 錯誤處理

為 API 請求實作健全的錯誤處理，並對暫時性故障實作重試邏輯。

### 3. 批量處理

處理多個蛋白質時，使用適當速率限制的並行下載。

### 4. 版本管理

始終在程式碼中指定和追蹤資料庫版本（當前：v4）。

---

## 錯誤處理

### 常見 HTTP 狀態碼

| 代碼 | 含義 | 動作 |
|------|---------|--------|
| 200 | 成功 | 正常處理回應 |
| 404 | 未找到 | 此 UniProt ID 沒有 AlphaFold 預測 |
| 429 | 請求過多 | 實作速率限制並使用退避重試 |
| 500 | 伺服器錯誤 | 使用指數退避重試 |
| 503 | 服務不可用 | 稍後等待並重試 |

---

## 速率限制

### 建議

- 最多限制 **10 個並行請求**
- 在順序請求之間添加 **100-200ms 延遲**
- 使用 Google Cloud 進行批量下載，而非 REST API
- 在本地快取所有已下載的資料

---

## 額外資源

- **AlphaFold GitHub：** https://github.com/google-deepmind/alphafold
- **Google Cloud 文件：** https://cloud.google.com/datasets/alphafold
- **3D-Beacons 文件：** https://www.ebi.ac.uk/pdbe/pdbe-kb/3dbeacons/docs
- **Biopython 教學：** https://biopython.org/wiki/AlphaFold

## 版本歷史

- **v1**（2021）：初始版本，包含約 35 萬個結構
- **v2**（2022）：擴展至 2 億+ 結構
- **v3**（2023）：更新模型和擴展覆蓋範圍
- **v4**（2024）：當前版本，改進信心指標

## 引用

在出版物中使用 AlphaFold DB 時，請引用：

1. Jumper, J. et al. Highly accurate protein structure prediction with AlphaFold. Nature 596, 583–589 (2021).
2. Varadi, M. et al. AlphaFold Protein Structure Database in 2024: providing structure coverage for over 214 million protein sequences. Nucleic Acids Res. 52, D368–D375 (2024).
