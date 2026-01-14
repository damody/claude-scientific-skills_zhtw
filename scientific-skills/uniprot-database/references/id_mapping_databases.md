# UniProt ID 映射資料庫

UniProt ID 映射服務支援的完整資料庫列表。呼叫 ID 映射 API 時使用這些資料庫名稱。

## 程式化檢索資料庫列表

```python
import requests
response = requests.get("https://rest.uniprot.org/configure/idmapping/fields")
databases = response.json()
```

## UniProt 資料庫

### UniProtKB
- `UniProtKB_AC-ID` - UniProt 登錄號和 ID
- `UniProtKB` - UniProt 知識庫
- `UniProtKB-Swiss-Prot` - 已審核（Swiss-Prot）
- `UniProtKB-TrEMBL` - 未審核（TrEMBL）
- `UniParc` - UniProt 檔案庫
- `UniRef50` - UniRef 50% 同一性叢集
- `UniRef90` - UniRef 90% 同一性叢集
- `UniRef100` - UniRef 100% 同一性叢集

## 序列資料庫

### 核苷酸序列
- `EMBL` - EMBL/GenBank/DDBJ
- `EMBL-CDS` - EMBL 編碼序列
- `RefSeq_Nucleotide` - RefSeq 核苷酸序列
- `CCDS` - 共識 CDS

### 蛋白質序列
- `RefSeq_Protein` - RefSeq 蛋白質序列
- `PIR` - 蛋白質資訊資源

## 基因資料庫

- `GeneID` - Entrez Gene
- `Gene_Name` - 基因名稱
- `Gene_Synonym` - 基因同義詞
- `Gene_OrderedLocusName` - 有序位點名稱
- `Gene_ORFName` - ORF 名稱

## 基因組資料庫

### 一般
- `Ensembl` - Ensembl
- `EnsemblGenomes` - Ensembl Genomes
- `EnsemblGenomes_PRO` - Ensembl Genomes 蛋白質
- `EnsemblGenomes_TRS` - Ensembl Genomes 轉錄本
- `Ensembl_PRO` - Ensembl 蛋白質
- `Ensembl_TRS` - Ensembl 轉錄本

### 物種特定
- `KEGG` - KEGG Genes
- `PATRIC` - PATRIC
- `UCSC` - UCSC Genome Browser
- `VectorBase` - VectorBase
- `WBParaSite` - WormBase ParaSite

## 結構資料庫

- `PDB` - 蛋白質資料庫（Protein Data Bank）
- `AlphaFoldDB` - AlphaFold 資料庫
- `BMRB` - 生物磁共振資料庫
- `PDBsum` - PDB 摘要
- `SASBDB` - 小角散射生物資料庫
- `SMR` - SWISS-MODEL 儲存庫

## 蛋白質家族和結構域資料庫

- `InterPro` - InterPro
- `Pfam` - Pfam 蛋白質家族
- `PROSITE` - PROSITE
- `SMART` - SMART 結構域
- `CDD` - 保守結構域資料庫
- `HAMAP` - HAMAP
- `PANTHER` - PANTHER
- `PRINTS` - PRINTS
- `ProDom` - ProDom
- `SFLD` - 結構-功能連結資料庫
- `SUPFAM` - SUPERFAMILY
- `TIGRFAMs` - TIGRFAMs

## 物種特定資料庫

### 模式生物
- `MGI` - 小鼠基因組資訊學
- `RGD` - 大鼠基因組資料庫
- `FlyBase` - FlyBase（果蠅）
- `WormBase` - WormBase（線蟲）
- `Xenbase` - Xenbase（非洲爪蟾）
- `ZFIN` - 斑馬魚資訊網路
- `dictyBase` - dictyBase（盤基網柄菌）
- `EcoGene` - EcoGene（大腸桿菌）
- `SGD` - 酵母菌基因組資料庫
- `PomBase` - PomBase（粟酒裂殖酵母）
- `TAIR` - 阿拉伯芥資訊資源

### 人類特定
- `HGNC` - HUGO 基因命名委員會
- `CCDS` - 共識編碼序列資料庫

## 路徑資料庫

- `Reactome` - Reactome
- `BioCyc` - BioCyc
- `PlantReactome` - Plant Reactome
- `SIGNOR` - SIGNOR
- `SignaLink` - SignaLink

## 酵素與代謝

- `EC` - 酵素委員會編號
- `BRENDA` - BRENDA 酵素資料庫
- `SABIO-RK` - SABIO-RK（生化反應）
- `MetaCyc` - MetaCyc

## 疾病與表型資料庫

- `OMIM` - 線上人類孟德爾遺傳
- `MIM` - MIM（與 OMIM 相同）
- `OrphaNet` - Orphanet（罕見疾病）
- `DisGeNET` - DisGeNET
- `MalaCards` - MalaCards
- `CTD` - 比較毒理基因組學資料庫
- `OpenTargets` - Open Targets

## 藥物與化學資料庫

- `ChEMBL` - ChEMBL
- `DrugBank` - DrugBank
- `DrugCentral` - DrugCentral
- `GuidetoPHARMACOLOGY` - 藥理學指南
- `SwissLipids` - SwissLipids

## 基因表達資料庫

- `Bgee` - Bgee 基因表達
- `ExpressionAtlas` - Expression Atlas
- `Genevisible` - Genevisible
- `CleanEx` - CleanEx

## 蛋白質組學資料庫

- `PRIDE` - PRIDE 蛋白質組學
- `PeptideAtlas` - PeptideAtlas
- `ProteomicsDB` - ProteomicsDB
- `CPTAC` - CPTAC
- `jPOST` - jPOST
- `MassIVE` - MassIVE
- `MaxQB` - MaxQB
- `PaxDb` - PaxDb
- `TopDownProteomics` - Top Down Proteomics

## 蛋白質-蛋白質交互作用

- `STRING` - STRING
- `BioGRID` - BioGRID
- `IntAct` - IntAct
- `MINT` - MINT
- `DIP` - 交互蛋白質資料庫
- `ComplexPortal` - Complex Portal

## 本體論

- `GO` - 基因本體論
- `GeneTree` - Ensembl GeneTree
- `HOGENOM` - HOGENOM
- `HOVERGEN` - HOVERGEN
- `KO` - KEGG Orthology
- `OMA` - OMA 直系同源
- `OrthoDB` - OrthoDB
- `TreeFam` - TreeFam

## 其他專業資料庫

### 醣基化
- `GlyConnect` - GlyConnect
- `GlyGen` - GlyGen

### 蛋白質修飾
- `PhosphoSitePlus` - PhosphoSitePlus
- `iPTMnet` - iPTMnet

### 抗體
- `Antibodypedia` - Antibodypedia
- `DNASU` - DNASU

### 蛋白質定位
- `COMPARTMENTS` - COMPARTMENTS
- `NeXtProt` - NeXtProt（人類蛋白質）

### 演化與系統發生
- `eggNOG` - eggNOG
- `GeneTree` - Ensembl GeneTree
- `InParanoid` - InParanoid

### 技術資源
- `PRO` - 蛋白質本體論
- `GenomeRNAi` - GenomeRNAi
- `PubMed` - PubMed 文獻參考

## 常見映射場景

### 範例 1：UniProt 到 PDB
```python
from_db = "UniProtKB_AC-ID"
to_db = "PDB"
ids = ["P01308", "P04637"]
```

### 範例 2：基因名稱到 UniProt
```python
from_db = "Gene_Name"
to_db = "UniProtKB"
ids = ["BRCA1", "TP53", "INSR"]
```

### 範例 3：UniProt 到 Ensembl
```python
from_db = "UniProtKB_AC-ID"
to_db = "Ensembl"
ids = ["P12345"]
```

### 範例 4：RefSeq 到 UniProt
```python
from_db = "RefSeq_Protein"
to_db = "UniProtKB"
ids = ["NP_000207.1"]
```

### 範例 5：UniProt 到 GO 術語
```python
from_db = "UniProtKB_AC-ID"
to_db = "GO"
ids = ["P01308"]
```

## 使用注意事項

1. **資料庫名稱區分大小寫**：使用列出的確切名稱

2. **多對多映射**：一個 ID 可能映射到多個目標 ID

3. **失敗的映射**：某些 ID 可能沒有映射；檢查結果中的 `failedIds` 欄位

4. **批次大小限制**：每個作業最多 100,000 個 ID

5. **結果過期**：結果儲存 7 天

6. **雙向映射**：大多數資料庫支援雙向映射

## API 端點

### 取得可用資料庫
```
GET https://rest.uniprot.org/configure/idmapping/fields
```

### 提交映射作業
```
POST https://rest.uniprot.org/idmapping/run
Content-Type: application/x-www-form-urlencoded

from={from_db}&to={to_db}&ids={comma_separated_ids}
```

### 檢查作業狀態
```
GET https://rest.uniprot.org/idmapping/status/{jobId}
```

### 取得結果
```
GET https://rest.uniprot.org/idmapping/results/{jobId}
```

## 資源

- ID 映射工具：https://www.uniprot.org/id-mapping
- API 文件：https://www.uniprot.org/help/id_mapping
- 程式化存取：https://www.uniprot.org/help/api_idmapping
