# gget 資料庫資訊

gget 模組查詢的資料庫概述，包括更新頻率和重要注意事項。

## 重要說明

gget 查詢的資料庫會持續更新，這有時會改變其結構。gget 模組每兩週自動測試一次，並在必要時更新以匹配新的資料庫結構。請始終保持 gget 更新：

```bash
pip install --upgrade gget
```

## 資料庫目錄

### 基因體參考資料庫

#### Ensembl
- **使用於：** gget ref、gget search、gget info、gget seq
- **描述：** 具有脊椎動物和無脊椎動物物種註釋的完整基因體資料庫
- **更新頻率：** 定期發布（編號版本）；約每 3 個月發布新版本
- **存取方式：** FTP 下載、REST API
- **網站：** https://www.ensembl.org/
- **注意事項：**
  - 支援脊椎動物和無脊椎動物基因體
  - 可指定版本號以確保可重現性
  - 常見物種有快捷方式（'human'、'mouse'）

#### UCSC Genome Browser
- **使用於：** gget blat
- **描述：** 具有 BLAT 比對工具的基因體瀏覽器資料庫
- **更新頻率：** 隨新組裝定期更新
- **存取方式：** 網路服務 API
- **網站：** https://genome.ucsc.edu/
- **注意事項：**
  - 提供多個基因體組裝（hg38、mm39 等）
  - BLAT 針對脊椎動物基因體最佳化

### 蛋白質與結構資料庫

#### UniProt
- **使用於：** gget info、gget seq（胺基酸序列）、gget elm
- **描述：** 通用蛋白質資源，完整的蛋白質序列和功能資訊
- **更新頻率：** 定期發布（Swiss-Prot 每週、TrEMBL 每月）
- **存取方式：** REST API
- **網站：** https://www.uniprot.org/
- **注意事項：**
  - Swiss-Prot：手動註釋和審核
  - TrEMBL：自動註釋

#### NCBI（國家生物技術資訊中心）
- **使用於：** gget info、gget bgee（非 Ensembl 物種）
- **描述：** 具有廣泛交叉參考的基因和蛋白質資料庫
- **更新頻率：** 持續更新
- **存取方式：** E-utilities API
- **網站：** https://www.ncbi.nlm.nih.gov/
- **資料庫：** Gene、Protein、RefSeq

#### RCSB PDB（蛋白質資料庫）
- **使用於：** gget pdb
- **描述：** 蛋白質和核酸 3D 結構資料儲存庫
- **更新頻率：** 每週更新
- **存取方式：** REST API
- **網站：** https://www.rcsb.org/
- **注意事項：**
  - 實驗確定的結構（X 射線、NMR、冷凍電鏡）
  - 包含實驗和出版物的中繼資料

#### ELM（真核線性基序）
- **使用於：** gget elm
- **描述：** 真核蛋白質功能位點資料庫
- **更新頻率：** 定期更新
- **存取方式：** 下載的資料庫（透過 gget setup elm）
- **網站：** http://elm.eu.org/
- **注意事項：**
  - 首次使用前需要本地下載
  - 包含經驗證的基序和模式

### 序列相似性資料庫

#### BLAST 資料庫（NCBI）
- **使用於：** gget blast
- **描述：** 用於 BLAST 搜尋的預格式化資料庫
- **更新頻率：** 定期更新
- **存取方式：** NCBI BLAST API
- **資料庫：**
  - **核苷酸：** nt（所有 GenBank）、refseq_rna、pdbnt
  - **蛋白質：** nr（非冗餘）、swissprot、pdbaa、refseq_protein
- **注意事項：**
  - nt 和 nr 是非常大的資料庫
  - 考慮使用專門資料庫以獲得更快、更專注的搜尋

### 表達與相關性資料庫

#### ARCHS4
- **使用於：** gget archs4
- **描述：** 公開可用 RNA-seq 資料的大規模挖掘
- **更新頻率：** 定期更新新樣本
- **存取方式：** HTTP API
- **網站：** https://maayanlab.cloud/archs4/
- **資料：**
  - 人類和小鼠 RNA-seq 資料
  - 相關性矩陣
  - 組織表達圖譜
- **引用：** Lachmann et al., Nature Communications, 2018

#### CZ CELLxGENE Discover
- **使用於：** gget cellxgene
- **描述：** 來自多項研究的單細胞 RNA-seq 資料
- **更新頻率：** 持續添加新資料集
- **存取方式：** Census API（透過 cellxgene-census 套件）
- **網站：** https://cellxgene.cziscience.com/
- **資料：**
  - 單細胞 RNA-seq 計數矩陣
  - 細胞類型註釋
  - 組織和疾病中繼資料
- **注意事項：**
  - 需要 gget setup cellxgene
  - 基因符號區分大小寫
  - 可能不支援最新 Python 版本

#### Bgee
- **使用於：** gget bgee
- **描述：** 基因表達和同源性資料庫
- **更新頻率：** 定期發布
- **存取方式：** REST API
- **網站：** https://www.bgee.org/
- **資料：**
  - 跨組織和發育階段的基因表達
  - 跨物種的同源性關係
- **引用：** Bastian et al., 2021

### 功能與路徑資料庫

#### Enrichr / modEnrichr
- **使用於：** gget enrichr
- **描述：** 基因集富集分析網路服務
- **更新頻率：** 底層資料庫定期更新
- **存取方式：** REST API
- **網站：** https://maayanlab.cloud/Enrichr/
- **包含的資料庫：**
  - KEGG 路徑
  - 基因本體（GO）
  - 轉錄因子標的（ChEA）
  - 疾病關聯（GWAS Catalog）
  - 細胞類型標記（PanglaoDB）
- **注意事項：**
  - 支援多種模式生物
  - 可提供背景基因列表進行自訂富集

### 疾病與藥物資料庫

#### Open Targets
- **使用於：** gget opentargets
- **描述：** 疾病-標的關聯的整合平台
- **更新頻率：** 定期發布（每季）
- **存取方式：** GraphQL API
- **網站：** https://www.opentargets.org/
- **資料：**
  - 疾病關聯
  - 藥物資訊和臨床試驗
  - 標的可處理性
  - 藥物基因體學
  - 基因表達
  - DepMap 基因-疾病效應
  - 蛋白質-蛋白質交互作用

#### cBioPortal
- **使用於：** gget cbio
- **描述：** 癌症基因體學資料入口
- **更新頻率：** 持續添加新研究
- **存取方式：** Web API、可下載資料集
- **網站：** https://www.cbioportal.org/
- **資料：**
  - 突變、拷貝數變異、結構變異
  - 基因表達
  - 臨床資料
- **注意事項：**
  - 大型資料集；建議快取
  - 提供多種癌症類型和研究

#### COSMIC（癌症體細胞突變目錄）
- **使用於：** gget cosmic
- **描述：** 完整的癌症突變資料庫
- **更新頻率：** 定期發布
- **存取方式：** 下載（需要帳號和商業使用授權）
- **網站：** https://cancer.sanger.ac.uk/cosmic
- **資料：**
  - 癌症中的體細胞突變
  - 基因普查
  - 細胞系資料
  - 藥物抗性突變
- **重要提示：**
  - 學術使用免費
  - 商業使用需要授權費用
  - 需要 COSMIC 帳號憑證
  - 查詢前必須下載資料庫

### AI 與預測服務

#### AlphaFold2（DeepMind）
- **使用於：** gget alphafold
- **描述：** 蛋白質結構預測的深度學習模型
- **模型版本：** 用於本地執行的簡化版本
- **存取方式：** 本地運算（需透過 gget setup 下載模型）
- **網站：** https://alphafold.ebi.ac.uk/
- **注意事項：**
  - 需要約 4GB 模型參數下載
  - 需要安裝 OpenMM
  - 運算密集
  - Python 版本有特定要求

#### OpenAI API
- **使用於：** gget gpt
- **描述：** 大型語言模型 API
- **更新頻率：** 定期發布新模型
- **存取方式：** REST API（需要 API 金鑰）
- **網站：** https://openai.com/
- **注意事項：**
  - 預設模型：gpt-3.5-turbo
  - 免費層級限於帳號建立後 3 個月
  - 設定計費限制以控制成本

## 資料一致性與可重現性

### 版本控制
確保分析可重現性：

1. **指定資料庫版本/版次：**
   ```python
   # 使用特定 Ensembl 版本
   gget.ref("homo_sapiens", release=110)

   # 使用特定 Census 版本
   gget.cellxgene(gene=["PAX7"], census_version="2023-07-25")
   ```

2. **記錄 gget 版本：**
   ```python
   import gget
   print(gget.__version__)
   ```

3. **儲存原始資料：**
   ```python
   # 始終儲存結果以確保可重現性
   results = gget.search(["ACE2"], species="homo_sapiens")
   results.to_csv("search_results_2025-01-15.csv", index=False)
   ```

### 處理資料庫更新

1. **定期更新 gget：**
   - 每兩週更新 gget 以匹配資料庫結構變化
   - 檢查發行說明了解重大變更

2. **錯誤處理：**
   - 資料庫結構變化可能導致暫時失敗
   - 檢查 GitHub issues：https://github.com/pachterlab/gget/issues
   - 如果發生錯誤，更新 gget

3. **API 速率限制：**
   - 為大規模查詢實作延遲
   - 盡可能使用本地資料庫（DIAMOND、COSMIC）
   - 快取結果以避免重複查詢

## 特定資料庫最佳實踐

### Ensembl
- 使用物種快捷方式（'human'、'mouse'）以方便操作
- 指定版本號以確保可重現性
- 使用 `gget ref --list_species` 檢查可用物種

### UniProt
- UniProt ID 比基因名稱更穩定
- Swiss-Prot 註釋是手動策展的，更可靠
- 僅在需要時使用 gget info 中的 PDB 旗標（增加執行時間）

### BLAST/BLAT
- 從預設參數開始，然後最佳化
- 使用專門資料庫（swissprot、refseq_protein）進行專注搜尋
- 根據查詢長度考慮 E 值截止

### 表達資料庫
- CELLxGENE 中基因符號區分大小寫
- ARCHS4 相關性資料基於共表達模式
- 解讀結果時考慮組織特異性

### 癌症資料庫
- cBioPortal：本地快取資料以進行重複分析
- COSMIC：下載適合您需求的資料庫子集
- 商業使用請遵守授權協議

## 引用

使用 gget 時，請引用 gget 出版物和底層資料庫：

**gget：**
Luebbert, L. & Pachter, L. (2023). Efficient querying of genomic reference databases with gget. Bioinformatics. https://doi.org/10.1093/bioinformatics/btac836

**特定資料庫引用：** 請查看 references/ 目錄或資料庫網站以獲取適當的引用。
