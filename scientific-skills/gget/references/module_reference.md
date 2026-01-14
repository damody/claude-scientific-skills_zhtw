# gget 模組參考

所有 gget 模組的完整參數參考。

## 參考與基因資訊模組

### gget ref
擷取 Ensembl 參考基因體 FTP 和中繼資料。

**參數：**
| 參數 | 類型 | 描述 | 預設值 |
|-----------|------|-------------|---------|
| `species` | str | 屬名_種名格式或快捷方式（'human'、'mouse'） | 必要 |
| `-w/--which` | str | 回傳的檔案類型：gtf、cdna、dna、cds、cdrna、pep | 全部 |
| `-r/--release` | int | Ensembl 版本號 | 最新 |
| `-od/--out_dir` | str | 輸出目錄路徑 | None |
| `-o/--out` | str | 結果的 JSON 檔案路徑 | None |
| `-l/--list_species` | flag | 列出可用脊椎動物物種 | False |
| `-liv/--list_iv_species` | flag | 列出可用無脊椎動物物種 | False |
| `-ftp` | flag | 僅回傳 FTP 連結 | False |
| `-d/--download` | flag | 下載檔案（需要 curl） | False |
| `-q/--quiet` | flag | 抑制進度資訊 | False |

**回傳：** 包含 FTP 連結、Ensembl 版本號、發布日期、檔案大小的 JSON

---

### gget search
在 Ensembl 中依名稱或描述搜尋基因。

**參數：**
| 參數 | 類型 | 描述 | 預設值 |
|-----------|------|-------------|---------|
| `searchwords` | str/list | 搜尋詞（不區分大小寫） | 必要 |
| `-s/--species` | str | 目標物種或核心資料庫名稱 | 必要 |
| `-r/--release` | int | Ensembl 版本號 | 最新 |
| `-t/--id_type` | str | 回傳 'gene' 或 'transcript' | 'gene' |
| `-ao/--andor` | str | 'or'（任何詞）或 'and'（所有詞） | 'or' |
| `-l/--limit` | int | 回傳的最大結果數 | None |
| `-o/--out` | str | 輸出檔案路徑（CSV/JSON） | None |

**回傳：** ensembl_id、gene_name、ensembl_description、ext_ref_description、biotype、URL

---

### gget info
從 Ensembl、UniProt 和 NCBI 取得完整基因/轉錄本中繼資料。

**參數：**
| 參數 | 類型 | 描述 | 預設值 |
|-----------|------|-------------|---------|
| `ens_ids` | str/list | Ensembl ID（也支援 WormBase、Flybase） | 必要 |
| `-o/--out` | str | 輸出檔案路徑（CSV/JSON） | None |
| `-n/--ncbi` | bool | 停用 NCBI 資料擷取 | False |
| `-u/--uniprot` | bool | 停用 UniProt 資料擷取 | False |
| `-pdb` | bool | 包含 PDB 識別碼 | False |
| `-csv` | flag | 回傳 CSV 格式（CLI） | False |
| `-q/--quiet` | flag | 抑制進度顯示 | False |

**Python 專用：**
- `save=True`：儲存輸出到目前目錄
- `wrap_text=True`：格式化 dataframe 並換行文字

**注意：** 同時處理 >1000 個 ID 可能導致伺服器錯誤。

**回傳：** UniProt ID、NCBI 基因 ID、基因名稱、同義詞、蛋白質名稱、描述、biotype、標準轉錄本

---

### gget seq
以 FASTA 格式擷取核苷酸或胺基酸序列。

**參數：**
| 參數 | 類型 | 描述 | 預設值 |
|-----------|------|-------------|---------|
| `ens_ids` | str/list | Ensembl 識別碼 | 必要 |
| `-o/--out` | str | 輸出檔案路徑 | stdout |
| `-t/--translate` | flag | 取得胺基酸序列 | False |
| `-iso/--isoforms` | flag | 回傳所有轉錄本變體 | False |
| `-q/--quiet` | flag | 抑制進度資訊 | False |

**資料來源：** Ensembl（核苷酸）、UniProt（胺基酸）

**回傳：** FASTA 格式序列

---

## 序列分析與比對模組

### gget blast
對標準資料庫進行 BLAST 序列搜尋。

**參數：**
| 參數 | 類型 | 描述 | 預設值 |
|-----------|------|-------------|---------|
| `sequence` | str | 序列或 FASTA/.txt 路徑 | 必要 |
| `-p/--program` | str | blastn、blastp、blastx、tblastn、tblastx | 自動偵測 |
| `-db/--database` | str | nt、refseq_rna、pdbnt、nr、swissprot、pdbaa、refseq_protein | nt 或 nr |
| `-l/--limit` | int | 回傳的最大命中數 | 50 |
| `-e/--expect` | float | E 值截止 | 10.0 |
| `-lcf/--low_comp_filt` | flag | 啟用低複雜度過濾 | False |
| `-mbo/--megablast_off` | flag | 停用 MegaBLAST（僅 blastn） | False |
| `-o/--out` | str | 輸出檔案路徑 | None |
| `-q/--quiet` | flag | 抑制進度 | False |

**回傳：** Description、Scientific Name、Common Name、Taxid、Max Score、Total Score、Query Coverage

---

### gget blat
使用 UCSC BLAT 尋找基因體位置。

**參數：**
| 參數 | 類型 | 描述 | 預設值 |
|-----------|------|-------------|---------|
| `sequence` | str | 序列或 FASTA/.txt 路徑 | 必要 |
| `-st/--seqtype` | str | 'DNA'、'protein'、'translated%20RNA'、'translated%20DNA' | 自動偵測 |
| `-a/--assembly` | str | 目標組裝（hg38、mm39、taeGut2 等） | 'human'/hg38 |
| `-o/--out` | str | 輸出檔案路徑 | None |
| `-csv` | flag | 回傳 CSV 格式（CLI） | False |
| `-q/--quiet` | flag | 抑制進度 | False |

**回傳：** 基因體、查詢大小、比對起始/結束、匹配數、錯配數、比對百分比

---

### gget muscle
使用 Muscle5 比對多個序列。

**參數：**
| 參數 | 類型 | 描述 | 預設值 |
|-----------|------|-------------|---------|
| `fasta` | str/list | 序列或 FASTA 檔案路徑 | 必要 |
| `-o/--out` | str | 輸出檔案路徑 | stdout |
| `-s5/--super5` | flag | 使用 Super5 演算法（更快，大型資料集） | False |
| `-q/--quiet` | flag | 抑制進度 | False |

**回傳：** ClustalW 格式比對或比對 FASTA（.afa）

---

### gget diamond
快速本地蛋白質/翻譯 DNA 比對。

**參數：**
| 參數 | 類型 | 描述 | 預設值 |
|-----------|------|-------------|---------|
| `query` | str/list | 查詢序列或 FASTA 檔案 | 必要 |
| `--reference` | str/list | 參考序列或 FASTA 檔案 | 必要 |
| `--sensitivity` | str | fast、mid-sensitive、sensitive、more-sensitive、very-sensitive、ultra-sensitive | very-sensitive |
| `--threads` | int | CPU 執行緒 | 1 |
| `--diamond_binary` | str | DIAMOND 安裝路徑 | 自動偵測 |
| `--diamond_db` | str | 儲存資料庫以供重複使用 | None |
| `--translated` | flag | 啟用核苷酸到胺基酸比對 | False |
| `-o/--out` | str | 輸出檔案路徑 | None |
| `-csv` | flag | CSV 格式（CLI） | False |
| `-q/--quiet` | flag | 抑制進度 | False |

**回傳：** 同一性 %、序列長度、匹配位置、缺口開口、E 值、位元分數

---

## 結構與蛋白質分析模組

### gget pdb
查詢 RCSB 蛋白質資料庫。

**參數：**
| 參數 | 類型 | 描述 | 預設值 |
|-----------|------|-------------|---------|
| `pdb_id` | str | PDB 識別碼（例如 '7S7U'） | 必要 |
| `-r/--resource` | str | pdb、entry、pubmed、assembly、entity types | 'pdb' |
| `-i/--identifier` | str | 組裝、實體或鏈 ID | None |
| `-o/--out` | str | 輸出檔案路徑 | stdout |

**回傳：** PDB 格式（結構）或 JSON（中繼資料）

---

### gget alphafold
使用 AlphaFold2 預測 3D 蛋白質結構。

**設定：** 需要 OpenMM 和 `gget setup alphafold`（約 4GB 下載）

**參數：**
| 參數 | 類型 | 描述 | 預設值 |
|-----------|------|-------------|---------|
| `sequence` | str/list | 胺基酸序列或 FASTA 檔案 | 必要 |
| `-mr/--multimer_recycles` | int | 多聚體的迴圈迭代 | 3 |
| `-o/--out` | str | 輸出資料夾路徑 | 時間戳記 |
| `-mfm/--multimer_for_monomer` | flag | 對單體套用多聚體模型 | False |
| `-r/--relax` | flag | 對頂級模型進行 AMBER 弛豫 | False |
| `-q/--quiet` | flag | 抑制進度 | False |

**Python 專用：**
- `plot` (bool)：產生 3D 視覺化（預設：True）
- `show_sidechains` (bool)：包含側鏈（預設：True）

**注意：** 多個序列自動觸發多聚體建模

**回傳：** PDB 結構檔案、JSON 比對誤差資料、可選 3D 圖表

---

### gget elm
預測真核線性基序。

**設定：** 需要 `gget setup elm`

**參數：**
| 參數 | 類型 | 描述 | 預設值 |
|-----------|------|-------------|---------|
| `sequence` | str | 胺基酸序列或 UniProt 登錄號 | 必要 |
| `-s/--sensitivity` | str | DIAMOND 比對敏感度 | very-sensitive |
| `-t/--threads` | int | 執行緒數 | 1 |
| `-bin/--diamond_binary` | str | DIAMOND 二進位檔路徑 | 自動偵測 |
| `-o/--out` | str | 輸出目錄路徑 | None |
| `-u/--uniprot` | flag | 輸入是 UniProt 登錄號 | False |
| `-e/--expand` | flag | 包含蛋白質名稱、生物體、參考文獻 | False |
| `-csv` | flag | CSV 格式（CLI） | False |
| `-q/--quiet` | flag | 抑制進度 | False |

**回傳：** 兩個輸出：
1. **ortholog_df**：來自同源蛋白質的基序
2. **regex_df**：在輸入序列中匹配的基序

---

## 表達與疾病資料模組

### gget archs4
查詢 ARCHS4 獲取基因相關性或組織表達。

**參數：**
| 參數 | 類型 | 描述 | 預設值 |
|-----------|------|-------------|---------|
| `gene` | str | 基因符號或 Ensembl ID | 必要 |
| `-w/--which` | str | 'correlation' 或 'tissue' | 'correlation' |
| `-s/--species` | str | 'human' 或 'mouse'（僅組織） | 'human' |
| `-o/--out` | str | 輸出檔案路徑 | None |
| `-e/--ensembl` | flag | 輸入是 Ensembl ID | False |
| `-csv` | flag | CSV 格式（CLI） | False |
| `-q/--quiet` | flag | 抑制進度 | False |

**回傳：**
- **correlation**：基因符號、Pearson 相關係數（前 100 名）
- **tissue**：組織 ID、最小值/Q1/中位數/Q3/最大值表達

---

### gget cellxgene
查詢 CZ CELLxGENE Discover Census 獲取單細胞資料。

**設定：** 需要 `gget setup cellxgene`

**參數：**
| 參數 | 類型 | 描述 | 預設值 |
|-----------|------|-------------|---------|
| `--gene` (-g) | list | 基因名稱或 Ensembl ID（區分大小寫！） | 必要 |
| `--tissue` | list | 組織類型 | None |
| `--cell_type` | list | 細胞類型 | None |
| `--species` (-s) | str | 'homo_sapiens' 或 'mus_musculus' | 'homo_sapiens' |
| `--census_version` (-cv) | str | "stable"、"latest" 或日期版本 | "stable" |
| `-o/--out` | str | 輸出檔案路徑（CLI 必要） | 必要 |
| `--ensembl` (-e) | flag | 使用 Ensembl ID | False |
| `--meta_only` (-mo) | flag | 僅回傳中繼資料 | False |
| `-q/--quiet` | flag | 抑制進度 | False |

**額外篩選器：** disease、development_stage、sex、assay、dataset_id、donor_id、ethnicity、suspension_type

**重要：** 基因符號區分大小寫（人類 'PAX7'，小鼠 'Pax7'）

**回傳：** 帶計數矩陣和中繼資料的 AnnData 物件

---

### gget enrichr
使用 Enrichr/modEnrichr 執行富集分析。

**參數：**
| 參數 | 類型 | 描述 | 預設值 |
|-----------|------|-------------|---------|
| `genes` | list | 基因符號或 Ensembl ID | 必要 |
| `-db/--database` | str | 參考資料庫或快捷方式 | 必要 |
| `-s/--species` | str | human、mouse、fly、yeast、worm、fish | 'human' |
| `-bkg_l/--background_list` | list | 背景基因 | None |
| `-o/--out` | str | 輸出檔案路徑 | None |
| `-ko/--kegg_out` | str | KEGG 路徑圖像目錄 | None |

**Python 專用：**
- `plot` (bool)：產生圖形結果

**資料庫快捷方式：**
- 'pathway' → KEGG_2021_Human
- 'transcription' → ChEA_2016
- 'ontology' → GO_Biological_Process_2021
- 'diseases_drugs' → GWAS_Catalog_2019
- 'celltypes' → PanglaoDB_Augmented_2021

**回傳：** 路徑/功能關聯及調整後 p 值、重疊基因計數

---

### gget bgee
從 Bgee 擷取同源性和表達。

**參數：**
| 參數 | 類型 | 描述 | 預設值 |
|-----------|------|-------------|---------|
| `ens_id` | str/list | Ensembl 或 NCBI 基因 ID | 必要 |
| `-t/--type` | str | 'orthologs' 或 'expression' | 'orthologs' |
| `-o/--out` | str | 輸出檔案路徑 | None |
| `-csv` | flag | CSV 格式（CLI） | False |
| `-q/--quiet` | flag | 抑制進度 | False |

**注意：** 當 `type='expression'` 時支援多個 ID

**回傳：**
- **orthologs**：跨物種基因及 ID、名稱、分類資訊
- **expression**：解剖學實體、信心分數、表達狀態

---

### gget opentargets
從 OpenTargets 擷取疾病/藥物關聯。

**參數：**
| 參數 | 類型 | 描述 | 預設值 |
|-----------|------|-------------|---------|
| `ens_id` | str | Ensembl 基因 ID | 必要 |
| `-r/--resource` | str | diseases、drugs、tractability、pharmacogenetics、expression、depmap、interactions | 'diseases' |
| `-l/--limit` | int | 最大結果數 | None |
| `-o/--out` | str | 輸出檔案路徑 | None |
| `-csv` | flag | CSV 格式（CLI） | False |
| `-q/--quiet` | flag | 抑制進度 | False |

**資源特定篩選器：**
- drugs：`--filter_disease`
- pharmacogenetics：`--filter_drug`
- expression/depmap：`--filter_tissue`、`--filter_anat_sys`、`--filter_organ`
- interactions：`--filter_protein_a`、`--filter_protein_b`、`--filter_gene_b`

**回傳：** 疾病/藥物關聯、可處理性、藥物基因體學、表達、DepMap、交互作用

---

### gget cbio
從 cBioPortal 繪製癌症基因體學熱圖。

**子命令：** search、plot

**search 參數：**
| 參數 | 類型 | 描述 | 預設值 |
|-----------|------|-------------|---------|
| `keywords` | list | 搜尋詞 | 必要 |

**plot 參數：**
| 參數 | 類型 | 描述 | 預設值 |
|-----------|------|-------------|---------|
| `-s/--study_ids` | list | cBioPortal 研究 ID | 必要 |
| `-g/--genes` | list | 基因名稱或 Ensembl ID | 必要 |
| `-st/--stratification` | str | tissue、cancer_type、cancer_type_detailed、study_id、sample | None |
| `-vt/--variation_type` | str | mutation_occurrences、cna_nonbinary、sv_occurrences、cna_occurrences、Consequence | None |
| `-f/--filter` | str | 依欄位值篩選（例如 'study_id:msk_impact_2017'） | None |
| `-dd/--data_dir` | str | 快取目錄 | ./gget_cbio_cache |
| `-fd/--figure_dir` | str | 輸出目錄 | ./gget_cbio_figures |
| `-t/--title` | str | 自訂圖表標題 | None |
| `-dpi` | int | 解析度 | 100 |
| `-q/--quiet` | flag | 抑制進度 | False |
| `-nc/--no_confirm` | flag | 跳過下載確認 | False |
| `-sh/--show` | flag | 在視窗中顯示圖表 | False |

**回傳：** PNG 熱圖圖像

---

### gget cosmic
搜尋 COSMIC 資料庫的癌症突變。

**重要：** 商業使用需要授權費用。需要 COSMIC 帳號。

**查詢參數：**
| 參數 | 類型 | 描述 | 預設值 |
|-----------|------|-------------|---------|
| `searchterm` | str | 基因名稱、Ensembl ID、突變、樣本 ID | 必要 |
| `-ctp/--cosmic_tsv_path` | str | COSMIC TSV 檔案路徑 | 必要 |
| `-l/--limit` | int | 最大結果數 | 100 |
| `-csv` | flag | CSV 格式（CLI） | False |

**下載參數：**
| 參數 | 類型 | 描述 | 預設值 |
|-----------|------|-------------|---------|
| `-d/--download_cosmic` | flag | 啟動下載模式 | False |
| `-gm/--gget_mutate` | flag | 建立 gget mutate 版本 | False |
| `-cp/--cosmic_project` | str | cancer、census、cell_line、resistance、genome_screen、targeted_screen | None |
| `-cv/--cosmic_version` | str | COSMIC 版本 | 最新 |
| `-gv/--grch_version` | int | 人類參考基因體（37 或 38） | None |
| `--email` | str | COSMIC 帳號電子郵件 | 必要 |
| `--password` | str | COSMIC 帳號密碼 | 必要 |

**注意：** 首次使用者必須下載資料庫

**回傳：** COSMIC 突變資料

---

## 額外工具

### gget mutate
產生突變核苷酸序列。

**參數：**
| 參數 | 類型 | 描述 | 預設值 |
|-----------|------|-------------|---------|
| `sequences` | str/list | FASTA 檔案或序列 | 必要 |
| `-m/--mutations` | str/df | CSV/TSV 檔案或 DataFrame | 必要 |
| `-mc/--mut_column` | str | 突變欄位名稱 | 'mutation' |
| `-sic/--seq_id_column` | str | 序列 ID 欄位 | 'seq_ID' |
| `-mic/--mut_id_column` | str | 突變 ID 欄位 | None |
| `-k/--k` | int | 側翼序列長度 | 30 |
| `-o/--out` | str | 輸出 FASTA 檔案路徑 | stdout |
| `-q/--quiet` | flag | 抑制進度 | False |

**回傳：** FASTA 格式的突變序列

---

### gget gpt
使用 OpenAI 的 API 產生文字。

**設定：** 需要 `gget setup gpt` 和 OpenAI API 金鑰

**參數：**
| 參數 | 類型 | 描述 | 預設值 |
|-----------|------|-------------|---------|
| `prompt` | str | 生成的文字輸入 | 必要 |
| `api_key` | str | OpenAI API 金鑰 | 必要 |
| `model` | str | OpenAI 模型名稱 | gpt-3.5-turbo |
| `temperature` | float | 取樣溫度（0-2） | 1.0 |
| `top_p` | float | 核取樣 | 1.0 |
| `max_tokens` | int | 產生的最大 token 數 | None |
| `frequency_penalty` | float | 頻率懲罰（0-2） | 0 |
| `presence_penalty` | float | 存在懲罰（0-2） | 0 |

**重要：** 免費層級限於 3 個月。設定計費限制。

**回傳：** 產生的文字字串

---

### gget setup
為模組安裝/下載相依套件。

**參數：**
| 參數 | 類型 | 描述 | 預設值 |
|-----------|------|-------------|---------|
| `module` | str | 模組名稱 | 必要 |
| `-o/--out` | str | 輸出資料夾（僅限 elm） | 套件安裝資料夾 |
| `-q/--quiet` | flag | 抑制進度 | False |

**需要設定的模組：**
- `alphafold` - 下載約 4GB 模型參數
- `cellxgene` - 安裝 cellxgene-census
- `elm` - 下載本地 ELM 資料庫
- `gpt` - 配置 OpenAI 整合

**回傳：** None（安裝相依套件）
