# UniProt API 欄位參考

用於自訂 UniProt API 查詢的完整可用欄位列表。使用這些欄位配合 `fields` 參數來只檢索您需要的資料。

## 使用方式

在查詢中加入 fields 參數：
```
https://rest.uniprot.org/uniprotkb/search?query=insulin&fields=accession,gene_names,organism_name,length
```

多個欄位以逗號分隔。逗號後不加空格。

## 核心欄位

### 識別資訊
- `accession` - 主要登錄號（例如：P12345）
- `id` - 條目名稱（例如：INSR_HUMAN）
- `uniprotkb_id` - 與 id 相同
- `entryType` - REVIEWED（Swiss-Prot）或 UNREVIEWED（TrEMBL）

### 蛋白質名稱
- `protein_name` - 推薦名稱和替代蛋白質名稱
- `gene_names` - 基因名稱
- `gene_primary` - 主要基因名稱
- `gene_synonym` - 基因同義詞
- `gene_oln` - 有序位點名稱（Ordered locus names）
- `gene_orf` - ORF 名稱

### 生物體資訊
- `organism_name` - 生物體學名
- `organism_id` - NCBI 分類學識別碼
- `lineage` - 分類學譜系
- `virus_hosts` - 病毒宿主生物體（用於病毒蛋白質）

### 序列資訊
- `sequence` - 胺基酸序列
- `length` - 序列長度
- `mass` - 分子量（道爾頓）
- `fragment` - 條目是否為片段
- `checksum` - 序列 CRC64 校驗碼

## 註解欄位

### 功能與生物學
- `cc_function` - 功能描述
- `cc_catalytic_activity` - 催化活性
- `cc_activity_regulation` - 活性調控
- `cc_pathway` - 代謝路徑資訊
- `cc_cofactor` - 輔因子資訊

### 交互作用與定位
- `cc_interaction` - 蛋白質-蛋白質交互作用
- `cc_subunit` - 亞基結構
- `cc_subcellular_location` - 亞細胞定位
- `cc_tissue_specificity` - 組織特異性
- `cc_developmental_stage` - 發育階段表達

### 疾病與表型
- `cc_disease` - 疾病關聯
- `cc_disruption_phenotype` - 破壞表型
- `cc_allergen` - 過敏原資訊
- `cc_toxic_dose` - 毒性劑量資訊

### 轉譯後修飾
- `cc_ptm` - 轉譯後修飾
- `cc_mass_spectrometry` - 質譜資料

### 其他註解
- `cc_alternative_products` - 替代產物（異構體）
- `cc_polymorphism` - 多型性資訊
- `cc_rna_editing` - RNA 編輯
- `cc_caution` - 注意事項
- `cc_miscellaneous` - 雜項資訊
- `cc_similarity` - 序列相似性
- `cc_sequence_caution` - 序列注意事項
- `cc_web_resource` - 網路資源

## 特徵欄位（ft_）

### 分子加工
- `ft_signal` - 訊號肽（Signal peptide）
- `ft_transit` - 轉運肽（Transit peptide）
- `ft_init_met` - 起始甲硫胺酸
- `ft_propep` - 前肽（Propeptide）
- `ft_chain` - 鏈（成熟蛋白質）
- `ft_peptide` - 肽段

### 區域與位點
- `ft_domain` - 結構域（Domain）
- `ft_repeat` - 重複序列
- `ft_ca_bind` - 鈣結合
- `ft_zn_fing` - 鋅指結構
- `ft_dna_bind` - DNA 結合
- `ft_np_bind` - 核苷酸結合
- `ft_region` - 興趣區域
- `ft_coiled` - 捲曲螺旋（Coiled coil）
- `ft_motif` - 短序列模體
- `ft_compbias` - 組成偏差

### 位點與修飾
- `ft_act_site` - 活性位點
- `ft_metal` - 金屬結合
- `ft_binding` - 結合位點
- `ft_site` - 位點
- `ft_mod_res` - 修飾殘基
- `ft_lipid` - 脂化作用
- `ft_carbohyd` - 醣基化
- `ft_disulfid` - 雙硫鍵
- `ft_crosslnk` - 交聯

### 結構特徵
- `ft_helix` - 螺旋（Helix）
- `ft_strand` - β 股（Beta strand）
- `ft_turn` - 轉角（Turn）
- `ft_transmem` - 跨膜區域
- `ft_intramem` - 膜內區域
- `ft_topo_dom` - 拓撲結構域

### 變異與衝突
- `ft_variant` - 天然變異
- `ft_var_seq` - 替代序列
- `ft_mutagen` - 突變實驗
- `ft_unsure` - 不確定殘基
- `ft_conflict` - 序列衝突
- `ft_non_cons` - 非連續殘基
- `ft_non_ter` - 非末端殘基
- `ft_non_std` - 非標準殘基

## 基因本體論（GO）

- `go` - 所有 GO 術語
- `go_p` - 生物過程（Biological process）
- `go_c` - 細胞組件（Cellular component）
- `go_f` - 分子功能（Molecular function）
- `go_id` - GO 術語識別碼

## 交叉參考（xref_）

### 序列資料庫
- `xref_embl` - EMBL/GenBank/DDBJ
- `xref_refseq` - RefSeq
- `xref_ccds` - CCDS
- `xref_pir` - PIR

### 3D 結構資料庫
- `xref_pdb` - 蛋白質資料庫（Protein Data Bank）
- `xref_pcddb` - PCD 資料庫
- `xref_alphafolddb` - AlphaFold 資料庫
- `xref_smr` - SWISS-MODEL 儲存庫

### 蛋白質家族/結構域資料庫
- `xref_interpro` - InterPro
- `xref_pfam` - Pfam
- `xref_prosite` - PROSITE
- `xref_smart` - SMART

### 基因組資料庫
- `xref_ensembl` - Ensembl
- `xref_ensemblgenomes` - Ensembl Genomes
- `xref_geneid` - Entrez Gene
- `xref_kegg` - KEGG

### 物種特定資料庫
- `xref_mgi` - MGI（小鼠）
- `xref_rgd` - RGD（大鼠）
- `xref_flybase` - FlyBase（果蠅）
- `xref_wormbase` - WormBase（線蟲）
- `xref_xenbase` - Xenbase（蛙）
- `xref_zfin` - ZFIN（斑馬魚）

### 路徑資料庫
- `xref_reactome` - Reactome
- `xref_signor` - SIGNOR
- `xref_signalink` - SignaLink

### 疾病資料庫
- `xref_disgenet` - DisGeNET
- `xref_malacards` - MalaCards
- `xref_omim` - OMIM
- `xref_orphanet` - Orphanet

### 藥物資料庫
- `xref_chembl` - ChEMBL
- `xref_drugbank` - DrugBank
- `xref_guidetopharmacology` - 藥理學指南

### 表達資料庫
- `xref_bgee` - Bgee
- `xref_expressionetatlas` - Expression Atlas
- `xref_genevisible` - Genevisible

## 中繼資料欄位

### 日期
- `date_created` - 條目建立日期
- `date_modified` - 最後修改日期
- `date_sequence_modified` - 最後序列修改日期

### 證據與品質
- `annotation_score` - 註解評分（1-5）
- `protein_existence` - 蛋白質存在等級
- `reviewed` - 條目是否已審核（Swiss-Prot）

### 文獻
- `lit_pubmed_id` - PubMed 識別碼
- `lit_doi` - DOI 識別碼

### 蛋白質組學
- `proteome` - 蛋白質組識別碼
- `tools` - 用於註解的工具

## 程式化檢索可用欄位

使用設定端點取得所有可用欄位：
```bash
curl https://rest.uniprot.org/configure/uniprotkb/result-fields
```

或使用 Python：
```python
import requests
response = requests.get("https://rest.uniprot.org/configure/uniprotkb/result-fields")
fields = response.json()
```

## 常見欄位組合

### 基本蛋白質資訊
```
fields=accession,id,protein_name,gene_names,organism_name,length
```

### 序列與結構
```
fields=accession,sequence,length,mass,xref_pdb,xref_alphafolddb
```

### 功能註解
```
fields=accession,protein_name,cc_function,cc_catalytic_activity,cc_pathway,go
```

### 疾病資訊
```
fields=accession,protein_name,gene_names,cc_disease,xref_omim,xref_malacards
```

### 表達模式
```
fields=accession,gene_names,cc_tissue_specificity,cc_developmental_stage,xref_bgee
```

### 完整註解
```
fields=accession,id,protein_name,gene_names,organism_name,sequence,length,cc_*,ft_*,go,xref_pdb
```

## 注意事項

1. **萬用字元**：某些欄位支援萬用字元（例如 `cc_*` 表示所有註解欄位，`ft_*` 表示所有特徵）

2. **效能**：請求較少欄位可改善回應時間並減少頻寬

3. **格式相依性**：某些欄位可能根據輸出格式（JSON vs TSV）有不同的格式化方式

4. **空值**：沒有資料的欄位可能從回應中省略（JSON）或為空（TSV）

5. **陣列 vs 字串**：在 JSON 格式中，許多欄位回傳物件陣列而非簡單字串

## 資源

- 互動式欄位探索器：https://www.uniprot.org/api-documentation
- API 欄位端點：https://rest.uniprot.org/configure/uniprotkb/result-fields
- 回傳欄位文件：https://www.uniprot.org/help/return_fields
