# HMDB 資料欄位參考

本文件提供有關 HMDB 代謝物條目中可用資料欄位的詳細資訊。

## 代謝物條目結構

每個 HMDB 代謝物條目包含 130+ 個資料欄位，組織成幾個類別：

### 化學資料欄位

**識別：**
- `accession`：主要 HMDB ID（例如 HMDB0000001）
- `secondary_accessions`：合併條目的先前 HMDB ID
- `name`：主要代謝物名稱
- `synonyms`：替代名稱和常用名稱
- `chemical_formula`：分子式（例如 C6H12O6）
- `average_molecular_weight`：平均分子量（道爾頓）
- `monoisotopic_molecular_weight`：單同位素分子量

**結構表示：**
- `smiles`：簡化分子輸入行輸入系統字串
- `inchi`：國際化學識別碼字串
- `inchikey`：用於快速查找的雜湊 InChI
- `iupac_name`：IUPAC 系統命名
- `traditional_iupac`：傳統 IUPAC 名稱

**化學性質：**
- `state`：物理狀態（固體、液體、氣體）
- `charge`：淨分子電荷
- `logp`：辛醇-水分配係數（實驗/預測）
- `pka_strongest_acidic`：最強酸性 pKa 值
- `pka_strongest_basic`：最強鹼性 pKa 值
- `polar_surface_area`：拓撲極性表面積（TPSA）
- `refractivity`：莫耳折射率
- `polarizability`：分子極化率
- `rotatable_bond_count`：可旋轉鍵數量
- `acceptor_count`：氫鍵受體數量
- `donor_count`：氫鍵給體數量

**化學分類法：**
- `kingdom`：化學界（例如有機化合物）
- `super_class`：化學超類
- `class`：化學類別
- `sub_class`：化學子類
- `direct_parent`：直接化學母體
- `alternative_parents`：替代母體分類
- `substituents`：存在的化學取代基
- `description`：化合物的文字描述

### 生物資料欄位

**代謝物來源：**
- `origin`：代謝物來源（內源性、外源性、藥物代謝物、食物成分）
- `biofluid_locations`：發現的生物體液（血液、尿液、唾液、腦脊液等）
- `tissue_locations`：發現的組織（肝臟、腎臟、大腦、肌肉等）
- `cellular_locations`：亞細胞位置（細胞質、粒線體、膜等）

**生物標本資訊：**
- `biospecimen`：生物標本類型
- `status`：偵測狀態（已偵測、預期、預測）
- `concentration`：濃度範圍及單位
- `concentration_references`：濃度資料的引用

**正常和異常濃度：**
對於每種生物體液（血液、尿液、唾液、腦脊液、糞便、汗液）：
- 正常濃度值和範圍
- 單位（μM、mg/L 等）
- 年齡和性別考量
- 異常濃度指標
- 臨床意義

### 代謝途徑和酵素資訊

**代謝途徑：**
- `pathways`：相關代謝途徑列表
  - 代謝途徑名稱
  - SMPDB ID（小分子代謝途徑資料庫 ID）
  - KEGG 代謝途徑 ID
  - 代謝途徑類別

**酵素反應：**
- `protein_associations`：酵素和轉運蛋白
  - 蛋白質名稱
  - 基因名稱
  - Uniprot ID
  - GenBank ID
  - 蛋白質類型（酵素、轉運蛋白、載體等）
  - 酵素反應
  - 酵素動力學（Km 值）

**生化背景：**
- `reactions`：涉及代謝物的生化反應
- `reaction_enzymes`：催化反應的酵素
- `cofactors`：所需輔因子
- `inhibitors`：已知的酵素抑制劑

### 疾病和生物標記關聯

**疾病連結：**
- `diseases`：相關疾病和病況
  - 疾病名稱
  - OMIM ID（人類孟德爾遺傳線上資料庫）
  - 疾病類別
  - 參考文獻和證據

**生物標記資訊：**
- `biomarker_status`：化合物是否為已知生物標記
- `biomarker_applications`：臨床應用
- `biomarker_for`：用作生物標記的疾病或病況

### 光譜資料

**NMR 光譜：**
- `nmr_spectra`：核磁共振資料
  - 光譜類型（1D ¹H、¹³C、2D COSY、HSQC 等）
  - 光譜儀頻率（MHz）
  - 使用的溶劑
  - 溫度
  - pH
  - 含化學位移和多重性的峰值列表
  - FID（自由感應衰減）檔案

**質譜：**
- `ms_spectra`：質譜資料
  - 光譜類型（MS、MS-MS、LC-MS、GC-MS）
  - 離子化模式（正、負、中性）
  - 碰撞能量
  - 儀器類型
  - 峰值列表（m/z、強度、註釋）
  - 預測與實驗標記

**層析法：**
- `chromatography`：層析性質
  - 滯留時間
  - 管柱類型
  - 流動相
  - 方法詳情

### 外部資料庫連結

**資料庫交叉參考：**
- `kegg_id`：KEGG 化合物 ID
- `pubchem_compound_id`：PubChem CID
- `pubchem_substance_id`：PubChem SID
- `chebi_id`：生物興趣化學實體 ID
- `chemspider_id`：ChemSpider ID
- `drugbank_id`：DrugBank 登錄號（如適用）
- `foodb_id`：FooDB ID（如為食物成分）
- `knapsack_id`：KNApSAcK ID
- `metacyc_id`：MetaCyc ID
- `bigg_id`：BiGG 模型 ID
- `wikipedia_id`：維基百科頁面連結
- `metlin_id`：METLIN ID
- `vmh_id`：虛擬代謝人類 ID
- `fbonto_id`：FlyBase 本體 ID

**蛋白質資料庫連結：**
- `uniprot_id`：相關蛋白質的 UniProt 登錄號
- `genbank_id`：相關基因的 GenBank ID
- `pdb_id`：蛋白質結構的蛋白質資料庫 ID

### 文獻和證據

**參考文獻：**
- `general_references`：關於代謝物的一般參考文獻
  - PubMed ID
  - 參考文獻
  - 引用
- `synthesis_reference`：合成方法和參考文獻
- `protein_references`：蛋白質關聯的參考文獻
- `pathway_references`：代謝途徑參與的參考文獻

### 本體論和分類

**本體論術語：**
- `ontology_terms`：相關本體論分類
  - 術語名稱
  - 本體論來源（ChEBI、MeSH 等）
  - 術語 ID
  - 定義

### 資料品質和來源

**中繼資料：**
- `creation_date`：條目建立日期
- `update_date`：條目最後更新日期
- `version`：HMDB 版本號
- `status`：條目狀態（已偵測、預期、預測）
- `evidence`：偵測/存在的證據等級

## XML 結構範例

以 XML 格式下載 HMDB 資料時，結構遵循此模式：

```xml
<metabolite>
  <accession>HMDB0000001</accession>
  <name>1-Methylhistidine</name>
  <chemical_formula>C7H11N3O2</chemical_formula>
  <average_molecular_weight>169.1811</average_molecular_weight>
  <monoisotopic_molecular_weight>169.085126436</monoisotopic_molecular_weight>
  <smiles>CN1C=NC(CC(=O)O)=C1</smiles>
  <inchi>InChI=1S/C7H11N3O2/c1-10-4-8-3-5(10)2-7(11)12/h3-4H,2H2,1H3,(H,11,12)</inchi>
  <inchikey>BRMWTNUJHUMWMS-UHFFFAOYSA-N</inchikey>

  <biospecimen_locations>
    <biospecimen>Blood</biospecimen>
    <biospecimen>Urine</biospecimen>
  </biospecimen_locations>

  <pathways>
    <pathway>
      <name>Histidine Metabolism</name>
      <smpdb_id>SMP0000044</smpdb_id>
      <kegg_map_id>map00340</kegg_map_id>
    </pathway>
  </pathways>

  <diseases>
    <disease>
      <name>Carnosinemia</name>
      <omim_id>212200</omim_id>
    </disease>
  </diseases>

  <normal_concentrations>
    <concentration>
      <biospecimen>Blood</biospecimen>
      <concentration_value>3.8</concentration_value>
      <concentration_units>uM</concentration_units>
    </concentration>
  </normal_concentrations>
</metabolite>
```

## 查詢特定欄位

以程式化方式處理 HMDB 資料時：

**用於代謝物鑑定：**
- 依 `accession`、`name`、`synonyms`、`inchi`、`smiles` 查詢

**用於化學相似性：**
- 使用 `smiles`、`inchi`、`inchikey`、`molecular_weight`、`chemical_formula`

**用於生物標記發現：**
- 依 `diseases`、`biomarker_status`、`normal_concentrations`、`abnormal_concentrations` 過濾

**用於代謝途徑分析：**
- 擷取 `pathways`、`protein_associations`、`reactions`

**用於光譜比對：**
- 與 `nmr_spectra`、`ms_spectra` 峰值列表比較

**用於跨資料庫整合：**
- 使用外部 ID 映射：`kegg_id`、`pubchem_compound_id`、`chebi_id` 等

## 欄位完整性

並非每個代謝物都填入所有欄位：

- **高度完整欄位**（>90% 條目）：accession、name、chemical_formula、molecular_weight、smiles、inchi
- **中度完整**（50-90%）：biospecimen_locations、tissue_locations、pathways
- **可變完整**（10-50%）：濃度資料、疾病關聯、蛋白質關聯
- **稀疏完整**（<10%）：實驗 NMR/MS 光譜、詳細動力學資料

預測和計算資料（例如預測的 MS 光譜、預測的濃度）在可用時補充實驗資料。
