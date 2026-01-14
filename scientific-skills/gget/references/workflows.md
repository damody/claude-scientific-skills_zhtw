# gget 工作流程範例

展示如何結合多個 gget 模組完成常見生物資訊學任務的擴展工作流程範例。

## 目錄
1. [完整基因分析流程](#完整基因分析流程)
2. [比較結構生物學](#比較結構生物學)
3. [癌症基因體學分析](#癌症基因體學分析)
4. [單細胞表達分析](#單細胞表達分析)
5. [建立參考轉錄體](#建立參考轉錄體)
6. [突變影響評估](#突變影響評估)
7. [藥物標的發現](#藥物標的發現)

---

## 完整基因分析流程

從發現到功能註釋的完整基因分析。

```python
import gget
import pandas as pd

# 步驟 1：搜尋感興趣的基因
print("步驟 1：搜尋 GABA 受體基因...")
search_results = gget.search(["GABA", "receptor", "alpha"],
                             species="homo_sapiens",
                             andor="and")
print(f"找到 {len(search_results)} 個基因")

# 步驟 2：取得詳細資訊
print("\n步驟 2：取得詳細資訊...")
gene_ids = search_results["ensembl_id"].tolist()[:5]  # 前 5 個基因
gene_info = gget.info(gene_ids, pdb=True)
print(gene_info[["ensembl_id", "gene_name", "uniprot_id", "description"]])

# 步驟 3：擷取序列
print("\n步驟 3：擷取序列...")
nucleotide_seqs = gget.seq(gene_ids)
protein_seqs = gget.seq(gene_ids, translate=True)

# 儲存序列
with open("gaba_receptors_nt.fasta", "w") as f:
    f.write(nucleotide_seqs)
with open("gaba_receptors_aa.fasta", "w") as f:
    f.write(protein_seqs)

# 步驟 4：取得表達資料
print("\n步驟 4：取得組織表達...")
for gene_id, gene_name in zip(gene_ids, gene_info["gene_name"]):
    expr_data = gget.archs4(gene_name, which="tissue")
    print(f"\n{gene_name} 表達：")
    print(expr_data.head())

# 步驟 5：尋找相關基因
print("\n步驟 5：尋找相關基因...")
correlated = gget.archs4(gene_info["gene_name"].iloc[0], which="correlation")
correlated_top = correlated.head(20)
print(correlated_top)

# 步驟 6：對相關基因進行富集分析
print("\n步驟 6：執行富集分析...")
gene_list = correlated_top["gene_symbol"].tolist()
enrichment = gget.enrichr(gene_list, database="ontology", plot=True)
print(enrichment.head(10))

# 步驟 7：取得疾病關聯
print("\n步驟 7：取得疾病關聯...")
for gene_id, gene_name in zip(gene_ids[:3], gene_info["gene_name"][:3]):
    diseases = gget.opentargets(gene_id, resource="diseases", limit=5)
    print(f"\n{gene_name} 疾病關聯：")
    print(diseases)

# 步驟 8：檢查同源基因
print("\n步驟 8：尋找同源基因...")
orthologs = gget.bgee(gene_ids[0], type="orthologs")
print(orthologs)

print("\n完整基因分析流程完成！")
```

---

## 比較結構生物學

跨物種比較蛋白質結構並分析功能基序。

```python
import gget

# 定義要比較的基因
human_gene = "ENSG00000169174"  # PCSK9
mouse_gene = "ENSMUSG00000044254"  # Pcsk9

print("比較結構生物學工作流程")
print("=" * 50)

# 步驟 1：取得基因資訊
print("\n1. 取得基因資訊...")
human_info = gget.info([human_gene])
mouse_info = gget.info([mouse_gene])

print(f"人類：{human_info['gene_name'].iloc[0]}")
print(f"小鼠：{mouse_info['gene_name'].iloc[0]}")

# 步驟 2：擷取蛋白質序列
print("\n2. 擷取蛋白質序列...")
human_seq = gget.seq(human_gene, translate=True)
mouse_seq = gget.seq(mouse_gene, translate=True)

# 儲存到檔案以進行比對
with open("pcsk9_sequences.fasta", "w") as f:
    f.write(human_seq)
    f.write("\n")
    f.write(mouse_seq)

# 步驟 3：比對序列
print("\n3. 比對序列...")
alignment = gget.muscle("pcsk9_sequences.fasta")
print("比對完成。以 ClustalW 格式視覺化：")
print(alignment)

# 步驟 4：從 PDB 取得現有結構
print("\n4. 搜尋 PDB 中的現有結構...")
# 使用 BLAST 依序列搜尋
pdb_results = gget.blast(human_seq, database="pdbaa", limit=5)
print("頂級 PDB 匹配：")
print(pdb_results[["Description", "Max Score", "Query Coverage"]])

# 下載頂級結構
if len(pdb_results) > 0:
    # 從描述中提取 PDB ID（通常格式為："PDB|XXXX|..."）
    pdb_id = pdb_results.iloc[0]["Description"].split("|")[1]
    print(f"\n下載 PDB 結構：{pdb_id}")
    gget.pdb(pdb_id, save=True)

# 步驟 5：使用 AlphaFold 預測結構
print("\n5. 使用 AlphaFold 預測結構...")
# 注意：這需要 gget setup alphafold 且運算密集
# 取消註解以執行：
# human_structure = gget.alphafold(human_seq, plot=True)
# mouse_structure = gget.alphafold(mouse_seq, plot=True)
print("（AlphaFold 預測已跳過 - 取消註解以執行）")

# 步驟 6：識別功能基序
print("\n6. 使用 ELM 識別功能基序...")
# 注意：需要 gget setup elm
# 取消註解以執行：
# human_ortholog_df, human_regex_df = gget.elm(human_seq)
# print("人類 PCSK9 功能基序：")
# print(human_regex_df)
print("（ELM 分析已跳過 - 取消註解以執行）")

# 步驟 7：取得同源性資訊
print("\n7. 從 Bgee 取得同源性資訊...")
orthologs = gget.bgee(human_gene, type="orthologs")
print("PCSK9 同源基因：")
print(orthologs)

print("\n比較結構生物學工作流程完成！")
```

---

## 癌症基因體學分析

分析癌症相關基因及其突變。

```python
import gget
import matplotlib.pyplot as plt

print("癌症基因體學分析工作流程")
print("=" * 50)

# 步驟 1：搜尋癌症相關基因
print("\n1. 搜尋乳癌基因...")
genes = gget.search(["breast", "cancer", "BRCA"],
                    species="homo_sapiens",
                    andor="or",
                    limit=20)
print(f"找到 {len(genes)} 個基因")

# 專注於特定基因
target_genes = ["BRCA1", "BRCA2", "TP53", "PIK3CA", "ESR1"]
print(f"\n分析：{', '.join(target_genes)}")

# 步驟 2：取得基因資訊
print("\n2. 取得基因資訊...")
gene_search = []
for gene in target_genes:
    result = gget.search([gene], species="homo_sapiens", limit=1)
    if len(result) > 0:
        gene_search.append(result.iloc[0])

gene_df = pd.DataFrame(gene_search)
gene_ids = gene_df["ensembl_id"].tolist()

# 步驟 3：取得疾病關聯
print("\n3. 從 OpenTargets 取得疾病關聯...")
for gene_id, gene_name in zip(gene_ids, target_genes):
    print(f"\n{gene_name} 疾病關聯：")
    diseases = gget.opentargets(gene_id, resource="diseases", limit=3)
    print(diseases[["disease_name", "overall_score"]])

# 步驟 4：取得藥物關聯
print("\n4. 取得藥物關聯...")
for gene_id, gene_name in zip(gene_ids[:3], target_genes[:3]):
    print(f"\n{gene_name} 藥物關聯：")
    drugs = gget.opentargets(gene_id, resource="drugs", limit=3)
    if len(drugs) > 0:
        print(drugs[["drug_name", "drug_type", "max_phase_for_all_diseases"]])

# 步驟 5：搜尋 cBioPortal 研究
print("\n5. 搜尋 cBioPortal 乳癌研究...")
studies = gget.cbio_search(["breast", "cancer"])
print(f"找到 {len(studies)} 個研究")
print(studies[:5])

# 步驟 6：建立癌症基因體學熱圖
print("\n6. 建立癌症基因體學熱圖...")
if len(studies) > 0:
    # 選擇相關研究
    selected_studies = studies[:2]  # 前 2 個研究

    gget.cbio_plot(
        selected_studies,
        target_genes,
        stratification="cancer_type",
        variation_type="mutation_occurrences",
        show=False
    )
    print("熱圖已儲存至 ./gget_cbio_figures/")

# 步驟 7：查詢 COSMIC 資料庫（需要設定）
print("\n7. 查詢 COSMIC 資料庫...")
# 注意：需要 COSMIC 帳號和資料庫下載
# 取消註解以執行：
# for gene in target_genes[:2]:
#     cosmic_results = gget.cosmic(
#         gene,
#         cosmic_tsv_path="cosmic_cancer.tsv",
#         limit=10
#     )
#     print(f"\n{gene} 在 COSMIC 中的突變：")
#     print(cosmic_results)
print("（COSMIC 查詢已跳過 - 需要資料庫下載）")

# 步驟 8：富集分析
print("\n8. 執行路徑富集...")
enrichment = gget.enrichr(target_genes, database="pathway", plot=True)
print("\n頂級富集路徑：")
print(enrichment.head(10))

print("\n癌症基因體學分析完成！")
```

---

## 單細胞表達分析

分析特定細胞類型和組織的單細胞 RNA-seq 資料。

```python
import gget
import scanpy as sc

print("單細胞表達分析工作流程")
print("=" * 50)

# 注意：需要 gget setup cellxgene

# 步驟 1：定義感興趣的基因和細胞類型
genes_of_interest = ["ACE2", "TMPRSS2", "CD4", "CD8A"]
tissue = "lung"
cell_types = ["type ii pneumocyte", "macrophage", "t cell"]

print(f"\n分析基因：{', '.join(genes_of_interest)}")
print(f"組織：{tissue}")
print(f"細胞類型：{', '.join(cell_types)}")

# 步驟 2：首先取得中繼資料
print("\n1. 擷取中繼資料...")
metadata = gget.cellxgene(
    gene=genes_of_interest,
    tissue=tissue,
    species="homo_sapiens",
    meta_only=True
)
print(f"找到 {len(metadata)} 個資料集")
print(metadata.head())

# 步驟 3：下載計數矩陣
print("\n2. 下載單細胞資料...")
# 注意：這可能是大量下載
adata = gget.cellxgene(
    gene=genes_of_interest,
    tissue=tissue,
    species="homo_sapiens",
    census_version="stable"
)
print(f"AnnData 形狀：{adata.shape}")
print(f"基因：{adata.n_vars}")
print(f"細胞：{adata.n_obs}")

# 步驟 4：使用 scanpy 進行基本 QC 和過濾
print("\n3. 執行品質控制...")
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)
print(f"QC 後 - 細胞：{adata.n_obs}，基因：{adata.n_vars}")

# 步驟 5：正規化和對數轉換
print("\n4. 正規化資料...")
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# 步驟 6：計算基因表達統計
print("\n5. 計算表達統計...")
for gene in genes_of_interest:
    if gene in adata.var_names:
        expr = adata[:, gene].X.toarray().flatten()
        print(f"\n{gene} 表達：")
        print(f"  平均值：{expr.mean():.3f}")
        print(f"  中位數：{np.median(expr):.3f}")
        print(f"  表達百分比：{(expr > 0).sum() / len(expr) * 100:.1f}%")

# 步驟 7：從 ARCHS4 取得組織表達以進行比較
print("\n6. 從 ARCHS4 取得批量組織表達...")
for gene in genes_of_interest:
    tissue_expr = gget.archs4(gene, which="tissue")
    lung_expr = tissue_expr[tissue_expr["tissue"] == "lung"]
    if len(lung_expr) > 0:
        print(f"\n{gene} 在肺部（ARCHS4）：")
        print(f"  中位數：{lung_expr['median'].iloc[0]:.3f}")

# 步驟 8：富集分析
print("\n7. 執行富集分析...")
enrichment = gget.enrichr(genes_of_interest, database="celltypes", plot=True)
print("\n頂級細胞類型關聯：")
print(enrichment.head(10))

# 步驟 9：取得疾病關聯
print("\n8. 取得疾病關聯...")
for gene in genes_of_interest:
    gene_search = gget.search([gene], species="homo_sapiens", limit=1)
    if len(gene_search) > 0:
        gene_id = gene_search["ensembl_id"].iloc[0]
        diseases = gget.opentargets(gene_id, resource="diseases", limit=3)
        print(f"\n{gene} 疾病關聯：")
        print(diseases[["disease_name", "overall_score"]])

print("\n單細胞表達分析完成！")
```

---

## 建立參考轉錄體

準備用於 RNA-seq 分析流程的參考資料。

```bash
#!/bin/bash
# 參考轉錄體建立工作流程

echo "參考轉錄體建立工作流程"
echo "=========================================="

# 步驟 1：列出可用物種
echo -e "\n1. 列出可用物種..."
gget ref --list_species > available_species.txt
echo "可用物種已儲存至 available_species.txt"

# 步驟 2：下載人類參考檔案
echo -e "\n2. 下載人類參考檔案..."
SPECIES="homo_sapiens"
RELEASE=110  # 指定版本以確保可重現性

# 下載 GTF 註釋
echo "下載 GTF 註釋..."
gget ref -w gtf -r $RELEASE -d $SPECIES -o human_ref_gtf.json

# 下載 cDNA 序列
echo "下載 cDNA 序列..."
gget ref -w cdna -r $RELEASE -d $SPECIES -o human_ref_cdna.json

# 下載蛋白質序列
echo "下載蛋白質序列..."
gget ref -w pep -r $RELEASE -d $SPECIES -o human_ref_pep.json

# 步驟 3：建立 kallisto 索引（如果已安裝 kallisto）
echo -e "\n3. 建立 kallisto 索引..."
if command -v kallisto &> /dev/null; then
    # 從下載取得 cDNA FASTA 檔案
    CDNA_FILE=$(ls *.cdna.all.fa.gz)
    if [ -f "$CDNA_FILE" ]; then
        kallisto index -i transcriptome.idx $CDNA_FILE
        echo "Kallisto 索引已建立：transcriptome.idx"
    else
        echo "找不到 cDNA FASTA 檔案"
    fi
else
    echo "未安裝 kallisto，跳過索引建立"
fi

# 步驟 4：下載基因體用於比對方法
echo -e "\n4. 下載基因體序列..."
gget ref -w dna -r $RELEASE -d $SPECIES -o human_ref_dna.json

# 步驟 5：取得感興趣基因的資訊
echo -e "\n5. 取得特定基因的資訊..."
gget search -s $SPECIES "TP53 BRCA1 BRCA2" -o key_genes.csv

echo -e "\n參考轉錄體建立完成！"
```

```python
# Python 版本
import gget
import json

print("參考轉錄體建立工作流程")
print("=" * 50)

# 配置
species = "homo_sapiens"
release = 110
genes_of_interest = ["TP53", "BRCA1", "BRCA2", "MYC", "EGFR"]

# 步驟 1：取得參考資訊
print("\n1. 取得參考資訊...")
ref_info = gget.ref(species, release=release)

# 儲存參考資訊
with open("reference_info.json", "w") as f:
    json.dump(ref_info, f, indent=2)
print("參考資訊已儲存至 reference_info.json")

# 步驟 2：下載特定檔案
print("\n2. 下載參考檔案...")
# GTF 註釋
gget.ref(species, which="gtf", release=release, download=True)
# cDNA 序列
gget.ref(species, which="cdna", release=release, download=True)

# 步驟 3：取得感興趣基因的資訊
print(f"\n3. 取得 {len(genes_of_interest)} 個基因的資訊...")
gene_data = []
for gene in genes_of_interest:
    result = gget.search([gene], species=species, limit=1)
    if len(result) > 0:
        gene_data.append(result.iloc[0])

# 取得詳細資訊
if gene_data:
    gene_ids = [g["ensembl_id"] for g in gene_data]
    detailed_info = gget.info(gene_ids)
    detailed_info.to_csv("genes_of_interest_info.csv", index=False)
    print("基因資訊已儲存至 genes_of_interest_info.csv")

# 步驟 4：取得序列
print("\n4. 擷取序列...")
sequences_nt = gget.seq(gene_ids)
sequences_aa = gget.seq(gene_ids, translate=True)

with open("key_genes_nucleotide.fasta", "w") as f:
    f.write(sequences_nt)
with open("key_genes_protein.fasta", "w") as f:
    f.write(sequences_aa)

print("\n參考轉錄體建立完成！")
print(f"已建立的檔案：")
print("  - reference_info.json")
print("  - genes_of_interest_info.csv")
print("  - key_genes_nucleotide.fasta")
print("  - key_genes_protein.fasta")
```

---

## 突變影響評估

分析基因突變對蛋白質結構和功能的影響。

```python
import gget
import pandas as pd

print("突變影響評估工作流程")
print("=" * 50)

# 定義要分析的突變
mutations = [
    {"gene": "TP53", "mutation": "c.818G>A", "description": "R273H 熱點"},
    {"gene": "EGFR", "mutation": "c.2573T>G", "description": "L858R 活化"},
]

# 步驟 1：取得基因資訊
print("\n1. 取得基因資訊...")
for mut in mutations:
    results = gget.search([mut["gene"]], species="homo_sapiens", limit=1)
    if len(results) > 0:
        mut["ensembl_id"] = results["ensembl_id"].iloc[0]
        print(f"{mut['gene']}：{mut['ensembl_id']}")

# 步驟 2：取得序列
print("\n2. 擷取野生型序列...")
for mut in mutations:
    # 取得核苷酸序列
    nt_seq = gget.seq(mut["ensembl_id"])
    mut["wt_sequence"] = nt_seq

    # 取得蛋白質序列
    aa_seq = gget.seq(mut["ensembl_id"], translate=True)
    mut["wt_protein"] = aa_seq

# 步驟 3：產生突變序列
print("\n3. 產生突變序列...")
# 建立 gget mutate 的突變 dataframe
mut_df = pd.DataFrame({
    "seq_ID": [m["gene"] for m in mutations],
    "mutation": [m["mutation"] for m in mutations]
})

# 對每個突變
for mut in mutations:
    # 從 FASTA 提取序列
    lines = mut["wt_sequence"].split("\n")
    seq = "".join(lines[1:])

    # 建立單一突變 df
    single_mut = pd.DataFrame({
        "seq_ID": [mut["gene"]],
        "mutation": [mut["mutation"]]
    })

    # 產生突變序列
    mutated = gget.mutate([seq], mutations=single_mut)
    mut["mutated_sequence"] = mutated

print("已產生突變序列")

# 步驟 4：取得現有結構資訊
print("\n4. 取得結構資訊...")
for mut in mutations:
    # 取得帶 PDB ID 的資訊
    info = gget.info([mut["ensembl_id"]], pdb=True)

    if "pdb_id" in info.columns and pd.notna(info["pdb_id"].iloc[0]):
        pdb_ids = info["pdb_id"].iloc[0].split(";")
        print(f"\n{mut['gene']} PDB 結構：{', '.join(pdb_ids[:3])}")

        # 下載第一個結構
        if len(pdb_ids) > 0:
            pdb_id = pdb_ids[0].strip()
            mut["pdb_id"] = pdb_id
            gget.pdb(pdb_id, save=True)
    else:
        print(f"\n{mut['gene']}：沒有可用的 PDB 結構")
        mut["pdb_id"] = None

# 步驟 5：使用 AlphaFold 預測結構（可選）
print("\n5. 使用 AlphaFold 預測結構...")
# 注意：需要 gget setup alphafold 且運算密集
# 取消註解以執行：
# for mut in mutations:
#     print(f"預測 {mut['gene']} 野生型結構...")
#     wt_structure = gget.alphafold(mut["wt_protein"])
#
#     print(f"預測 {mut['gene']} 突變體結構...")
#     # 需要先翻譯突變序列
#     # mutant_structure = gget.alphafold(mutated_protein)
print("（AlphaFold 預測已跳過 - 取消註解以執行）")

# 步驟 6：尋找功能基序
print("\n6. 識別功能基序...")
# 注意：需要 gget setup elm
# 取消註解以執行：
# for mut in mutations:
#     ortholog_df, regex_df = gget.elm(mut["wt_protein"])
#     print(f"\n{mut['gene']} 功能基序：")
#     print(regex_df)
print("（ELM 分析已跳過 - 取消註解以執行）")

# 步驟 7：取得疾病關聯
print("\n7. 取得疾病關聯...")
for mut in mutations:
    diseases = gget.opentargets(
        mut["ensembl_id"],
        resource="diseases",
        limit=5
    )
    print(f"\n{mut['gene']}（{mut['description']}）疾病關聯：")
    print(diseases[["disease_name", "overall_score"]])

# 步驟 8：查詢 COSMIC 突變頻率
print("\n8. 查詢 COSMIC 資料庫...")
# 注意：需要 COSMIC 資料庫下載
# 取消註解以執行：
# for mut in mutations:
#     cosmic_results = gget.cosmic(
#         mut["mutation"],
#         cosmic_tsv_path="cosmic_cancer.tsv",
#         limit=10
#     )
#     print(f"\n{mut['gene']} {mut['mutation']} 在 COSMIC 中：")
#     print(cosmic_results)
print("（COSMIC 查詢已跳過 - 需要資料庫下載）")

print("\n突變影響評估完成！")
```

---

## 藥物標的發現

識別和驗證特定疾病的潛在藥物標的。

```python
import gget
import pandas as pd

print("藥物標的發現工作流程")
print("=" * 50)

# 步驟 1：搜尋疾病相關基因
disease = "alzheimer"
print(f"\n1. 搜尋 {disease} 疾病基因...")
genes = gget.search([disease], species="homo_sapiens", limit=50)
print(f"找到 {len(genes)} 個潛在基因")

# 步驟 2：取得詳細資訊
print("\n2. 取得詳細基因資訊...")
gene_ids = genes["ensembl_id"].tolist()[:20]  # 前 20 個
gene_info = gget.info(gene_ids[:10])  # 限制以避免超時

# 步驟 3：從 OpenTargets 取得疾病關聯
print("\n3. 取得疾病關聯...")
disease_scores = []
for gene_id, gene_name in zip(gene_info["ensembl_id"], gene_info["gene_name"]):
    diseases = gget.opentargets(gene_id, resource="diseases", limit=10)

    # 篩選阿茲海默症
    alzheimer = diseases[diseases["disease_name"].str.contains("Alzheimer", case=False, na=False)]

    if len(alzheimer) > 0:
        disease_scores.append({
            "ensembl_id": gene_id,
            "gene_name": gene_name,
            "disease_score": alzheimer["overall_score"].max()
        })

disease_df = pd.DataFrame(disease_scores).sort_values("disease_score", ascending=False)
print("\n頂級疾病相關基因：")
print(disease_df.head(10))

# 步驟 4：取得可處理性資訊
print("\n4. 評估標的可處理性...")
top_targets = disease_df.head(5)
for _, row in top_targets.iterrows():
    tractability = gget.opentargets(
        row["ensembl_id"],
        resource="tractability"
    )
    print(f"\n{row['gene_name']} 可處理性：")
    print(tractability)

# 步驟 5：取得表達資料
print("\n5. 取得組織表達資料...")
for _, row in top_targets.iterrows():
    # 從 OpenTargets 取得腦部表達
    expression = gget.opentargets(
        row["ensembl_id"],
        resource="expression",
        filter_tissue="brain"
    )
    print(f"\n{row['gene_name']} 腦部表達：")
    print(expression)

    # 從 ARCHS4 取得組織表達
    tissue_expr = gget.archs4(row["gene_name"], which="tissue")
    brain_expr = tissue_expr[tissue_expr["tissue"].str.contains("brain", case=False, na=False)]
    print(f"ARCHS4 腦部表達：")
    print(brain_expr)

# 步驟 6：檢查現有藥物
print("\n6. 檢查現有藥物...")
for _, row in top_targets.iterrows():
    drugs = gget.opentargets(row["ensembl_id"], resource="drugs", limit=5)
    print(f"\n{row['gene_name']} 藥物關聯：")
    if len(drugs) > 0:
        print(drugs[["drug_name", "drug_type", "max_phase_for_all_diseases"]])
    else:
        print("未找到藥物")

# 步驟 7：取得蛋白質-蛋白質交互作用
print("\n7. 取得蛋白質-蛋白質交互作用...")
for _, row in top_targets.iterrows():
    interactions = gget.opentargets(
        row["ensembl_id"],
        resource="interactions",
        limit=10
    )
    print(f"\n{row['gene_name']} 交互作用：")
    if len(interactions) > 0:
        print(interactions[["gene_b_symbol", "interaction_score"]])

# 步驟 8：富集分析
print("\n8. 執行路徑富集...")
gene_list = top_targets["gene_name"].tolist()
enrichment = gget.enrichr(gene_list, database="pathway", plot=True)
print("\n頂級富集路徑：")
print(enrichment.head(10))

# 步驟 9：取得結構資訊
print("\n9. 取得結構資訊...")
for _, row in top_targets.iterrows():
    info = gget.info([row["ensembl_id"]], pdb=True)

    if "pdb_id" in info.columns and pd.notna(info["pdb_id"].iloc[0]):
        pdb_ids = info["pdb_id"].iloc[0].split(";")
        print(f"\n{row['gene_name']} PDB 結構：{', '.join(pdb_ids[:3])}")
    else:
        print(f"\n{row['gene_name']}：沒有可用的 PDB 結構")
        # 可以使用 AlphaFold 預測
        print(f"  考慮使用 AlphaFold 預測")

# 步驟 10：產生標的摘要報告
print("\n10. 產生標的摘要報告...")
report = []
for _, row in top_targets.iterrows():
    report.append({
        "基因": row["gene_name"],
        "Ensembl ID": row["ensembl_id"],
        "疾病分數": row["disease_score"],
        "標的狀態": "高優先級"
    })

report_df = pd.DataFrame(report)
report_df.to_csv("drug_targets_report.csv", index=False)
print("\n標的報告已儲存至 drug_targets_report.csv")

print("\n藥物標的發現工作流程完成！")
```

---

## 工作流程開發技巧

### 錯誤處理
```python
import gget

def safe_gget_call(func, *args, **kwargs):
    """帶錯誤處理的 gget 呼叫包裝器"""
    try:
        result = func(*args, **kwargs)
        return result
    except Exception as e:
        print(f"{func.__name__} 發生錯誤：{str(e)}")
        return None

# 使用方式
result = safe_gget_call(gget.search, ["ACE2"], species="homo_sapiens")
if result is not None:
    print(result)
```

### 速率限制
```python
import time
import gget

def rate_limited_queries(gene_ids, delay=1):
    """帶速率限制的多基因查詢"""
    results = []
    for i, gene_id in enumerate(gene_ids):
        print(f"查詢 {i+1}/{len(gene_ids)}：{gene_id}")
        result = gget.info([gene_id])
        results.append(result)

        if i < len(gene_ids) - 1:  # 最後一個查詢後不需要等待
            time.sleep(delay)

    return pd.concat(results, ignore_index=True)
```

### 快取結果
```python
import os
import pickle
import gget

def cached_gget(cache_file, func, *args, **kwargs):
    """快取 gget 結果以避免重複查詢"""
    if os.path.exists(cache_file):
        print(f"從快取載入：{cache_file}")
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    result = func(*args, **kwargs)

    with open(cache_file, "wb") as f:
        pickle.dump(result, f)
    print(f"已儲存至快取：{cache_file}")

    return result

# 使用方式
result = cached_gget("ace2_info.pkl", gget.info, ["ENSG00000130234"])
```

---

這些工作流程展示了如何結合多個 gget 模組進行全面的生物資訊學分析。請根據您的特定研究問題和資料類型進行調整。
