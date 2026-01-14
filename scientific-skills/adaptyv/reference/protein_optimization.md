# 蛋白質序列優化

## 概述

在提交蛋白質序列進行實驗測試之前，使用計算工具優化序列以改善表現、溶解度和穩定性。這種預篩選可降低實驗成本並提高成功率。

## 常見的蛋白質表現問題

### 1. 未配對的半胱胺酸

**問題：**
- 未配對的半胱胺酸形成不需要的二硫鍵
- 導致聚集和錯誤摺疊
- 降低表現產量和穩定性

**解決方案：**
- 除非功能需要，否則移除未配對的半胱胺酸
- 適當配對半胱胺酸以形成結構性二硫鍵
- 在非關鍵位置替換為絲胺酸或丙胺酸

**範例：**
```python
# 檢查半胱胺酸配對
from Bio.Seq import Seq

def check_cysteines(sequence):
    cys_count = sequence.count('C')
    if cys_count % 2 != 0:
        print(f"警告：半胱胺酸數量為奇數（{cys_count}）")
    return cys_count
```

### 2. 過度疏水性

**問題：**
- 長疏水區塊促進聚集
- 暴露的疏水殘基導致蛋白質團聚
- 在水性緩衝液中溶解度差

**解決方案：**
- 維持平衡的親疏水性圖譜
- 在結構域之間使用短而靈活的連接子
- 減少表面暴露的疏水殘基

**指標：**
- Kyte-Doolittle 親疏水性圖
- GRAVY 分數（平均親疏水性）
- pSAE（溶劑可及疏水殘基百分比）

### 3. 低溶解度

**問題：**
- 蛋白質在表現或純化過程中沉澱
- 包涵體形成
- 下游處理困難

**解決方案：**
- 使用溶解度預測工具進行預篩選
- 應用序列優化演算法
- 必要時添加增溶標籤

## 優化的計算工具

### NetSolP - 初始溶解度篩選

**目的：** 快速溶解度預測以篩選序列。

**方法：** 基於大腸桿菌表現資料訓練的機器學習模型。

**用法：**
```python
# 安裝：uv pip install requests
import requests

def predict_solubility_netsolp(sequence):
    """使用 NetSolP 網路服務預測蛋白質溶解度"""
    url = "https://services.healthtech.dtu.dk/services/NetSolP-1.0/api/predict"

    data = {
        "sequence": sequence,
        "format": "fasta"
    }

    response = requests.post(url, data=data)
    return response.json()

# 範例
sequence = "MKVLWAALLGLLGAAA..."
result = predict_solubility_netsolp(sequence)
print(f"溶解度分數：{result['score']}")
```

**解讀：**
- 分數 > 0.5：可能可溶
- 分數 < 0.5：可能不可溶
- 用於更昂貴預測前的初始篩選

**使用時機：**
- 大型文庫的首次篩選
- 設計序列的快速驗證
- 優先選擇實驗測試的序列

### SoluProt - 全面溶解度預測

**目的：** 更高準確度的進階溶解度預測。

**方法：** 結合序列和結構特徵的深度學習模型。

**用法：**
```python
# 安裝：uv pip install soluprot
from soluprot import predict_solubility

def screen_variants_soluprot(sequences):
    """篩選多個序列的溶解度"""
    results = []
    for name, seq in sequences.items():
        score = predict_solubility(seq)
        results.append({
            'name': name,
            'sequence': seq,
            'solubility_score': score,
            'predicted_soluble': score > 0.6
        })
    return results

# 範例
sequences = {
    'variant_1': 'MKVLW...',
    'variant_2': 'MATGV...'
}

results = screen_variants_soluprot(sequences)
soluble_variants = [r for r in results if r['predicted_soluble']]
```

**解讀：**
- 分數 > 0.6：高溶解度信賴度
- 分數 0.4-0.6：不確定，可能需要優化
- 分數 < 0.4：可能有問題

**使用時機：**
- NetSolP 初始篩選後
- 需要更高預測準確度時
- 在投入昂貴的合成/測試前

### SolubleMPNN - 序列重設計

**目的：** 重設計蛋白質序列以提高溶解度同時維持功能。

**方法：** 建議突變以增加溶解度的圖神經網路。

**用法：**
```python
# 安裝：uv pip install soluble-mpnn
from soluble_mpnn import optimize_sequence

def optimize_for_solubility(sequence, structure_pdb=None):
    """
    重設計序列以改善溶解度

    參數：
        sequence：原始胺基酸序列
        structure_pdb：可選的 PDB 檔案用於結構感知設計

    回傳：
        按預測溶解度排名的優化序列變體
    """

    variants = optimize_sequence(
        sequence=sequence,
        structure=structure_pdb,
        num_variants=10,
        temperature=0.1  # 較低 = 較保守的突變
    )

    return variants

# 範例
original_seq = "MKVLWAALLGLLGAAA..."
optimized_variants = optimize_for_solubility(original_seq)

for i, variant in enumerate(optimized_variants):
    print(f"變體 {i+1}：")
    print(f"  序列：{variant['sequence']}")
    print(f"  溶解度分數：{variant['solubility_score']}")
    print(f"  突變：{variant['mutations']}")
```

**設計策略：**
- **保守**（temperature=0.1）：最小變化，較安全
- **中等**（temperature=0.3）：變化與安全性的平衡
- **積極**（temperature=0.5）：更多突變，較高風險

**使用時機：**
- 序列優化的主要工具
- 改善問題序列的預設起點
- 生成多樣的可溶變體

**最佳實務：**
- 每個序列生成 10-50 個變體
- 可用時使用結構資訊（提高準確度）
- 驗證關鍵功能殘基是否保留
- 測試多個溫度設定

### ESM（進化規模建模）- 序列可能性

**目的：** 根據進化模式評估蛋白質序列的「自然程度」。

**方法：** 在數百萬個自然序列上訓練的蛋白質語言模型。

**用法：**
```python
# 安裝：uv pip install fair-esm
import torch
from esm import pretrained

def score_sequence_esm(sequence):
    """
    計算序列的 ESM 可能性分數
    較高分數表示更自然/穩定的序列
    """

    model, alphabet = pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()

    data = [("protein", sequence)]
    _, _, batch_tokens = batch_converter(data)

    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33])
        token_logprobs = results["logits"].log_softmax(dim=-1)

    # 計算困惑度作為序列品質指標
    sequence_score = token_logprobs.mean().item()

    return sequence_score

# 範例 - 比較變體
sequences = {
    'original': 'MKVLW...',
    'optimized_1': 'MKVLS...',
    'optimized_2': 'MKVLA...'
}

for name, seq in sequences.items():
    score = score_sequence_esm(seq)
    print(f"{name}：ESM 分數 = {score:.3f}")
```

**解讀：**
- 較高分數 → 更「自然」的序列
- 用於避免不可能的突變
- 與功能需求取得平衡

**使用時機：**
- 篩選合成設計
- 比較 SolubleMPNN 變體
- 確保序列不過於人工
- 避免表現瓶頸

**與設計整合：**
```python
def rank_variants_by_esm(variants):
    """按 ESM 可能性排名蛋白質變體"""
    scored = []
    for v in variants:
        esm_score = score_sequence_esm(v['sequence'])
        v['esm_score'] = esm_score
        scored.append(v)

    # 按溶解度和 ESM 分數的組合排序
    scored.sort(
        key=lambda x: x['solubility_score'] * x['esm_score'],
        reverse=True
    )

    return scored
```

### ipTM - 介面穩定性（AlphaFold-Multimer）

**目的：** 評估蛋白質-蛋白質介面穩定性和結合信賴度。

**方法：** 來自 AlphaFold-Multimer 預測的介面預測 TM 分數。

**用法：**
```python
# 需要安裝 AlphaFold-Multimer
# 或使用 ColabFold 以獲得更簡便的存取

def predict_interface_stability(protein_a_seq, protein_b_seq):
    """
    使用 AlphaFold-Multimer 預測介面穩定性

    回傳 ipTM 分數：較高 = 更穩定的介面
    """
    from colabfold import run_alphafold_multimer

    sequences = {
        'chainA': protein_a_seq,
        'chainB': protein_b_seq
    }

    result = run_alphafold_multimer(sequences)

    return {
        'ipTM': result['iptm'],
        'pTM': result['ptm'],
        'pLDDT': result['plddt']
    }

# 抗體-抗原結合範例
antibody_seq = "EVQLVESGGGLVQPGG..."
antigen_seq = "MKVLWAALLGLLGAAA..."

stability = predict_interface_stability(antibody_seq, antigen_seq)
print(f"介面 pTM：{stability['ipTM']:.3f}")

# 解讀
if stability['ipTM'] > 0.7:
    print("高信賴度介面")
elif stability['ipTM'] > 0.5:
    print("中等信賴度介面")
else:
    print("低信賴度介面 - 可能需要重設計")
```

**解讀：**
- ipTM > 0.7：強預測介面
- ipTM 0.5-0.7：中等介面信賴度
- ipTM < 0.5：弱介面，考慮重設計

**使用時機：**
- 抗體-抗原設計
- 蛋白質-蛋白質交互作用工程
- 驗證結合介面
- 比較介面變體

### pSAE - 溶劑可及疏水殘基

**目的：** 量化促進聚集的暴露疏水殘基。

**方法：** 計算疏水殘基佔據的溶劑可及表面積（SASA）百分比。

**用法：**
```python
# 需要結構（PDB 檔案或 AlphaFold 預測）
# 安裝：uv pip install biopython

from Bio.PDB import PDBParser, DSSP
import numpy as np

def calculate_psae(pdb_file):
    """
    計算溶劑可及疏水殘基百分比（pSAE）

    較低 pSAE = 較好的溶解度
    """

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_file)

    # 執行 DSSP 以取得溶劑可及性
    model = structure[0]
    dssp = DSSP(model, pdb_file, acc_array='Wilke')

    hydrophobic = ['ALA', 'VAL', 'ILE', 'LEU', 'MET', 'PHE', 'TRP', 'PRO']

    total_sasa = 0
    hydrophobic_sasa = 0

    for residue in dssp:
        res_name = residue[1]
        rel_accessibility = residue[3]

        total_sasa += rel_accessibility
        if res_name in hydrophobic:
            hydrophobic_sasa += rel_accessibility

    psae = (hydrophobic_sasa / total_sasa) * 100

    return psae

# 範例
pdb_file = "protein_structure.pdb"
psae_score = calculate_psae(pdb_file)
print(f"pSAE：{psae_score:.2f}%")

# 解讀
if psae_score < 25:
    print("預期良好溶解度")
elif psae_score < 35:
    print("中等溶解度")
else:
    print("高聚集風險")
```

**解讀：**
- pSAE < 25%：低聚集風險
- pSAE 25-35%：中等風險
- pSAE > 35%：高聚集風險

**使用時機：**
- 分析設計的結構
- AlphaFold 後驗證
- 識別聚集熱點
- 指導表面突變

## 推薦的優化工作流程

### 步驟 1：初始篩選（快速）

```python
def initial_screening(sequences):
    """
    使用 NetSolP 進行快速首次篩選
    過濾明顯有問題的序列
    """
    passed = []
    for name, seq in sequences.items():
        netsolp_score = predict_solubility_netsolp(seq)
        if netsolp_score > 0.5:
            passed.append((name, seq))

    return passed
```

### 步驟 2：詳細評估（中等）

```python
def detailed_assessment(filtered_sequences):
    """
    使用 SoluProt 和 ESM 進行更徹底的分析
    按多個標準排名序列
    """
    results = []
    for name, seq in filtered_sequences:
        soluprot_score = predict_solubility(seq)
        esm_score = score_sequence_esm(seq)

        combined_score = soluprot_score * 0.7 + esm_score * 0.3

        results.append({
            'name': name,
            'sequence': seq,
            'soluprot': soluprot_score,
            'esm': esm_score,
            'combined': combined_score
        })

    results.sort(key=lambda x: x['combined'], reverse=True)
    return results
```

### 步驟 3：序列優化（如需要）

```python
def optimize_problematic_sequences(sequences_needing_optimization):
    """
    使用 SolubleMPNN 重設計問題序列
    回傳改進的變體
    """
    optimized = []
    for name, seq in sequences_needing_optimization:
        # 生成多個變體
        variants = optimize_sequence(
            sequence=seq,
            num_variants=10,
            temperature=0.2
        )

        # 使用 ESM 評分變體
        for variant in variants:
            variant['esm_score'] = score_sequence_esm(variant['sequence'])

        # 保留最佳變體
        variants.sort(
            key=lambda x: x['solubility_score'] * x['esm_score'],
            reverse=True
        )

        optimized.extend(variants[:3])  # 每個序列的前 3 個變體

    return optimized
```

### 步驟 4：基於結構的驗證（針對關鍵序列）

```python
def structure_validation(top_candidates):
    """
    預測結構並計算頂級候選者的 pSAE
    實驗測試前的最終驗證
    """
    validated = []
    for candidate in top_candidates:
        # 使用 AlphaFold 預測結構
        structure_pdb = predict_structure_alphafold(candidate['sequence'])

        # 計算 pSAE
        psae = calculate_psae(structure_pdb)

        candidate['psae'] = psae
        candidate['pass_structure_check'] = psae < 30

        validated.append(candidate)

    return validated
```

### 完整工作流程範例

```python
def complete_optimization_pipeline(initial_sequences):
    """
    端對端優化流程

    輸入：{name: sequence} 的字典
    輸出：優化、驗證序列的排名列表
    """

    print("步驟 1：使用 NetSolP 進行初始篩選...")
    filtered = initial_screening(initial_sequences)
    print(f"  通過：{len(filtered)}/{len(initial_sequences)}")

    print("步驟 2：使用 SoluProt 和 ESM 進行詳細評估...")
    assessed = detailed_assessment(filtered)

    # 分為良好和需優化
    good_sequences = [s for s in assessed if s['soluprot'] > 0.6]
    needs_optimization = [s for s in assessed if s['soluprot'] <= 0.6]

    print(f"  良好序列：{len(good_sequences)}")
    print(f"  需要優化：{len(needs_optimization)}")

    if needs_optimization:
        print("步驟 3：使用 SolubleMPNN 優化問題序列...")
        optimized = optimize_problematic_sequences(needs_optimization)
        all_sequences = good_sequences + optimized
    else:
        all_sequences = good_sequences

    print("步驟 4：針對頂級候選者進行基於結構的驗證...")
    top_20 = all_sequences[:20]
    final_validated = structure_validation(top_20)

    # 最終排名
    final_validated.sort(
        key=lambda x: (
            x['pass_structure_check'],
            x['combined'],
            -x['psae']
        ),
        reverse=True
    )

    return final_validated

# 使用方式
initial_library = {
    'variant_1': 'MKVLWAALLGLLGAAA...',
    'variant_2': 'MATGVLWAALLGLLGA...',
    # ... 更多序列
}

optimized_library = complete_optimization_pipeline(initial_library)

# 提交頂級序列至 Adaptyv
top_sequences_for_testing = optimized_library[:50]
```

## 最佳實務摘要

1. **總是在實驗測試前進行預篩選**
2. **首先使用 NetSolP** 快速篩選大型文庫
3. **應用 SolubleMPNN** 作為預設優化工具
4. **使用 ESM 驗證** 以避免不自然的序列
5. **計算 pSAE** 進行基於結構的驗證
6. **每個設計測試多個變體** 以考慮預測的不確定性
7. **保留對照組** - 包含野生型或已知良好的序列
8. **迭代** - 使用實驗結果來精進預測

## 與 Adaptyv 整合

計算優化後，提交序列至 Adaptyv：

```python
# 優化流程後
optimized_sequences = complete_optimization_pipeline(initial_library)

# 準備 FASTA 格式
fasta_content = ""
for seq_data in optimized_sequences[:50]:  # 前 50 名
    fasta_content += f">{seq_data['name']}\n{seq_data['sequence']}\n"

# 提交至 Adaptyv
import requests
response = requests.post(
    "https://kq5jp7qj7wdqklhsxmovkzn4l40obksv.lambda-url.eu-central-1.on.aws/experiments",
    headers={"Authorization": f"Bearer {api_key}"},
    json={
        "sequences": fasta_content,
        "experiment_type": "expression",
        "metadata": {
            "optimization_method": "SolubleMPNN_ESM_pipeline",
            "computational_scores": [s['combined'] for s in optimized_sequences[:50]]
        }
    }
)
```

## 故障排除

**問題：所有序列在溶解度預測上都得分很低**
- 檢查序列是否包含不尋常的胺基酸
- 驗證 FASTA 格式是否正確
- 考慮蛋白質家族是否本身溶解度就低
- 儘管預測不佳，可能需要實驗驗證

**問題：SolubleMPNN 改變了功能重要的殘基**
- 提供結構檔案以保留空間約束
- 遮蔽關鍵殘基使其不被突變
- 降低溫度參數以進行保守變更
- 手動還原有問題的突變

**問題：優化後 ESM 分數低**
- 優化可能過於激進
- 嘗試在 SolubleMPNN 中使用較低的溫度
- 在溶解度和自然性之間取得平衡
- 考慮某些優化可能需要非自然突變

**問題：預測與實驗結果不符**
- 預測是機率性的，不是確定性的
- 宿主系統和條件會影響表現
- 某些蛋白質可能需要實驗驗證
- 將預測用作富集而非絕對篩選
