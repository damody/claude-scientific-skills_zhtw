# Medchem 規則和篩選器目錄

medchem 中所有可用藥物化學規則、結構警示和篩選器的完整目錄。

## 目錄

1. [類藥性規則](#類藥性規則)
2. [先導化合物類規則](#先導化合物類規則)
3. [片段規則](#片段規則)
4. [CNS 規則](#cns-規則)
5. [結構警示篩選器](#結構警示篩選器)
6. [化學基團模式](#化學基團模式)

---

## 類藥性規則

### 五規則（Rule of Five，Lipinski）

**參考文獻：** Lipinski et al., Adv Drug Deliv Rev (1997) 23:3-25

**目的：** 預測口服生物利用度

**標準：**
- 分子量 ≤ 500 Da
- LogP ≤ 5
- 氫鍵供體 ≤ 5
- 氫鍵受體 ≤ 10

**用法：**
```python
mc.rules.basic_rules.rule_of_five(mol)
```

**備註：**
- 藥物發現中最廣泛使用的篩選器之一
- 約 90% 的口服活性藥物符合這些規則
- 存在例外，特別是天然產物和抗生素

---

### Veber 規則（Rule of Veber）

**參考文獻：** Veber et al., J Med Chem (2002) 45:2615-2623

**目的：** 口服生物利用度的額外標準

**標準：**
- 可旋轉鍵 ≤ 10
- 拓撲極性表面積（TPSA）≤ 140 Å²

**用法：**
```python
mc.rules.basic_rules.rule_of_veber(mol)
```

**備註：**
- 補充五規則
- TPSA 與細胞滲透性相關
- 可旋轉鍵影響分子柔性

---

### 藥物規則（Rule of Drug）

**目的：** 組合的類藥性評估

**標準：**
- 通過五規則
- 通過 Veber 規則
- 不包含 PAINS 子結構

**用法：**
```python
mc.rules.basic_rules.rule_of_drug(mol)
```

---

### REOS（Rapid Elimination Of Swill）

**參考文獻：** Walters & Murcko, Adv Drug Deliv Rev (2002) 54:255-271

**目的：** 篩選掉不太可能成為藥物的化合物

**標準：**
- 分子量：200-500 Da
- LogP：-5 到 5
- 氫鍵供體：0-5
- 氫鍵受體：0-10

**用法：**
```python
mc.rules.basic_rules.rule_of_reos(mol)
```

---

### 黃金三角形（Golden Triangle）

**參考文獻：** Johnson et al., J Med Chem (2009) 52:5487-5500

**目的：** 平衡親脂性和分子量

**標準：**
- 200 ≤ MW ≤ 50 × LogP + 400
- LogP：-2 到 5

**用法：**
```python
mc.rules.basic_rules.golden_triangle(mol)
```

**備註：**
- 定義最佳的物理化學空間
- 在 MW vs LogP 圖上視覺表示類似三角形

---

## 先導化合物類規則

### Oprea 規則（Rule of Oprea）

**參考文獻：** Oprea et al., J Chem Inf Comput Sci (2001) 41:1308-1315

**目的：** 識別用於最佳化的先導化合物類化合物

**標準：**
- 分子量：200-350 Da
- LogP：-2 到 4
- 可旋轉鍵 ≤ 7
- 環數 ≤ 4

**用法：**
```python
mc.rules.basic_rules.rule_of_oprea(mol)
```

**原理：** 先導化合物應該在最佳化過程中有「成長空間」

---

### 先導化合物類規則（軟性）

**目的：** 寬鬆的先導化合物類標準

**標準：**
- 分子量：250-450 Da
- LogP：-3 到 4
- 可旋轉鍵 ≤ 10

**用法：**
```python
mc.rules.basic_rules.rule_of_leadlike_soft(mol)
```

---

### 先導化合物類規則（嚴格）

**目的：** 嚴格的先導化合物類標準

**標準：**
- 分子量：200-350 Da
- LogP：-2 到 3.5
- 可旋轉鍵 ≤ 7
- 環數：1-3

**用法：**
```python
mc.rules.basic_rules.rule_of_leadlike_strict(mol)
```

---

## 片段規則

### 三規則（Rule of Three）

**參考文獻：** Congreve et al., Drug Discov Today (2003) 8:876-877

**目的：** 篩選片段庫，用於基於片段的藥物發現（fragment-based drug discovery）

**標準：**
- 分子量 ≤ 300 Da
- LogP ≤ 3
- 氫鍵供體 ≤ 3
- 氫鍵受體 ≤ 3
- 可旋轉鍵 ≤ 3
- 極性表面積 ≤ 60 Å²

**用法：**
```python
mc.rules.basic_rules.rule_of_three(mol)
```

**備註：**
- 片段在最佳化過程中被發展成先導化合物
- 較低的複雜性允許更多的起點

---

## CNS 規則

### CNS 規則（Rule of CNS）

**目的：** 中樞神經系統類藥性

**標準：**
- 分子量 ≤ 450 Da
- LogP：-1 到 5
- 氫鍵供體 ≤ 2
- TPSA ≤ 90 Å²

**用法：**
```python
mc.rules.basic_rules.rule_of_cns(mol)
```

**原理：**
- 血腦屏障（blood-brain barrier）穿透需要特定性質
- 較低的 TPSA 和 HBD 計數改善 BBB 滲透性
- 嚴格的約束反映 CNS 的挑戰

---

## 結構警示篩選器

### PAINS（Pan Assay INterference compoundS）

**參考文獻：** Baell & Holloway, J Med Chem (2010) 53:2719-2740

**目的：** 識別干擾檢測的化合物

**類別：**
- 鄰苯二酚（Catechols）
- 醌類（Quinones）
- 羅丹寧（Rhodanines）
- 羥基苯基腙（Hydroxyphenylhydrazones）
- 烷基/芳基醛（Alkyl/aryl aldehydes）
- 麥可受體（特定模式）

**用法：**
```python
mc.rules.basic_rules.pains_filter(mol)
# 如果未發現 PAINS 則返回 True
```

**備註：**
- PAINS 化合物通過非特異性機制在多個檢測中顯示活性
- 篩選活動中常見的假陽性
- 應在先導化合物選擇中降低優先級

---

### 常見警示篩選器（Common Alerts Filters）

**來源：** 源自 ChEMBL 策展和藥物化學文獻

**目的：** 標記常見的有問題結構模式

**警示類別：**
1. **反應性基團**
   - 環氧化物（Epoxides）
   - 氮雜環丙烷（Aziridines）
   - 醯鹵（Acid halides）
   - 異氰酸酯（Isocyanates）

2. **代謝不穩定性**
   - 聯氨（Hydrazines）
   - 硫脲（Thioureas）
   - 苯胺（某些模式）

3. **聚集體**
   - 多芳環系統
   - 長脂肪鏈

4. **毒性基團**
   - 硝基芳烴
   - 芳香族 N-氧化物
   - 某些雜環

**用法：**
```python
alert_filter = mc.structural.CommonAlertsFilters()
has_alerts, details = alert_filter.check_mol(mol)
```

**返回格式：**
```python
{
    "has_alerts": True,
    "alert_details": ["reactive_epoxide", "metabolic_hydrazine"],
    "num_alerts": 2
}
```

---

### NIBR 篩選器

**來源：** Novartis Institutes for BioMedical Research

**目的：** 工業藥物化學篩選規則

**特點：**
- 從 Novartis 經驗開發的專有篩選器集
- 平衡類藥性與實用藥物化學
- 包括結構警示和性質篩選器

**用法：**
```python
nibr_filter = mc.structural.NIBRFilters()
results = nibr_filter(mols=mol_list, n_jobs=-1)
```

**返回格式：** 布林值列表（True = 通過）

---

### Lilly 扣分篩選器

**參考文獻：** 基於 Eli Lilly 藥物化學規則

**來源：** 18 年累積的 275 個結構模式

**目的：** 識別檢測干擾和有問題的功能

**機制：**
- 每個匹配的模式增加扣分
- 扣分 >100 的分子被拒絕
- 某些模式增加 10-50 分，其他模式增加 100+（立即拒絕）

**扣分類別：**

1. **高扣分（>50）：**
   - 已知的毒性基團
   - 高反應性功能
   - 強金屬螯合劑

2. **中扣分（20-50）：**
   - 代謝不穩定性
   - 易聚集結構
   - 頻繁命中者

3. **低扣分（5-20）：**
   - 小問題
   - 依賴情境的問題

**用法：**
```python
lilly_filter = mc.structural.LillyDemeritsFilters()
results = lilly_filter(mols=mol_list, n_jobs=-1)
```

**返回格式：**
```python
{
    "demerits": 35,
    "passes": True,  # （扣分 ≤ 100）
    "matched_patterns": [
        {"pattern": "phenolic_ester", "demerits": 20},
        {"pattern": "aniline_derivative", "demerits": 15}
    ]
}
```

---

## 化學基團模式

### 鉸鏈結合物（Hinge Binders）

**目的：** 識別激酶鉸鏈結合基序

**常見模式：**
- 氨基吡啶
- 氨基嘧啶
- 吲唑
- 苯并咪唑

**用法：**
```python
group = mc.groups.ChemicalGroup(groups=["hinge_binders"])
has_hinge = group.has_match(mol_list)
```

**應用：** 激酶抑制劑設計

---

### 磷酸結合物（Phosphate Binders）

**目的：** 識別磷酸結合基團

**常見模式：**
- 特定幾何構型中的鹼性胺
- 胍基
- 精胺酸模擬物

**用法：**
```python
group = mc.groups.ChemicalGroup(groups=["phosphate_binders"])
```

**應用：** 激酶抑制劑、磷酸酶抑制劑

---

### 麥可受體（Michael Acceptors）

**目的：** 識別親電子麥可受體基團

**常見模式：**
- α,β-不飽和羰基
- α,β-不飽和腈
- 乙烯基碸
- 丙烯醯胺

**用法：**
```python
group = mc.groups.ChemicalGroup(groups=["michael_acceptors"])
```

**備註：**
- 對於共價抑制劑可能是理想的
- 在篩選中經常被標記為反應性警示

---

### 反應性基團（Reactive Groups）

**目的：** 識別一般反應性功能

**常見模式：**
- 環氧化物
- 氮雜環丙烷
- 醯鹵
- 異氰酸酯
- 磺醯氯

**用法：**
```python
group = mc.groups.ChemicalGroup(groups=["reactive_groups"])
```

---

## 自訂 SMARTS 模式

使用 SMARTS 定義自訂結構模式：

```python
custom_patterns = {
    "my_warhead": "[C;H0](=O)C(F)(F)F",  # 三氟甲基酮
    "my_scaffold": "c1ccc2c(c1)ncc(n2)N",  # 氨基苯并咪唑
}

group = mc.groups.ChemicalGroup(
    groups=["hinge_binders"],
    custom_smarts=custom_patterns
)
```

---

## 篩選器選擇指南

### 初始篩選（高通量）

建議篩選器：
- 五規則
- PAINS 篩選器
- 常見警示（寬鬆設定）

```python
rfilter = mc.rules.RuleFilters(rule_list=["rule_of_five", "pains_filter"])
alert_filter = mc.structural.CommonAlertsFilters()
```

---

### 苗頭化合物到先導化合物（Hit-to-Lead）

建議篩選器：
- Oprea 規則或先導化合物類（軟性）
- NIBR 篩選器
- Lilly 扣分

```python
rfilter = mc.rules.RuleFilters(rule_list=["rule_of_oprea"])
nibr_filter = mc.structural.NIBRFilters()
lilly_filter = mc.structural.LillyDemeritsFilters()
```

---

### 先導化合物最佳化

建議篩選器：
- 藥物規則
- 先導化合物類（嚴格）
- 完整結構警示分析
- 複雜度篩選器

```python
rfilter = mc.rules.RuleFilters(rule_list=["rule_of_drug", "rule_of_leadlike_strict"])
alert_filter = mc.structural.CommonAlertsFilters()
complexity_filter = mc.complexity.ComplexityFilter(max_complexity=400)
```

---

### CNS 靶標

建議篩選器：
- CNS 規則
- 減少的 PAINS 標準（CNS 專用）
- BBB 滲透性約束

```python
rfilter = mc.rules.RuleFilters(rule_list=["rule_of_cns"])
constraints = mc.constraints.Constraints(
    tpsa_max=90,
    hbd_max=2,
    mw_range=(300, 450)
)
```

---

### 基於片段的藥物發現

建議篩選器：
- 三規則
- 最小複雜度
- 基本反應性基團檢查

```python
rfilter = mc.rules.RuleFilters(rule_list=["rule_of_three"])
complexity_filter = mc.complexity.ComplexityFilter(max_complexity=250)
```

---

## 重要考量

### 假陽性和假陰性

**篩選器是指南，不是絕對規則：**

1. **假陽性**（好藥物被標記）：
   - 約 10% 的上市藥物未通過五規則
   - 天然產物經常違反標準規則
   - 前藥故意打破規則
   - 抗生素和抗病毒藥物經常不合規

2. **假陰性**（壞化合物通過）：
   - 通過篩選器並不保證成功
   - 未捕獲靶標特定問題
   - 體內性質未完全預測

### 情境特定應用

**不同情境需要不同標準：**

- **靶標類別：** 激酶 vs GPCRs vs 離子通道有不同的最佳空間
- **藥物類型：** 小分子 vs PROTACs vs 分子膠
- **給藥途徑：** 口服 vs 靜脈注射 vs 局部
- **疾病領域：** CNS vs 腫瘤學 vs 感染性疾病
- **階段：** 篩選 vs 苗頭化合物到先導化合物 vs 先導化合物最佳化

### 與機器學習結合

現代方法將規則與 ML 結合：

```python
# 基於規則的預篩選
rule_results = mc.rules.RuleFilters(rule_list=["rule_of_five"])(mols)
filtered_mols = [mol for mol, r in zip(mols, rule_results) if r["passes"]]

# 對篩選集進行 ML 模型評分
ml_scores = ml_model.predict(filtered_mols)

# 組合決策
final_candidates = [
    mol for mol, score in zip(filtered_mols, ml_scores)
    if score > threshold
]
```

---

## 參考文獻

1. Lipinski CA et al. Adv Drug Deliv Rev (1997) 23:3-25
2. Veber DF et al. J Med Chem (2002) 45:2615-2623
3. Oprea TI et al. J Chem Inf Comput Sci (2001) 41:1308-1315
4. Congreve M et al. Drug Discov Today (2003) 8:876-877
5. Baell JB & Holloway GA. J Med Chem (2010) 53:2719-2740
6. Johnson TW et al. J Med Chem (2009) 52:5487-5500
7. Walters WP & Murcko MA. Adv Drug Deliv Rev (2002) 54:255-271
8. Hann MM & Oprea TI. Curr Opin Chem Biol (2004) 8:255-263
9. Rishton GM. Drug Discov Today (1997) 2:382-384
