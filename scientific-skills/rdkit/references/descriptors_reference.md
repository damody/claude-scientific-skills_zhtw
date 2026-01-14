# RDKit 分子描述子參考

RDKit `Descriptors` 模組中可用分子描述子的完整參考。

## 用法

```python
from rdkit import Chem
from rdkit.Chem import Descriptors

mol = Chem.MolFromSmiles('CCO')

# 計算單一描述子
mw = Descriptors.MolWt(mol)

# 一次計算所有描述子
all_desc = Descriptors.CalcMolDescriptors(mol)
```

## 分子量和質量

### MolWt
分子的平均分子量。
```python
Descriptors.MolWt(mol)
```

### ExactMolWt
使用同位素組成的精確分子量。
```python
Descriptors.ExactMolWt(mol)
```

### HeavyAtomMolWt
忽略氫的平均分子量。
```python
Descriptors.HeavyAtomMolWt(mol)
```

## 親脂性

### MolLogP
Wildman-Crippen LogP（辛醇-水分配係數）。
```python
Descriptors.MolLogP(mol)
```

### MolMR
Wildman-Crippen 莫耳折射率。
```python
Descriptors.MolMR(mol)
```

## 極性表面積

### TPSA
基於片段貢獻的拓撲極性表面積（TPSA）。
```python
Descriptors.TPSA(mol)
```

### LabuteASA
Labute 近似表面積（ASA）。
```python
Descriptors.LabuteASA(mol)
```

## 氫鍵

### NumHDonors
氫鍵供體數量（N-H 和 O-H）。
```python
Descriptors.NumHDonors(mol)
```

### NumHAcceptors
氫鍵受體數量（N 和 O）。
```python
Descriptors.NumHAcceptors(mol)
```

### NOCount
N 和 O 原子數量。
```python
Descriptors.NOCount(mol)
```

### NHOHCount
N-H 和 O-H 鍵數量。
```python
Descriptors.NHOHCount(mol)
```

## 原子計數

### HeavyAtomCount
重原子數量（非氫）。
```python
Descriptors.HeavyAtomCount(mol)
```

### NumHeteroatoms
雜原子數量（非 C 和非 H）。
```python
Descriptors.NumHeteroatoms(mol)
```

### NumValenceElectrons
價電子總數。
```python
Descriptors.NumValenceElectrons(mol)
```

### NumRadicalElectrons
自由基電子數量。
```python
Descriptors.NumRadicalElectrons(mol)
```

## 環描述子

### RingCount
環數量。
```python
Descriptors.RingCount(mol)
```

### NumAromaticRings
芳香環數量。
```python
Descriptors.NumAromaticRings(mol)
```

### NumSaturatedRings
飽和環數量。
```python
Descriptors.NumSaturatedRings(mol)
```

### NumAliphaticRings
脂肪族（非芳香）環數量。
```python
Descriptors.NumAliphaticRings(mol)
```

### NumAromaticCarbocycles
芳香碳環數量（僅含碳的環）。
```python
Descriptors.NumAromaticCarbocycles(mol)
```

### NumAromaticHeterocycles
芳香雜環數量（含雜原子的環）。
```python
Descriptors.NumAromaticHeterocycles(mol)
```

### NumSaturatedCarbocycles
飽和碳環數量。
```python
Descriptors.NumSaturatedCarbocycles(mol)
```

### NumSaturatedHeterocycles
飽和雜環數量。
```python
Descriptors.NumSaturatedHeterocycles(mol)
```

### NumAliphaticCarbocycles
脂肪族碳環數量。
```python
Descriptors.NumAliphaticCarbocycles(mol)
```

### NumAliphaticHeterocycles
脂肪族雜環數量。
```python
Descriptors.NumAliphaticHeterocycles(mol)
```

## 可旋轉鍵

### NumRotatableBonds
可旋轉鍵數量（柔性）。
```python
Descriptors.NumRotatableBonds(mol)
```

## 芳香原子

### NumAromaticAtoms
芳香原子數量。
```python
Descriptors.NumAromaticAtoms(mol)
```

## 比例描述子

### FractionCsp3
sp3 雜化碳的比例。
```python
Descriptors.FractionCsp3(mol)
```

## 複雜度描述子

### BertzCT
Bertz 複雜度指數。
```python
Descriptors.BertzCT(mol)
```

### Ipc
資訊含量（複雜度度量）。
```python
Descriptors.Ipc(mol)
```

## Kappa 形狀指數

基於圖不變量的分子形狀描述子。

### Kappa1
第一 kappa 形狀指數。
```python
Descriptors.Kappa1(mol)
```

### Kappa2
第二 kappa 形狀指數。
```python
Descriptors.Kappa2(mol)
```

### Kappa3
第三 kappa 形狀指數。
```python
Descriptors.Kappa3(mol)
```

## Chi 連接性指數

分子連接性指數。

### Chi0, Chi1, Chi2, Chi3, Chi4
簡單 chi 連接性指數。
```python
Descriptors.Chi0(mol)
Descriptors.Chi1(mol)
Descriptors.Chi2(mol)
Descriptors.Chi3(mol)
Descriptors.Chi4(mol)
```

### Chi0n, Chi1n, Chi2n, Chi3n, Chi4n
價態修正 chi 連接性指數。
```python
Descriptors.Chi0n(mol)
Descriptors.Chi1n(mol)
Descriptors.Chi2n(mol)
Descriptors.Chi3n(mol)
Descriptors.Chi4n(mol)
```

### Chi0v, Chi1v, Chi2v, Chi3v, Chi4v
價態 chi 連接性指數。
```python
Descriptors.Chi0v(mol)
Descriptors.Chi1v(mol)
Descriptors.Chi2v(mol)
Descriptors.Chi3v(mol)
Descriptors.Chi4v(mol)
```

## Hall-Kier Alpha

### HallKierAlpha
Hall-Kier alpha 值（分子柔性）。
```python
Descriptors.HallKierAlpha(mol)
```

## Balaban J 指數

### BalabanJ
Balaban J 指數（分支描述子）。
```python
Descriptors.BalabanJ(mol)
```

## EState 指數

電拓撲態指數。

### MaxEStateIndex
最大 E-state 值。
```python
Descriptors.MaxEStateIndex(mol)
```

### MinEStateIndex
最小 E-state 值。
```python
Descriptors.MinEStateIndex(mol)
```

### MaxAbsEStateIndex
最大絕對 E-state 值。
```python
Descriptors.MaxAbsEStateIndex(mol)
```

### MinAbsEStateIndex
最小絕對 E-state 值。
```python
Descriptors.MinAbsEStateIndex(mol)
```

## 部分電荷

### MaxPartialCharge
最大部分電荷。
```python
Descriptors.MaxPartialCharge(mol)
```

### MinPartialCharge
最小部分電荷。
```python
Descriptors.MinPartialCharge(mol)
```

### MaxAbsPartialCharge
最大絕對部分電荷。
```python
Descriptors.MaxAbsPartialCharge(mol)
```

### MinAbsPartialCharge
最小絕對部分電荷。
```python
Descriptors.MinAbsPartialCharge(mol)
```

## 指紋密度

測量分子指紋的密度。

### FpDensityMorgan1
半徑 1 的 Morgan 指紋密度。
```python
Descriptors.FpDensityMorgan1(mol)
```

### FpDensityMorgan2
半徑 2 的 Morgan 指紋密度。
```python
Descriptors.FpDensityMorgan2(mol)
```

### FpDensityMorgan3
半徑 3 的 Morgan 指紋密度。
```python
Descriptors.FpDensityMorgan3(mol)
```

## PEOE VSA 描述子

軌域電負度部分均衡（PEOE）VSA 描述子。

### PEOE_VSA1 到 PEOE_VSA14
使用部分電荷和表面積貢獻的 MOE 類型描述子。
```python
Descriptors.PEOE_VSA1(mol)
# ... 到 PEOE_VSA14
```

## SMR VSA 描述子

莫耳折射率 VSA 描述子。

### SMR_VSA1 到 SMR_VSA10
使用 MR 貢獻和表面積的 MOE 類型描述子。
```python
Descriptors.SMR_VSA1(mol)
# ... 到 SMR_VSA10
```

## SLogP VSA 描述子

LogP VSA 描述子。

### SLogP_VSA1 到 SLogP_VSA12
使用 LogP 貢獻和表面積的 MOE 類型描述子。
```python
Descriptors.SLogP_VSA1(mol)
# ... 到 SLogP_VSA12
```

## EState VSA 描述子

### EState_VSA1 到 EState_VSA11
使用 E-state 指數和表面積的 MOE 類型描述子。
```python
Descriptors.EState_VSA1(mol)
# ... 到 EState_VSA11
```

## VSA 描述子

范德華表面積描述子。

### VSA_EState1 到 VSA_EState10
EState VSA 描述子。
```python
Descriptors.VSA_EState1(mol)
# ... 到 VSA_EState10
```

## BCUT 描述子

Burden-CAS-University of Texas 特徵值描述子。

### BCUT2D_MWHI
以分子量加權的 Burden 矩陣最高特徵值。
```python
Descriptors.BCUT2D_MWHI(mol)
```

### BCUT2D_MWLOW
以分子量加權的 Burden 矩陣最低特徵值。
```python
Descriptors.BCUT2D_MWLOW(mol)
```

### BCUT2D_CHGHI
以部分電荷加權的最高特徵值。
```python
Descriptors.BCUT2D_CHGHI(mol)
```

### BCUT2D_CHGLO
以部分電荷加權的最低特徵值。
```python
Descriptors.BCUT2D_CHGLO(mol)
```

### BCUT2D_LOGPHI
以 LogP 加權的最高特徵值。
```python
Descriptors.BCUT2D_LOGPHI(mol)
```

### BCUT2D_LOGPLOW
以 LogP 加權的最低特徵值。
```python
Descriptors.BCUT2D_LOGPLOW(mol)
```

### BCUT2D_MRHI
以莫耳折射率加權的最高特徵值。
```python
Descriptors.BCUT2D_MRHI(mol)
```

### BCUT2D_MRLOW
以莫耳折射率加權的最低特徵值。
```python
Descriptors.BCUT2D_MRLOW(mol)
```

## 自相關描述子

### AUTOCORR2D
2D 自相關描述子（如果啟用）。
測量性質空間分布的各種自相關指數。

## MQN 描述子

分子量子數 - 42 個簡單描述子。

### mqn1 到 mqn42
計數各種分子特徵的整數描述子。
```python
# 透過 CalcMolDescriptors 存取
desc = Descriptors.CalcMolDescriptors(mol)
mqns = {k: v for k, v in desc.items() if k.startswith('mqn')}
```

## QED

### qed
類藥性定量估計。
```python
Descriptors.qed(mol)
```

## Lipinski 五規則

使用 Lipinski 標準檢查類藥性：

```python
def lipinski_rule_of_five(mol):
    mw = Descriptors.MolWt(mol) <= 500
    logp = Descriptors.MolLogP(mol) <= 5
    hbd = Descriptors.NumHDonors(mol) <= 5
    hba = Descriptors.NumHAcceptors(mol) <= 10
    return mw and logp and hbd and hba
```

## 批次描述子計算

一次計算所有描述子：

```python
from rdkit import Chem
from rdkit.Chem import Descriptors

mol = Chem.MolFromSmiles('CCO')

# 取得所有描述子為字典
all_descriptors = Descriptors.CalcMolDescriptors(mol)

# 存取特定描述子
mw = all_descriptors['MolWt']
logp = all_descriptors['MolLogP']

# 取得可用描述子名稱列表
from rdkit.Chem import Descriptors
descriptor_names = [desc[0] for desc in Descriptors._descList]
```

## 描述子類別總結

1. **物理化學**：MolWt、MolLogP、MolMR、TPSA
2. **拓撲**：BertzCT、BalabanJ、Kappa 指數
3. **電子**：部分電荷、E-state 指數
4. **形狀**：Kappa 指數、BCUT 描述子
5. **連接性**：Chi 指數
6. **2D 指紋**：FpDensity 描述子
7. **原子計數**：重原子、雜原子、環
8. **類藥性**：QED、Lipinski 參數
9. **柔性**：NumRotatableBonds、HallKierAlpha
10. **表面積**：VSA 基礎描述子

## 常見用例

### 類藥性篩選

```python
def screen_druglikeness(mol):
    return {
        'MW': Descriptors.MolWt(mol),
        'LogP': Descriptors.MolLogP(mol),
        'HBD': Descriptors.NumHDonors(mol),
        'HBA': Descriptors.NumHAcceptors(mol),
        'TPSA': Descriptors.TPSA(mol),
        'RotBonds': Descriptors.NumRotatableBonds(mol),
        'AromaticRings': Descriptors.NumAromaticRings(mol),
        'QED': Descriptors.qed(mol)
    }
```

### 先導類過濾

```python
def is_leadlike(mol):
    mw = 250 <= Descriptors.MolWt(mol) <= 350
    logp = Descriptors.MolLogP(mol) <= 3.5
    rot_bonds = Descriptors.NumRotatableBonds(mol) <= 7
    return mw and logp and rot_bonds
```

### 多樣性分析

```python
def molecular_complexity(mol):
    return {
        'BertzCT': Descriptors.BertzCT(mol),
        'NumRings': Descriptors.RingCount(mol),
        'NumRotBonds': Descriptors.NumRotatableBonds(mol),
        'FractionCsp3': Descriptors.FractionCsp3(mol),
        'NumAromaticRings': Descriptors.NumAromaticRings(mol)
    }
```

## 提示

1. **使用批次計算**以避免多個描述子的冗餘計算
2. **檢查 None** - 某些描述子可能對無效分子返回 None
3. **正規化描述子**用於機器學習應用
4. **選擇相關描述子** - 並非所有 200+ 個描述子都對每個任務有用
5. **單獨考慮 3D 描述子**（需要 3D 座標）
6. **驗證範圍** - 檢查描述子值是否在預期範圍內
