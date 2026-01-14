# PyHealth 醫療代碼轉換

## 概述

醫療資料使用多種編碼系統與標準。PyHealth 的 MedCode 模組透過本體查詢與跨系統對應，實現醫療編碼系統之間的轉換與對應。

## 核心類別

### InnerMap
處理系統內的本體查詢與階層導航。

**關鍵功能：**
- 帶屬性（名稱、描述）的代碼查詢
- 祖先/後代階層遍歷
- 代碼標準化與轉換
- 父子關係導航

### CrossMap
管理不同編碼標準之間的跨系統對應。

**關鍵功能：**
- 編碼系統之間的轉換
- 多對多關係處理
- 階層層級指定（用於藥物）
- 雙向對應支援

## 支援的編碼系統

### 診斷代碼

**ICD-9-CM（國際疾病分類第九版臨床修訂版）**
- 舊版診斷編碼系統
- 具有 3-5 位數代碼的階層結構
- 2015 年前在美國醫療系統使用
- 使用方式：`from pyhealth.medcode import InnerMap`
  - `icd9_map = InnerMap.load("ICD9CM")`

**ICD-10-CM（國際疾病分類第十版臨床修訂版）**
- 目前的診斷編碼標準
- 英數字代碼（3-7 個字元）
- 比 ICD-9 更精細
- 使用方式：`from pyhealth.medcode import InnerMap`
  - `icd10_map = InnerMap.load("ICD10CM")`

**CCSCM（ICD-CM 臨床分類軟體）**
- 將 ICD 代碼分組為具臨床意義的類別
- 降低分析維度
- 單層與多層階層
- 使用方式：`from pyhealth.medcode import CrossMap`
  - `icd_to_ccs = CrossMap.load("ICD9CM", "CCSCM")`

### 處置代碼

**ICD-9-PROC（ICD-9 處置代碼）**
- 住院處置分類
- 3-4 位數數字代碼
- 舊版系統（2015 年前）
- 使用方式：`from pyhealth.medcode import InnerMap`
  - `icd9proc_map = InnerMap.load("ICD9PROC")`

**ICD-10-PROC（ICD-10 處置編碼系統）**
- 目前的處置編碼標準
- 7 個字元的英數字代碼
- 比 ICD-9-PROC 更詳細
- 使用方式：`from pyhealth.medcode import InnerMap`
  - `icd10proc_map = InnerMap.load("ICD10PROC")`

**CCSPROC（處置臨床分類軟體）**
- 將處置代碼分組為類別
- 簡化處置分析
- 使用方式：`from pyhealth.medcode import CrossMap`
  - `proc_to_ccs = CrossMap.load("ICD9PROC", "CCSPROC")`

### 藥物代碼

**NDC（國家藥物代碼）**
- 美國 FDA 藥物識別系統
- 10 或 11 位數代碼
- 產品層級特異性（製造商、劑量、包裝）
- 使用方式：`from pyhealth.medcode import InnerMap`
  - `ndc_map = InnerMap.load("NDC")`

**RxNorm**
- 標準化藥物術語
- 標準化藥物名稱與關係
- 連結多個藥物詞彙表
- 使用方式：`from pyhealth.medcode import CrossMap`
  - `ndc_to_rxnorm = CrossMap.load("NDC", "RXNORM")`

**ATC（解剖學治療學化學分類系統）**
- WHO 藥物分類系統
- 5 層階層：
  - **第 1 層**：解剖學主要群組（1 個字母）
  - **第 2 層**：治療亞群（2 位數）
  - **第 3 層**：藥理亞群（1 個字母）
  - **第 4 層**：化學亞群（1 個字母）
  - **第 5 層**：化學物質（2 位數）
- 範例："C03CA01" = Furosemide（呋塞米）
  - C = 心血管系統
  - C03 = 利尿劑
  - C03C = 高效利尿劑
  - C03CA = 磺胺類
  - C03CA01 = Furosemide

**使用方式：**
```python
from pyhealth.medcode import CrossMap
ndc_to_atc = CrossMap.load("NDC", "ATC")
atc_codes = ndc_to_atc.map("00074-3799-13", level=3)  # 取得 ATC 第 3 層
```

## 常見操作

### InnerMap 操作

**1. 代碼查詢**
```python
from pyhealth.medcode import InnerMap

icd9_map = InnerMap.load("ICD9CM")
info = icd9_map.lookup("428.0")  # 心臟衰竭
# 回傳：名稱、描述、額外屬性
```

**2. 祖先遍歷**
```python
# 取得階層中的所有父代碼
ancestors = icd9_map.get_ancestors("428.0")
# 回傳：["428", "420-429", "390-459"]
```

**3. 後代遍歷**
```python
# 取得所有子代碼
descendants = icd9_map.get_descendants("428")
# 回傳：["428.0", "428.1", "428.2", ...]
```

**4. 代碼標準化**
```python
# 標準化代碼格式
standard_code = icd9_map.standardize("4280")  # 回傳 "428.0"
```

### CrossMap 操作

**1. 直接轉換**
```python
from pyhealth.medcode import CrossMap

# ICD-9-CM 轉 CCS
icd_to_ccs = CrossMap.load("ICD9CM", "CCSCM")
ccs_codes = icd_to_ccs.map("82101")  # 冠狀動脈粥狀硬化
# 回傳：["101"]  # 冠狀動脈粥狀硬化的 CCS 類別
```

**2. 階層藥物對應**
```python
# 不同層級的 NDC 轉 ATC
ndc_to_atc = CrossMap.load("NDC", "ATC")

# 取得特定 ATC 層級
atc_level_1 = ndc_to_atc.map("00074-3799-13", level=1)  # 解剖學群組
atc_level_3 = ndc_to_atc.map("00074-3799-13", level=3)  # 藥理學
atc_level_5 = ndc_to_atc.map("00074-3799-13", level=5)  # 化學物質
```

**3. 雙向對應**
```python
# 任一方向的對應
rxnorm_to_ndc = CrossMap.load("RXNORM", "NDC")
ndc_codes = rxnorm_to_ndc.map("197381")  # 取得 RxNorm 對應的所有 NDC 代碼
```

## 工作流程範例

### 範例 1：標準化並分組診斷
```python
from pyhealth.medcode import InnerMap, CrossMap

# 載入對應表
icd9_map = InnerMap.load("ICD9CM")
icd_to_ccs = CrossMap.load("ICD9CM", "CCSCM")

# 處理診斷代碼
raw_codes = ["4280", "428.0", "42800"]

standardized = [icd9_map.standardize(code) for code in raw_codes]
# 全部變成 "428.0"

ccs_categories = [icd_to_ccs.map(code)[0] for code in standardized]
# 全部對應到 CCS 類別 "108"（心臟衰竭）
```

### 範例 2：藥物分類分析
```python
from pyhealth.medcode import CrossMap

# 將 NDC 對應到 ATC 進行藥物類別分析
ndc_to_atc = CrossMap.load("NDC", "ATC")

patient_drugs = ["00074-3799-13", "00074-7286-01", "00456-0765-01"]

# 取得治療亞群（ATC 第 2 層）
drug_classes = []
for ndc in patient_drugs:
    atc_codes = ndc_to_atc.map(ndc, level=2)
    if atc_codes:
        drug_classes.append(atc_codes[0])

# 分析藥物類別分佈
```

### 範例 3：ICD-9 轉 ICD-10 遷移
```python
from pyhealth.medcode import CrossMap

# 載入 ICD-9 轉 ICD-10 對應
icd9_to_icd10 = CrossMap.load("ICD9CM", "ICD10CM")

# 轉換歷史 ICD-9 代碼
icd9_code = "428.0"
icd10_codes = icd9_to_icd10.map(icd9_code)
# 回傳：["I50.9", "I50.1", ...]  # 多個可能的 ICD-10 代碼

# 處理一對多對應
for icd10_code in icd10_codes:
    print(f"ICD-9 {icd9_code} -> ICD-10 {icd10_code}")
```

## 與資料集整合

醫療代碼轉換與 PyHealth 資料集無縫整合：

```python
from pyhealth.datasets import MIMIC4Dataset
from pyhealth.medcode import CrossMap

# 載入資料集
dataset = MIMIC4Dataset(root="/path/to/data")

# 載入代碼對應
icd_to_ccs = CrossMap.load("ICD10CM", "CCSCM")

# 處理病人診斷
for patient in dataset.iter_patients():
    for visit in patient.visits:
        diagnosis_events = [e for e in visit.events if e.vocabulary == "ICD10CM"]

        for event in diagnosis_events:
            ccs_codes = icd_to_ccs.map(event.code)
            print(f"診斷 {event.code} -> CCS {ccs_codes}")
```

## 使用情境

### 臨床研究
- 跨不同編碼系統標準化診斷
- 分組相關疾病以識別隊列
- 協調使用不同標準的多中心研究

### 藥物安全分析
- 按治療類別分類藥物
- 在類別層級識別藥物交互作用
- 分析多重用藥模式

### 醫療分析
- 降低診斷/處置維度
- 建立有意義的臨床類別
- 跨編碼系統變更進行縱向分析

### 機器學習
- 建立一致的特徵表示
- 處理訓練/測試資料中的詞彙不匹配
- 生成階層嵌入

## 最佳實務

1. **對應前務必標準化代碼**以確保格式一致
2. **適當處理一對多對應**（某些代碼對應到多個目標）
3. 對應藥物時**明確指定 ATC 層級**以避免歧義
4. **使用 CCS 類別**降低診斷/處置維度
5. **驗證對應結果**因為某些代碼可能沒有直接轉換
6. **記錄代碼版本**（ICD-9 vs ICD-10）以維護資料來源
