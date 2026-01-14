# PyHealth 臨床預測任務

## 概述

PyHealth 提供 20+ 個預定義的臨床預測任務，用於常見的醫療 AI 應用。每個任務函數將原始病人資料轉換為結構化的輸入-輸出配對，用於模型訓練。

## 任務函數結構

所有任務函數繼承自 `BaseTask` 並提供：

- **input_schema**：定義輸入特徵（診斷、藥物、檢驗等）
- **output_schema**：定義預測目標（標籤、數值）
- **pre_filter()**：可選的病人/就診過濾邏輯

**使用模式：**
```python
from pyhealth.datasets import MIMIC4Dataset
from pyhealth.tasks import mortality_prediction_mimic4_fn

dataset = MIMIC4Dataset(root="/path/to/data")
sample_dataset = dataset.set_task(mortality_prediction_mimic4_fn)
```

## 電子健康記錄（EHR）任務

### 死亡率預測

**目的：** 預測病人在下次就診或指定時間範圍內的死亡風險

**MIMIC-III Mortality**（`mortality_prediction_mimic3_fn`）
- 預測下次住院就診時的死亡
- 二元分類任務
- 輸入：歷史診斷、處置、藥物
- 輸出：二元標籤（死亡/存活）

**MIMIC-IV Mortality**（`mortality_prediction_mimic4_fn`）
- MIMIC-IV 資料集的更新版本
- 增強的特徵集
- 改善的標籤品質

**eICU Mortality**（`mortality_prediction_eicu_fn`）
- 多中心加護病房死亡率預測
- 考慮醫院層級變異

**OMOP Mortality**（`mortality_prediction_omop_fn`）
- 標準化死亡率預測
- 適用於 OMOP 通用資料模型

**In-Hospital Mortality**（`inhospital_mortality_prediction_mimic4_fn`）
- 預測當次住院期間的死亡
- 即時風險評估
- 比下次就診死亡率有更早的預測窗口

**StageNet Mortality**（`mortality_prediction_mimic4_fn_stagenet`）
- StageNet 模型架構專用
- 時間階段感知預測

### 再入院預測

**目的：** 識別在指定時間範圍內（通常 30 天）有再入院風險的病人

**MIMIC-III Readmission**（`readmission_prediction_mimic3_fn`）
- 30 天再入院預測
- 二元分類
- 輸入：診斷歷史、藥物、人口統計
- 輸出：二元標籤（再入院/未再入院）

**MIMIC-IV Readmission**（`readmission_prediction_mimic4_fn`）
- 增強的再入院特徵
- 改善的時間建模

**eICU Readmission**（`readmission_prediction_eicu_fn`）
- 加護病房專用再入院風險
- 多中心資料

**OMOP Readmission**（`readmission_prediction_omop_fn`）
- 標準化再入院預測

### 住院天數預測

**目的：** 估計住院時間長度，用於資源規劃和病人管理

**MIMIC-III Length of Stay**（`length_of_stay_prediction_mimic3_fn`）
- 迴歸任務
- 輸入：入院診斷、生命徵象、人口統計
- 輸出：連續值（天數）

**MIMIC-IV Length of Stay**（`length_of_stay_prediction_mimic4_fn`）
- 增強的住院天數預測特徵
- 更好的時間粒度

**eICU Length of Stay**（`length_of_stay_prediction_eicu_fn`）
- 加護病房停留時間預測
- 多醫院資料

**OMOP Length of Stay**（`length_of_stay_prediction_omop_fn`）
- 標準化住院天數預測

### 藥物推薦

**目的：** 根據病人歷史和當前狀況建議適當的藥物

**MIMIC-III Drug Recommendation**（`drug_recommendation_mimic3_fn`）
- 多標籤分類
- 輸入：診斷、先前藥物、人口統計
- 輸出：推薦藥物代碼集合
- 考慮藥物交互作用

**MIMIC-IV Drug Recommendation**（`drug_recommendation_mimic4_fn`）
- 更新的藥物資料
- 增強的交互作用建模

**eICU Drug Recommendation**（`drug_recommendation_eicu_fn`）
- 重症照護藥物推薦

**OMOP Drug Recommendation**（`drug_recommendation_omop_fn`）
- 標準化藥物推薦

**關鍵考量：**
- 處理多重用藥場景
- 多標籤預測（每位病人多種藥物）
- 可與 SafeDrug/GAMENet 模型整合以獲得安全感知推薦

## 專門臨床任務

### 醫療編碼

**MIMIC-III ICD-9 Coding**（`icd9_coding_mimic3_fn`）
- 將 ICD-9 診斷/處置代碼分配給臨床筆記
- 多標籤文字分類
- 輸入：臨床文字/文件
- 輸出：ICD-9 代碼集合
- 支援診斷和處置編碼

### 病人連結

**MIMIC-III Patient Linking**（`patient_linkage_mimic3_fn`）
- 記錄匹配與去重
- 二元分類（是否為同一病人）
- 輸入：兩筆記錄的人口統計和臨床特徵
- 輸出：匹配機率

## 生理訊號任務

### 睡眠分期

**目的：** 從腦電圖/生理訊號分類睡眠階段，用於睡眠障礙診斷

**ISRUC Sleep Staging**（`sleep_staging_isruc_fn`）
- 多類別分類（清醒、N1、N2、N3、REM）
- 輸入：多通道腦電圖訊號
- 輸出：每個 epoch 的睡眠階段（通常 30 秒）

**SleepEDF Sleep Staging**（`sleep_staging_sleepedf_fn`）
- 標準睡眠分期任務
- PSG 訊號處理

**SHHS Sleep Staging**（`sleep_staging_shhs_fn`）
- 大規模睡眠研究資料
- 族群層級睡眠分析

**標準化標籤：**
- 清醒（W）
- 非快速眼動期第 1 階段（N1）
- 非快速眼動期第 2 階段（N2）
- 非快速眼動期第 3 階段（N3/深度睡眠）
- 快速眼動期（REM）

### 腦電圖分析

**Abnormality Detection**（`abnormality_detection_tuab_fn`）
- 二元分類（正常/異常腦電圖）
- 臨床篩檢應用
- 輸入：多通道腦電圖記錄
- 輸出：二元標籤

**Event Detection**（`event_detection_tuev_fn`）
- 識別特定腦電圖事件（棘波、癲癇發作）
- 多類別分類
- 輸入：腦電圖時間序列
- 輸出：事件類型與時間

**Seizure Detection**（`seizure_detection_tusz_fn`）
- 專門的癲癇發作檢測
- 對癲癇監測至關重要
- 輸入：連續腦電圖
- 輸出：癲癇發作/非癲癇發作分類

## 醫學影像任務

### COVID-19 胸部 X 光分類

**COVID-19 CXR**（`covid_classification_cxr_fn`）
- 多類別影像分類
- 類別：COVID-19、細菌性肺炎、病毒性肺炎、正常
- 輸入：胸部 X 光影像
- 輸出：疾病分類

## 文字任務

### 醫療轉錄分類

**Medical Specialty Classification**（`medical_transcription_classification_fn`）
- 按醫療專科分類臨床筆記
- 多類別文字分類
- 輸入：臨床轉錄文字
- 輸出：醫療專科（心臟科、神經科等）

## 自訂任務建立

### 建立自訂任務

透過指定輸入/輸出 schema 定義自訂預測任務：

```python
from pyhealth.tasks import BaseTask

def custom_task_fn(patient):
    """自訂預測任務"""

    # 定義輸入特徵
    samples = []

    for i, visit in enumerate(patient.visits):
        # 如果歷史不足則跳過
        if i < 2:
            continue

        # 從歷史就診建立輸入
        input_info = {
            "diagnoses": [],
            "medications": [],
            "procedures": []
        }

        # 從先前就診收集特徵
        for past_visit in patient.visits[:i]:
            for event in past_visit.events:
                if event.vocabulary == "ICD10CM":
                    input_info["diagnoses"].append(event.code)
                elif event.vocabulary == "NDC":
                    input_info["medications"].append(event.code)

        # 定義預測目標
        # 範例：預測當次就診的特定結果
        output_info = {
            "label": 1 if some_condition else 0
        }

        samples.append({
            "patient_id": patient.patient_id,
            "visit_id": visit.visit_id,
            "input_info": input_info,
            "output_info": output_info
        })

    return samples

# 套用自訂任務
sample_dataset = dataset.set_task(custom_task_fn)
```

### 任務函數元件

1. **輸入 Schema 定義**
   - 指定要提取的特徵
   - 定義特徵類型（代碼、序列、數值）
   - 設定時間窗口

2. **輸出 Schema 定義**
   - 定義預測目標
   - 設定標籤類型（二元、多類別、多標籤、迴歸）
   - 指定評估指標

3. **過濾邏輯**
   - 排除資料不足的病人/就診
   - 套用納入/排除標準
   - 處理缺失資料

4. **樣本生成**
   - 建立輸入-輸出配對
   - 維護病人/就診識別碼
   - 保持時間順序

## 任務選擇指南

### 臨床預測任務
**使用時機：** 處理結構化 EHR 資料（診斷、藥物、處置）

**資料集：** MIMIC-III、MIMIC-IV、eICU、OMOP

**常見任務：**
- 死亡率預測用於風險分層
- 再入院預測用於照護轉銜規劃
- 住院天數用於資源配置
- 藥物推薦用於臨床決策支援

### 訊號處理任務
**使用時機：** 處理生理時間序列資料

**資料集：** SleepEDF、SHHS、ISRUC、TUEV、TUAB、TUSZ

**常見任務：**
- 睡眠分期用於睡眠障礙診斷
- 腦電圖異常檢測用於篩檢
- 癲癇發作檢測用於癲癇監測

### 影像任務
**使用時機：** 處理醫學影像

**資料集：** COVID-19 CXR

**常見任務：**
- 從放射線影像進行疾病分類
- 異常檢測

### 文字任務
**使用時機：** 處理臨床筆記和文件

**資料集：** Medical Transcriptions、MIMIC-III（含筆記）

**常見任務：**
- 從臨床文字進行醫療編碼
- 專科分類
- 臨床資訊提取

## 任務輸出結構

所有任務函數回傳 `SampleDataset`，包含：

```python
sample = {
    "patient_id": "unique_patient_id",
    "visit_id": "unique_visit_id",  # 如適用
    "input_info": {
        # 輸入特徵（診斷、藥物等）
    },
    "output_info": {
        # 預測目標（標籤、數值）
    }
}
```

## 與模型整合

任務定義模型的輸入/輸出契約：

```python
from pyhealth.datasets import MIMIC4Dataset
from pyhealth.tasks import mortality_prediction_mimic4_fn
from pyhealth.models import Transformer

# 1. 建立任務專用資料集
dataset = MIMIC4Dataset(root="/path/to/data")
sample_dataset = dataset.set_task(mortality_prediction_mimic4_fn)

# 2. 模型自動適應任務 schema
model = Transformer(
    dataset=sample_dataset,
    feature_keys=["diagnoses", "medications"],
    mode="binary",  # 與任務輸出匹配
)
```

## 最佳實務

1. **將任務與臨床問題匹配**：可用時選擇預定義任務以進行標準化基準測試

2. **考慮時間窗口**：確保有足夠的歷史進行有意義的預測

3. **處理類別不平衡**：許多臨床結果很罕見（死亡率、再入院）

4. **驗證臨床相關性**：確保預測窗口與臨床決策時間軸一致

5. **使用適當的指標**：不同任務需要不同的評估指標（二元用 AUROC，多類別用 macro-F1）

6. **記錄排除標準**：追蹤哪些病人/就診被過濾及原因

7. **保護病人隱私**：始終使用去識別資料並遵循 HIPAA/GDPR 指南
