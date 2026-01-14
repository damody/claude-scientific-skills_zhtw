# PyHealth 資料集與資料結構

## 核心資料結構

### Event（事件）
具有以下屬性的個別醫療事件：
- **code**：醫療代碼（診斷、藥物、處置、檢驗）
- **vocabulary**：編碼系統（ICD-9-CM、NDC、LOINC 等）
- **timestamp**：事件發生時間
- **value**：數值（用於檢驗、生命徵象）
- **unit**：測量單位

### Patient（病人）
按時間順序組織的跨就診事件集合。每個病人包含：
- **patient_id**：唯一識別碼
- **birth_datetime**：出生日期
- **gender**：病人性別
- **ethnicity**：病人種族
- **visits**：就診物件列表

### Visit（就診）
包含以下內容的醫療照護encounter：
- **visit_id**：唯一識別碼
- **encounter_time**：就診時間戳記
- **discharge_time**：出院時間戳記
- **visit_type**：encounter類型（住院、門診、急診）
- **events**：此次就診期間的事件列表

## BaseDataset 類別

**關鍵方法：**
- `get_patient(patient_id)`：擷取單一病人記錄
- `iter_patients()`：遍歷所有病人
- `stats()`：取得資料集統計（病人數、就診數、事件數）
- `set_task(task_fn)`：定義預測任務

## 可用資料集

### 電子健康記錄（EHR）資料集

**MIMIC-III Dataset**（`MIMIC3Dataset`）
- 來自 Beth Israel Deaconess 醫學中心的加護病房資料
- 40,000+ 重症病人
- 診斷、處置、藥物、檢驗結果
- 使用方式：`from pyhealth.datasets import MIMIC3Dataset`

**MIMIC-IV Dataset**（`MIMIC4Dataset`）
- 更新版本，包含 70,000+ 病人
- 改善的資料品質與涵蓋範圍
- 增強的人口統計與臨床細節
- 使用方式：`from pyhealth.datasets import MIMIC4Dataset`

**eICU Dataset**（`eICUDataset`）
- 多中心重症照護資料庫
- 來自 200+ 醫院的 200,000+ 入院記錄
- 跨機構的標準化加護病房資料
- 使用方式：`from pyhealth.datasets import eICUDataset`

**OMOP Dataset**（`OMOPDataset`）
- Observational Medical Outcomes Partnership 格式
- 標準化通用資料模型
- 跨醫療系統的互通性
- 使用方式：`from pyhealth.datasets import OMOPDataset`

**EHRShot Dataset**（`EHRShotDataset`）
- 少樣本學習基準資料集
- 專門用於測試模型泛化能力
- 使用方式：`from pyhealth.datasets import EHRShotDataset`

### 生理訊號資料集

**睡眠腦電圖資料集：**
- `SleepEDFDataset`：用於睡眠分期的 Sleep-EDF 資料庫
- `SHHSDataset`：睡眠心臟健康研究資料
- `ISRUCDataset`：ISRUC-Sleep 資料庫

**Temple University 腦電圖資料集：**
- `TUEVDataset`：異常腦電圖事件檢測
- `TUABDataset`：異常/正常腦電圖分類
- `TUSZDataset`：癲癇發作檢測

**所有訊號資料集支援：**
- 多通道腦電圖訊號
- 標準化採樣率
- 專家標註
- 睡眠階段或異常標籤

### 醫學影像資料集

**COVID-19 CXR Dataset**（`COVID19CXRDataset`）
- 用於 COVID-19 分類的胸部 X 光影像
- 多類別標籤（COVID-19、肺炎、正常）
- 使用方式：`from pyhealth.datasets import COVID19CXRDataset`

### 文字資料集

**Medical Transcriptions Dataset**（`MedicalTranscriptionsDataset`）
- 臨床筆記與轉錄文字
- 醫療專科分類
- 文字預測任務
- 使用方式：`from pyhealth.datasets import MedicalTranscriptionsDataset`

**Cardiology Dataset**（`CardiologyDataset`）
- 心臟科病人記錄
- 心血管疾病預測
- 使用方式：`from pyhealth.datasets import CardiologyDataset`

### 預處理資料集

**MIMIC Extract Dataset**（`MIMICExtractDataset`）
- 預先提取的 MIMIC 特徵
- 可直接使用的基準測試資料
- 減少預處理需求
- 使用方式：`from pyhealth.datasets import MIMICExtractDataset`

## SampleDataset 類別

將原始資料集轉換為任務特定格式的樣本。

**用途：** 將病人層級資料轉換為模型可用的輸入/輸出配對

**關鍵屬性：**
- `input_schema`：定義輸入資料結構
- `output_schema`：定義目標標籤/預測
- `samples`：處理後的樣本列表

**使用模式：**
```python
# 在 BaseDataset 上設定任務後
sample_dataset = dataset.set_task(task_fn)
```

## 資料分割函數

**病人層級分割**（`split_by_patient`）
- 確保同一病人不會出現在多個分割中
- 防止資料洩漏
- 推薦用於臨床預測任務

**就診層級分割**（`split_by_visit`）
- 按個別就診進行分割
- 允許同一病人跨分割（請謹慎使用）

**樣本層級分割**（`split_by_sample`）
- 隨機樣本分割
- 最靈活但可能造成洩漏

**參數：**
- `dataset`：要分割的 SampleDataset
- `ratios`：分割比例元組（例如 [0.7, 0.1, 0.2]）
- `seed`：用於可重現性的隨機種子

## 常見工作流程

```python
from pyhealth.datasets import MIMIC4Dataset
from pyhealth.tasks import mortality_prediction_mimic4_fn
from pyhealth.datasets import split_by_patient

# 1. 載入資料集
dataset = MIMIC4Dataset(root="/path/to/data")

# 2. 設定預測任務
sample_dataset = dataset.set_task(mortality_prediction_mimic4_fn)

# 3. 分割資料
train, val, test = split_by_patient(sample_dataset, [0.7, 0.1, 0.2])

# 4. 取得統計資訊
print(dataset.stats())
```

## 效能備註

- PyHealth 處理醫療資料的速度**比 pandas 快 3 倍**
- 針對大規模 EHR 資料集進行優化
- 記憶體高效的病人迭代
- 向量化特徵提取操作
