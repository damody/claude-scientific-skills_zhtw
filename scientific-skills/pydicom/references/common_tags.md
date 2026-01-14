# 常用 DICOM 標籤參考

本文件提供按類別組織的常用 DICOM 標籤完整列表。標籤可以在 pydicom 中使用屬性記號（例如 `ds.PatientName`）或標籤元組記號（例如 `ds[0x0010, 0x0010]`）存取。

## 病患資訊標籤

| 標籤 | 名稱 | 類型 | 說明 |
|-----|------|------|-------------|
| (0010,0010) | PatientName | PN | 病患全名 |
| (0010,0020) | PatientID | LO | 病患主要識別碼 |
| (0010,0030) | PatientBirthDate | DA | 出生日期 (YYYYMMDD) |
| (0010,0032) | PatientBirthTime | TM | 出生時間 (HHMMSS) |
| (0010,0040) | PatientSex | CS | 病患性別 (M, F, O) |
| (0010,1010) | PatientAge | AS | 病患年齡 (格式: nnnD/W/M/Y) |
| (0010,1020) | PatientSize | DS | 病患身高（公尺） |
| (0010,1030) | PatientWeight | DS | 病患體重（公斤） |
| (0010,1040) | PatientAddress | LO | 病患郵寄地址 |
| (0010,2160) | EthnicGroup | SH | 病患族群 |
| (0010,4000) | PatientComments | LT | 關於病患的其他註解 |

## 檢查資訊標籤

| 標籤 | 名稱 | 類型 | 說明 |
|-----|------|------|-------------|
| (0020,000D) | StudyInstanceUID | UI | 檢查的唯一識別碼 |
| (0008,0020) | StudyDate | DA | 檢查開始日期 (YYYYMMDD) |
| (0008,0030) | StudyTime | TM | 檢查開始時間 (HHMMSS) |
| (0008,1030) | StudyDescription | LO | 檢查說明 |
| (0020,0010) | StudyID | SH | 使用者或機構定義的檢查識別碼 |
| (0008,0050) | AccessionNumber | SH | RIS 產生的檢查識別碼 |
| (0008,0090) | ReferringPhysicianName | PN | 病患轉介醫師姓名 |
| (0008,1060) | NameOfPhysiciansReadingStudy | PN | 判讀檢查的醫師姓名 |
| (0008,1080) | AdmittingDiagnosesDescription | LO | 入院診斷說明 |

## 系列資訊標籤

| 標籤 | 名稱 | 類型 | 說明 |
|-----|------|------|-------------|
| (0020,000E) | SeriesInstanceUID | UI | 系列的唯一識別碼 |
| (0020,0011) | SeriesNumber | IS | 此系列的數字識別碼 |
| (0008,103E) | SeriesDescription | LO | 系列說明 |
| (0008,0060) | Modality | CS | 設備類型 (CT, MR, US 等) |
| (0008,0021) | SeriesDate | DA | 系列開始日期 (YYYYMMDD) |
| (0008,0031) | SeriesTime | TM | 系列開始時間 (HHMMSS) |
| (0018,0015) | BodyPartExamined | CS | 檢查的身體部位 |
| (0018,5100) | PatientPosition | CS | 病患位置 (HFS, FFS 等) |
| (0020,0060) | Laterality | CS | 成對身體部位的側向性 (R, L) |

## 影像資訊標籤

| 標籤 | 名稱 | 類型 | 說明 |
|-----|------|------|-------------|
| (0008,0018) | SOPInstanceUID | UI | 此實例的唯一識別碼 |
| (0020,0013) | InstanceNumber | IS | 識別此影像的編號 |
| (0008,0008) | ImageType | CS | 影像識別特徵 |
| (0008,0023) | ContentDate | DA | 內容建立日期 (YYYYMMDD) |
| (0008,0033) | ContentTime | TM | 內容建立時間 (HHMMSS) |
| (0020,0032) | ImagePositionPatient | DS | 影像位置 (x, y, z)，單位 mm |
| (0020,0037) | ImageOrientationPatient | DS | 影像列/欄的方向餘弦 |
| (0020,1041) | SliceLocation | DS | 影像平面的相對位置 |
| (0018,0050) | SliceThickness | DS | 切片厚度，單位 mm |
| (0018,0088) | SpacingBetweenSlices | DS | 切片間距，單位 mm |

## 像素資料標籤

| 標籤 | 名稱 | 類型 | 說明 |
|-----|------|------|-------------|
| (7FE0,0010) | PixelData | OB/OW | 影像的實際像素資料 |
| (0028,0010) | Rows | US | 影像列數 |
| (0028,0011) | Columns | US | 影像欄數 |
| (0028,0100) | BitsAllocated | US | 每個像素樣本分配的位元數 |
| (0028,0101) | BitsStored | US | 每個像素樣本儲存的位元數 |
| (0028,0102) | HighBit | US | 像素樣本的最高有效位元 |
| (0028,0103) | PixelRepresentation | US | 0=無符號, 1=有符號 |
| (0028,0002) | SamplesPerPixel | US | 每像素樣本數 (1 或 3) |
| (0028,0004) | PhotometricInterpretation | CS | 色彩空間 (MONOCHROME2, RGB 等) |
| (0028,0006) | PlanarConfiguration | US | 彩色像素資料排列 |
| (0028,0030) | PixelSpacing | DS | 物理間距 [列, 欄]，單位 mm |
| (0028,0008) | NumberOfFrames | IS | 多幀影像的幀數 |
| (0028,0034) | PixelAspectRatio | IS | 垂直與水平像素的比率 |

## 視窗與顯示標籤

| 標籤 | 名稱 | 類型 | 說明 |
|-----|------|------|-------------|
| (0028,1050) | WindowCenter | DS | 顯示視窗中心 |
| (0028,1051) | WindowWidth | DS | 顯示視窗寬度 |
| (0028,1052) | RescaleIntercept | DS | 輸出 = m*SV + b 中的 b |
| (0028,1053) | RescaleSlope | DS | 輸出 = m*SV + b 中的 m |
| (0028,1054) | RescaleType | LO | 重新縮放類型 (HU 等) |
| (0028,1055) | WindowCenterWidthExplanation | LO | 視窗值的說明 |
| (0028,3010) | VOILUTSequence | SQ | VOI LUT 說明 |

## CT 特定標籤

| 標籤 | 名稱 | 類型 | 說明 |
|-----|------|------|-------------|
| (0018,0060) | KVP | DS | 峰值千伏特 |
| (0018,1030) | ProtocolName | LO | 掃描協定名稱 |
| (0018,1100) | ReconstructionDiameter | DS | 重建圓直徑 |
| (0018,1110) | DistanceSourceToDetector | DS | 距離，單位 mm |
| (0018,1111) | DistanceSourceToPatient | DS | 距離，單位 mm |
| (0018,1120) | GantryDetectorTilt | DS | 機架傾斜角度 |
| (0018,1130) | TableHeight | DS | 檢查床高度，單位 mm |
| (0018,1150) | ExposureTime | IS | 曝光時間，單位 ms |
| (0018,1151) | XRayTubeCurrent | IS | X 光管電流，單位 mA |
| (0018,1152) | Exposure | IS | 曝光量，單位 mAs |
| (0018,1160) | FilterType | SH | X 光濾片材質 |
| (0018,1210) | ConvolutionKernel | SH | 重建演算法 |

## MR 特定標籤

| 標籤 | 名稱 | 類型 | 說明 |
|-----|------|------|-------------|
| (0018,0080) | RepetitionTime | DS | TR，單位 ms |
| (0018,0081) | EchoTime | DS | TE，單位 ms |
| (0018,0082) | InversionTime | DS | TI，單位 ms |
| (0018,0083) | NumberOfAverages | DS | 資料平均次數 |
| (0018,0084) | ImagingFrequency | DS | 頻率，單位 MHz |
| (0018,0085) | ImagedNucleus | SH | 成像的原子核 (1H 等) |
| (0018,0086) | EchoNumbers | IS | 回波編號 |
| (0018,0087) | MagneticFieldStrength | DS | 磁場強度，單位 Tesla |
| (0018,0088) | SpacingBetweenSlices | DS | 間距，單位 mm |
| (0018,0089) | NumberOfPhaseEncodingSteps | IS | 相位編碼步數 |
| (0018,0091) | EchoTrainLength | IS | 回波串長度 |
| (0018,0093) | PercentSampling | DS | 採集矩陣採樣比例 |
| (0018,0094) | PercentPhaseFieldOfView | DS | 相位與頻率 FOV 比率 |
| (0018,1030) | ProtocolName | LO | 掃描協定名稱 |
| (0018,1314) | FlipAngle | DS | 翻轉角度 |

## 檔案詮釋資訊標籤

| 標籤 | 名稱 | 類型 | 說明 |
|-----|------|------|-------------|
| (0002,0000) | FileMetaInformationGroupLength | UL | 檔案詮釋資訊長度 |
| (0002,0001) | FileMetaInformationVersion | OB | 檔案詮釋資訊版本 |
| (0002,0002) | MediaStorageSOPClassUID | UI | SOP 類別 UID |
| (0002,0003) | MediaStorageSOPInstanceUID | UI | SOP 實例 UID |
| (0002,0010) | TransferSyntaxUID | UI | 傳輸語法 UID |
| (0002,0012) | ImplementationClassUID | UI | 實作類別 UID |
| (0002,0013) | ImplementationVersionName | SH | 實作版本名稱 |

## 設備標籤

| 標籤 | 名稱 | 類型 | 說明 |
|-----|------|------|-------------|
| (0008,0070) | Manufacturer | LO | 設備製造商 |
| (0008,0080) | InstitutionName | LO | 機構名稱 |
| (0008,0081) | InstitutionAddress | ST | 機構地址 |
| (0008,1010) | StationName | SH | 設備站名稱 |
| (0008,1040) | InstitutionalDepartmentName | LO | 部門名稱 |
| (0008,1050) | PerformingPhysicianName | PN | 執行檢查的醫師 |
| (0008,1070) | OperatorsName | PN | 操作員姓名 |
| (0008,1090) | ManufacturerModelName | LO | 型號名稱 |
| (0018,1000) | DeviceSerialNumber | LO | 設備序號 |
| (0018,1020) | SoftwareVersions | LO | 軟體版本 |

## 時間標籤

| 標籤 | 名稱 | 類型 | 說明 |
|-----|------|------|-------------|
| (0008,0012) | InstanceCreationDate | DA | 實例建立日期 |
| (0008,0013) | InstanceCreationTime | TM | 實例建立時間 |
| (0008,0022) | AcquisitionDate | DA | 擷取開始日期 |
| (0008,0032) | AcquisitionTime | TM | 擷取開始時間 |
| (0008,002A) | AcquisitionDateTime | DT | 擷取日期和時間 |

## DICOM 值表示法 (VR)

DICOM 中使用的常見值表示法類型：

- **AE**：應用程式實體（最多 16 字元）
- **AS**：年齡字串（nnnD/W/M/Y）
- **CS**：代碼字串（最多 16 字元）
- **DA**：日期（YYYYMMDD）
- **DS**：十進位字串
- **DT**：日期時間（YYYYMMDDHHMMSS.FFFFFF&ZZXX）
- **IS**：整數字串
- **LO**：長字串（最多 64 字元）
- **LT**：長文字（最多 10240 字元）
- **PN**：人名
- **SH**：短字串（最多 16 字元）
- **SQ**：項目序列
- **ST**：短文字（最多 1024 字元）
- **TM**：時間（HHMMSS.FFFFFF）
- **UI**：唯一識別碼（UID）
- **UL**：無符號長整數（4 位元組）
- **US**：無符號短整數（2 位元組）
- **OB**：其他位元組字串
- **OW**：其他字組字串

## 使用範例

### 按名稱存取標籤
```python
patient_name = ds.PatientName
study_date = ds.StudyDate
modality = ds.Modality
```

### 按編號存取標籤
```python
patient_name = ds[0x0010, 0x0010].value
study_date = ds[0x0008, 0x0020].value
modality = ds[0x0008, 0x0060].value
```

### 檢查標籤是否存在
```python
if hasattr(ds, 'PatientName'):
    print(ds.PatientName)

# 或使用 'in' 運算子
if (0x0010, 0x0010) in ds:
    print(ds[0x0010, 0x0010].value)
```

### 使用預設值安全存取
```python
patient_name = getattr(ds, 'PatientName', 'Unknown')
study_desc = ds.get('StudyDescription', 'No description')
```

## 參考資源

- DICOM 標準：https://www.dicomstandard.org/
- DICOM 標籤瀏覽器：https://dicom.innolitics.com/ciods
- Pydicom 文件：https://pydicom.github.io/pydicom/
