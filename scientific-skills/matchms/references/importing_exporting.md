# Matchms 匯入和匯出參考

本文件詳細說明 matchms 中用於載入和儲存質譜資料的所有檔案格式支援。

## 匯入質譜

Matchms 提供專用函數從各種檔案格式載入質譜。所有匯入函數都返回生成器，以實現大型檔案的記憶體高效處理。

### 常見匯入模式

```python
from matchms.importing import load_from_mgf

# 載入質譜（返回生成器）
spectra_generator = load_from_mgf("spectra.mgf")

# 轉換為列表進行處理
spectra = list(spectra_generator)
```

## 支援的匯入格式

### MGF（Mascot 通用格式）

**函數**：`load_from_mgf(filename, metadata_harmonization=True)`

**描述**：從 MGF 檔案載入質譜，這是質譜資料交換的常見格式。

**參數**：
- `filename` (str)：MGF 檔案路徑
- `metadata_harmonization` (bool, 預設=True)：應用自動元資料鍵協調

**範例**：
```python
from matchms.importing import load_from_mgf

# 帶元資料協調載入
spectra = list(load_from_mgf("data.mgf"))

# 不帶協調載入
spectra = list(load_from_mgf("data.mgf", metadata_harmonization=False))
```

**MGF 格式**：基於文字的格式，包含 BEGIN IONS/END IONS 區塊，內含元資料和峰值列表。

---

### MSP（NIST 質譜庫格式）

**函數**：`load_from_msp(filename, metadata_harmonization=True)`

**描述**：從 MSP 檔案載入質譜，常用於質譜庫。

**參數**：
- `filename` (str)：MSP 檔案路徑
- `metadata_harmonization` (bool, 預設=True)：應用自動元資料協調

**範例**：
```python
from matchms.importing import load_from_msp

spectra = list(load_from_msp("library.msp"))
```

**MSP 格式**：基於文字的格式，包含 Name/MW/Comment 欄位，後接峰值列表。

---

### mzML（質譜標記語言）

**函數**：`load_from_mzml(filename, ms_level=2, metadata_harmonization=True)`

**描述**：從 mzML 檔案載入質譜，這是原始質譜資料的標準 XML 格式。

**參數**：
- `filename` (str)：mzML 檔案路徑
- `ms_level` (int, 預設=2)：要擷取的 MS 層級（1 表示 MS1，2 表示 MS2/串聯）
- `metadata_harmonization` (bool, 預設=True)：應用自動元資料協調

**範例**：
```python
from matchms.importing import load_from_mzml

# 載入 MS2 質譜（預設）
ms2_spectra = list(load_from_mzml("data.mzML"))

# 載入 MS1 質譜
ms1_spectra = list(load_from_mzml("data.mzML", ms_level=1))
```

**mzML 格式**：基於 XML 的標準格式，包含原始儀器資料和豐富的元資料。

---

### mzXML

**函數**：`load_from_mzxml(filename, ms_level=2, metadata_harmonization=True)`

**描述**：從 mzXML 檔案載入質譜，這是較早的質譜資料 XML 格式。

**參數**：
- `filename` (str)：mzXML 檔案路徑
- `ms_level` (int, 預設=2)：要擷取的 MS 層級
- `metadata_harmonization` (bool, 預設=True)：應用自動元資料協調

**範例**：
```python
from matchms.importing import load_from_mzxml

spectra = list(load_from_mzxml("data.mzXML"))
```

**mzXML 格式**：基於 XML 的格式，mzML 的前身。

---

### JSON（GNPS 格式）

**函數**：`load_from_json(filename, metadata_harmonization=True)`

**描述**：從 JSON 檔案載入質譜，特別是 GNPS 相容的 JSON 格式。

**參數**：
- `filename` (str)：JSON 檔案路徑
- `metadata_harmonization` (bool, 預設=True)：應用自動元資料協調

**範例**：
```python
from matchms.importing import load_from_json

spectra = list(load_from_json("spectra.json"))
```

**JSON 格式**：結構化 JSON，包含質譜元資料和峰值陣列。

---

### Pickle（Python 序列化）

**函數**：`load_from_pickle(filename)`

**描述**：從 pickle 檔案載入先前儲存的 matchms Spectrum 物件。快速載入預處理的質譜。

**參數**：
- `filename` (str)：pickle 檔案路徑

**範例**：
```python
from matchms.importing import load_from_pickle

spectra = list(load_from_pickle("processed_spectra.pkl"))
```

**使用案例**：儲存和載入預處理質譜以加快後續分析。

---

### USI（通用質譜識別碼）

**函數**：`load_from_usi(usi)`

**描述**：從代謝體學 USI 參考載入單一質譜。

**參數**：
- `usi` (str)：通用質譜識別碼字串

**範例**：
```python
from matchms.importing import load_from_usi

usi = "mzspec:GNPS:TASK-...:spectrum..."
spectrum = load_from_usi(usi)
```

**USI 格式**：用於從線上儲存庫存取質譜的標準化識別碼。

---

## 匯出質譜

Matchms 提供函數將處理後的質譜儲存為各種格式以供分享和存檔。

### MGF 匯出

**函數**：`save_as_mgf(spectra, filename, write_mode='w')`

**描述**：將質譜儲存為 MGF 格式。

**參數**：
- `spectra` (list)：要儲存的 Spectrum 物件列表
- `filename` (str)：輸出檔案路徑
- `write_mode` (str, 預設='w')：檔案寫入模式（'w' 為寫入，'a' 為附加）

**範例**：
```python
from matchms.exporting import save_as_mgf

save_as_mgf(processed_spectra, "output.mgf")
```

---

### MSP 匯出

**函數**：`save_as_msp(spectra, filename, write_mode='w')`

**描述**：將質譜儲存為 MSP 格式。

**參數**：
- `spectra` (list)：要儲存的 Spectrum 物件列表
- `filename` (str)：輸出檔案路徑
- `write_mode` (str, 預設='w')：檔案寫入模式

**範例**：
```python
from matchms.exporting import save_as_msp

save_as_msp(library_spectra, "library.msp")
```

---

### JSON 匯出

**函數**：`save_as_json(spectra, filename, write_mode='w')`

**描述**：將質譜儲存為 JSON 格式（GNPS 相容）。

**參數**：
- `spectra` (list)：要儲存的 Spectrum 物件列表
- `filename` (str)：輸出檔案路徑
- `write_mode` (str, 預設='w')：檔案寫入模式

**範例**：
```python
from matchms.exporting import save_as_json

save_as_json(spectra, "spectra.json")
```

---

### Pickle 匯出

**函數**：`save_as_pickle(spectra, filename)`

**描述**：將質譜儲存為 Python pickle 檔案。保留所有 Spectrum 屬性，載入速度最快。

**參數**：
- `spectra` (list)：要儲存的 Spectrum 物件列表
- `filename` (str)：輸出檔案路徑

**範例**：
```python
from matchms.exporting import save_as_pickle

save_as_pickle(processed_spectra, "processed.pkl")
```

**優點**：
- 快速儲存和載入
- 保留精確的 Spectrum 狀態
- 無格式轉換開銷

**缺點**：
- 非人類可讀
- Python 特定（不可移植到其他語言）
- Pickle 格式可能在不同 Python 版本間不相容

---

## 完整匯入/匯出工作流程

### 預處理和儲存流程

```python
from matchms.importing import load_from_mgf
from matchms.exporting import save_as_mgf, save_as_pickle
from matchms.filtering import default_filters, normalize_intensities
from matchms.filtering import select_by_relative_intensity

# 載入原始質譜
spectra = list(load_from_mgf("raw_data.mgf"))

# 處理質譜
processed = []
for spectrum in spectra:
    spectrum = default_filters(spectrum)
    spectrum = normalize_intensities(spectrum)
    spectrum = select_by_relative_intensity(spectrum, intensity_from=0.01)
    if spectrum is not None:
        processed.append(spectrum)

# 儲存處理後的質譜（MGF 用於分享）
save_as_mgf(processed, "processed_data.mgf")

# 儲存為 pickle 以便快速重新載入
save_as_pickle(processed, "processed_data.pkl")
```

### 格式轉換

```python
from matchms.importing import load_from_mzml
from matchms.exporting import save_as_mgf, save_as_msp

# 將 mzML 轉換為 MGF
spectra = list(load_from_mzml("data.mzML", ms_level=2))
save_as_mgf(spectra, "data.mgf")

# 轉換為 MSP 庫格式
save_as_msp(spectra, "data.msp")
```

### 從多個檔案載入

```python
from matchms.importing import load_from_mgf
import glob

# 載入目錄中的所有 MGF 檔案
all_spectra = []
for mgf_file in glob.glob("data/*.mgf"):
    spectra = list(load_from_mgf(mgf_file))
    all_spectra.extend(spectra)

print(f"Loaded {len(all_spectra)} spectra from multiple files")
```

### 記憶體高效處理

```python
from matchms.importing import load_from_mgf
from matchms.exporting import save_as_mgf
from matchms.filtering import default_filters, normalize_intensities

# 處理大型檔案而不全部載入記憶體
def process_spectrum(spectrum):
    spectrum = default_filters(spectrum)
    spectrum = normalize_intensities(spectrum)
    return spectrum

# 串流處理
with open("output.mgf", 'w') as outfile:
    for spectrum in load_from_mgf("large_file.mgf"):
        processed = process_spectrum(spectrum)
        if processed is not None:
            # 立即寫入而不儲存在記憶體中
            save_as_mgf([processed], outfile, write_mode='a')
```

## 格式選擇指南

**MGF**：
- ✓ 廣泛支援
- ✓ 人類可讀
- ✓ 適合資料分享
- ✓ 中等檔案大小
- 最適用於：資料交換、GNPS 上傳、發表資料

**MSP**：
- ✓ 質譜庫標準
- ✓ 人類可讀
- ✓ 良好的元資料支援
- 最適用於：參考庫、NIST 格式相容性

**JSON**：
- ✓ 結構化格式
- ✓ GNPS 相容
- ✓ 易於程式解析
- 最適用於：網頁應用程式、GNPS 整合、結構化資料

**Pickle**：
- ✓ 最快的儲存/載入
- ✓ 保留精確狀態
- ✗ 不可移植到其他語言
- ✗ 非人類可讀
- 最適用於：中間處理、僅限 Python 的工作流程

**mzML/mzXML**：
- ✓ 原始儀器資料
- ✓ 豐富的元資料
- ✓ 產業標準
- ✗ 大檔案大小
- ✗ 解析較慢
- 最適用於：原始資料存檔、多層級 MS 資料

## 元資料協調

`metadata_harmonization` 參數（在大多數匯入函數中可用）自動標準化元資料鍵：

```python
# 不帶協調
spectrum = load_from_mgf("data.mgf", metadata_harmonization=False)
# 可能有："PRECURSOR_MZ"、"Precursor_mz"、"precursormz"

# 帶協調（預設）
spectrum = load_from_mgf("data.mgf", metadata_harmonization=True)
# 標準化為："precursor_mz"
```

**建議**：保持協調啟用（預設），以便在不同資料來源之間一致存取元資料。

## 檔案格式規格

有關詳細的格式規格：
- **MGF**：http://www.matrixscience.com/help/data_file_help.html
- **MSP**：https://chemdata.nist.gov/mass-spc/ms-search/
- **mzML**：http://www.psidev.info/mzML
- **GNPS JSON**：https://gnps.ucsd.edu/

## 進一步閱讀

有關完整的 API 文件：
https://matchms.readthedocs.io/en/latest/api/matchms.importing.html
https://matchms.readthedocs.io/en/latest/api/matchms.exporting.html

