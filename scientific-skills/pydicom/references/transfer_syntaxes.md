# DICOM 傳輸語法參考

本文件提供 DICOM 傳輸語法和壓縮格式的完整參考。傳輸語法定義了 DICOM 資料的編碼方式，包括位元組順序、壓縮方法和其他編碼規則。

## 概述

傳輸語法 UID 指定：
1. **位元組順序**：Little Endian 或 Big Endian
2. **值表示法 (VR)**：隱式或顯式
3. **壓縮**：無，或特定壓縮演算法

## 未壓縮傳輸語法

### 隱式 VR Little Endian (1.2.840.10008.1.2)
- **預設**傳輸語法
- 值表示法為隱式（未明確編碼）
- Little Endian 位元組順序
- **Pydicom 常數**：`pydicom.uid.ImplicitVRLittleEndian`

**使用方式：**
```python
import pydicom
ds.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
```

### 顯式 VR Little Endian (1.2.840.10008.1.2.1)
- **最常見**的傳輸語法
- 值表示法為顯式
- Little Endian 位元組順序
- **Pydicom 常數**：`pydicom.uid.ExplicitVRLittleEndian`

**使用方式：**
```python
ds.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
```

### 顯式 VR Big Endian (1.2.840.10008.1.2.2) - 已停用
- 值表示法為顯式
- Big Endian 位元組順序
- **已棄用** - 不建議用於新實作
- **Pydicom 常數**：`pydicom.uid.ExplicitVRBigEndian`

## JPEG 壓縮

### JPEG Baseline (Process 1) (1.2.840.10008.1.2.4.50)
- **有損**壓縮
- 僅 8 位元樣本
- 最廣泛支援的 JPEG 格式
- **Pydicom 常數**：`pydicom.uid.JPEGBaseline8Bit`

**相依套件：** 需要 `pylibjpeg` 或 `pillow`

**使用方式：**
```python
# 壓縮
ds.compress(pydicom.uid.JPEGBaseline8Bit)

# 解壓縮
ds.decompress()
```

### JPEG Extended (Process 2 & 4) (1.2.840.10008.1.2.4.51)
- **有損**壓縮
- 8 位元和 12 位元樣本
- **Pydicom 常數**：`pydicom.uid.JPEGExtended12Bit`

### JPEG Lossless, Non-Hierarchical (Process 14) (1.2.840.10008.1.2.4.57)
- **無損**壓縮
- 一階預測
- **Pydicom 常數**：`pydicom.uid.JPEGLossless`

**相依套件：** 需要 `pylibjpeg-libjpeg` 或 `gdcm`

### JPEG Lossless, Non-Hierarchical, First-Order Prediction (1.2.840.10008.1.2.4.70)
- **無損**壓縮
- 使用 Process 14 Selection Value 1
- **Pydicom 常數**：`pydicom.uid.JPEGLosslessSV1`

**使用方式：**
```python
# 壓縮為 JPEG Lossless
ds.compress(pydicom.uid.JPEGLossless)
```

### JPEG-LS Lossless (1.2.840.10008.1.2.4.80)
- **無損**壓縮
- 低複雜度，良好壓縮率
- **Pydicom 常數**：`pydicom.uid.JPEGLSLossless`

**相依套件：** 需要 `pylibjpeg-libjpeg` 或 `gdcm`

### JPEG-LS Lossy (Near-Lossless) (1.2.840.10008.1.2.4.81)
- **近無損**壓縮
- 允許控制精度損失
- **Pydicom 常數**：`pydicom.uid.JPEGLSNearLossless`

## JPEG 2000 壓縮

### JPEG 2000 Lossless Only (1.2.840.10008.1.2.4.90)
- **無損**壓縮
- 基於小波的壓縮
- 比 JPEG Lossless 有更好的壓縮率
- **Pydicom 常數**：`pydicom.uid.JPEG2000Lossless`

**相依套件：** 需要 `pylibjpeg-openjpeg`、`gdcm` 或 `pillow`

**使用方式：**
```python
# 壓縮為 JPEG 2000 Lossless
ds.compress(pydicom.uid.JPEG2000Lossless)
```

### JPEG 2000 (1.2.840.10008.1.2.4.91)
- **有損或無損**壓縮
- 基於小波的壓縮
- 低位元率時高品質
- **Pydicom 常數**：`pydicom.uid.JPEG2000`

**相依套件：** 需要 `pylibjpeg-openjpeg`、`gdcm` 或 `pillow`

### JPEG 2000 Part 2 Multi-component Lossless (1.2.840.10008.1.2.4.92)
- **無損**壓縮
- 支援多分量影像
- **Pydicom 常數**：`pydicom.uid.JPEG2000MCLossless`

### JPEG 2000 Part 2 Multi-component (1.2.840.10008.1.2.4.93)
- **有損或無損**壓縮
- 支援多分量影像
- **Pydicom 常數**：`pydicom.uid.JPEG2000MC`

## RLE 壓縮

### RLE Lossless (1.2.840.10008.1.2.5)
- **無損**壓縮
- 行程編碼
- 簡單、快速的演算法
- 適合有重複值的影像
- **Pydicom 常數**：`pydicom.uid.RLELossless`

**相依套件：** pydicom 內建（不需要額外套件）

**使用方式：**
```python
# 使用 RLE 壓縮
ds.compress(pydicom.uid.RLELossless)

# 解壓縮
ds.decompress()
```

## Deflated 傳輸語法

### Deflated Explicit VR Little Endian (1.2.840.10008.1.2.1.99)
- 對整個資料集使用 ZLIB 壓縮
- 不常使用
- **Pydicom 常數**：`pydicom.uid.DeflatedExplicitVRLittleEndian`

## MPEG 壓縮

### MPEG2 Main Profile @ Main Level (1.2.840.10008.1.2.4.100)
- **有損**視訊壓縮
- 用於多幀影像/視訊
- **Pydicom 常數**：`pydicom.uid.MPEG2MPML`

### MPEG2 Main Profile @ High Level (1.2.840.10008.1.2.4.101)
- **有損**視訊壓縮
- 比 MPML 更高解析度
- **Pydicom 常數**：`pydicom.uid.MPEG2MPHL`

### MPEG-4 AVC/H.264 High Profile (1.2.840.10008.1.2.4.102-106)
- **有損**視訊壓縮
- 各種層級 (BD, 2D, 3D, Stereo)
- 現代視訊編解碼器

## 檢查傳輸語法

### 識別目前傳輸語法
```python
import pydicom

ds = pydicom.dcmread('image.dcm')

# 取得傳輸語法 UID
ts_uid = ds.file_meta.TransferSyntaxUID
print(f"傳輸語法 UID：{ts_uid}")

# 取得人類可讀的名稱
print(f"傳輸語法名稱：{ts_uid.name}")

# 檢查是否已壓縮
print(f"是否已壓縮：{ts_uid.is_compressed}")
```

### 常見檢查
```python
# 檢查是否為 little endian
if ts_uid.is_little_endian:
    print("Little Endian")

# 檢查是否為隱式 VR
if ts_uid.is_implicit_VR:
    print("隱式 VR")

# 檢查壓縮類型
if 'JPEG' in ts_uid.name:
    print("JPEG 壓縮")
elif 'JPEG2000' in ts_uid.name:
    print("JPEG 2000 壓縮")
elif 'RLE' in ts_uid.name:
    print("RLE 壓縮")
```

## 解壓縮

### 自動解壓縮
Pydicom 在存取 `pixel_array` 時可以自動解壓縮像素資料：

```python
import pydicom

# 讀取壓縮的 DICOM
ds = pydicom.dcmread('compressed.dcm')

# 像素資料會自動解壓縮
pixel_array = ds.pixel_array  # 如有需要會解壓縮
```

### 手動解壓縮
```python
import pydicom

ds = pydicom.dcmread('compressed.dcm')

# 就地解壓縮
ds.decompress()

# 現在儲存為未壓縮
ds.save_as('uncompressed.dcm', write_like_original=False)
```

## 壓縮

### 壓縮 DICOM 檔案
```python
import pydicom

ds = pydicom.dcmread('uncompressed.dcm')

# 使用 JPEG 2000 Lossless 壓縮
ds.compress(pydicom.uid.JPEG2000Lossless)
ds.save_as('compressed_j2k.dcm')

# 使用 RLE Lossless 壓縮（無額外相依套件）
ds.compress(pydicom.uid.RLELossless)
ds.save_as('compressed_rle.dcm')

# 使用 JPEG Baseline 壓縮（有損）
ds.compress(pydicom.uid.JPEGBaseline8Bit)
ds.save_as('compressed_jpeg.dcm')
```

### 使用自訂編碼參數壓縮
```python
import pydicom
from pydicom.encoders import JPEGLSLosslessEncoder

ds = pydicom.dcmread('uncompressed.dcm')

# 使用自訂參數壓縮
ds.compress(pydicom.uid.JPEGLSLossless, encoding_plugin='pylibjpeg')
```

## 安裝壓縮處理器

不同的傳輸語法需要不同的 Python 套件：

### JPEG Baseline/Extended
```bash
pip install pylibjpeg pylibjpeg-libjpeg
# 或
pip install pillow
```

### JPEG Lossless/JPEG-LS
```bash
pip install pylibjpeg pylibjpeg-libjpeg
# 或
pip install python-gdcm
```

### JPEG 2000
```bash
pip install pylibjpeg pylibjpeg-openjpeg
# 或
pip install python-gdcm
# 或
pip install pillow
```

### RLE
不需要額外套件 - pydicom 內建

### 完整安裝
```bash
# 安裝所有常見處理器
pip install pylibjpeg pylibjpeg-libjpeg pylibjpeg-openjpeg python-gdcm
```

## 檢查可用的處理器

```python
import pydicom

# 列出可用的像素資料處理器
from pydicom.pixel_data_handlers.util import get_pixel_data_handlers
handlers = get_pixel_data_handlers()

print("可用的處理器：")
for handler in handlers:
    print(f"  - {handler.__name__}")
```

## 最佳實務

1. **建立新檔案時使用顯式 VR Little Endian** 以獲得最大相容性
2. **使用 JPEG 2000 Lossless** 以獲得良好的壓縮率且無品質損失
3. **如果無法安裝額外相依套件，使用 RLE Lossless**
4. **處理前檢查傳輸語法** 以確保您有正確的處理器
5. **測試解壓縮** 在部署前確保已安裝所有必要的套件
6. **盡可能保留原始**傳輸語法，使用 `write_like_original=True`
7. **選擇有損壓縮時考慮檔案大小** vs 品質的權衡
8. **對診斷影像使用無損壓縮** 以維持臨床品質

## 常見問題

### 問題：「無法解碼像素資料」
**原因：** 缺少壓縮處理器
**解決方案：** 安裝適當的套件（請參閱上方的安裝壓縮處理器）

### 問題：「不支援的傳輸語法」
**原因：** 罕見或不支援的壓縮格式
**解決方案：** 嘗試安裝 `python-gdcm`，它支援更多格式

### 問題：「像素資料已解壓縮但看起來不正確」
**原因：** 可能需要應用 VOI LUT 或重新縮放
**解決方案：** 使用 `apply_voi_lut()` 或應用 `RescaleSlope`/`RescaleIntercept`

## 參考資源

- DICOM 標準第 5 部分（資料結構和編碼）：https://dicom.nema.org/medical/dicom/current/output/chtml/part05/PS3.5.html
- Pydicom 傳輸語法文件：https://pydicom.github.io/pydicom/stable/guides/user/transfer_syntaxes.html
- Pydicom 壓縮指南：https://pydicom.github.io/pydicom/stable/old/image_data_compression.html
