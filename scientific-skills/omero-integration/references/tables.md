# OMERO 表格

此參考涵蓋使用 OMERO.tables 在 OMERO 中建立和管理結構化表格資料。

## OMERO.tables 概述

OMERO.tables 提供了一種儲存與 OMERO 物件關聯的結構化表格資料的方式。表格以 HDF5 檔案儲存，可以有效率地查詢。常見用途包括：

- 儲存影像的定量測量
- 記錄分析結果
- 追蹤實驗中繼資料
- 將測量連結到特定影像或 ROI

## 欄位類型

OMERO.tables 支援多種欄位類型：

- **LongColumn**：整數值（64 位元）
- **DoubleColumn**：浮點數值
- **StringColumn**：文字資料（固定最大長度）
- **BoolColumn**：布林值
- **LongArrayColumn**：整數陣列
- **DoubleArrayColumn**：浮點數陣列
- **FileColumn**：OMERO 檔案參考
- **ImageColumn**：OMERO 影像參考
- **RoiColumn**：OMERO ROI 參考
- **WellColumn**：OMERO 孔參考

## 建立表格

### 基本表格建立

```python
from random import random
import omero.grid

# 建立唯一表格名稱
table_name = f"MyAnalysisTable_{random()}"

# 定義欄位（初始化用空資料）
col1 = omero.grid.LongColumn('ImageID', 'Image identifier', [])
col2 = omero.grid.DoubleColumn('MeanIntensity', 'Mean pixel intensity', [])
col3 = omero.grid.StringColumn('Category', 'Classification', 64, [])

columns = [col1, col2, col3]

# 取得資源並建立表格
resources = conn.c.sf.sharedResources()
repository_id = resources.repositories().descriptions[0].getId().getValue()
table = resources.newTable(repository_id, table_name)

# 使用欄位定義初始化表格
table.initialize(columns)
```

### 新增資料到表格

```python
# 準備資料
image_ids = [1, 2, 3, 4, 5]
intensities = [123.4, 145.2, 98.7, 156.3, 132.8]
categories = ["Good", "Good", "Poor", "Excellent", "Good"]

# 建立資料欄位
data_col1 = omero.grid.LongColumn('ImageID', 'Image identifier', image_ids)
data_col2 = omero.grid.DoubleColumn('MeanIntensity', 'Mean pixel intensity', intensities)
data_col3 = omero.grid.StringColumn('Category', 'Classification', 64, categories)

data = [data_col1, data_col2, data_col3]

# 將資料新增到表格
table.addData(data)

# 取得檔案參考
orig_file = table.getOriginalFile()
table.close()  # 完成後務必關閉表格
```

### 將表格連結到資料集

```python
# 從表格建立檔案註解
orig_file_id = orig_file.id.val
file_ann = omero.model.FileAnnotationI()
file_ann.setFile(omero.model.OriginalFileI(orig_file_id, False))
file_ann = conn.getUpdateService().saveAndReturnObject(file_ann)

# 連結到資料集
link = omero.model.DatasetAnnotationLinkI()
link.setParent(omero.model.DatasetI(dataset_id, False))
link.setChild(omero.model.FileAnnotationI(file_ann.getId().getValue(), False))
conn.getUpdateService().saveAndReturnObject(link)

print(f"Linked table to dataset {dataset_id}")
```

## 欄位類型詳解

### Long 欄位（整數）

```python
# 整數值欄位
image_ids = [101, 102, 103, 104, 105]
col = omero.grid.LongColumn('ImageID', 'Image identifier', image_ids)
```

### Double 欄位（浮點數）

```python
# 浮點數值欄位
measurements = [12.34, 56.78, 90.12, 34.56, 78.90]
col = omero.grid.DoubleColumn('Measurement', 'Value in microns', measurements)
```

### String 欄位（文字）

```python
# 文字欄位（需要最大長度）
labels = ["Control", "Treatment A", "Treatment B", "Control", "Treatment A"]
col = omero.grid.StringColumn('Condition', 'Experimental condition', 64, labels)
```

### Boolean 欄位

```python
# 布林值欄位
flags = [True, False, True, True, False]
col = omero.grid.BoolColumn('QualityPass', 'Passes quality control', flags)
```

### Image 欄位（影像參考）

```python
# 連結到 OMERO 影像的欄位
image_ids = [101, 102, 103, 104, 105]
col = omero.grid.ImageColumn('Image', 'Source image', image_ids)
```

### ROI 欄位（ROI 參考）

```python
# 連結到 OMERO ROI 的欄位
roi_ids = [201, 202, 203, 204, 205]
col = omero.grid.RoiColumn('ROI', 'Associated ROI', roi_ids)
```

### 陣列欄位

```python
# Double 陣列欄位
histogram_data = [
    [10, 20, 30, 40],
    [15, 25, 35, 45],
    [12, 22, 32, 42]
]
col = omero.grid.DoubleArrayColumn('Histogram', 'Intensity histogram', histogram_data)

# Long 陣列欄位
bin_counts = [[5, 10, 15], [8, 12, 16], [6, 11, 14]]
col = omero.grid.LongArrayColumn('Bins', 'Histogram bins', bin_counts)
```

## 讀取表格資料

### 開啟現有表格

```python
# 按名稱取得表格檔案
orig_table_file = conn.getObject("OriginalFile",
                                 attributes={'name': table_name})

# 開啟表格
resources = conn.c.sf.sharedResources()
table = resources.openTable(orig_table_file._obj)

print(f"Opened table: {table.getOriginalFile().getName().getValue()}")
print(f"Number of rows: {table.getNumberOfRows()}")
```

### 讀取所有資料

```python
# 取得欄位標頭
print("Columns:")
for col in table.getHeaders():
    print(f"  {col.name}: {col.description}")

# 讀取所有資料
row_count = table.getNumberOfRows()
data = table.readCoordinates(range(row_count))

# 顯示資料
for col in data.columns:
    print(f"\nColumn: {col.name}")
    for value in col.values:
        print(f"  {value}")

table.close()
```

### 讀取特定列

```python
# 讀取第 10-20 列
start = 10
stop = 20
data = table.read(list(range(table.getHeaders().__len__())), start, stop)

for col in data.columns:
    print(f"Column: {col.name}")
    for value in col.values:
        print(f"  {value}")
```

### 讀取特定欄位

```python
# 只讀取欄位 0 和 2
column_indices = [0, 2]
start = 0
stop = table.getNumberOfRows()

data = table.read(column_indices, start, stop)

for col in data.columns:
    print(f"Column: {col.name}")
    print(f"Values: {col.values}")
```

## 查詢表格

### 條件查詢

```python
# 查詢 MeanIntensity > 100 的列
row_count = table.getNumberOfRows()

query_rows = table.getWhereList(
    "(MeanIntensity > 100)",
    variables={},
    start=0,
    stop=row_count,
    step=0
)

print(f"Found {len(query_rows)} matching rows")

# 讀取符合的列
data = table.readCoordinates(query_rows)

for col in data.columns:
    print(f"\n{col.name}:")
    for value in col.values:
        print(f"  {value}")
```

### 複雜查詢

```python
# 使用 AND 的多個條件
query_rows = table.getWhereList(
    "(MeanIntensity > 100) & (MeanIntensity < 150)",
    variables={},
    start=0,
    stop=row_count,
    step=0
)

# 使用 OR 的多個條件
query_rows = table.getWhereList(
    "(Category == 'Good') | (Category == 'Excellent')",
    variables={},
    start=0,
    stop=row_count,
    step=0
)

# 字串匹配
query_rows = table.getWhereList(
    "(Category == 'Good')",
    variables={},
    start=0,
    stop=row_count,
    step=0
)
```

## 完整範例：影像分析結果

```python
from omero.gateway import BlitzGateway
import omero.grid
import omero.model
import numpy as np

HOST = 'omero.example.com'
PORT = 4064
USERNAME = 'user'
PASSWORD = 'pass'

with BlitzGateway(USERNAME, PASSWORD, host=HOST, port=PORT) as conn:
    # 取得資料集
    dataset = conn.getObject("Dataset", dataset_id)
    print(f"Analyzing dataset: {dataset.getName()}")

    # 從影像收集測量
    image_ids = []
    mean_intensities = []
    max_intensities = []
    cell_counts = []

    for image in dataset.listChildren():
        image_ids.append(image.getId())

        # 取得像素資料
        pixels = image.getPrimaryPixels()
        plane = pixels.getPlane(0, 0, 0)  # Z=0, C=0, T=0

        # 計算統計資料
        mean_intensities.append(float(np.mean(plane)))
        max_intensities.append(float(np.max(plane)))

        # 模擬細胞計數（實際應來自分析）
        cell_counts.append(np.random.randint(50, 200))

    # 建立表格
    table_name = f"Analysis_Results_{dataset.getId()}"

    # 定義欄位
    col1 = omero.grid.ImageColumn('Image', 'Source image', [])
    col2 = omero.grid.DoubleColumn('MeanIntensity', 'Mean pixel value', [])
    col3 = omero.grid.DoubleColumn('MaxIntensity', 'Maximum pixel value', [])
    col4 = omero.grid.LongColumn('CellCount', 'Number of cells detected', [])

    # 初始化表格
    resources = conn.c.sf.sharedResources()
    repository_id = resources.repositories().descriptions[0].getId().getValue()
    table = resources.newTable(repository_id, table_name)
    table.initialize([col1, col2, col3, col4])

    # 新增資料
    data_col1 = omero.grid.ImageColumn('Image', 'Source image', image_ids)
    data_col2 = omero.grid.DoubleColumn('MeanIntensity', 'Mean pixel value',
                                        mean_intensities)
    data_col3 = omero.grid.DoubleColumn('MaxIntensity', 'Maximum pixel value',
                                        max_intensities)
    data_col4 = omero.grid.LongColumn('CellCount', 'Number of cells detected',
                                      cell_counts)

    table.addData([data_col1, data_col2, data_col3, data_col4])

    # 取得檔案並關閉表格
    orig_file = table.getOriginalFile()
    table.close()

    # 連結到資料集
    orig_file_id = orig_file.id.val
    file_ann = omero.model.FileAnnotationI()
    file_ann.setFile(omero.model.OriginalFileI(orig_file_id, False))
    file_ann = conn.getUpdateService().saveAndReturnObject(file_ann)

    link = omero.model.DatasetAnnotationLinkI()
    link.setParent(omero.model.DatasetI(dataset_id, False))
    link.setChild(omero.model.FileAnnotationI(file_ann.getId().getValue(), False))
    conn.getUpdateService().saveAndReturnObject(link)

    print(f"Created and linked table with {len(image_ids)} rows")

    # 查詢結果
    table = resources.openTable(orig_file)

    high_cell_count_rows = table.getWhereList(
        "(CellCount > 100)",
        variables={},
        start=0,
        stop=table.getNumberOfRows(),
        step=0
    )

    print(f"Images with >100 cells: {len(high_cell_count_rows)}")

    # 讀取這些列
    data = table.readCoordinates(high_cell_count_rows)
    for i in range(len(high_cell_count_rows)):
        img_id = data.columns[0].values[i]
        count = data.columns[3].values[i]
        print(f"  Image {img_id}: {count} cells")

    table.close()
```

## 從物件擷取表格

### 尋找附加到資料集的表格

```python
# 取得資料集
dataset = conn.getObject("Dataset", dataset_id)

# 列出檔案註解
for ann in dataset.listAnnotations():
    if isinstance(ann, omero.gateway.FileAnnotationWrapper):
        file_obj = ann.getFile()
        file_name = file_obj.getName()

        # 檢查是否為表格（可能有特定命名模式）
        if "Table" in file_name or file_name.endswith(".h5"):
            print(f"Found table: {file_name} (ID: {file_obj.getId()})")

            # 開啟並檢查
            resources = conn.c.sf.sharedResources()
            table = resources.openTable(file_obj._obj)

            print(f"  Rows: {table.getNumberOfRows()}")
            print(f"  Columns:")
            for col in table.getHeaders():
                print(f"    {col.name}")

            table.close()
```

## 更新表格

### 附加列

```python
# 開啟現有表格
resources = conn.c.sf.sharedResources()
table = resources.openTable(orig_file._obj)

# 準備新資料
new_image_ids = [106, 107]
new_intensities = [88.9, 92.3]
new_categories = ["Good", "Excellent"]

# 建立資料欄位
data_col1 = omero.grid.LongColumn('ImageID', '', new_image_ids)
data_col2 = omero.grid.DoubleColumn('MeanIntensity', '', new_intensities)
data_col3 = omero.grid.StringColumn('Category', '', 64, new_categories)

# 附加資料
table.addData([data_col1, data_col2, data_col3])

print(f"New row count: {table.getNumberOfRows()}")
table.close()
```

## 刪除表格

### 刪除表格檔案

```python
# 取得檔案物件
orig_file = conn.getObject("OriginalFile", file_id)

# 刪除檔案（同時刪除表格）
conn.deleteObjects("OriginalFile", [file_id], wait=True)
print(f"Deleted table file {file_id}")
```

### 取消表格與物件的連結

```python
# 尋找註解連結
dataset = conn.getObject("Dataset", dataset_id)

for ann in dataset.listAnnotations():
    if isinstance(ann, omero.gateway.FileAnnotationWrapper):
        if "Table" in ann.getFile().getName():
            # 刪除連結（保留表格，移除關聯）
            conn.deleteObjects("DatasetAnnotationLink",
                             [ann.link.getId()],
                             wait=True)
            print(f"Unlinked table from dataset")
```

## 最佳實務

1. **描述性名稱**：使用有意義的表格和欄位名稱
2. **關閉表格**：使用後務必關閉表格
3. **字串長度**：為字串欄位設定適當的最大長度
4. **連結到物件**：將表格附加到相關的資料集或專案
5. **使用參考**：為物件參考使用 ImageColumn、RoiColumn
6. **高效查詢**：使用 getWhereList() 而非讀取所有資料
7. **文件記錄**：為欄位新增描述
8. **版本控制**：在表格名稱或中繼資料中包含版本資訊
9. **批次操作**：以批次新增資料以獲得更好的效能
10. **錯誤處理**：檢查 None 傳回值並處理異常

## 常見模式

### ROI 測量表格

```python
# ROI 測量的表格結構
columns = [
    omero.grid.ImageColumn('Image', 'Source image', []),
    omero.grid.RoiColumn('ROI', 'Measured ROI', []),
    omero.grid.LongColumn('ChannelIndex', 'Channel number', []),
    omero.grid.DoubleColumn('Area', 'ROI area in pixels', []),
    omero.grid.DoubleColumn('MeanIntensity', 'Mean intensity', []),
    omero.grid.DoubleColumn('IntegratedDensity', 'Sum of intensities', []),
    omero.grid.StringColumn('CellType', 'Cell classification', 32, [])
]
```

### 時間序列資料表格

```python
# 時間序列測量的表格結構
columns = [
    omero.grid.ImageColumn('Image', 'Time series image', []),
    omero.grid.LongColumn('Timepoint', 'Time index', []),
    omero.grid.DoubleColumn('Timestamp', 'Time in seconds', []),
    omero.grid.DoubleColumn('Value', 'Measured value', []),
    omero.grid.StringColumn('Measurement', 'Type of measurement', 64, [])
]
```

### 篩選結果表格

```python
# 篩選板分析的表格結構
columns = [
    omero.grid.WellColumn('Well', 'Plate well', []),
    omero.grid.LongColumn('FieldIndex', 'Field number', []),
    omero.grid.DoubleColumn('CellCount', 'Number of cells', []),
    omero.grid.DoubleColumn('Viability', 'Percent viable', []),
    omero.grid.StringColumn('Phenotype', 'Observed phenotype', 128, []),
    omero.grid.BoolColumn('Hit', 'Hit in screen', [])
]
```
