# 腳本與批次操作

此參考涵蓋建立用於伺服器端處理和批次操作的 OMERO.scripts。

## OMERO.scripts 概述

OMERO.scripts 是在 OMERO 伺服器上執行的 Python 腳本，可從 OMERO 用戶端（web、insight、CLI）呼叫。它們作為擴展 OMERO 功能的外掛程式運作。

### 主要功能

- **伺服器端執行**：腳本在伺服器上執行，避免資料傳輸
- **用戶端整合**：可從任何 OMERO 用戶端呼叫，具有自動生成的 UI
- **參數處理**：定義具有驗證的輸入參數
- **結果回報**：將影像、檔案或訊息傳回用戶端
- **批次處理**：高效處理多張影像或資料集

## 基本腳本結構

### 最小腳本範本

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import omero
from omero.gateway import BlitzGateway
import omero.scripts as scripts
from omero.rtypes import rlong, rstring, robject

def run_script():
    """
    主腳本函數。
    """
    # 腳本定義
    client = scripts.client(
        'Script_Name.py',
        """
        描述此腳本的功能。
        """,

        # 輸入參數
        scripts.String("Data_Type", optional=False, grouping="1",
                      description="選擇影像來源",
                      values=[rstring('Dataset'), rstring('Image')],
                      default=rstring('Dataset')),

        scripts.Long("IDs", optional=False, grouping="2",
                    description="資料集或影像 ID").ofType(rlong(0)),

        # 輸出
        namespaces=[omero.constants.namespaces.NSDYNAMIC],
        version="1.0"
    )

    try:
        # 取得連線
        conn = BlitzGateway(client_obj=client)

        # 取得腳本參數
        script_params = client.getInputs(unwrap=True)
        data_type = script_params["Data_Type"]
        ids = script_params["IDs"]

        # 處理資料
        message = process_data(conn, data_type, ids)

        # 傳回結果
        client.setOutput("Message", rstring(message))

    finally:
        client.closeSession()

def process_data(conn, data_type, ids):
    """
    根據參數處理影像。
    """
    # 在此實作
    return "Processing complete"

if __name__ == "__main__":
    run_script()
```

## 腳本參數

### 參數類型

```python
# 字串參數
scripts.String("Name", optional=False,
              description="輸入名稱")

# 具有選項的字串
scripts.String("Mode", optional=False,
              values=[rstring('Fast'), rstring('Accurate')],
              default=rstring('Fast'))

# 整數參數
scripts.Long("ImageID", optional=False,
            description="要處理的影像").ofType(rlong(0))

# 整數列表
scripts.List("ImageIDs", optional=False,
            description="多張影像").ofType(rlong(0))

# 浮點數參數
scripts.Float("Threshold", optional=True,
             description="閾值",
             min=0.0, max=1.0, default=0.5)

# 布林參數
scripts.Bool("SaveResults", optional=True,
            description="將結果儲存到 OMERO",
            default=True)
```

### 參數分組

```python
# 將相關參數分組
scripts.String("Data_Type", grouping="1",
              description="來源類型",
              values=[rstring('Dataset'), rstring('Image')])

scripts.Long("Dataset_ID", grouping="1.1",
            description="資料集 ID").ofType(rlong(0))

scripts.List("Image_IDs", grouping="1.2",
            description="影像 ID").ofType(rlong(0))
```

## 存取輸入資料

### 取得腳本參數

```python
# 在 run_script() 內部
client = scripts.client(...)

# 以 Python 物件取得參數
script_params = client.getInputs(unwrap=True)

# 存取個別參數
data_type = script_params.get("Data_Type", "Image")
image_ids = script_params.get("Image_IDs", [])
threshold = script_params.get("Threshold", 0.5)
save_results = script_params.get("SaveResults", True)
```

### 從參數取得影像

```python
def get_images_from_params(conn, script_params):
    """
    根據腳本參數取得影像物件。
    """
    images = []

    data_type = script_params["Data_Type"]

    if data_type == "Dataset":
        dataset_id = script_params["Dataset_ID"]
        dataset = conn.getObject("Dataset", dataset_id)
        if dataset:
            images = list(dataset.listChildren())

    elif data_type == "Image":
        image_ids = script_params["Image_IDs"]
        for image_id in image_ids:
            image = conn.getObject("Image", image_id)
            if image:
                images.append(image)

    return images
```

## 處理影像

### 批次影像處理

```python
def process_images(conn, images, threshold):
    """
    處理多張影像。
    """
    results = []

    for image in images:
        print(f"Processing: {image.getName()}")

        # 取得像素資料
        pixels = image.getPrimaryPixels()
        size_z = image.getSizeZ()
        size_c = image.getSizeC()
        size_t = image.getSizeT()

        # 處理每個平面
        for z in range(size_z):
            for c in range(size_c):
                for t in range(size_t):
                    plane = pixels.getPlane(z, c, t)

                    # 套用閾值
                    binary = (plane > threshold).astype(np.uint8)

                    # 計算特徵
                    feature_count = count_features(binary)

                    results.append({
                        'image_id': image.getId(),
                        'image_name': image.getName(),
                        'z': z, 'c': c, 't': t,
                        'feature_count': feature_count
                    })

    return results
```

## 生成輸出

### 傳回訊息

```python
# 簡單訊息
message = "Processed 10 images successfully"
client.setOutput("Message", rstring(message))

# 詳細訊息
message = "Results:\n"
for result in results:
    message += f"Image {result['image_id']}: {result['count']} cells\n"
client.setOutput("Message", rstring(message))
```

### 傳回影像

```python
# 傳回新建立的影像
new_image = conn.createImageFromNumpySeq(...)
client.setOutput("New_Image", robject(new_image._obj))
```

### 傳回檔案

```python
# 建立並傳回檔案註解
file_ann = conn.createFileAnnfromLocalFile(
    output_file_path,
    mimetype="text/csv",
    ns="analysis.results"
)

client.setOutput("Result_File", robject(file_ann._obj))
```

### 傳回表格

```python
# 建立 OMERO 表格並傳回
resources = conn.c.sf.sharedResources()
table = create_results_table(resources, results)
orig_file = table.getOriginalFile()
table.close()

# 建立檔案註解
file_ann = omero.model.FileAnnotationI()
file_ann.setFile(orig_file)
file_ann = conn.getUpdateService().saveAndReturnObject(file_ann)

client.setOutput("Results_Table", robject(file_ann._obj))
```

## 完整範例腳本

### 範例 1：最大強度投影

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import omero
from omero.gateway import BlitzGateway
import omero.scripts as scripts
from omero.rtypes import rlong, rstring, robject
import numpy as np

def run_script():
    client = scripts.client(
        'Maximum_Intensity_Projection.py',
        """
        從 Z 堆疊影像建立最大強度投影。
        """,

        scripts.String("Data_Type", optional=False, grouping="1",
                      description="處理影像來源",
                      values=[rstring('Dataset'), rstring('Image')],
                      default=rstring('Image')),

        scripts.List("IDs", optional=False, grouping="2",
                    description="資料集或影像 ID").ofType(rlong(0)),

        scripts.Bool("Link_to_Source", optional=True, grouping="3",
                    description="將結果連結到來源資料集",
                    default=True),

        version="1.0"
    )

    try:
        conn = BlitzGateway(client_obj=client)
        script_params = client.getInputs(unwrap=True)

        # 取得影像
        images = get_images(conn, script_params)
        created_images = []

        for image in images:
            print(f"Processing: {image.getName()}")

            # 建立 MIP
            mip_image = create_mip(conn, image)
            if mip_image:
                created_images.append(mip_image)

        # 回報結果
        if created_images:
            message = f"Created {len(created_images)} MIP images"
            # 傳回第一張影像以供顯示
            client.setOutput("Message", rstring(message))
            client.setOutput("Result", robject(created_images[0]._obj))
        else:
            client.setOutput("Message", rstring("No images created"))

    finally:
        client.closeSession()

def get_images(conn, script_params):
    """從腳本參數取得影像。"""
    images = []
    data_type = script_params["Data_Type"]
    ids = script_params["IDs"]

    if data_type == "Dataset":
        for dataset_id in ids:
            dataset = conn.getObject("Dataset", dataset_id)
            if dataset:
                images.extend(list(dataset.listChildren()))
    else:
        for image_id in ids:
            image = conn.getObject("Image", image_id)
            if image:
                images.append(image)

    return images

def create_mip(conn, source_image):
    """建立最大強度投影。"""
    pixels = source_image.getPrimaryPixels()
    size_z = source_image.getSizeZ()
    size_c = source_image.getSizeC()
    size_t = source_image.getSizeT()

    if size_z == 1:
        print("  Skipping (single Z-section)")
        return None

    def plane_gen():
        for c in range(size_c):
            for t in range(size_t):
                # 取得 Z 堆疊
                z_stack = []
                for z in range(size_z):
                    plane = pixels.getPlane(z, c, t)
                    z_stack.append(plane)

                # 最大投影
                max_proj = np.max(z_stack, axis=0)
                yield max_proj

    # 建立新影像
    new_image = conn.createImageFromNumpySeq(
        plane_gen(),
        f"{source_image.getName()}_MIP",
        1, size_c, size_t,
        description="Maximum intensity projection",
        dataset=source_image.getParent()
    )

    return new_image

if __name__ == "__main__":
    run_script()
```

### 範例 2：批次 ROI 分析

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import omero
from omero.gateway import BlitzGateway
import omero.scripts as scripts
from omero.rtypes import rlong, rstring, robject
import omero.grid

def run_script():
    client = scripts.client(
        'Batch_ROI_Analysis.py',
        """
        分析多張影像的 ROI 並建立結果表格。
        """,

        scripts.Long("Dataset_ID", optional=False,
                    description="具有影像和 ROI 的資料集").ofType(rlong(0)),

        scripts.Long("Channel_Index", optional=True,
                    description="要分析的通道（0 索引）",
                    default=0, min=0),

        version="1.0"
    )

    try:
        conn = BlitzGateway(client_obj=client)
        script_params = client.getInputs(unwrap=True)

        dataset_id = script_params["Dataset_ID"]
        channel_index = script_params["Channel_Index"]

        # 取得資料集
        dataset = conn.getObject("Dataset", dataset_id)
        if not dataset:
            client.setOutput("Message", rstring("Dataset not found"))
            return

        # 分析 ROI
        results = analyze_rois(conn, dataset, channel_index)

        # 建立表格
        table_file = create_results_table(conn, dataset, results)

        # 回報
        message = f"Analyzed {len(results)} ROIs from {dataset.getName()}"
        client.setOutput("Message", rstring(message))
        client.setOutput("Results_Table", robject(table_file._obj))

    finally:
        client.closeSession()

def analyze_rois(conn, dataset, channel_index):
    """分析資料集影像中的所有 ROI。"""
    roi_service = conn.getRoiService()
    results = []

    for image in dataset.listChildren():
        result = roi_service.findByImage(image.getId(), None)

        if not result.rois:
            continue

        # 取得形狀 ID
        shape_ids = []
        for roi in result.rois:
            for shape in roi.copyShapes():
                shape_ids.append(shape.id.val)

        # 取得統計資料
        stats = roi_service.getShapeStatsRestricted(
            shape_ids, 0, 0, [channel_index]
        )

        # 儲存結果
        for i, stat in enumerate(stats):
            results.append({
                'image_id': image.getId(),
                'image_name': image.getName(),
                'shape_id': shape_ids[i],
                'mean': stat.mean[channel_index],
                'min': stat.min[channel_index],
                'max': stat.max[channel_index],
                'sum': stat.sum[channel_index],
                'area': stat.pointsCount[channel_index]
            })

    return results

def create_results_table(conn, dataset, results):
    """從結果建立 OMERO 表格。"""
    # 準備資料
    image_ids = [r['image_id'] for r in results]
    shape_ids = [r['shape_id'] for r in results]
    means = [r['mean'] for r in results]
    mins = [r['min'] for r in results]
    maxs = [r['max'] for r in results]
    sums = [r['sum'] for r in results]
    areas = [r['area'] for r in results]

    # 建立表格
    resources = conn.c.sf.sharedResources()
    repository_id = resources.repositories().descriptions[0].getId().getValue()
    table = resources.newTable(repository_id, f"ROI_Analysis_{dataset.getId()}")

    # 定義欄位
    columns = [
        omero.grid.ImageColumn('Image', 'Source image', []),
        omero.grid.LongColumn('ShapeID', 'ROI shape ID', []),
        omero.grid.DoubleColumn('Mean', 'Mean intensity', []),
        omero.grid.DoubleColumn('Min', 'Min intensity', []),
        omero.grid.DoubleColumn('Max', 'Max intensity', []),
        omero.grid.DoubleColumn('Sum', 'Integrated density', []),
        omero.grid.LongColumn('Area', 'Area in pixels', [])
    ]
    table.initialize(columns)

    # 新增資料
    data = [
        omero.grid.ImageColumn('Image', 'Source image', image_ids),
        omero.grid.LongColumn('ShapeID', 'ROI shape ID', shape_ids),
        omero.grid.DoubleColumn('Mean', 'Mean intensity', means),
        omero.grid.DoubleColumn('Min', 'Min intensity', mins),
        omero.grid.DoubleColumn('Max', 'Max intensity', maxs),
        omero.grid.DoubleColumn('Sum', 'Integrated density', sums),
        omero.grid.LongColumn('Area', 'Area in pixels', areas)
    ]
    table.addData(data)

    orig_file = table.getOriginalFile()
    table.close()

    # 連結到資料集
    file_ann = omero.model.FileAnnotationI()
    file_ann.setFile(orig_file)
    file_ann = conn.getUpdateService().saveAndReturnObject(file_ann)

    link = omero.model.DatasetAnnotationLinkI()
    link.setParent(dataset._obj)
    link.setChild(file_ann)
    conn.getUpdateService().saveAndReturnObject(link)

    return file_ann

if __name__ == "__main__":
    run_script()
```

## 腳本部署

### 安裝位置

腳本應放置在 OMERO 伺服器腳本目錄中：
```
OMERO_DIR/lib/scripts/
```

### 建議結構

```
lib/scripts/
├── analysis/
│   ├── Cell_Counter.py
│   └── ROI_Analyzer.py
├── export/
│   ├── Export_Images.py
│   └── Export_ROIs.py
└── util/
    └── Helper_Functions.py
```

### 測試腳本

```bash
# 測試腳本語法
python Script_Name.py

# 上傳到 OMERO
omero script upload Script_Name.py

# 列出腳本
omero script list

# 從 CLI 執行腳本
omero script launch Script_ID Dataset_ID=123
```

## 最佳實務

1. **錯誤處理**：務必使用 try-finally 關閉會話
2. **進度更新**：為長時間操作列印狀態訊息
3. **參數驗證**：在處理前檢查參數
4. **記憶體管理**：以批次處理大型資料集
5. **文件記錄**：包含清晰的描述和參數文件
6. **版本控制**：在腳本中包含版本號
7. **命名空間**：為輸出使用適當的命名空間
8. **傳回物件**：傳回建立的物件以供用戶端顯示
9. **日誌記錄**：使用 print() 記錄到伺服器日誌
10. **測試**：使用各種輸入組合進行測試

## 常見模式

### 進度回報

```python
total = len(images)
for idx, image in enumerate(images):
    print(f"Processing {idx + 1}/{total}: {image.getName()}")
    # 處理影像
```

### 錯誤收集

```python
errors = []
for image in images:
    try:
        process_image(image)
    except Exception as e:
        errors.append(f"{image.getName()}: {str(e)}")

if errors:
    message = "Completed with errors:\n" + "\n".join(errors)
else:
    message = "All images processed successfully"
```

### 資源清理

```python
try:
    # 腳本處理
    pass
finally:
    # 清理臨時檔案
    if os.path.exists(temp_file):
        os.remove(temp_file)
    client.closeSession()
```
