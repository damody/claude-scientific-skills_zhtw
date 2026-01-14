# 資料存取與擷取

此參考涵蓋瀏覽 OMERO 的階層式資料結構和擷取物件。

## OMERO 資料階層

### 標準階層

```
Project（專案）
  └─ Dataset（資料集）
       └─ Image（影像）
```

### 篩選階層

```
Screen（篩選器）
  └─ Plate（板）
       └─ Well（孔）
            └─ WellSample（孔樣本）
                 └─ Image（影像）
```

## 列出物件

### 列出專案

```python
# 列出當前使用者的所有專案
for project in conn.listProjects():
    print(f"Project: {project.getName()} (ID: {project.getId()})")
```

### 使用篩選器列出專案

```python
# 取得當前使用者和群組
my_exp_id = conn.getUser().getId()
default_group_id = conn.getEventContext().groupId

# 使用篩選器列出專案
for project in conn.getObjects("Project", opts={
    'owner': my_exp_id,                    # 按所有者篩選
    'group': default_group_id,             # 按群組篩選
    'order_by': 'lower(obj.name)',         # 按字母順序排序
    'limit': 10,                           # 限制結果
    'offset': 0                            # 分頁偏移
}):
    print(f"Project: {project.getName()}")
```

### 列出資料集

```python
# 列出所有資料集
for dataset in conn.getObjects("Dataset"):
    print(f"Dataset: {dataset.getName()} (ID: {dataset.getId()})")

# 列出孤立資料集（不在任何專案中）
for dataset in conn.getObjects("Dataset", opts={'orphaned': True}):
    print(f"Orphaned Dataset: {dataset.getName()}")
```

### 列出影像

```python
# 列出所有影像
for image in conn.getObjects("Image"):
    print(f"Image: {image.getName()} (ID: {image.getId()})")

# 列出特定資料集中的影像
dataset_id = 123
for image in conn.getObjects("Image", opts={'dataset': dataset_id}):
    print(f"Image: {image.getName()}")

# 列出孤立影像
for image in conn.getObjects("Image", opts={'orphaned': True}):
    print(f"Orphaned Image: {image.getName()}")
```

## 透過 ID 擷取物件

### 取得單一物件

```python
# 透過 ID 取得專案
project = conn.getObject("Project", project_id)
if project:
    print(f"Project: {project.getName()}")
else:
    print("Project not found")

# 透過 ID 取得資料集
dataset = conn.getObject("Dataset", dataset_id)

# 透過 ID 取得影像
image = conn.getObject("Image", image_id)
```

### 透過 ID 取得多個物件

```python
# 一次取得多個專案
project_ids = [1, 2, 3, 4, 5]
projects = conn.getObjects("Project", project_ids)

for project in projects:
    print(f"Project: {project.getName()}")
```

### 支援的物件類型

`getObject()` 和 `getObjects()` 方法支援：
- `"Project"`
- `"Dataset"`
- `"Image"`
- `"Screen"`
- `"Plate"`
- `"Well"`
- `"Roi"`
- `"Annotation"`（以及特定類型：`"TagAnnotation"`、`"FileAnnotation"` 等）
- `"Experimenter"`
- `"ExperimenterGroup"`
- `"Fileset"`

## 按屬性查詢

### 按名稱查詢物件

```python
# 尋找具有特定名稱的影像
images = conn.getObjects("Image", attributes={"name": "sample_001.tif"})

for image in images:
    print(f"Found image: {image.getName()} (ID: {image.getId()})")

# 尋找具有特定名稱的資料集
datasets = conn.getObjects("Dataset", attributes={"name": "Control Group"})
```

### 按值查詢註解

```python
# 尋找具有特定文字值的標籤
tags = conn.getObjects("TagAnnotation",
                      attributes={"textValue": "experiment_tag"})

for tag in tags:
    print(f"Tag: {tag.getValue()}")

# 尋找對應註解
map_anns = conn.getObjects("MapAnnotation",
                          attributes={"ns": "custom.namespace"})
```

## 瀏覽階層

### 向下瀏覽（父到子）

```python
# 專案 → 資料集 → 影像
project = conn.getObject("Project", project_id)

for dataset in project.listChildren():
    print(f"Dataset: {dataset.getName()}")

    for image in dataset.listChildren():
        print(f"  Image: {image.getName()}")
```

### 向上瀏覽（子到父）

```python
# 影像 → 資料集 → 專案
image = conn.getObject("Image", image_id)

# 取得父資料集
dataset = image.getParent()
if dataset:
    print(f"Dataset: {dataset.getName()}")

    # 取得父專案
    project = dataset.getParent()
    if project:
        print(f"Project: {project.getName()}")
```

### 完整階層遍歷

```python
# 遍歷完整專案階層
for project in conn.getObjects("Project", opts={'order_by': 'lower(obj.name)'}):
    print(f"Project: {project.getName()} (ID: {project.getId()})")

    for dataset in project.listChildren():
        image_count = dataset.countChildren()
        print(f"  Dataset: {dataset.getName()} ({image_count} images)")

        for image in dataset.listChildren():
            print(f"    Image: {image.getName()}")
            print(f"      Size: {image.getSizeX()} x {image.getSizeY()}")
            print(f"      Channels: {image.getSizeC()}")
```

## 篩選資料存取

### 列出篩選器和板

```python
# 列出所有篩選器
for screen in conn.getObjects("Screen"):
    print(f"Screen: {screen.getName()} (ID: {screen.getId()})")

    # 列出篩選器中的板
    for plate in screen.listChildren():
        print(f"  Plate: {plate.getName()} (ID: {plate.getId()})")
```

### 存取板的孔

```python
# 取得板
plate = conn.getObject("Plate", plate_id)

# 板中繼資料
print(f"Plate: {plate.getName()}")
print(f"Grid size: {plate.getGridSize()}")  # 例如 96 孔板的 (8, 12)
print(f"Number of fields: {plate.getNumberOfFields()}")

# 遍歷孔
for well in plate.listChildren():
    print(f"Well at row {well.row}, column {well.column}")

    # 計算孔中的影像（視野）
    field_count = well.countWellSample()
    print(f"  Number of fields: {field_count}")

    # 存取孔中的影像
    for index in range(field_count):
        image = well.getImage(index)
        print(f"    Field {index}: {image.getName()}")
```

### 直接孔存取

```python
# 透過行和列取得特定孔
well = plate.getWell(row=0, column=0)  # 左上角孔

# 從孔取得影像
if well.countWellSample() > 0:
    image = well.getImage(0)  # 第一個視野
    print(f"Image: {image.getName()}")
```

### 孔樣本存取

```python
# 直接存取孔樣本
for well in plate.listChildren():
    for ws in well.listChildren():  # ws = WellSample
        image = ws.getImage()
        print(f"WellSample {ws.getId()}: {image.getName()}")
```

## 影像屬性

### 基本尺寸

```python
image = conn.getObject("Image", image_id)

# 像素尺寸
print(f"X: {image.getSizeX()}")
print(f"Y: {image.getSizeY()}")
print(f"Z: {image.getSizeZ()} (Z-sections)")
print(f"C: {image.getSizeC()} (Channels)")
print(f"T: {image.getSizeT()} (Time points)")

# 影像類型
print(f"Type: {image.getPixelsType()}")  # 例如 'uint16'、'uint8'
```

### 物理尺寸

```python
# 取得具有單位的像素大小（OMERO 5.1.0+）
size_x_obj = image.getPixelSizeX(units=True)
size_y_obj = image.getPixelSizeY(units=True)
size_z_obj = image.getPixelSizeZ(units=True)

print(f"Pixel Size X: {size_x_obj.getValue()} {size_x_obj.getSymbol()}")
print(f"Pixel Size Y: {size_y_obj.getValue()} {size_y_obj.getSymbol()}")
print(f"Pixel Size Z: {size_z_obj.getValue()} {size_z_obj.getSymbol()}")

# 取得浮點數（微米）
size_x = image.getPixelSizeX()  # 以 µm 為單位傳回浮點數
size_y = image.getPixelSizeY()
size_z = image.getPixelSizeZ()
```

### 通道資訊

```python
# 遍歷通道
for channel in image.getChannels():
    print(f"Channel {channel.getLabel()}:")
    print(f"  Color: {channel.getColor().getRGB()}")
    print(f"  Lookup Table: {channel.getLut()}")
    print(f"  Wavelength: {channel.getEmissionWave()}")
```

### 影像中繼資料

```python
# 擷取日期
acquired = image.getAcquisitionDate()
if acquired:
    print(f"Acquired: {acquired}")

# 描述
description = image.getDescription()
if description:
    print(f"Description: {description}")

# 所有者和群組
details = image.getDetails()
print(f"Owner: {details.getOwner().getFullName()}")
print(f"Username: {details.getOwner().getOmeName()}")
print(f"Group: {details.getGroup().getName()}")
print(f"Created: {details.getCreationEvent().getTime()}")
```

## 物件所有權和權限

### 取得所有者資訊

```python
# 取得物件所有者
obj = conn.getObject("Dataset", dataset_id)
owner = obj.getDetails().getOwner()

print(f"Owner ID: {owner.getId()}")
print(f"Username: {owner.getOmeName()}")
print(f"Full Name: {owner.getFullName()}")
print(f"Email: {owner.getEmail()}")
```

### 取得群組資訊

```python
# 取得物件的群組
obj = conn.getObject("Image", image_id)
group = obj.getDetails().getGroup()

print(f"Group: {group.getName()} (ID: {group.getId()})")
```

### 按所有者篩選

```python
# 取得特定使用者的物件
user_id = 5
datasets = conn.getObjects("Dataset", opts={'owner': user_id})

for dataset in datasets:
    print(f"Dataset: {dataset.getName()}")
```

## 進階查詢

### 分頁

```python
# 分頁瀏覽大型結果集
page_size = 50
offset = 0

while True:
    images = list(conn.getObjects("Image", opts={
        'limit': page_size,
        'offset': offset,
        'order_by': 'obj.id'
    }))

    if not images:
        break

    for image in images:
        print(f"Image: {image.getName()}")

    offset += page_size
```

### 排序結果

```python
# 按名稱排序（不區分大小寫）
projects = conn.getObjects("Project", opts={
    'order_by': 'lower(obj.name)'
})

# 按 ID 排序（升序）
datasets = conn.getObjects("Dataset", opts={
    'order_by': 'obj.id'
})

# 按名稱排序（降序）
images = conn.getObjects("Image", opts={
    'order_by': 'lower(obj.name) desc'
})
```

### 組合篩選器

```python
# 使用多個篩選器的複雜查詢
my_exp_id = conn.getUser().getId()
default_group_id = conn.getEventContext().groupId

images = conn.getObjects("Image", opts={
    'owner': my_exp_id,
    'group': default_group_id,
    'dataset': dataset_id,
    'order_by': 'lower(obj.name)',
    'limit': 100,
    'offset': 0
})
```

## 計算物件

### 計算子物件

```python
# 計算資料集中的影像
dataset = conn.getObject("Dataset", dataset_id)
image_count = dataset.countChildren()
print(f"Dataset contains {image_count} images")

# 計算專案中的資料集
project = conn.getObject("Project", project_id)
dataset_count = project.countChildren()
print(f"Project contains {dataset_count} datasets")
```

### 計算註解

```python
# 計算物件上的註解
image = conn.getObject("Image", image_id)
annotation_count = image.countAnnotations()
print(f"Image has {annotation_count} annotations")
```

## 孤立物件

### 尋找孤立資料集

```python
# 未連結到任何專案的資料集
orphaned_datasets = conn.getObjects("Dataset", opts={'orphaned': True})

print("Orphaned Datasets:")
for dataset in orphaned_datasets:
    print(f"  {dataset.getName()} (ID: {dataset.getId()})")
    print(f"    Owner: {dataset.getDetails().getOwner().getOmeName()}")
    print(f"    Images: {dataset.countChildren()}")
```

### 尋找孤立影像

```python
# 不在任何資料集中的影像
orphaned_images = conn.getObjects("Image", opts={'orphaned': True})

print("Orphaned Images:")
for image in orphaned_images:
    print(f"  {image.getName()} (ID: {image.getId()})")
```

### 尋找孤立板

```python
# 不在任何篩選器中的板
orphaned_plates = conn.getObjects("Plate", opts={'orphaned': True})

for plate in orphaned_plates:
    print(f"Orphaned Plate: {plate.getName()}")
```

## 完整範例

```python
from omero.gateway import BlitzGateway

# 連線詳情
HOST = 'omero.example.com'
PORT = 4064
USERNAME = 'user'
PASSWORD = 'pass'

# 連線並查詢資料
with BlitzGateway(USERNAME, PASSWORD, host=HOST, port=PORT) as conn:
    # 取得使用者上下文
    user = conn.getUser()
    group = conn.getGroupFromContext()

    print(f"Connected as {user.getName()} in group {group.getName()}")
    print()

    # 列出專案及其資料集和影像
    for project in conn.getObjects("Project", opts={'limit': 5}):
        print(f"Project: {project.getName()} (ID: {project.getId()})")

        for dataset in project.listChildren():
            image_count = dataset.countChildren()
            print(f"  Dataset: {dataset.getName()} ({image_count} images)")

            # 顯示前 3 張影像
            for idx, image in enumerate(dataset.listChildren()):
                if idx >= 3:
                    print(f"    ... and {image_count - 3} more")
                    break
                print(f"    Image: {image.getName()}")
                print(f"      Size: {image.getSizeX()}x{image.getSizeY()}")
                print(f"      Channels: {image.getSizeC()}, Z: {image.getSizeZ()}")

        print()
```

## 最佳實務

1. **使用上下文管理器**：務必使用 `with` 陳述式進行自動連線清理
2. **限制結果**：對大型資料集使用 `limit` 和 `offset`
3. **提早篩選**：應用篩選器以減少資料傳輸
4. **檢查 None**：在使用前務必檢查 `getObject()` 是否傳回 None
5. **高效遍歷**：使用 `listChildren()` 而非單獨查詢
6. **載入前計數**：使用 `countChildren()` 決定是否載入資料
7. **群組上下文**：在跨群組查詢前設定適當的群組上下文
8. **分頁**：對大型結果集實作分頁
9. **物件重用**：快取經常存取的物件以減少查詢
10. **錯誤處理**：將查詢包裝在 try-except 區塊中以確保穩健性
