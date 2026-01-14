# 感興趣區域 (ROI)

此參考涵蓋在 OMERO 中建立、擷取和分析 ROI。

## ROI 概述

OMERO 中的 ROI（感興趣區域）是幾何形狀的容器，用於標記影像上的特定區域。每個 ROI 可以包含多個形狀，且形狀可以針對特定的 Z 切面和時間點。

### 支援的形狀類型

- **Rectangle**：矩形區域
- **Ellipse**：圓形和橢圓形區域
- **Line**：線段
- **Point**：單點
- **Polygon**：多點多邊形
- **Mask**：基於像素的遮罩
- **Polyline**：多段線

## 建立 ROI

### 輔助函數

```python
from omero.rtypes import rdouble, rint, rstring
import omero.model

def create_roi(conn, image, shapes):
    """
    建立 ROI 並將其連結到形狀。

    參數：
        conn：BlitzGateway 連線
        image：Image 物件
        shapes：形狀物件列表

    傳回：
        已儲存的 ROI 物件
    """
    roi = omero.model.RoiI()
    roi.setImage(image._obj)

    for shape in shapes:
        roi.addShape(shape)

    updateService = conn.getUpdateService()
    return updateService.saveAndReturnObject(roi)

def rgba_to_int(red, green, blue, alpha=255):
    """
    將 RGBA 值（0-255）轉換為 OMERO 的整數編碼。

    參數：
        red, green, blue, alpha：顏色值（0-255）

    傳回：
        整數顏色值
    """
    return int.from_bytes([red, green, blue, alpha],
                          byteorder='big', signed=True)
```

### 矩形 ROI

```python
from omero.rtypes import rdouble, rint, rstring
import omero.model

# 取得影像
image = conn.getObject("Image", image_id)

# 定義位置和大小
x, y = 50, 100
width, height = 200, 150
z, t = 0, 0  # Z 切面和時間點

# 建立矩形
rect = omero.model.RectangleI()
rect.x = rdouble(x)
rect.y = rdouble(y)
rect.width = rdouble(width)
rect.height = rdouble(height)
rect.theZ = rint(z)
rect.theT = rint(t)

# 設定標籤和顏色
rect.textValue = rstring("Cell Region")
rect.fillColor = rint(rgba_to_int(255, 0, 0, 50))    # 紅色，半透明
rect.strokeColor = rint(rgba_to_int(255, 255, 0, 255))  # 黃色邊框

# 建立 ROI
roi = create_roi(conn, image, [rect])
print(f"Created ROI ID: {roi.getId().getValue()}")
```

### 橢圓 ROI

```python
# 中心位置和半徑
center_x, center_y = 250, 250
radius_x, radius_y = 100, 75
z, t = 0, 0

# 建立橢圓
ellipse = omero.model.EllipseI()
ellipse.x = rdouble(center_x)
ellipse.y = rdouble(center_y)
ellipse.radiusX = rdouble(radius_x)
ellipse.radiusY = rdouble(radius_y)
ellipse.theZ = rint(z)
ellipse.theT = rint(t)
ellipse.textValue = rstring("Nucleus")
ellipse.fillColor = rint(rgba_to_int(0, 255, 0, 50))

# 建立 ROI
roi = create_roi(conn, image, [ellipse])
```

### 線段 ROI

```python
# 線段端點
x1, y1 = 100, 100
x2, y2 = 300, 200
z, t = 0, 0

# 建立線段
line = omero.model.LineI()
line.x1 = rdouble(x1)
line.y1 = rdouble(y1)
line.x2 = rdouble(x2)
line.y2 = rdouble(y2)
line.theZ = rint(z)
line.theT = rint(t)
line.textValue = rstring("Measurement Line")
line.strokeColor = rint(rgba_to_int(0, 0, 255, 255))

# 建立 ROI
roi = create_roi(conn, image, [line])
```

### 點 ROI

```python
# 點位置
x, y = 150, 150
z, t = 0, 0

# 建立點
point = omero.model.PointI()
point.x = rdouble(x)
point.y = rdouble(y)
point.theZ = rint(z)
point.theT = rint(t)
point.textValue = rstring("Feature Point")

# 建立 ROI
roi = create_roi(conn, image, [point])
```

### 多邊形 ROI

```python
from omero.model.enums import UnitsLength

# 以字串定義頂點 "x1,y1 x2,y2 x3,y3 ..."
vertices = "10,20 50,150 200,200 250,75"
z, t = 0, 0

# 建立多邊形
polygon = omero.model.PolygonI()
polygon.points = rstring(vertices)
polygon.theZ = rint(z)
polygon.theT = rint(t)
polygon.textValue = rstring("Cell Outline")

# 設定顏色和線條寬度
polygon.fillColor = rint(rgba_to_int(255, 0, 255, 50))
polygon.strokeColor = rint(rgba_to_int(255, 255, 0, 255))
polygon.strokeWidth = omero.model.LengthI(2, UnitsLength.PIXEL)

# 建立 ROI
roi = create_roi(conn, image, [polygon])
```

### 遮罩 ROI

```python
import numpy as np
import struct
import math

def create_mask_bytes(mask_array, bytes_per_pixel=1):
    """
    將二進位遮罩陣列轉換為 OMERO 的位元打包位元組。

    參數：
        mask_array：二進位 numpy 陣列（0 和 1）
        bytes_per_pixel：1 或 2

    傳回：
        OMERO 遮罩的位元組陣列
    """
    if bytes_per_pixel == 2:
        divider = 16.0
        format_string = "H"
        byte_factor = 0.5
    elif bytes_per_pixel == 1:
        divider = 8.0
        format_string = "B"
        byte_factor = 1
    else:
        raise ValueError("bytes_per_pixel must be 1 or 2")

    mask_bytes = mask_array.astype(np.uint8).tobytes()
    steps = math.ceil(len(mask_bytes) / divider)
    packed_mask = []

    for i in range(int(steps)):
        binary = mask_bytes[i * int(divider):
                           i * int(divider) + int(divider)]
        format_str = str(int(byte_factor * len(binary))) + format_string
        binary = struct.unpack(format_str, binary)
        s = "".join(str(bit) for bit in binary)
        packed_mask.append(int(s, 2))

    return bytearray(packed_mask)

# 建立二進位遮罩（1 和 0）
mask_w, mask_h = 100, 100
mask_array = np.fromfunction(
    lambda x, y: ((x - 50)**2 + (y - 50)**2) < 40**2,  # 圓形
    (mask_w, mask_h)
)

# 打包遮罩
mask_packed = create_mask_bytes(mask_array, bytes_per_pixel=1)

# 遮罩位置
mask_x, mask_y = 50, 50
z, t, c = 0, 0, 0

# 建立遮罩
mask = omero.model.MaskI()
mask.setX(rdouble(mask_x))
mask.setY(rdouble(mask_y))
mask.setWidth(rdouble(mask_w))
mask.setHeight(rdouble(mask_h))
mask.setTheZ(rint(z))
mask.setTheT(rint(t))
mask.setTheC(rint(c))
mask.setBytes(mask_packed)
mask.textValue = rstring("Segmentation Mask")

# 設定顏色
from omero.gateway import ColorHolder
mask_color = ColorHolder()
mask_color.setRed(255)
mask_color.setGreen(0)
mask_color.setBlue(0)
mask_color.setAlpha(100)
mask.setFillColor(rint(mask_color.getInt()))

# 建立 ROI
roi = create_roi(conn, image, [mask])
```

## 單一 ROI 中的多個形狀

```python
# 為同一 ROI 建立多個形狀
shapes = []

# 矩形
rect = omero.model.RectangleI()
rect.x = rdouble(100)
rect.y = rdouble(100)
rect.width = rdouble(50)
rect.height = rdouble(50)
rect.theZ = rint(0)
rect.theT = rint(0)
shapes.append(rect)

# 橢圓
ellipse = omero.model.EllipseI()
ellipse.x = rdouble(125)
ellipse.y = rdouble(125)
ellipse.radiusX = rdouble(20)
ellipse.radiusY = rdouble(20)
ellipse.theZ = rint(0)
ellipse.theT = rint(0)
shapes.append(ellipse)

# 建立具有兩個形狀的單一 ROI
roi = create_roi(conn, image, shapes)
```

## 擷取 ROI

### 取得影像的所有 ROI

```python
# 取得 ROI 服務
roi_service = conn.getRoiService()

# 尋找影像的所有 ROI
result = roi_service.findByImage(image_id, None)

print(f"Found {len(result.rois)} ROIs")

for roi in result.rois:
    print(f"ROI ID: {roi.getId().getValue()}")
    print(f"  Number of shapes: {len(roi.copyShapes())}")
```

### 解析 ROI 形狀

```python
import omero.model

result = roi_service.findByImage(image_id, None)

for roi in result.rois:
    roi_id = roi.getId().getValue()
    print(f"ROI ID: {roi_id}")

    for shape in roi.copyShapes():
        shape_id = shape.getId().getValue()
        z = shape.getTheZ().getValue() if shape.getTheZ() else None
        t = shape.getTheT().getValue() if shape.getTheT() else None

        # 取得標籤
        label = ""
        if shape.getTextValue():
            label = shape.getTextValue().getValue()

        print(f"  Shape ID: {shape_id}, Z: {z}, T: {t}, Label: {label}")

        # 類型特定解析
        if isinstance(shape, omero.model.RectangleI):
            x = shape.getX().getValue()
            y = shape.getY().getValue()
            width = shape.getWidth().getValue()
            height = shape.getHeight().getValue()
            print(f"    Rectangle: ({x}, {y}) {width}x{height}")

        elif isinstance(shape, omero.model.EllipseI):
            x = shape.getX().getValue()
            y = shape.getY().getValue()
            rx = shape.getRadiusX().getValue()
            ry = shape.getRadiusY().getValue()
            print(f"    Ellipse: center ({x}, {y}), radii ({rx}, {ry})")

        elif isinstance(shape, omero.model.PointI):
            x = shape.getX().getValue()
            y = shape.getY().getValue()
            print(f"    Point: ({x}, {y})")

        elif isinstance(shape, omero.model.LineI):
            x1 = shape.getX1().getValue()
            y1 = shape.getY1().getValue()
            x2 = shape.getX2().getValue()
            y2 = shape.getY2().getValue()
            print(f"    Line: ({x1}, {y1}) to ({x2}, {y2})")

        elif isinstance(shape, omero.model.PolygonI):
            points = shape.getPoints().getValue()
            print(f"    Polygon: {points}")

        elif isinstance(shape, omero.model.MaskI):
            x = shape.getX().getValue()
            y = shape.getY().getValue()
            width = shape.getWidth().getValue()
            height = shape.getHeight().getValue()
            print(f"    Mask: ({x}, {y}) {width}x{height}")
```

## 分析 ROI 強度

### 取得 ROI 形狀的統計資料

```python
# 從 ROI 取得所有形狀
roi_service = conn.getRoiService()
result = roi_service.findByImage(image_id, None)

shape_ids = []
for roi in result.rois:
    for shape in roi.copyShapes():
        shape_ids.append(shape.id.val)

# 定義位置
z, t = 0, 0
channel_index = 0

# 取得統計資料
stats = roi_service.getShapeStatsRestricted(
    shape_ids, z, t, [channel_index]
)

# 顯示統計資料
for i, stat in enumerate(stats):
    shape_id = shape_ids[i]
    print(f"Shape {shape_id} statistics:")
    print(f"  Points Count: {stat.pointsCount[channel_index]}")
    print(f"  Min: {stat.min[channel_index]}")
    print(f"  Mean: {stat.mean[channel_index]}")
    print(f"  Max: {stat.max[channel_index]}")
    print(f"  Sum: {stat.sum[channel_index]}")
    print(f"  Std Dev: {stat.stdDev[channel_index]}")
```

### 提取 ROI 內的像素值

```python
import numpy as np

# 取得影像和 ROI
image = conn.getObject("Image", image_id)
result = roi_service.findByImage(image_id, None)

# 取得第一個矩形形狀
roi = result.rois[0]
rect = roi.copyShapes()[0]

# 取得矩形邊界
x = int(rect.getX().getValue())
y = int(rect.getY().getValue())
width = int(rect.getWidth().getValue())
height = int(rect.getHeight().getValue())
z = rect.getTheZ().getValue()
t = rect.getTheT().getValue()

# 取得像素資料
pixels = image.getPrimaryPixels()

# 提取每個通道的區域
for c in range(image.getSizeC()):
    # 取得平面
    plane = pixels.getPlane(z, c, t)

    # 提取 ROI 區域
    roi_region = plane[y:y+height, x:x+width]

    print(f"Channel {c}:")
    print(f"  Mean intensity: {np.mean(roi_region)}")
    print(f"  Max intensity: {np.max(roi_region)}")
```

## 修改 ROI

### 更新形狀屬性

```python
# 取得 ROI 和形狀
result = roi_service.findByImage(image_id, None)
roi = result.rois[0]
shape = roi.copyShapes()[0]

# 修改形狀（範例：變更矩形大小）
if isinstance(shape, omero.model.RectangleI):
    shape.setWidth(rdouble(150))
    shape.setHeight(rdouble(100))
    shape.setTextValue(rstring("Updated Rectangle"))

# 儲存變更
updateService = conn.getUpdateService()
updated_roi = updateService.saveAndReturnObject(roi._obj)
```

### 從 ROI 移除形狀

```python
result = roi_service.findByImage(image_id, None)

for roi in result.rois:
    for shape in roi.copyShapes():
        # 檢查條件（例如，按標籤移除）
        if (shape.getTextValue() and
            shape.getTextValue().getValue() == "test-Ellipse"):

            print(f"Removing shape {shape.getId().getValue()}")
            roi.removeShape(shape)

            # 儲存修改後的 ROI
            updateService = conn.getUpdateService()
            roi = updateService.saveAndReturnObject(roi)
```

## 刪除 ROI

### 刪除單一 ROI

```python
# 透過 ID 刪除 ROI
roi_id = 123
conn.deleteObjects("Roi", [roi_id], wait=True)
print(f"Deleted ROI {roi_id}")
```

### 刪除影像的所有 ROI

```python
# 取得影像的所有 ROI ID
result = roi_service.findByImage(image_id, None)
roi_ids = [roi.getId().getValue() for roi in result.rois]

# 刪除全部
if roi_ids:
    conn.deleteObjects("Roi", roi_ids, wait=True)
    print(f"Deleted {len(roi_ids)} ROIs")
```

## 批次 ROI 建立

### 為多張影像建立 ROI

```python
# 取得影像
dataset = conn.getObject("Dataset", dataset_id)

for image in dataset.listChildren():
    # 在每張影像的中心建立矩形
    x = image.getSizeX() // 2 - 50
    y = image.getSizeY() // 2 - 50

    rect = omero.model.RectangleI()
    rect.x = rdouble(x)
    rect.y = rdouble(y)
    rect.width = rdouble(100)
    rect.height = rdouble(100)
    rect.theZ = rint(0)
    rect.theT = rint(0)
    rect.textValue = rstring("Auto ROI")

    roi = create_roi(conn, image, [rect])
    print(f"Created ROI for image {image.getName()}")
```

### 跨 Z 堆疊建立 ROI

```python
image = conn.getObject("Image", image_id)
size_z = image.getSizeZ()

# 在每個 Z 切面建立矩形
shapes = []
for z in range(size_z):
    rect = omero.model.RectangleI()
    rect.x = rdouble(100)
    rect.y = rdouble(100)
    rect.width = rdouble(50)
    rect.height = rdouble(50)
    rect.theZ = rint(z)
    rect.theT = rint(0)
    shapes.append(rect)

# 單一 ROI 具有跨 Z 的形狀
roi = create_roi(conn, image, shapes)
```

## 完整範例

```python
from omero.gateway import BlitzGateway
from omero.rtypes import rdouble, rint, rstring
import omero.model

HOST = 'omero.example.com'
PORT = 4064
USERNAME = 'user'
PASSWORD = 'pass'

def rgba_to_int(r, g, b, a=255):
    return int.from_bytes([r, g, b, a], byteorder='big', signed=True)

with BlitzGateway(USERNAME, PASSWORD, host=HOST, port=PORT) as conn:
    # 取得影像
    image = conn.getObject("Image", image_id)
    print(f"Processing: {image.getName()}")

    # 建立多個 ROI
    updateService = conn.getUpdateService()

    # ROI 1：矩形
    roi1 = omero.model.RoiI()
    roi1.setImage(image._obj)

    rect = omero.model.RectangleI()
    rect.x = rdouble(50)
    rect.y = rdouble(50)
    rect.width = rdouble(100)
    rect.height = rdouble(100)
    rect.theZ = rint(0)
    rect.theT = rint(0)
    rect.textValue = rstring("Cell 1")
    rect.strokeColor = rint(rgba_to_int(255, 0, 0, 255))

    roi1.addShape(rect)
    roi1 = updateService.saveAndReturnObject(roi1)
    print(f"Created ROI 1: {roi1.getId().getValue()}")

    # ROI 2：橢圓
    roi2 = omero.model.RoiI()
    roi2.setImage(image._obj)

    ellipse = omero.model.EllipseI()
    ellipse.x = rdouble(200)
    ellipse.y = rdouble(150)
    ellipse.radiusX = rdouble(40)
    ellipse.radiusY = rdouble(30)
    ellipse.theZ = rint(0)
    ellipse.theT = rint(0)
    ellipse.textValue = rstring("Cell 2")
    ellipse.strokeColor = rint(rgba_to_int(0, 255, 0, 255))

    roi2.addShape(ellipse)
    roi2 = updateService.saveAndReturnObject(roi2)
    print(f"Created ROI 2: {roi2.getId().getValue()}")

    # 擷取並分析
    roi_service = conn.getRoiService()
    result = roi_service.findByImage(image_id, None)

    shape_ids = []
    for roi in result.rois:
        for shape in roi.copyShapes():
            shape_ids.append(shape.id.val)

    # 取得統計資料
    stats = roi_service.getShapeStatsRestricted(shape_ids, 0, 0, [0])

    for i, stat in enumerate(stats):
        print(f"Shape {shape_ids[i]}:")
        print(f"  Mean intensity: {stat.mean[0]:.2f}")
```

## 最佳實務

1. **組織形狀**：將相關形狀分組到單一 ROI
2. **標記形狀**：使用 textValue 進行識別
3. **設定 Z 和 T**：務必指定 Z 切面和時間點
4. **顏色編碼**：對形狀類型使用一致的顏色
5. **驗證座標**：確保形狀在影像範圍內
6. **批次建立**：盡可能在單一交易中建立多個 ROI
7. **刪除未使用**：移除臨時或測試 ROI
8. **匯出資料**：將 ROI 統計資料儲存在表格中以供後續分析
9. **版本控制**：在註解中記錄 ROI 建立方法
10. **效能**：使用形狀統計服務而非手動像素提取
