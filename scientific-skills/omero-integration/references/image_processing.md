# 影像處理與渲染

此參考涵蓋存取原始像素資料、影像渲染，以及在 OMERO 中建立新影像。

## 存取原始像素資料

### 取得單一平面

```python
# 取得影像
image = conn.getObject("Image", image_id)

# 取得尺寸
size_z = image.getSizeZ()
size_c = image.getSizeC()
size_t = image.getSizeT()

# 取得像素物件
pixels = image.getPrimaryPixels()

# 取得單一平面（傳回 NumPy 陣列）
z, c, t = 0, 0, 0  # 第一個 Z 切面、通道和時間點
plane = pixels.getPlane(z, c, t)

print(f"Shape: {plane.shape}")
print(f"Data type: {plane.dtype.name}")
print(f"Min: {plane.min()}, Max: {plane.max()}")
```

### 取得多個平面

```python
import numpy as np

# 取得特定通道和時間點的 Z 堆疊
pixels = image.getPrimaryPixels()
c, t = 0, 0  # 第一個通道和時間點

# 建立 (z, c, t) 座標列表
zct_list = [(z, c, t) for z in range(size_z)]

# 一次取得所有平面
planes = pixels.getPlanes(zct_list)

# 堆疊成 3D 陣列
z_stack = np.array([p for p in planes])
print(f"Z-stack shape: {z_stack.shape}")
```

### 取得超立方體（5D 資料的子集）

```python
# 取得 5D 資料的子集（Z、C、T）
zct_list = []
for z in range(size_z // 2, size_z):  # Z 的後半部分
    for c in range(size_c):           # 所有通道
        for t in range(size_t):       # 所有時間點
            zct_list.append((z, c, t))

# 取得平面
planes = pixels.getPlanes(zct_list)

# 處理每個平面
for i, plane in enumerate(planes):
    z, c, t = zct_list[i]
    print(f"Plane Z={z}, C={c}, T={t}: Min={plane.min()}, Max={plane.max()}")
```

### 取得區塊（感興趣區域）

```python
# 定義區塊座標
x, y = 50, 50          # 左上角
width, height = 100, 100  # 區塊大小
tile = (x, y, width, height)

# 取得特定 Z、C、T 的區塊
z, c, t = 0, 0, 0
zct_list = [(z, c, t, tile)]

tiles = pixels.getTiles(zct_list)
tile_data = tiles[0]

print(f"Tile shape: {tile_data.shape}")  # 應該是 (height, width)
```

### 取得多個區塊

```python
# 從 Z 堆疊取得區塊
c, t = 0, 0
tile = (50, 50, 100, 100)  # x, y, width, height

# 建立包含區塊的列表
zct_list = [(z, c, t, tile) for z in range(size_z)]

tiles = pixels.getTiles(zct_list)

for i, tile_data in enumerate(tiles):
    print(f"Tile Z={i}: {tile_data.shape}, Min={tile_data.min()}")
```

## 影像直方圖

### 取得直方圖

```python
# 取得第一個通道的直方圖
channel_index = 0
num_bins = 256
z, t = 0, 0

histogram = image.getHistogram([channel_index], num_bins, False, z, t)
print(f"Histogram bins: {len(histogram)}")
print(f"First 10 bins: {histogram[:10]}")
```

### 多通道直方圖

```python
# 取得所有通道的直方圖
channels = list(range(image.getSizeC()))
histograms = image.getHistogram(channels, 256, False, 0, 0)

for c, hist in enumerate(histograms):
    print(f"Channel {c}: Total pixels = {sum(hist)}")
```

## 影像渲染

### 使用當前設定渲染影像

```python
from PIL import Image
from io import BytesIO

# 取得影像
image = conn.getObject("Image", image_id)

# 在特定 Z 和 T 渲染
z = image.getSizeZ() // 2  # 中間 Z 切面
t = 0

rendered_image = image.renderImage(z, t)
# rendered_image 是 PIL Image 物件
rendered_image.save("rendered_image.jpg")
```

### 取得縮圖

```python
from PIL import Image
from io import BytesIO

# 取得縮圖（使用當前渲染設定）
thumbnail_data = image.getThumbnail()

# 轉換為 PIL Image
thumbnail = Image.open(BytesIO(thumbnail_data))
thumbnail.save("thumbnail.jpg")

# 取得特定縮圖大小
thumbnail_data = image.getThumbnail(size=(96, 96))
thumbnail = Image.open(BytesIO(thumbnail_data))
```

## 渲染設定

### 查看當前設定

```python
# 顯示渲染設定
print("Current Rendering Settings:")
print(f"Grayscale mode: {image.isGreyscaleRenderingModel()}")
print(f"Default Z: {image.getDefaultZ()}")
print(f"Default T: {image.getDefaultT()}")
print()

# 通道設定
print("Channel Settings:")
for idx, channel in enumerate(image.getChannels()):
    print(f"Channel {idx + 1}:")
    print(f"  Label: {channel.getLabel()}")
    print(f"  Color: {channel.getColor().getHtml()}")
    print(f"  Active: {channel.isActive()}")
    print(f"  Window: {channel.getWindowStart()} - {channel.getWindowEnd()}")
    print(f"  Min/Max: {channel.getWindowMin()} - {channel.getWindowMax()}")
```

### 設定渲染模式

```python
# 切換到灰階渲染
image.setGreyscaleRenderingModel()

# 切換到彩色渲染
image.setColorRenderingModel()
```

### 設定啟用通道

```python
# 啟用特定通道（1 索引）
image.setActiveChannels([1, 3])  # 只啟用通道 1 和 3

# 啟用所有通道
all_channels = list(range(1, image.getSizeC() + 1))
image.setActiveChannels(all_channels)

# 啟用單一通道
image.setActiveChannels([2])
```

### 設定通道顏色

```python
# 設定通道顏色（十六進位格式）
channels = [1, 2, 3]
colors = ['FF0000', '00FF00', '0000FF']  # 紅、綠、藍

image.setActiveChannels(channels, colors=colors)

# 使用 None 保持現有顏色
colors = ['FF0000', None, '0000FF']  # 保持通道 2 的顏色
image.setActiveChannels(channels, colors=colors)
```

### 設定通道視窗（強度範圍）

```python
# 設定通道的強度視窗
channels = [1, 2]
windows = [
    [100.0, 500.0],  # 通道 1：100-500
    [50.0, 300.0]    # 通道 2：50-300
]

image.setActiveChannels(channels, windows=windows)

# 使用 None 保持現有視窗
windows = [[100.0, 500.0], [None, None]]
image.setActiveChannels(channels, windows=windows)
```

### 設定預設 Z 和 T

```python
# 設定預設 Z 切面和時間點
image.setDefaultZ(5)
image.setDefaultT(0)

# 使用預設值渲染
rendered_image = image.renderImage(z=None, t=None)
rendered_image.save("default_rendering.jpg")
```

## 渲染個別通道

### 分別渲染每個通道

```python
# 設定灰階模式
image.setGreyscaleRenderingModel()

z = image.getSizeZ() // 2
t = 0

# 渲染每個通道
for c in range(1, image.getSizeC() + 1):
    image.setActiveChannels([c])
    rendered = image.renderImage(z, t)
    rendered.save(f"channel_{c}.jpg")
```

### 渲染多通道複合

```python
# 前 3 個通道的彩色複合
image.setColorRenderingModel()
channels = [1, 2, 3]
colors = ['FF0000', '00FF00', '0000FF']  # RGB

image.setActiveChannels(channels, colors=colors)
rendered = image.renderImage(z, t)
rendered.save("rgb_composite.jpg")
```

## 影像投影

### 最大強度投影

```python
# 設定投影類型
image.setProjection('intmax')

# 渲染（投影跨越所有 Z）
z, t = 0, 0  # 投影時忽略 Z
rendered = image.renderImage(z, t)
rendered.save("max_projection.jpg")

# 重設為正常渲染
image.setProjection('normal')
```

### 平均強度投影

```python
image.setProjection('intmean')
rendered = image.renderImage(z, t)
rendered.save("mean_projection.jpg")
image.setProjection('normal')
```

### 可用投影類型

- `'normal'`：無投影（預設）
- `'intmax'`：最大強度投影
- `'intmean'`：平均強度投影
- `'intmin'`：最小強度投影（如支援）

## 儲存和重設渲染設定

### 儲存當前設定為預設

```python
# 修改渲染設定
image.setActiveChannels([1, 2])
image.setDefaultZ(5)

# 儲存為新預設
image.saveDefaults()
```

### 重設為匯入設定

```python
# 重設為原始匯入設定
image.resetDefaults(save=True)
```

## 從 NumPy 陣列建立影像

### 建立簡單影像

```python
import numpy as np

# 建立範例資料
size_x, size_y = 512, 512
size_z, size_c, size_t = 10, 2, 1

# 生成平面
def plane_generator():
    """產生平面的生成器"""
    for z in range(size_z):
        for c in range(size_c):
            for t in range(size_t):
                # 建立合成資料
                plane = np.random.randint(0, 255, (size_y, size_x), dtype=np.uint8)
                yield plane

# 建立影像
image = conn.createImageFromNumpySeq(
    plane_generator(),
    "Test Image",
    size_z, size_c, size_t,
    description="Image created from NumPy arrays",
    dataset=None
)

print(f"Created image ID: {image.getId()}")
```

### 從硬編碼陣列建立影像

```python
from numpy import array, int8

# 定義尺寸
size_x, size_y = 5, 4
size_z, size_c, size_t = 1, 2, 1

# 建立平面
plane1 = array(
    [[0, 1, 2, 3, 4],
     [5, 6, 7, 8, 9],
     [0, 1, 2, 3, 4],
     [5, 6, 7, 8, 9]],
    dtype=int8
)

plane2 = array(
    [[5, 6, 7, 8, 9],
     [0, 1, 2, 3, 4],
     [5, 6, 7, 8, 9],
     [0, 1, 2, 3, 4]],
    dtype=int8
)

planes = [plane1, plane2]

def plane_gen():
    for p in planes:
        yield p

# 建立影像
desc = "Image created from hard-coded arrays"
image = conn.createImageFromNumpySeq(
    plane_gen(),
    "numpy_image",
    size_z, size_c, size_t,
    description=desc,
    dataset=None
)

print(f"Created image: {image.getName()} (ID: {image.getId()})")
```

### 在資料集中建立影像

```python
# 取得目標資料集
dataset = conn.getObject("Dataset", dataset_id)

# 建立影像
image = conn.createImageFromNumpySeq(
    plane_generator(),
    "New Analysis Result",
    size_z, size_c, size_t,
    description="Result from analysis pipeline",
    dataset=dataset  # 新增到資料集
)
```

### 建立衍生影像

```python
# 取得來源影像
source = conn.getObject("Image", source_image_id)
size_z = source.getSizeZ()
size_c = source.getSizeC()
size_t = source.getSizeT()
dataset = source.getParent()

pixels = source.getPrimaryPixels()
new_size_c = 1  # 平均通道

def plane_gen():
    """將通道平均在一起"""
    for z in range(size_z):
        for c in range(new_size_c):
            for t in range(size_t):
                # 取得多個通道
                channel0 = pixels.getPlane(z, 0, t)
                channel1 = pixels.getPlane(z, 1, t)

                # 組合
                new_plane = (channel0.astype(float) + channel1.astype(float)) / 2
                new_plane = new_plane.astype(channel0.dtype)

                yield new_plane

# 建立新影像
desc = "Averaged channels from source image"
derived = conn.createImageFromNumpySeq(
    plane_gen(),
    f"{source.getName()}_averaged",
    size_z, new_size_c, size_t,
    description=desc,
    dataset=dataset
)

print(f"Created derived image: {derived.getId()}")
```

## 設定物理尺寸

### 設定具有單位的像素大小

```python
from omero.model.enums import UnitsLength
import omero.model

# 取得影像
image = conn.getObject("Image", image_id)

# 建立單位物件
pixel_size_x = omero.model.LengthI(0.325, UnitsLength.MICROMETER)
pixel_size_y = omero.model.LengthI(0.325, UnitsLength.MICROMETER)
pixel_size_z = omero.model.LengthI(1.0, UnitsLength.MICROMETER)

# 取得像素物件
pixels = image.getPrimaryPixels()._obj

# 設定物理大小
pixels.setPhysicalSizeX(pixel_size_x)
pixels.setPhysicalSizeY(pixel_size_y)
pixels.setPhysicalSizeZ(pixel_size_z)

# 儲存變更
conn.getUpdateService().saveObject(pixels)

print("Updated pixel dimensions")
```

### 可用長度單位

來自 `omero.model.enums.UnitsLength`：
- `ANGSTROM`
- `NANOMETER`
- `MICROMETER`
- `MILLIMETER`
- `CENTIMETER`
- `METER`
- `PIXEL`

### 為新影像設定像素大小

```python
from omero.model.enums import UnitsLength
import omero.model

# 建立影像
image = conn.createImageFromNumpySeq(
    plane_generator(),
    "New Image with Dimensions",
    size_z, size_c, size_t
)

# 設定像素大小
pixel_size = omero.model.LengthI(0.5, UnitsLength.MICROMETER)
pixels = image.getPrimaryPixels()._obj
pixels.setPhysicalSizeX(pixel_size)
pixels.setPhysicalSizeY(pixel_size)

z_size = omero.model.LengthI(2.0, UnitsLength.MICROMETER)
pixels.setPhysicalSizeZ(z_size)

conn.getUpdateService().saveObject(pixels)
```

## 完整範例：影像處理管線

```python
from omero.gateway import BlitzGateway
import numpy as np

HOST = 'omero.example.com'
PORT = 4064
USERNAME = 'user'
PASSWORD = 'pass'

with BlitzGateway(USERNAME, PASSWORD, host=HOST, port=PORT) as conn:
    # 取得來源影像
    source = conn.getObject("Image", source_image_id)
    print(f"Processing: {source.getName()}")

    # 取得尺寸
    size_x = source.getSizeX()
    size_y = source.getSizeY()
    size_z = source.getSizeZ()
    size_c = source.getSizeC()
    size_t = source.getSizeT()

    pixels = source.getPrimaryPixels()

    # 處理：Z 方向的最大強度投影
    def plane_gen():
        for c in range(size_c):
            for t in range(size_t):
                # 取得此 C、T 的所有 Z 平面
                z_stack = []
                for z in range(size_z):
                    plane = pixels.getPlane(z, c, t)
                    z_stack.append(plane)

                # 最大投影
                max_proj = np.max(z_stack, axis=0)
                yield max_proj

    # 建立結果影像（單一 Z 切面）
    result = conn.createImageFromNumpySeq(
        plane_gen(),
        f"{source.getName()}_MIP",
        1, size_c, size_t,  # 投影為 Z=1
        description="Maximum intensity projection",
        dataset=source.getParent()
    )

    print(f"Created MIP image: {result.getId()}")

    # 複製像素大小（只有 X 和 Y，投影沒有 Z）
    from omero.model.enums import UnitsLength
    import omero.model

    source_pixels = source.getPrimaryPixels()._obj
    result_pixels = result.getPrimaryPixels()._obj

    result_pixels.setPhysicalSizeX(source_pixels.getPhysicalSizeX())
    result_pixels.setPhysicalSizeY(source_pixels.getPhysicalSizeY())

    conn.getUpdateService().saveObject(result_pixels)

    print("Processing complete")
```

## 處理不同資料類型

### 處理各種像素類型

```python
# 取得像素類型
pixel_type = image.getPixelsType()
print(f"Pixel type: {pixel_type}")

# 常見類型：uint8、uint16、uint32、int8、int16、int32、float、double

# 取得具有正確 dtype 的平面
plane = pixels.getPlane(z, c, t)
print(f"NumPy dtype: {plane.dtype}")

# 如需處理則轉換
if plane.dtype == np.uint16:
    # 轉換為浮點數進行處理
    plane_float = plane.astype(np.float32)
    # 處理...
    # 轉換回來
    result = plane_float.astype(np.uint16)
```

### 處理大型影像

```python
# 以區塊處理大型影像以節省記憶體
tile_size = 512
size_x = image.getSizeX()
size_y = image.getSizeY()

for y in range(0, size_y, tile_size):
    for x in range(0, size_x, tile_size):
        # 取得區塊尺寸
        w = min(tile_size, size_x - x)
        h = min(tile_size, size_y - y)
        tile = (x, y, w, h)

        # 取得區塊資料
        zct_list = [(z, c, t, tile)]
        tile_data = pixels.getTiles(zct_list)[0]

        # 處理區塊
        # ...
```

## 最佳實務

1. **使用生成器**：建立影像時使用生成器以避免將所有資料載入記憶體
2. **指定資料類型**：將 NumPy dtypes 與 OMERO 像素類型匹配
3. **設定物理尺寸**：務必為新影像設定像素大小
4. **分塊處理大型影像**：以區塊處理大型影像以管理記憶體
5. **關閉連線**：完成後務必關閉連線
6. **渲染效率**：渲染多張影像時快取渲染設定
7. **通道索引**：記住渲染時通道是 1 索引，像素存取時是 0 索引
8. **儲存設定**：如果應該永久保存則儲存渲染設定
9. **壓縮**：在 renderImage() 中使用 compression 參數以加快傳輸
10. **錯誤處理**：檢查 None 傳回值並處理異常
