# PyLabRobot 資源管理

## 概述

PyLabRobot 中的資源代表協定中使用的實驗室設備、耗材或元件。資源系統提供階層結構，用於管理微孔盤、吸頭架、槽、試管、載體和其他實驗室耗材，具有精確的空間定位和狀態追蹤。

## 資源基礎

### 什麼是資源？

資源代表：
- 一件實驗室耗材（微孔盤、吸頭架、槽、試管）
- 設備（液體處理器、微孔盤讀取器）
- 耗材的一部分（孔、吸頭）
- 耗材的容器（工作台、載體）

所有資源繼承自基礎 `Resource` 類別，並形成具有父子關係的樹狀結構（樹形）。

### 資源屬性

每個資源需要：
- **name**：資源的唯一識別碼
- **size_x, size_y, size_z**：以毫米為單位的尺寸（長方體表示）
- **location**：相對於父資源原點的座標（可選，分配時設定）

```python
from pylabrobot.resources import Resource

# 建立基本資源
resource = Resource(
    name="my_resource",
    size_x=127.76,  # mm
    size_y=85.48,   # mm
    size_z=14.5     # mm
)
```

## 資源類型

### 微孔盤

具有孔的微孔盤用於容納液體：

```python
from pylabrobot.resources import (
    Cos_96_DW_1mL,      # 96 孔盤，1mL 深孔
    Cos_96_DW_500ul,    # 96 孔盤，500µL
    Plate_384_Sq,       # 384 孔方形盤
    Cos_96_PCR          # 96 孔 PCR 盤
)

# 建立微孔盤
plate = Cos_96_DW_1mL(name="sample_plate")

# 存取孔
well_a1 = plate["A1"]                  # 單個孔
row_a = plate["A1:H1"]                 # 整行（A1-H1）
col_1 = plate["A1:A12"]                # 整列（A1-A12）
range_wells = plate["A1:C3"]           # 孔範圍
all_wells = plate.children             # 所有孔作為列表
```

### 吸頭架

容納移液器吸頭的容器：

```python
from pylabrobot.resources import (
    TIP_CAR_480_A00,    # 96 個標準吸頭
    HTF_L,              # Hamilton 吸頭，有濾芯
    TipRack             # 通用吸頭架
)

# 建立吸頭架
tip_rack = TIP_CAR_480_A00(name="tips")

# 存取吸頭
tip_a1 = tip_rack["A1"]                # 單個吸頭位置
tips_row = tip_rack["A1:H1"]           # 一排吸頭
tips_col = tip_rack["A1:A12"]          # 一列吸頭

# 檢查吸頭存在（需要啟用吸頭追蹤）
from pylabrobot.resources import set_tip_tracking
set_tip_tracking(True)

has_tip = tip_rack["A1"].tracker.has_tip
```

### 槽

用於試劑的儲液容器：

```python
from pylabrobot.resources import Trough_100ml

# 建立槽
trough = Trough_100ml(name="buffer")

# 存取通道
channel_1 = trough["channel_1"]
all_channels = trough.children
```

### 試管

單個試管或試管架：

```python
from pylabrobot.resources import Tube, TubeRack

# 建立試管架
tube_rack = TubeRack(name="samples")

# 存取試管
tube_a1 = tube_rack["A1"]
```

### 載體

容納微孔盤、吸頭或其他耗材的平台：

```python
from pylabrobot.resources import (
    PlateCarrier,
    TipCarrier,
    MFXCarrier
)

# 載體提供耗材位置
carrier = PlateCarrier(name="plate_carrier")

# 將微孔盤分配到載體
plate = Cos_96_DW_1mL(name="plate")
carrier.assign_child_resource(plate, location=(0, 0, 0))
```

## 工作台管理

### 使用工作台

工作台代表機器人的工作表面：

```python
from pylabrobot.resources import STARLetDeck, OTDeck

# Hamilton STARlet 工作台
deck = STARLetDeck()

# Opentrons OT-2 工作台
deck = OTDeck()
```

### 將資源分配到工作台

使用軌道或座標將資源分配到特定工作台位置：

```python
from pylabrobot.liquid_handling import LiquidHandler
from pylabrobot.resources import STARLetDeck, TIP_CAR_480_A00, Cos_96_DW_1mL

lh = LiquidHandler(backend=backend, deck=STARLetDeck())

# 使用軌道位置分配（Hamilton STAR）
tip_rack = TIP_CAR_480_A00(name="tips")
source_plate = Cos_96_DW_1mL(name="source")
dest_plate = Cos_96_DW_1mL(name="dest")

lh.deck.assign_child_resource(tip_rack, rails=1)
lh.deck.assign_child_resource(source_plate, rails=10)
lh.deck.assign_child_resource(dest_plate, rails=15)

# 使用座標分配（x, y, z 以 mm 為單位）
lh.deck.assign_child_resource(
    resource=tip_rack,
    location=(100, 200, 0)
)
```

### 取消分配資源

從工作台移除資源：

```python
# 取消分配特定資源
lh.deck.unassign_child_resource(tip_rack)

# 存取已分配的資源
all_resources = lh.deck.children
resource_names = [r.name for r in lh.deck.children]
```

## 座標系統

PyLabRobot 使用右手笛卡爾座標系統：

- **X 軸**：從左到右（向右增加）
- **Y 軸**：從前到後（向後增加）
- **Z 軸**：從下到上（向上增加）
- **原點**：父資源的左前下角

### 位置計算

```python
# 取得絕對位置（相對於工作台/根）
absolute_loc = plate.get_absolute_location()

# 取得相對於另一個資源的位置
relative_loc = well.get_location_wrt(deck)

# 取得相對於父資源的位置
parent_relative = plate.location
```

## 狀態管理

### 追蹤液體體積

追蹤孔和容器中的液體體積：

```python
from pylabrobot.resources import set_volume_tracking

# 全域啟用體積追蹤
set_volume_tracking(True)

# 設定孔中的液體
plate["A1"].tracker.set_liquids([
    (None, 200)  # (液體類型, 體積_µL)
])

# 多種液體
plate["A2"].tracker.set_liquids([
    ("water", 100),
    ("ethanol", 50)
])

# 取得目前體積
volume = plate["A1"].tracker.get_volume()  # 回傳總體積

# 取得液體
liquids = plate["A1"].tracker.get_liquids()  # 回傳 (類型, 體積) 元組列表
```

### 追蹤吸頭存在

追蹤吸頭架中哪些吸頭存在：

```python
from pylabrobot.resources import set_tip_tracking

# 全域啟用吸頭追蹤
set_tip_tracking(True)

# 檢查吸頭是否存在
has_tip = tip_rack["A1"].tracker.has_tip

# 使用 pick_up_tips/drop_tips 時自動追蹤吸頭
await lh.pick_up_tips(tip_rack["A1"])  # 標記吸頭為不存在
await lh.return_tips()                  # 標記吸頭為存在
```

## 序列化

### 儲存和載入資源

將資源定義和狀態儲存到 JSON：

```python
# 儲存資源定義
plate.save("plate_definition.json")

# 從 JSON 載入資源
from pylabrobot.resources import Plate
plate = Plate.load_from_json_file("plate_definition.json")

# 儲存工作台布局
lh.deck.save("deck_layout.json")

# 載入工作台布局
from pylabrobot.resources import Deck
deck = Deck.load_from_json_file("deck_layout.json")
```

### 狀態序列化

與定義分開儲存和還原資源狀態：

```python
# 儲存狀態（吸頭存在、液體體積）
state = plate.serialize_state()
with open("plate_state.json", "w") as f:
    json.dump(state, f)

# 載入狀態
with open("plate_state.json", "r") as f:
    state = json.load(f)
plate.load_state(state)

# 儲存階層中的所有狀態
all_states = lh.deck.serialize_all_state()

# 載入所有狀態
lh.deck.load_all_state(all_states)
```

## 自訂資源

### 定義自訂耗材

當內建資源不符合您的設備時建立自訂耗材：

```python
from pylabrobot.resources import Plate, Well

# 定義自訂微孔盤
class CustomPlate(Plate):
    def __init__(self, name: str):
        super().__init__(
            name=name,
            size_x=127.76,
            size_y=85.48,
            size_z=14.5,
            num_items_x=12,  # 12 列
            num_items_y=8,   # 8 行
            dx=9.0,          # 孔間距 X
            dy=9.0,          # 孔間距 Y
            dz=0.0,          # 孔間距 Z（通常為 0）
            item_dx=9.0,     # 孔中心之間的距離 X
            item_dy=9.0      # 孔中心之間的距離 Y
        )

# 使用自訂微孔盤
custom_plate = CustomPlate(name="my_custom_plate")
```

### 自訂孔

定義自訂孔幾何形狀：

```python
from pylabrobot.resources import Well

# 建立自訂孔
well = Well(
    name="custom_well",
    size_x=8.0,
    size_y=8.0,
    size_z=10.5,
    max_volume=200,      # µL
    bottom_shape="flat"  # 或 "v"、"u"
)
```

## 資源發現

### 尋找資源

導航資源階層：

```python
# 取得微孔盤中的所有孔
wells = plate.children

# 按名稱尋找資源
resource = lh.deck.get_resource("plate_name")

# 遍歷資源
for resource in lh.deck.children:
    print(f"{resource.name}: {resource.get_absolute_location()}")

# 按模式取得孔
wells_a = [w for w in plate.children if w.name.startswith("A")]
```

### 資源後設資料

存取資源資訊：

```python
# 資源屬性
print(f"名稱：{plate.name}")
print(f"尺寸：{plate.size_x} x {plate.size_y} x {plate.size_z} mm")
print(f"位置：{plate.get_absolute_location()}")
print(f"父資源：{plate.parent.name if plate.parent else None}")
print(f"子資源：{len(plate.children)}")

# 類型檢查
from pylabrobot.resources import Plate, TipRack
if isinstance(resource, Plate):
    print("這是一個微孔盤")
elif isinstance(resource, TipRack):
    print("這是一個吸頭架")
```

## 最佳實務

1. **唯一名稱**：為所有資源使用描述性的唯一名稱
2. **啟用追蹤**：開啟吸頭和體積追蹤以實現準確的狀態管理
3. **座標驗證**：驗證資源位置在工作台上不重疊
4. **狀態序列化**：儲存工作台布局和狀態以實現可重現的協定
5. **資源清理**：不再需要時取消分配資源
6. **自訂資源**：當內建選項不符合時定義自訂耗材
7. **文件**：記錄自訂資源的尺寸和屬性
8. **類型檢查**：在操作前使用 isinstance() 驗證資源類型
9. **階層導航**：使用父/子關係導航資源樹
10. **JSON 儲存**：將工作台布局儲存在 JSON 中以進行版本控制和共享

## 常見模式

### 完整工作台設置

```python
from pylabrobot.liquid_handling import LiquidHandler
from pylabrobot.liquid_handling.backends import STAR
from pylabrobot.resources import (
    STARLetDeck,
    TIP_CAR_480_A00,
    Cos_96_DW_1mL,
    Trough_100ml,
    set_tip_tracking,
    set_volume_tracking
)

# 啟用追蹤
set_tip_tracking(True)
set_volume_tracking(True)

# 初始化液體處理器
lh = LiquidHandler(backend=STAR(), deck=STARLetDeck())
await lh.setup()

# 定義資源
tip_rack_1 = TIP_CAR_480_A00(name="tips_1")
tip_rack_2 = TIP_CAR_480_A00(name="tips_2")
source_plate = Cos_96_DW_1mL(name="source")
dest_plate = Cos_96_DW_1mL(name="dest")
buffer = Trough_100ml(name="buffer")

# 分配到工作台
lh.deck.assign_child_resource(tip_rack_1, rails=1)
lh.deck.assign_child_resource(tip_rack_2, rails=2)
lh.deck.assign_child_resource(buffer, rails=5)
lh.deck.assign_child_resource(source_plate, rails=10)
lh.deck.assign_child_resource(dest_plate, rails=15)

# 設定初始體積
for well in source_plate.children:
    well.tracker.set_liquids([(None, 200)])

buffer["channel_1"].tracker.set_liquids([(None, 50000)])  # 50 mL

# 儲存工作台布局
lh.deck.save("my_protocol_deck.json")

# 儲存初始狀態
import json
with open("initial_state.json", "w") as f:
    json.dump(lh.deck.serialize_all_state(), f)
```

### 載入已儲存的工作台

```python
from pylabrobot.resources import Deck

# 從檔案載入工作台
deck = Deck.load_from_json_file("my_protocol_deck.json")

# 載入狀態
import json
with open("initial_state.json", "r") as f:
    state = json.load(f)
deck.load_all_state(state)

# 與液體處理器一起使用
lh = LiquidHandler(backend=STAR(), deck=deck)
await lh.setup()

# 按名稱存取資源
source_plate = deck.get_resource("source")
dest_plate = deck.get_resource("dest")
```

## 其他資源

- 資源文件：https://docs.pylabrobot.org/resources/introduction.html
- 自訂資源指南：https://docs.pylabrobot.org/resources/custom-resources.html
- API 參考：https://docs.pylabrobot.org/api/pylabrobot.resources.html
- 工作台布局：https://github.com/PyLabRobot/pylabrobot/tree/main/pylabrobot/resources/deck
