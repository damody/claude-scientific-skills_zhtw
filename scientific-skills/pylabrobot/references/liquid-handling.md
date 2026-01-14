# PyLabRobot 液體處理

## 概述

液體處理模組（`pylabrobot.liquid_handling`）提供控制液體處理機器人的統一介面。`LiquidHandler` 類別作為所有移液操作的主要介面，通過後端抽象在不同硬體平台上運作。

## 基本設置

### 初始化液體處理器

```python
from pylabrobot.liquid_handling import LiquidHandler
from pylabrobot.liquid_handling.backends import STAR
from pylabrobot.resources import STARLetDeck

# 建立使用 STAR 後端的液體處理器
lh = LiquidHandler(backend=STAR(), deck=STARLetDeck())
await lh.setup()

# 完成時
await lh.stop()
```

### 在後端之間切換

通過更換後端即可更改機器人，無需重寫協定：

```python
# Hamilton STAR
from pylabrobot.liquid_handling.backends import STAR
lh = LiquidHandler(backend=STAR(), deck=STARLetDeck())

# Opentrons OT-2
from pylabrobot.liquid_handling.backends import OpentronsBackend
lh = LiquidHandler(backend=OpentronsBackend(host="192.168.1.100"), deck=OTDeck())

# 模擬（不需要硬體）
from pylabrobot.liquid_handling.backends.simulation import ChatterboxBackend
lh = LiquidHandler(backend=ChatterboxBackend(), deck=STARLetDeck())
```

## 核心操作

### 吸頭管理

拾取和丟棄吸頭是液體處理操作的基礎：

```python
# 從特定位置拾取吸頭
await lh.pick_up_tips(tip_rack["A1"])           # 單個吸頭
await lh.pick_up_tips(tip_rack["A1:H1"])        # 一排 8 個吸頭
await lh.pick_up_tips(tip_rack["A1:A12"])       # 一列 12 個吸頭

# 丟棄吸頭
await lh.drop_tips()                             # 在目前位置丟棄
await lh.drop_tips(waste)                        # 在指定位置丟棄

# 將吸頭放回原架
await lh.return_tips()
```

**吸頭追蹤**：啟用自動吸頭追蹤以監控吸頭使用：

```python
from pylabrobot.resources import set_tip_tracking
set_tip_tracking(True)  # 全域啟用
```

### 吸取液體

從孔或容器中抽取液體：

```python
# 基本吸取
await lh.aspirate(plate["A1"], vols=100)         # 從 A1 吸取 100 µL

# 多個孔使用相同體積
await lh.aspirate(plate["A1:H1"], vols=100)      # 從每個孔吸取 100 µL

# 多個孔使用不同體積
await lh.aspirate(
    plate["A1:A3"],
    vols=[100, 150, 200]                          # 不同體積
)

# 進階參數
await lh.aspirate(
    plate["A1"],
    vols=100,
    flow_rate=50,                                 # µL/s
    liquid_height=5,                              # 距離底部 mm
    blow_out_air_volume=10                        # µL 空氣
)
```

### 分配液體

將液體分配到孔或容器中：

```python
# 基本分配
await lh.dispense(plate["A2"], vols=100)         # 分配 100 µL 到 A2

# 多個孔
await lh.dispense(plate["A1:H1"], vols=100)      # 分配 100 µL 到每個孔

# 不同體積
await lh.dispense(
    plate["A1:A3"],
    vols=[100, 150, 200]
)

# 進階參數
await lh.dispense(
    plate["A2"],
    vols=100,
    flow_rate=50,                                 # µL/s
    liquid_height=2,                              # 距離底部 mm
    blow_out_air_volume=10                        # µL 空氣
)
```

### 轉移液體

轉移將吸取和分配合併為單一操作：

```python
# 基本轉移
await lh.transfer(
    source=source_plate["A1"],
    dest=dest_plate["A1"],
    vols=100
)

# 多次轉移（相同吸頭）
await lh.transfer(
    source=source_plate["A1:H1"],
    dest=dest_plate["A1:H1"],
    vols=100
)

# 每個孔不同體積
await lh.transfer(
    source=source_plate["A1:A3"],
    dest=dest_plate["B1:B3"],
    vols=[50, 100, 150]
)

# 帶有吸頭處理
await lh.pick_up_tips(tip_rack["A1:H1"])
await lh.transfer(
    source=source_plate["A1:H12"],
    dest=dest_plate["A1:H12"],
    vols=100
)
await lh.drop_tips()
```

## 進階技術

### 連續稀釋

在微孔盤行或列中建立連續稀釋：

```python
# 2 倍連續稀釋
source_vols = [100, 50, 50, 50, 50, 50, 50, 50]
dest_vols = [0, 50, 50, 50, 50, 50, 50, 50]

# 先添加稀釋液
await lh.pick_up_tips(tip_rack["A1"])
await lh.transfer(
    source=buffer["A1"],
    dest=plate["A2:A8"],
    vols=50
)
await lh.drop_tips()

# 執行連續稀釋
await lh.pick_up_tips(tip_rack["A2"])
for i in range(7):
    await lh.aspirate(plate[f"A{i+1}"], vols=50)
    await lh.dispense(plate[f"A{i+2}"], vols=50)
    # 混合
    await lh.aspirate(plate[f"A{i+2}"], vols=50)
    await lh.dispense(plate[f"A{i+2}"], vols=50)
await lh.drop_tips()
```

### 微孔盤複製

將整個微孔盤布局複製到另一個微孔盤：

```python
# 設置吸頭
await lh.pick_up_tips(tip_rack["A1:H1"])

# 複製 96 孔盤（12 列）
for col in range(1, 13):
    await lh.transfer(
        source=source_plate[f"A{col}:H{col}"],
        dest=dest_plate[f"A{col}:H{col}"],
        vols=100
    )

await lh.drop_tips()
```

### 多通道移液

同時使用多個通道進行平行操作：

```python
# 8 通道轉移（整行）
await lh.pick_up_tips(tip_rack["A1:H1"])
await lh.transfer(
    source=source_plate["A1:H1"],
    dest=dest_plate["A1:H1"],
    vols=100
)
await lh.drop_tips()

# 使用 8 通道處理整個微孔盤
for col in range(1, 13):
    await lh.pick_up_tips(tip_rack[f"A{col}:H{col}"])
    await lh.transfer(
        source=source_plate[f"A{col}:H{col}"],
        dest=dest_plate[f"A{col}:H{col}"],
        vols=100
    )
    await lh.drop_tips()
```

### 混合液體

通過重複吸取和分配來混合液體：

```python
# 通過吸取/分配混合
await lh.pick_up_tips(tip_rack["A1"])

# 混合 5 次
for _ in range(5):
    await lh.aspirate(plate["A1"], vols=80)
    await lh.dispense(plate["A1"], vols=80)

await lh.drop_tips()
```

## 體積追蹤

自動追蹤孔中的液體體積：

```python
from pylabrobot.resources import set_volume_tracking

# 全域啟用體積追蹤
set_volume_tracking(True)

# 設定初始體積
plate["A1"].tracker.set_liquids([(None, 200)])  # 200 µL

# 吸取 100 µL 後
await lh.aspirate(plate["A1"], vols=100)
print(plate["A1"].tracker.get_volume())  # 100 µL

# 檢查剩餘體積
remaining = plate["A1"].tracker.get_volume()
```

## 液體類別

定義液體屬性以優化移液：

```python
# 液體類別控制吸取/分配參數
from pylabrobot.liquid_handling import LiquidClass

# 建立自訂液體類別
water = LiquidClass(
    name="Water",
    aspiration_flow_rate=100,
    dispense_flow_rate=150,
    aspiration_mix_flow_rate=100,
    dispense_mix_flow_rate=100,
    air_transport_retract_dist=10
)

# 與操作一起使用
await lh.aspirate(
    plate["A1"],
    vols=100,
    liquid_class=water
)
```

## 錯誤處理

處理液體處理操作中的錯誤：

```python
try:
    await lh.setup()
    await lh.pick_up_tips(tip_rack["A1"])
    await lh.transfer(source["A1"], dest["A1"], vols=100)
    await lh.drop_tips()
except Exception as e:
    print(f"液體處理錯誤：{e}")
    # 嘗試丟棄吸頭（如果持有）
    try:
        await lh.drop_tips()
    except:
        pass
finally:
    await lh.stop()
```

## 最佳實務

1. **始終設置和停止**：在操作前呼叫 `await lh.setup()`，完成時呼叫 `await lh.stop()`
2. **啟用追蹤**：使用吸頭追蹤和體積追蹤以實現準確的狀態管理
3. **吸頭管理**：始終在吸取前拾取吸頭，完成後丟棄
4. **流速**：根據液體黏度和容器類型調整流速
5. **液體高度**：設定適當的吸取/分配高度以避免飛濺
6. **錯誤處理**：使用 try/finally 區塊確保適當清理
7. **在模擬中測試**：在硬體上執行前使用 ChatterboxBackend 測試協定
8. **體積限制**：遵守吸頭體積限制和孔容量
9. **混合**：分配黏稠液體後或準確度至關重要時進行混合
10. **文件**：記錄液體類別和自訂參數以確保可重現性

## 常見模式

### 完整液體處理協定

```python
from pylabrobot.liquid_handling import LiquidHandler
from pylabrobot.liquid_handling.backends import STAR
from pylabrobot.resources import STARLetDeck, TIP_CAR_480_A00, Cos_96_DW_1mL
from pylabrobot.resources import set_tip_tracking, set_volume_tracking

# 啟用追蹤
set_tip_tracking(True)
set_volume_tracking(True)

# 初始化
lh = LiquidHandler(backend=STAR(), deck=STARLetDeck())
await lh.setup()

try:
    # 定義資源
    tip_rack = TIP_CAR_480_A00(name="tips")
    source = Cos_96_DW_1mL(name="source")
    dest = Cos_96_DW_1mL(name="dest")

    # 分配到工作台
    lh.deck.assign_child_resource(tip_rack, rails=1)
    lh.deck.assign_child_resource(source, rails=10)
    lh.deck.assign_child_resource(dest, rails=15)

    # 設定初始體積
    for well in source.children:
        well.tracker.set_liquids([(None, 200)])

    # 執行協定
    await lh.pick_up_tips(tip_rack["A1:H1"])
    await lh.transfer(
        source=source["A1:H12"],
        dest=dest["A1:H12"],
        vols=100
    )
    await lh.drop_tips()

finally:
    await lh.stop()
```

## 硬體專用說明

### Hamilton STAR

- 支援完整液體處理功能
- 使用 USB 連接進行通訊
- 直接執行韌體命令
- 支援 CO-RE（壓縮 O 型環擴展）吸頭

### Opentrons OT-2

- 需要 IP 位址進行網路連接
- 使用 HTTP API 進行通訊
- 僅限 8 通道和單通道移液器
- 與 STAR 相比工作台布局較簡單

### Tecan EVO

- 開發中支援
- 與 Hamilton STAR 類似的功能
- 請查看文件以了解目前相容性狀態

## 其他資源

- 官方液體處理指南：https://docs.pylabrobot.org/user_guide/basic.html
- API 參考：https://docs.pylabrobot.org/api/pylabrobot.liquid_handling.html
- 範例協定：https://github.com/PyLabRobot/pylabrobot/tree/main/examples
