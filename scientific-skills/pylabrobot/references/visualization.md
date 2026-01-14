# PyLabRobot 視覺化與模擬

## 概述

PyLabRobot 提供視覺化和模擬工具，用於在沒有實體硬體的情況下開發、測試和驗證實驗室協定。視覺化器提供工作台狀態的即時 3D 視覺化，而模擬後端則實現協定測試和驗證。

## 視覺化器

### 什麼是視覺化器？

PyLabRobot 視覺化器是一個基於瀏覽器的工具，它：
- 顯示工作台布局的 3D 視覺化
- 顯示即時吸頭存在和液體體積
- 與模擬和實體機器人配合使用
- 提供互動式工作台狀態檢視
- 實現視覺化協定驗證

### 啟動視覺化器

視覺化器作為網路伺服器運行並在瀏覽器中顯示：

```python
from pylabrobot.visualizer import Visualizer

# 建立視覺化器
vis = Visualizer()

# 啟動網路伺服器（自動開啟瀏覽器）
await vis.start()

# 停止視覺化器
await vis.stop()
```

**預設設定：**
- 連接埠：1234 (http://localhost:1234)
- 啟動時自動開啟瀏覽器

### 將液體處理器連接到視覺化器

```python
from pylabrobot.liquid_handling import LiquidHandler
from pylabrobot.liquid_handling.backends.simulation import ChatterboxBackend
from pylabrobot.resources import STARLetDeck
from pylabrobot.visualizer import Visualizer

# 建立視覺化器
vis = Visualizer()
await vis.start()

# 建立使用模擬後端的液體處理器
lh = LiquidHandler(
    backend=ChatterboxBackend(num_channels=8),
    deck=STARLetDeck()
)

# 將液體處理器連接到視覺化器
lh.visualizer = vis

await lh.setup()

# 現在所有操作都會即時視覺化
await lh.pick_up_tips(tip_rack["A1:H1"])
await lh.aspirate(plate["A1:H1"], vols=100)
await lh.dispense(plate["A2:H2"], vols=100)
await lh.drop_tips()
```

### 追蹤功能

#### 啟用追蹤

要讓視覺化器顯示吸頭和液體，請啟用追蹤：

```python
from pylabrobot.resources import set_tip_tracking, set_volume_tracking

# 全域啟用（在建立資源之前）
set_tip_tracking(True)
set_volume_tracking(True)
```

#### 設定初始液體

定義初始液體內容以供視覺化：

```python
# 在單個孔中設定液體
plate["A1"].tracker.set_liquids([
    (None, 200)  # (液體類型, 體積_µL)
])

# 在一個孔中設定多種液體
plate["A2"].tracker.set_liquids([
    ("water", 100),
    ("ethanol", 50)
])

# 在多個孔中設定液體
for well in plate["A1:H1"]:
    well.tracker.set_liquids([(None, 200)])

# 在整個微孔盤中設定液體
for well in plate.children:
    well.tracker.set_liquids([("sample", 150)])
```

#### 視覺化吸頭存在

```python
# 使用拾取/丟棄操作時自動追蹤吸頭
await lh.pick_up_tips(tip_rack["A1:H1"])  # 視覺化器中顯示吸頭不存在
await lh.return_tips()                     # 視覺化器中顯示吸頭存在
```

### 完整視覺化器範例

```python
from pylabrobot.liquid_handling import LiquidHandler
from pylabrobot.liquid_handling.backends.simulation import ChatterboxBackend
from pylabrobot.resources import (
    STARLetDeck,
    TIP_CAR_480_A00,
    Cos_96_DW_1mL,
    set_tip_tracking,
    set_volume_tracking
)
from pylabrobot.visualizer import Visualizer

# 啟用追蹤
set_tip_tracking(True)
set_volume_tracking(True)

# 建立視覺化器
vis = Visualizer()
await vis.start()

# 建立液體處理器
lh = LiquidHandler(
    backend=ChatterboxBackend(num_channels=8),
    deck=STARLetDeck()
)
lh.visualizer = vis
await lh.setup()

# 定義資源
tip_rack = TIP_CAR_480_A00(name="tips")
source_plate = Cos_96_DW_1mL(name="source")
dest_plate = Cos_96_DW_1mL(name="dest")

# 分配到工作台
lh.deck.assign_child_resource(tip_rack, rails=1)
lh.deck.assign_child_resource(source_plate, rails=10)
lh.deck.assign_child_resource(dest_plate, rails=15)

# 設定初始體積
for well in source_plate.children:
    well.tracker.set_liquids([("sample", 200)])

# 執行帶有視覺化的協定
await lh.pick_up_tips(tip_rack["A1:H1"])
await lh.transfer(
    source_plate["A1:H12"],
    dest_plate["A1:H12"],
    vols=100
)
await lh.drop_tips()

# 保持視覺化器開啟以檢視最終狀態
input("按 Enter 關閉視覺化器...")

# 清理
await lh.stop()
await vis.stop()
```

## 工作台布局編輯器

### 使用工作台編輯器

PyLabRobot 包含圖形化工作台布局編輯器：

**功能：**
- 視覺化工作台設計介面
- 拖放資源放置
- 編輯初始液體狀態
- 設定吸頭存在
- 以 JSON 儲存/載入布局

**使用方式：**
- 通過視覺化器介面存取
- 以圖形方式建立布局而非程式碼
- 匯出為 JSON 以供協定使用

### 載入工作台布局

```python
from pylabrobot.resources import Deck

# 從 JSON 檔案載入工作台
deck = Deck.load_from_json_file("my_deck_layout.json")

# 與液體處理器一起使用
lh = LiquidHandler(backend=backend, deck=deck)
await lh.setup()

# 資源已經分配
source = deck.get_resource("source")
dest = deck.get_resource("dest")
tip_rack = deck.get_resource("tips")
```

## 模擬

### ChatterboxBackend

ChatterboxBackend 模擬液體處理操作：

**功能：**
- 不需要硬體
- 驗證協定邏輯
- 追蹤吸頭和體積
- 支援所有液體處理操作
- 與視覺化器配合使用

**設置：**

```python
from pylabrobot.liquid_handling.backends.simulation import ChatterboxBackend

# 建立模擬後端
backend = ChatterboxBackend(
    num_channels=8  # 模擬 8 通道移液器
)

# 與液體處理器一起使用
lh = LiquidHandler(backend=backend, deck=STARLetDeck())
```

### 模擬使用情境

#### 協定開發

```python
async def develop_protocol():
    """使用模擬開發協定"""

    # 使用模擬進行開發
    lh = LiquidHandler(
        backend=ChatterboxBackend(),
        deck=STARLetDeck()
    )

    # 連接視覺化器
    vis = Visualizer()
    await vis.start()
    lh.visualizer = vis

    await lh.setup()

    try:
        # 開發和測試協定
        await lh.pick_up_tips(tip_rack["A1"])
        await lh.transfer(plate["A1"], plate["A2"], vols=100)
        await lh.drop_tips()

        print("協定開發完成！")

    finally:
        await lh.stop()
        await vis.stop()
```

#### 協定驗證

```python
async def validate_protocol():
    """不使用硬體驗證協定邏輯"""

    set_tip_tracking(True)
    set_volume_tracking(True)

    lh = LiquidHandler(
        backend=ChatterboxBackend(),
        deck=STARLetDeck()
    )
    await lh.setup()

    try:
        # 設置資源
        tip_rack = TIP_CAR_480_A00(name="tips")
        plate = Cos_96_DW_1mL(name="plate")

        lh.deck.assign_child_resource(tip_rack, rails=1)
        lh.deck.assign_child_resource(plate, rails=10)

        # 設定初始狀態
        for well in plate.children:
            well.tracker.set_liquids([(None, 200)])

        # 執行協定
        await lh.pick_up_tips(tip_rack["A1:H1"])

        # 測試不同體積
        test_volumes = [50, 100, 150]
        for i, vol in enumerate(test_volumes):
            await lh.transfer(
                plate[f"A{i+1}:H{i+1}"],
                plate[f"A{i+4}:H{i+4}"],
                vols=vol
            )

        await lh.drop_tips()

        # 驗證體積
        for i, vol in enumerate(test_volumes):
            for row in "ABCDEFGH":
                well = plate[f"{row}{i+4}"]
                actual_vol = well.tracker.get_volume()
                assert actual_vol == vol, f"{well.name} 中體積不匹配"

        print("協定驗證通過！")

    finally:
        await lh.stop()
```

#### 測試邊緣情況

```python
async def test_edge_cases():
    """在模擬中測試協定邊緣情況"""

    lh = LiquidHandler(
        backend=ChatterboxBackend(),
        deck=STARLetDeck()
    )
    await lh.setup()

    try:
        # 測試 1：從空孔吸取
        try:
            await lh.aspirate(empty_plate["A1"], vols=100)
            print("X 應該對空孔拋出錯誤")
        except Exception as e:
            print(f"正確拋出錯誤：{e}")

        # 測試 2：孔溢出
        try:
            await lh.dispense(small_well, vols=1000)  # 太多
            print("X 應該對溢出拋出錯誤")
        except Exception as e:
            print(f"正確拋出錯誤：{e}")

        # 測試 3：吸頭容量
        try:
            await lh.aspirate(large_volume_well, vols=2000)  # 超過吸頭容量
            print("X 應該對吸頭容量拋出錯誤")
        except Exception as e:
            print(f"正確拋出錯誤：{e}")

    finally:
        await lh.stop()
```

### CI/CD 整合

使用模擬進行自動化測試：

```python
# test_protocols.py
import pytest
from pylabrobot.liquid_handling import LiquidHandler
from pylabrobot.liquid_handling.backends.simulation import ChatterboxBackend

@pytest.mark.asyncio
async def test_transfer_protocol():
    """測試液體轉移協定"""

    lh = LiquidHandler(
        backend=ChatterboxBackend(),
        deck=STARLetDeck()
    )
    await lh.setup()

    try:
        # 設置
        tip_rack = TIP_CAR_480_A00(name="tips")
        plate = Cos_96_DW_1mL(name="plate")

        lh.deck.assign_child_resource(tip_rack, rails=1)
        lh.deck.assign_child_resource(plate, rails=10)

        # 設定初始體積
        plate["A1"].tracker.set_liquids([(None, 200)])

        # 執行
        await lh.pick_up_tips(tip_rack["A1"])
        await lh.transfer(plate["A1"], plate["A2"], vols=100)
        await lh.drop_tips()

        # 斷言
        assert plate["A1"].tracker.get_volume() == 100
        assert plate["A2"].tracker.get_volume() == 100

    finally:
        await lh.stop()
```

## 最佳實務

1. **始終先使用模擬**：在硬體上運行前在模擬中開發和測試協定
2. **啟用追蹤**：開啟吸頭和體積追蹤以實現準確的視覺化
3. **設定初始狀態**：定義初始液體體積以進行真實的模擬
4. **視覺檢查**：使用視覺化器驗證工作台布局和協定執行
5. **驗證邏輯**：在模擬中測試邊緣情況和錯誤條件
6. **自動化測試**：將模擬整合到 CI/CD 管線
7. **儲存布局**：使用 JSON 儲存和共享工作台布局
8. **記錄狀態**：記錄初始狀態以實現可重現性
9. **互動式開發**：開發期間保持視覺化器開啟
10. **協定改進**：在硬體運行前在模擬中迭代

## 常見模式

### 開發到正式環境工作流程

```python
import os

# 配置
USE_HARDWARE = os.getenv("USE_HARDWARE", "false").lower() == "true"

# 建立適當的後端
if USE_HARDWARE:
    from pylabrobot.liquid_handling.backends import STAR
    backend = STAR()
    print("在 Hamilton STAR 硬體上運行")
else:
    from pylabrobot.liquid_handling.backends.simulation import ChatterboxBackend
    backend = ChatterboxBackend()
    print("在模擬模式下運行")

# 協定的其餘部分相同
lh = LiquidHandler(backend=backend, deck=STARLetDeck())

if not USE_HARDWARE:
    # 為模擬啟用視覺化器
    vis = Visualizer()
    await vis.start()
    lh.visualizer = vis

await lh.setup()

# 協定執行
# ...（硬體和模擬的相同程式碼）

# 執行方式：USE_HARDWARE=false python protocol.py  # 模擬
# 執行方式：USE_HARDWARE=true python protocol.py   # 硬體
```

### 視覺化協定驗證

```python
async def visual_verification():
    """使用視覺化驗證暫停運行協定"""

    vis = Visualizer()
    await vis.start()

    lh = LiquidHandler(
        backend=ChatterboxBackend(),
        deck=STARLetDeck()
    )
    lh.visualizer = vis
    await lh.setup()

    try:
        # 步驟 1
        await lh.pick_up_tips(tip_rack["A1:H1"])
        input("按 Enter 繼續...")

        # 步驟 2
        await lh.aspirate(source["A1:H1"], vols=100)
        input("按 Enter 繼續...")

        # 步驟 3
        await lh.dispense(dest["A1:H1"], vols=100)
        input("按 Enter 繼續...")

        # 步驟 4
        await lh.drop_tips()
        input("按 Enter 完成...")

    finally:
        await lh.stop()
        await vis.stop()
```

## 故障排除

### 視覺化器不更新

- 確保在操作前設定了 `lh.visualizer = vis`
- 檢查是否已全域啟用追蹤
- 驗證視覺化器正在運行（`vis.start()`）
- 如果連接中斷，請重新整理瀏覽器

### 追蹤不運作

```python
# 必須在建立資源之前啟用追蹤
set_tip_tracking(True)
set_volume_tracking(True)

# 然後建立資源
tip_rack = TIP_CAR_480_A00(name="tips")
plate = Cos_96_DW_1mL(name="plate")
```

### 模擬錯誤

- 模擬會驗證操作（例如，不能從空孔吸取）
- 使用 try/except 處理驗證錯誤
- 檢查初始狀態是否正確設定
- 驗證體積不超過容量

## 其他資源

- 視覺化器文件：https://docs.pylabrobot.org/user_guide/using-the-visualizer.html（如有）
- 模擬指南：https://docs.pylabrobot.org/user_guide/simulation.html（如有）
- API 參考：https://docs.pylabrobot.org/api/pylabrobot.visualizer.html
- GitHub 範例：https://github.com/PyLabRobot/pylabrobot/tree/main/examples
