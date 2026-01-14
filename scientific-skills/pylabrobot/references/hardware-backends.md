# PyLabRobot 硬體後端

## 概述

PyLabRobot 使用後端抽象系統，允許相同的協定程式碼在不同的液體處理機器人和平台上執行。後端處理設備特定的通訊，而 `LiquidHandler` 前端提供統一的介面。

## 後端架構

### 後端如何運作

1. **前端**：`LiquidHandler` 類別提供高階 API
2. **後端**：設備特定的類別處理硬體通訊
3. **協定**：相同的程式碼適用於不同的後端

```python
# 相同的協定程式碼
await lh.pick_up_tips(tip_rack["A1"])
await lh.aspirate(plate["A1"], vols=100)
await lh.dispense(plate["A2"], vols=100)
await lh.drop_tips()

# 適用於任何後端（STAR、Opentrons、模擬等）
```

### 後端介面

所有後端繼承自 `LiquidHandlerBackend` 並實現：
- `setup()`：初始化與硬體的連接
- `stop()`：關閉連接並清理
- 設備特定的命令方法（aspirate、dispense 等）

## 支援的後端

### Hamilton STAR（完整支援）

Hamilton STAR 和 STARlet 液體處理機器人具有完整的 PyLabRobot 支援。

**設置：**

```python
from pylabrobot.liquid_handling import LiquidHandler
from pylabrobot.liquid_handling.backends import STAR
from pylabrobot.resources import STARLetDeck

# 建立 STAR 後端
backend = STAR()

# 初始化液體處理器
lh = LiquidHandler(backend=backend, deck=STARLetDeck())
await lh.setup()
```

**平台支援：**
- Windows ✅
- macOS ✅
- Linux ✅
- Raspberry Pi ✅

**通訊：**
- USB 連接到機器人
- 直接韌體命令
- 不需要 Hamilton 軟體

**功能：**
- 完整液體處理操作
- CO-RE 吸頭支援
- 96 通道頭支援（如有配備）
- 溫度控制
- 載體和軌道式定位

**工作台類型：**
```python
from pylabrobot.resources import STARLetDeck, STARDeck

# 用於 STARlet（較小工作台）
deck = STARLetDeck()

# 用於 STAR（完整工作台）
deck = STARDeck()
```

**範例：**

```python
from pylabrobot.liquid_handling import LiquidHandler
from pylabrobot.liquid_handling.backends import STAR
from pylabrobot.resources import STARLetDeck, TIP_CAR_480_A00, Cos_96_DW_1mL

# 初始化
lh = LiquidHandler(backend=STAR(), deck=STARLetDeck())
await lh.setup()

# 定義資源
tip_rack = TIP_CAR_480_A00(name="tips")
plate = Cos_96_DW_1mL(name="plate")

# 分配到軌道
lh.deck.assign_child_resource(tip_rack, rails=1)
lh.deck.assign_child_resource(plate, rails=10)

# 執行協定
await lh.pick_up_tips(tip_rack["A1"])
await lh.transfer(plate["A1"], plate["A2"], vols=100)
await lh.drop_tips()

await lh.stop()
```

### Opentrons OT-2（支援）

Opentrons OT-2 通過 Opentrons HTTP API 支援。

**設置：**

```python
from pylabrobot.liquid_handling import LiquidHandler
from pylabrobot.liquid_handling.backends import OpentronsBackend
from pylabrobot.resources import OTDeck

# 建立 Opentrons 後端（需要機器人 IP 位址）
backend = OpentronsBackend(host="192.168.1.100")  # 替換為您機器人的 IP

# 初始化液體處理器
lh = LiquidHandler(backend=backend, deck=OTDeck())
await lh.setup()
```

**平台支援：**
- 任何可網路存取 OT-2 的平台

**通訊：**
- 透過網路的 HTTP API
- 需要機器人 IP 位址
- 不需要 Opentrons 應用程式

**功能：**
- 8 通道移液器支援
- 單通道移液器支援
- 標準 OT-2 工作台布局
- 座標式定位

**限制：**
- 使用較舊的 Opentrons HTTP API
- 某些功能可能比 STAR 有限

**範例：**

```python
from pylabrobot.liquid_handling import LiquidHandler
from pylabrobot.liquid_handling.backends import OpentronsBackend
from pylabrobot.resources import OTDeck

# 使用機器人 IP 初始化
lh = LiquidHandler(
    backend=OpentronsBackend(host="192.168.1.100"),
    deck=OTDeck()
)
await lh.setup()

# 載入工作台布局
lh.deck = Deck.load_from_json_file("opentrons_layout.json")

# 執行協定
await lh.pick_up_tips(tip_rack["A1"])
await lh.transfer(plate["A1"], plate["A2"], vols=100)
await lh.drop_tips()

await lh.stop()
```

### Tecan EVO（開發中）

Tecan EVO 液體處理機器人的支援正在開發中。

**目前狀態：**
- 開發中
- 基本命令可能可用
- 請查看文件以了解目前功能支援

**設置（當可用時）：**

```python
from pylabrobot.liquid_handling import LiquidHandler
from pylabrobot.liquid_handling.backends import TecanBackend
from pylabrobot.resources import TecanDeck

backend = TecanBackend()
lh = LiquidHandler(backend=backend, deck=TecanDeck())
```

### Hamilton Vantage（大部分支援）

Hamilton Vantage 具有「大部分」完整的支援。

**設置：**

```python
from pylabrobot.liquid_handling.backends import Vantage
from pylabrobot.resources import VantageDeck

lh = LiquidHandler(backend=Vantage(), deck=VantageDeck())
```

**功能：**
- 類似 STAR 支援
- 某些進階功能可能有限

## 模擬後端

### ChatterboxBackend（模擬）

使用模擬後端在沒有實體硬體的情況下測試協定。

**設置：**

```python
from pylabrobot.liquid_handling import LiquidHandler
from pylabrobot.liquid_handling.backends.simulation import ChatterboxBackend
from pylabrobot.resources import STARLetDeck

# 建立模擬後端
backend = ChatterboxBackend(num_channels=8)

# 初始化液體處理器
lh = LiquidHandler(backend=backend, deck=STARLetDeck())
await lh.setup()
```

**功能：**
- 不需要硬體
- 模擬所有液體處理操作
- 與視覺化器配合使用提供即時反饋
- 驗證協定邏輯
- 追蹤吸頭和體積

**使用情境：**
- 協定開發和測試
- 培訓和教育
- CI/CD 管線測試
- 無硬體存取時的除錯

**範例：**

```python
from pylabrobot.liquid_handling import LiquidHandler
from pylabrobot.liquid_handling.backends.simulation import ChatterboxBackend
from pylabrobot.resources import STARLetDeck, TIP_CAR_480_A00, Cos_96_DW_1mL
from pylabrobot.resources import set_tip_tracking, set_volume_tracking

# 啟用模擬追蹤
set_tip_tracking(True)
set_volume_tracking(True)

# 使用模擬後端初始化
lh = LiquidHandler(
    backend=ChatterboxBackend(num_channels=8),
    deck=STARLetDeck()
)
await lh.setup()

# 定義資源
tip_rack = TIP_CAR_480_A00(name="tips")
plate = Cos_96_DW_1mL(name="plate")

lh.deck.assign_child_resource(tip_rack, rails=1)
lh.deck.assign_child_resource(plate, rails=10)

# 設定初始體積
for well in plate.children:
    well.tracker.set_liquids([(None, 200)])

# 執行模擬協定
await lh.pick_up_tips(tip_rack["A1:H1"])
await lh.transfer(plate["A1:H1"], plate["A2:H2"], vols=100)
await lh.drop_tips()

# 檢查結果
print(f"A1 體積：{plate['A1'].tracker.get_volume()} µL")  # 100 µL
print(f"A2 體積：{plate['A2'].tracker.get_volume()} µL")  # 100 µL

await lh.stop()
```

## 切換後端

### 與後端無關的協定

編寫適用於任何後端的協定：

```python
def get_backend(robot_type: str):
    """建立適當後端的工廠函數"""
    if robot_type == "star":
        from pylabrobot.liquid_handling.backends import STAR
        return STAR()
    elif robot_type == "opentrons":
        from pylabrobot.liquid_handling.backends import OpentronsBackend
        return OpentronsBackend(host="192.168.1.100")
    elif robot_type == "simulation":
        from pylabrobot.liquid_handling.backends.simulation import ChatterboxBackend
        return ChatterboxBackend()
    else:
        raise ValueError(f"未知的機器人類型：{robot_type}")

def get_deck(robot_type: str):
    """建立適當工作台的工廠函數"""
    if robot_type == "star":
        from pylabrobot.resources import STARLetDeck
        return STARLetDeck()
    elif robot_type == "opentrons":
        from pylabrobot.resources import OTDeck
        return OTDeck()
    elif robot_type == "simulation":
        from pylabrobot.resources import STARLetDeck
        return STARLetDeck()
    else:
        raise ValueError(f"未知的機器人類型：{robot_type}")

# 在協定中使用
robot_type = "simulation"  # 根據需要更改為 "star" 或 "opentrons"
backend = get_backend(robot_type)
deck = get_deck(robot_type)

lh = LiquidHandler(backend=backend, deck=deck)
await lh.setup()

# 協定程式碼適用於任何後端
await lh.pick_up_tips(tip_rack["A1"])
await lh.transfer(plate["A1"], plate["A2"], vols=100)
await lh.drop_tips()
```

### 開發工作流程

1. **開發**：使用 ChatterboxBackend 編寫協定
2. **測試**：使用視覺化器執行以驗證邏輯
3. **驗證**：使用真實工作台布局在模擬中測試
4. **部署**：切換到硬體後端（STAR、Opentrons）

```python
# 開發
lh = LiquidHandler(backend=ChatterboxBackend(), deck=STARLetDeck())

# ... 開發協定 ...

# 正式環境（只需更改後端）
lh = LiquidHandler(backend=STAR(), deck=STARLetDeck())
```

## 後端配置

### 自訂後端參數

某些後端接受配置參數：

```python
# 帶有自訂參數的 Opentrons
backend = OpentronsBackend(
    host="192.168.1.100",
    port=31950  # 預設 Opentrons API 連接埠
)

# 帶有自訂通道的 ChatterboxBackend
backend = ChatterboxBackend(
    num_channels=8  # 8 通道模擬
)
```

### 連接故障排除

**Hamilton STAR：**
- 確保 USB 傳輸線已連接
- 檢查沒有其他軟體正在使用機器人
- 驗證韌體是最新的
- 在 macOS/Linux 上，可能需要 USB 權限

**Opentrons OT-2：**
- 驗證機器人 IP 位址正確
- 檢查網路連通性（ping 機器人）
- 確保機器人已開機
- 確認 Opentrons 應用程式沒有阻擋 API 存取

**一般：**
- 使用 `await lh.setup()` 測試連接
- 檢查錯誤訊息以了解具體問題
- 確保有適當的設備存取權限

## 後端特定功能

### Hamilton STAR 專用

```python
# 直接存取後端以使用硬體專用功能
star_backend = lh.backend

# Hamilton 專用命令（如需要）
# 大多數操作應通過 LiquidHandler 介面
```

### Opentrons 專用

```python
# Opentrons 專用配置
ot_backend = lh.backend

# 如需要可直接存取 OT-2 API（進階）
# 大多數操作應通過 LiquidHandler 介面
```

## 最佳實務

1. **抽象硬體**：盡可能編寫與後端無關的協定
2. **在模擬中測試**：始終先使用 ChatterboxBackend 測試
3. **工廠模式**：使用工廠函數建立後端
4. **錯誤處理**：優雅地處理連接錯誤
5. **文件**：記錄您的協定支援哪些後端
6. **配置**：使用配置檔案儲存後端參數
7. **版本控制**：追蹤後端版本和相容性
8. **清理**：始終呼叫 `await lh.stop()` 以釋放硬體
9. **單一連接**：同時只應有一個程式連接到硬體
10. **平台測試**：部署前在目標平台上測試

## 常見模式

### 多後端支援

```python
import asyncio
from typing import Literal

async def run_protocol(
    robot_type: Literal["star", "opentrons", "simulation"],
    visualize: bool = False
):
    """在指定後端上執行協定"""

    # 建立後端
    if robot_type == "star":
        from pylabrobot.liquid_handling.backends import STAR
        backend = STAR()
        deck = STARLetDeck()
    elif robot_type == "opentrons":
        from pylabrobot.liquid_handling.backends import OpentronsBackend
        backend = OpentronsBackend(host="192.168.1.100")
        deck = OTDeck()
    elif robot_type == "simulation":
        from pylabrobot.liquid_handling.backends.simulation import ChatterboxBackend
        backend = ChatterboxBackend()
        deck = STARLetDeck()

    # 初始化
    lh = LiquidHandler(backend=backend, deck=deck)
    await lh.setup()

    try:
        # 載入工作台布局（與後端無關）
        # lh.deck = Deck.load_from_json_file(f"{robot_type}_layout.json")

        # 執行協定（與後端無關）
        await lh.pick_up_tips(tip_rack["A1"])
        await lh.transfer(plate["A1"], plate["A2"], vols=100)
        await lh.drop_tips()

        print("協定成功完成！")

    finally:
        await lh.stop()

# 在不同後端上執行
await run_protocol("simulation")      # 在模擬中測試
await run_protocol("star")            # 在 Hamilton STAR 上執行
await run_protocol("opentrons")       # 在 Opentrons OT-2 上執行
```

## 其他資源

- 後端文件：https://docs.pylabrobot.org/user_guide/backends.html
- 支援的機器：https://docs.pylabrobot.org/user_guide/machines.html
- API 參考：https://docs.pylabrobot.org/api/pylabrobot.liquid_handling.backends.html
- GitHub 範例：https://github.com/PyLabRobot/pylabrobot/tree/main/examples
