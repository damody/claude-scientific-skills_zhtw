# PyLabRobot 材料處理設備

## 概述

PyLabRobot 整合材料處理設備，包括加熱震盪器、培養箱、離心機和幫浦。這些設備實現環境控制、樣品製備和超越基本液體處理的自動化工作流程。

## 加熱震盪器

### Hamilton HeaterShaker

Hamilton HeaterShaker 為微孔盤提供溫度控制和軌道震盪。

#### 設置

```python
from pylabrobot.heating_shaking import HeaterShaker
from pylabrobot.heating_shaking.hamilton import HamiltonHeaterShakerBackend

# 建立加熱震盪器
hs = HeaterShaker(
    name="heater_shaker_1",
    backend=HamiltonHeaterShakerBackend(),
    size_x=156.0,
    size_y=  156.0,
    size_z=18.0
)

await hs.setup()
```

#### 操作

**溫度控制：**

```python
# 設定溫度（攝氏度）
await hs.set_temperature(37)

# 取得目前溫度
temp = await hs.get_temperature()
print(f"目前溫度：{temp}°C")

# 關閉加熱
await hs.set_temperature(None)
```

**震盪控制：**

```python
# 開始震盪（RPM）
await hs.set_shake_rate(300)  # 300 RPM

# 停止震盪
await hs.set_shake_rate(0)
```

**微孔盤操作：**

```python
# 鎖定微孔盤位置
await hs.lock_plate()

# 解鎖微孔盤
await hs.unlock_plate()
```

#### 與液體處理器整合

```python
from pylabrobot.liquid_handling import LiquidHandler
from pylabrobot.liquid_handling.backends import STAR
from pylabrobot.resources import STARLetDeck

# 初始化設備
lh = LiquidHandler(backend=STAR(), deck=STARLetDeck())
hs = HeaterShaker(name="hs", backend=HamiltonHeaterShakerBackend())

await lh.setup()
await hs.setup()

try:
    # 將加熱震盪器分配到工作台
    lh.deck.assign_child_resource(hs, rails=8)

    # 準備樣品
    tip_rack = TIP_CAR_480_A00(name="tips")
    plate = Cos_96_DW_1mL(name="plate")

    lh.deck.assign_child_resource(tip_rack, rails=1)

    # 將微孔盤放置在加熱震盪器上
    hs.assign_child_resource(plate, location=(0, 0, 0))

    # 將試劑轉移到加熱震盪器上的微孔盤
    await lh.pick_up_tips(tip_rack["A1:H1"])
    await lh.transfer(reagent["A1:H1"], plate["A1:H1"], vols=100)
    await lh.drop_tips()

    # 鎖定微孔盤並開始培養
    await hs.lock_plate()
    await hs.set_temperature(37)
    await hs.set_shake_rate(300)

    # 培養
    import asyncio
    await asyncio.sleep(600)  # 10 分鐘

    # 停止震盪和加熱
    await hs.set_shake_rate(0)
    await hs.set_temperature(None)
    await hs.unlock_plate()

finally:
    await lh.stop()
    await hs.stop()
```

#### 多個加熱震盪器

HamiltonHeaterShakerBackend 處理多個單元：

```python
# 後端自動管理多個加熱震盪器
hs1 = HeaterShaker(name="hs1", backend=HamiltonHeaterShakerBackend())
hs2 = HeaterShaker(name="hs2", backend=HamiltonHeaterShakerBackend())

await hs1.setup()
await hs2.setup()

# 分配到不同的工作台位置
lh.deck.assign_child_resource(hs1, rails=8)
lh.deck.assign_child_resource(hs2, rails=12)

# 獨立控制
await hs1.set_temperature(37)
await hs2.set_temperature(42)
```

### Inheco ThermoShake

Inheco ThermoShake 提供溫度控制和震盪。

#### 設置

```python
from pylabrobot.heating_shaking import HeaterShaker
from pylabrobot.heating_shaking.inheco import InhecoThermoShakeBackend

hs = HeaterShaker(
    name="thermoshake",
    backend=InhecoThermoShakeBackend(),
    size_x=156.0,
    size_y=156.0,
    size_z=18.0
)

await hs.setup()
```

#### 操作

與 Hamilton HeaterShaker 類似：

```python
# 溫度控制
await hs.set_temperature(37)
temp = await hs.get_temperature()

# 震盪控制
await hs.set_shake_rate(300)

# 微孔盤鎖定
await hs.lock_plate()
await hs.unlock_plate()
```

## 培養箱

### Inheco 培養箱

PyLabRobot 支援各種 Inheco 培養箱型號用於控溫微孔盤儲存。

#### 支援的型號

- Inheco 單微孔盤培養箱
- Inheco 多微孔盤培養箱
- 其他 Inheco 溫度控制器

#### 設置

```python
from pylabrobot.temperature_control import TemperatureController
from pylabrobot.temperature_control.inheco import InhecoBackend

# 建立培養箱
incubator = TemperatureController(
    name="incubator",
    backend=InhecoBackend(),
    size_x=156.0,
    size_y=156.0,
    size_z=50.0
)

await incubator.setup()
```

#### 操作

```python
# 設定溫度
await incubator.set_temperature(37)

# 取得溫度
temp = await incubator.get_temperature()
print(f"培養箱溫度：{temp}°C")

# 關閉
await incubator.set_temperature(None)
```

### Thermo Fisher Cytomat 培養箱

Cytomat 培養箱提供具有溫度和 CO2 控制的自動化微孔盤儲存。

#### 設置

```python
from pylabrobot.incubation import Incubator
from pylabrobot.incubation.cytomat_backend import CytomatBackend

incubator = Incubator(
    name="cytomat",
    backend=CytomatBackend()
)

await incubator.setup()
```

#### 操作

```python
# 儲存微孔盤
await incubator.store_plate(plate_id="plate_001", position=1)

# 取出微孔盤
await incubator.retrieve_plate(position=1)

# 設定環境條件
await incubator.set_temperature(37)
await incubator.set_co2(5.0)  # 5% CO2
```

## 離心機

### Agilent VSpin

Agilent VSpin 是用於微孔盤處理的真空輔助離心機。

#### 設置

```python
from pylabrobot.centrifuge import Centrifuge
from pylabrobot.centrifuge.vspin import VSpinBackend

centrifuge = Centrifuge(
    name="vspin",
    backend=VSpinBackend()
)

await centrifuge.setup()
```

#### 操作

**門控制：**

```python
# 開門
await centrifuge.open_door()

# 關門
await centrifuge.close_door()

# 鎖門
await centrifuge.lock_door()

# 解鎖門
await centrifuge.unlock_door()
```

**轉桶定位：**

```python
# 將轉桶移至載入位置
await centrifuge.move_bucket_to_loading()

# 將轉桶移至原位
await centrifuge.move_bucket_to_home()
```

**旋轉：**

```python
# 運行離心機
await centrifuge.spin(
    speed=2000,      # RPM
    duration=300     # 秒
)

# 停止旋轉
await centrifuge.stop_spin()
```

#### 整合範例

```python
async def centrifuge_workflow():
    """完整離心工作流程"""

    lh = LiquidHandler(backend=STAR(), deck=STARLetDeck())
    centrifuge = Centrifuge(name="vspin", backend=VSpinBackend())

    await lh.setup()
    await centrifuge.setup()

    try:
        # 準備樣品
        await lh.pick_up_tips(tip_rack["A1:H1"])
        await lh.transfer(samples["A1:H12"], plate["A1:H12"], vols=200)
        await lh.drop_tips()

        # 載入離心機
        print("將微孔盤移至離心機")
        await centrifuge.open_door()
        await centrifuge.move_bucket_to_loading()
        input("微孔盤載入後按 Enter...")

        await centrifuge.move_bucket_to_home()
        await centrifuge.close_door()
        await centrifuge.lock_door()

        # 離心
        await centrifuge.spin(speed=2000, duration=300)

        # 卸載
        await centrifuge.unlock_door()
        await centrifuge.open_door()
        await centrifuge.move_bucket_to_loading()
        input("微孔盤移除後按 Enter...")

        await centrifuge.move_bucket_to_home()
        await centrifuge.close_door()

    finally:
        await lh.stop()
        await centrifuge.stop()
```

## 幫浦

### Cole Parmer Masterflex

PyLabRobot 支援 Cole Parmer Masterflex 蠕動幫浦用於流體輸送。

#### 設置

```python
from pylabrobot.pumps import Pump
from pylabrobot.pumps.cole_parmer import ColeParmerMasterflexBackend

pump = Pump(
    name="masterflex",
    backend=ColeParmerMasterflexBackend()
)

await pump.setup()
```

#### 操作

**運行幫浦：**

```python
# 運行一段時間
await pump.run_for_duration(
    duration=10,      # 秒
    speed=50          # 最大值的百分比
)

# 連續運行
await pump.start(speed=50)

# 停止幫浦
await pump.stop()
```

**基於體積的泵送：**

```python
# 泵送特定體積（需要校準）
await pump.pump_volume(
    volume=10,        # mL
    speed=50          # 最大值的百分比
)
```

#### 校準

```python
# 校準幫浦以提高體積準確度
# （需要已知體積測量）
await pump.run_for_duration(duration=60, speed=50)
actual_volume = 25.3  # mL 測量值

pump.calibrate(duration=60, speed=50, volume=actual_volume)
```

### Agrowtek 幫浦陣列

支援 Agrowtek 幫浦陣列用於多個同時流體輸送。

#### 設置

```python
from pylabrobot.pumps import PumpArray
from pylabrobot.pumps.agrowtek import AgrowtekBackend

pump_array = PumpArray(
    name="agrowtek",
    backend=AgrowtekBackend(),
    num_pumps=8
)

await pump_array.setup()
```

#### 操作

```python
# 運行特定幫浦
await pump_array.run_pump(
    pump_number=1,
    duration=10,
    speed=50
)

# 同時運行多個幫浦
await pump_array.run_pumps(
    pump_numbers=[1, 2, 3],
    duration=10,
    speed=50
)
```

## 多設備協定

### 複雜工作流程範例

```python
async def complex_workflow():
    """多設備自動化工作流程"""

    # 初始化所有設備
    lh = LiquidHandler(backend=STAR(), deck=STARLetDeck())
    hs = HeaterShaker(name="hs", backend=HamiltonHeaterShakerBackend())
    centrifuge = Centrifuge(name="vspin", backend=VSpinBackend())
    pump = Pump(name="pump", backend=ColeParmerMasterflexBackend())

    await lh.setup()
    await hs.setup()
    await centrifuge.setup()
    await pump.setup()

    try:
        # 1. 樣品製備
        await lh.pick_up_tips(tip_rack["A1:H1"])
        await lh.transfer(samples["A1:H12"], plate["A1:H12"], vols=100)
        await lh.drop_tips()

        # 2. 通過幫浦添加試劑
        await pump.pump_volume(volume=50, speed=50)

        # 3. 在加熱震盪器上混合
        await hs.lock_plate()
        await hs.set_temperature(37)
        await hs.set_shake_rate(300)
        await asyncio.sleep(600)  # 10 分鐘培養
        await hs.set_shake_rate(0)
        await hs.set_temperature(None)
        await hs.unlock_plate()

        # 4. 離心
        await centrifuge.open_door()
        # （載入微孔盤）
        await centrifuge.close_door()
        await centrifuge.spin(speed=2000, duration=180)
        await centrifuge.open_door()
        # （卸載微孔盤）

        # 5. 轉移上清液
        await lh.pick_up_tips(tip_rack["A2:H2"])
        await lh.transfer(
            plate["A1:H12"],
            output_plate["A1:H12"],
            vols=80
        )
        await lh.drop_tips()

    finally:
        await lh.stop()
        await hs.stop()
        await centrifuge.stop()
        await pump.stop()
```

## 最佳實務

1. **設備初始化**：在協定開始時設置所有設備
2. **順序操作**：材料處理通常需要順序步驟
3. **安全性**：手動微孔盤處理前始終解鎖/開門
4. **溫度平衡**：留出時間讓設備達到溫度
5. **錯誤處理**：使用 try/finally 優雅地處理設備錯誤
6. **狀態驗證**：操作前檢查設備狀態
7. **時間控制**：考慮設備特定的延遲（加熱、離心）
8. **維護**：遵循製造商的維護計畫
9. **校準**：定期校準幫浦和溫度控制器
10. **文件**：記錄所有設備設定和參數

## 常見模式

### 控溫培養

```python
async def incubate_with_shaking(
    plate,
    temperature: float,
    shake_rate: int,
    duration: int
):
    """使用溫度和震盪培養微孔盤"""

    hs = HeaterShaker(name="hs", backend=HamiltonHeaterShakerBackend())
    await hs.setup()

    try:
        # 將微孔盤分配到加熱震盪器
        hs.assign_child_resource(plate, location=(0, 0, 0))

        # 開始培養
        await hs.lock_plate()
        await hs.set_temperature(temperature)
        await hs.set_shake_rate(shake_rate)

        # 等待
        await asyncio.sleep(duration)

        # 停止
        await hs.set_shake_rate(0)
        await hs.set_temperature(None)
        await hs.unlock_plate()

    finally:
        await hs.stop()

# 在協定中使用
await incubate_with_shaking(
    plate=assay_plate,
    temperature=37,
    shake_rate=300,
    duration=600  # 10 分鐘
)
```

### 自動化微孔盤處理

```python
async def process_plates(plate_list: list):
    """處理多個微孔盤的工作流程"""

    lh = LiquidHandler(backend=STAR(), deck=STARLetDeck())
    hs = HeaterShaker(name="hs", backend=HamiltonHeaterShakerBackend())

    await lh.setup()
    await hs.setup()

    try:
        for i, plate in enumerate(plate_list):
            print(f"處理微孔盤 {i+1}/{len(plate_list)}")

            # 轉移樣品
            await lh.pick_up_tips(tip_rack[f"A{i+1}:H{i+1}"])
            await lh.transfer(
                source[f"A{i+1}:H{i+1}"],
                plate["A1:H1"],
                vols=100
            )
            await lh.drop_tips()

            # 培養
            hs.assign_child_resource(plate, location=(0, 0, 0))
            await hs.lock_plate()
            await hs.set_temperature(37)
            await hs.set_shake_rate(300)
            await asyncio.sleep(300)  # 5 分鐘
            await hs.set_shake_rate(0)
            await hs.set_temperature(None)
            await hs.unlock_plate()
            hs.unassign_child_resource(plate)

    finally:
        await lh.stop()
        await hs.stop()
```

## 其他資源

- 材料處理文件：https://docs.pylabrobot.org/user_guide/01_material-handling/
- 加熱震盪器：https://docs.pylabrobot.org/user_guide/01_material-handling/heating_shaking/
- API 參考：https://docs.pylabrobot.org/api/
- 支援的設備：https://docs.pylabrobot.org/user_guide/machines.html
