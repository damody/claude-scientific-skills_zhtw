# PyLabRobot 分析設備

## 概述

PyLabRobot 整合分析設備，包括微孔盤讀取器、天平和其他測量裝置。這使得結合液體處理與分析測量的自動化工作流程成為可能。

## 微孔盤讀取器

### BMG CLARIOstar (Plus)

BMG Labtech CLARIOstar 和 CLARIOstar Plus 是可測量吸光度、發光度和螢光的微孔盤讀取器。

#### 硬體設置

**實體連接：**
1. IEC C13 電源線連接到主電源
2. USB-B 傳輸線連接到電腦（設備端有安全螺絲）
3. 選配：RS-232 連接埠用於微孔盤堆疊單元

**通訊：**
- 通過 FTDI/USB-A 在韌體層級進行串列連接
- 跨平台支援（Windows、macOS、Linux）

#### 軟體設置

```python
from pylabrobot.plate_reading import PlateReader
from pylabrobot.plate_reading.clario_star_backend import CLARIOstarBackend

# 建立後端
backend = CLARIOstarBackend()

# 初始化微孔盤讀取器
pr = PlateReader(
    name="CLARIOstar",
    backend=backend,
    size_x=0.0,    # 微孔盤讀取器的物理尺寸不重要
    size_y=0.0,
    size_z=0.0
)

# 設置（初始化設備）
await pr.setup()

# 完成時
await pr.stop()
```

#### 基本操作

**開啟和關閉：**

```python
# 開啟載入托盤
await pr.open()

# （手動或機器人載入微孔盤）

# 關閉載入托盤
await pr.close()
```

**溫度控制：**

```python
# 設定溫度（攝氏度）
await pr.set_temperature(37)

# 注意：達到溫度較慢
# 在協定早期設定溫度
```

**讀取測量：**

```python
# 吸光度讀取
data = await pr.read_absorbance(wavelength=450)  # nm

# 發光度讀取
data = await pr.read_luminescence()

# 螢光讀取
data = await pr.read_fluorescence(
    excitation_wavelength=485,  # nm
    emission_wavelength=535     # nm
)
```

#### 資料格式

微孔盤讀取器方法回傳陣列資料：

```python
import numpy as np

# 讀取吸光度
data = await pr.read_absorbance(wavelength=450)

# data 通常是 2D 陣列（96 孔盤為 8x12）
print(f"資料形狀：{data.shape}")
print(f"孔 A1：{data[0][0]}")
print(f"孔 H12：{data[7][11]}")

# 轉換為 DataFrame 以便更容易處理
import pandas as pd
df = pd.DataFrame(data)
```

#### 與液體處理器整合

結合微孔盤讀取與液體處理：

```python
from pylabrobot.liquid_handling import LiquidHandler
from pylabrobot.liquid_handling.backends import STAR
from pylabrobot.resources import STARLetDeck
from pylabrobot.plate_reading import PlateReader
from pylabrobot.plate_reading.clario_star_backend import CLARIOstarBackend

# 初始化液體處理器
lh = LiquidHandler(backend=STAR(), deck=STARLetDeck())
await lh.setup()

# 初始化微孔盤讀取器
pr = PlateReader(name="CLARIOstar", backend=CLARIOstarBackend())
await pr.setup()

# 提早設定溫度
await pr.set_temperature(37)

try:
    # 使用液體處理器準備樣品
    tip_rack = TIP_CAR_480_A00(name="tips")
    reagent_plate = Cos_96_DW_1mL(name="reagents")
    assay_plate = Cos_96_DW_1mL(name="assay")

    lh.deck.assign_child_resource(tip_rack, rails=1)
    lh.deck.assign_child_resource(reagent_plate, rails=10)
    lh.deck.assign_child_resource(assay_plate, rails=15)

    # 轉移樣品
    await lh.pick_up_tips(tip_rack["A1:H1"])
    await lh.transfer(
        reagent_plate["A1:H12"],
        assay_plate["A1:H12"],
        vols=100
    )
    await lh.drop_tips()

    # 將微孔盤移到讀取器（手動或機械手臂）
    print("將分析微孔盤移到微孔盤讀取器")
    input("微孔盤載入後按 Enter...")

    # 讀取微孔盤
    await pr.open()
    # （此處載入微孔盤）
    await pr.close()

    data = await pr.read_absorbance(wavelength=450)
    print(f"吸光度資料：{data}")

finally:
    await lh.stop()
    await pr.stop()
```

#### 進階功能

**開發狀態：**

部分 CLARIOstar 功能正在開發中：
- 光譜掃描
- 注射器針頭控制
- 詳細測量參數配置
- 特定孔讀取模式

請查看目前文件以了解最新功能支援。

#### 最佳實務

1. **溫度控制**：提早設定溫度，因為加熱較慢
2. **微孔盤載入**：確保微孔盤在關閉前正確就位
3. **測量選擇**：為您的分析選擇適當的波長
4. **資料驗證**：檢查測量品質和預期範圍
5. **錯誤處理**：處理逾時和通訊錯誤
6. **維護**：依據製造商指南保持光學元件清潔

#### 範例：完整的微孔盤讀取工作流程

```python
async def run_plate_reading_assay():
    """包含樣品準備和讀取的完整工作流程"""

    # 初始化設備
    lh = LiquidHandler(backend=STAR(), deck=STARLetDeck())
    pr = PlateReader(name="CLARIOstar", backend=CLARIOstarBackend())

    await lh.setup()
    await pr.setup()

    # 設定微孔盤讀取器溫度
    await pr.set_temperature(37)

    try:
        # 定義資源
        tip_rack = TIP_CAR_480_A00(name="tips")
        samples = Cos_96_DW_1mL(name="samples")
        assay_plate = Cos_96_DW_1mL(name="assay")
        substrate = Trough_100ml(name="substrate")

        lh.deck.assign_child_resource(tip_rack, rails=1)
        lh.deck.assign_child_resource(substrate, rails=5)
        lh.deck.assign_child_resource(samples, rails=10)
        lh.deck.assign_child_resource(assay_plate, rails=15)

        # 轉移樣品
        await lh.pick_up_tips(tip_rack["A1:H1"])
        await lh.transfer(
            samples["A1:H12"],
            assay_plate["A1:H12"],
            vols=50
        )
        await lh.drop_tips()

        # 添加底物
        await lh.pick_up_tips(tip_rack["A2:H2"])
        for col in range(1, 13):
            await lh.transfer(
                substrate["channel_1"],
                assay_plate[f"A{col}:H{col}"],
                vols=50
            )
        await lh.drop_tips()

        # 培養（如需要）
        # await asyncio.sleep(300)  # 5 分鐘

        # 移至微孔盤讀取器
        print("將分析微孔盤轉移到 CLARIOstar")
        input("準備好後按 Enter...")

        await pr.open()
        input("微孔盤載入後按 Enter...")
        await pr.close()

        # 讀取吸光度
        data = await pr.read_absorbance(wavelength=450)

        # 處理結果
        import pandas as pd
        df = pd.DataFrame(
            data,
            index=[f"{r}" for r in "ABCDEFGH"],
            columns=[f"{c}" for c in range(1, 13)]
        )

        print("吸光度結果：")
        print(df)

        # 儲存結果
        df.to_csv("plate_reading_results.csv")

        return df

    finally:
        await lh.stop()
        await pr.stop()

# 執行分析
results = await run_plate_reading_assay()
```

## 天平

### Mettler Toledo 天平

PyLabRobot 支援 Mettler Toledo 天平進行質量測量。

#### 設置

```python
from pylabrobot.scales import Scale
from pylabrobot.scales.mettler_toledo_backend import MettlerToledoBackend

# 建立天平
scale = Scale(
    name="analytical_scale",
    backend=MettlerToledoBackend()
)

await scale.setup()
```

#### 操作

```python
# 取得重量測量
weight = await scale.get_weight()  # 回傳以公克為單位的重量
print(f"重量：{weight} g")

# 歸零（扣重）天平
await scale.tare()

# 取得多次測量
weights = []
for i in range(5):
    w = await scale.get_weight()
    weights.append(w)
    await asyncio.sleep(1)

average_weight = sum(weights) / len(weights)
print(f"平均重量：{average_weight} g")
```

#### 與液體處理器整合

```python
# 在協定期間稱量樣品
lh = LiquidHandler(backend=STAR(), deck=STARLetDeck())
scale = Scale(name="scale", backend=MettlerToledoBackend())

await lh.setup()
await scale.setup()

try:
    # 歸零天平
    await scale.tare()

    # 分配液體
    await lh.pick_up_tips(tip_rack["A1"])
    await lh.aspirate(reagent["A1"], vols=1000)

    # （移動到天平位置）

    # 分配並稱重
    await lh.dispense(container, vols=1000)
    weight = await scale.get_weight()

    print(f"分配重量：{weight} g")

    # 計算實際體積（假設水的密度 = 1 g/mL）
    actual_volume = weight * 1000  # 將 g 轉換為 µL
    print(f"實際體積：{actual_volume} µL")

    await lh.drop_tips()

finally:
    await lh.stop()
    await scale.stop()
```

## 其他分析設備

### 流式細胞儀

部分流式細胞儀整合正在開發中。請查看目前文件以了解支援狀態。

### 分光光度計

可能支援其他分光光度計型號。請查看文件以了解目前設備相容性。

## 多設備工作流程

### 協調多個設備

```python
async def multi_device_workflow():
    """協調液體處理器、微孔盤讀取器和天平"""

    # 初始化所有設備
    lh = LiquidHandler(backend=STAR(), deck=STARLetDeck())
    pr = PlateReader(name="CLARIOstar", backend=CLARIOstarBackend())
    scale = Scale(name="scale", backend=MettlerToledoBackend())

    await lh.setup()
    await pr.setup()
    await scale.setup()

    try:
        # 1. 稱量試劑
        await scale.tare()
        # （將容器放在天平上）
        reagent_weight = await scale.get_weight()

        # 2. 使用液體處理器準備樣品
        await lh.pick_up_tips(tip_rack["A1:H1"])
        await lh.transfer(source["A1:H12"], dest["A1:H12"], vols=100)
        await lh.drop_tips()

        # 3. 讀取微孔盤
        await pr.open()
        # （載入微孔盤）
        await pr.close()
        data = await pr.read_absorbance(wavelength=450)

        return {
            "reagent_weight": reagent_weight,
            "absorbance_data": data
        }

    finally:
        await lh.stop()
        await pr.stop()
        await scale.stop()
```

## 最佳實務

1. **設備初始化**：在協定開始時設置所有設備
2. **錯誤處理**：優雅地處理通訊錯誤
3. **清理**：始終對所有設備呼叫 `stop()`
4. **時間控制**：考慮設備特定的時間（溫度平衡、測量時間）
5. **校準**：遵循製造商的校準程序
6. **資料驗證**：驗證測量值在預期範圍內
7. **文件**：記錄設備設定和參數
8. **整合測試**：徹底測試多設備工作流程
9. **並行操作**：使用 async 在可能時重疊操作
10. **資料儲存**：儲存帶有後設資料（時間戳記、設定）的原始資料

## 常見模式

### 動力學微孔盤讀取

```python
async def kinetic_reading(num_reads: int, interval: int):
    """執行動力學微孔盤讀取"""

    pr = PlateReader(name="CLARIOstar", backend=CLARIOstarBackend())
    await pr.setup()

    try:
        await pr.set_temperature(37)
        await pr.open()
        # （載入微孔盤）
        await pr.close()

        results = []
        for i in range(num_reads):
            data = await pr.read_absorbance(wavelength=450)
            timestamp = time.time()
            results.append({
                "read_number": i + 1,
                "timestamp": timestamp,
                "data": data
            })

            if i < num_reads - 1:
                await asyncio.sleep(interval)

        return results

    finally:
        await pr.stop()

# 每 30 秒讀取一次，持續 10 分鐘
results = await kinetic_reading(num_reads=20, interval=30)
```

## 其他資源

- 微孔盤讀取文件：https://docs.pylabrobot.org/user_guide/02_analytical/
- BMG CLARIOstar 指南：https://docs.pylabrobot.org/user_guide/02_analytical/plate-reading/bmg-clariostar.html
- API 參考：https://docs.pylabrobot.org/api/pylabrobot.plate_reading.html
- 支援的設備：https://docs.pylabrobot.org/user_guide/machines.html
