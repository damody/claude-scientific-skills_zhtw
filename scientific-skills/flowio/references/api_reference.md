# FlowIO API 參考

## 概述

FlowIO 是用於讀取和寫入流式細胞儀標準（FCS）檔案的 Python 函式庫。它支援 FCS 版本 2.0、3.0 和 3.1，且依賴最少。

## 安裝

```bash
pip install flowio
```

支援 Python 3.9 及更高版本。

## 核心類別

### FlowData

用於處理 FCS 檔案的主要類別。

#### 建構函式

```python
FlowData(fcs_file,
         ignore_offset_error=False,
         ignore_offset_discrepancy=False,
         use_header_offsets=False,
         only_text=False,
         nextdata_offset=None,
         null_channel_list=None)
```

**參數：**
- `fcs_file`：檔案路徑（str）、Path 物件或檔案控制代碼
- `ignore_offset_error`（bool）：忽略位移錯誤（預設：False）
- `ignore_offset_discrepancy`（bool）：忽略 HEADER 和 TEXT 區段之間的位移差異（預設：False）
- `use_header_offsets`（bool）：使用 HEADER 區段位移而非 TEXT 區段（預設：False）
- `only_text`（bool）：僅解析 TEXT 區段，跳過 DATA 和 ANALYSIS（預設：False）
- `nextdata_offset`（int）：讀取多資料集檔案的位元組位移
- `null_channel_list`（list）：要排除的空通道 PnN 標籤列表

#### 屬性

**檔案資訊：**
- `name`：FCS 檔案名稱
- `file_size`：檔案大小（位元組）
- `version`：FCS 版本（例如：'3.0'、'3.1'）
- `header`：包含 HEADER 區段資訊的字典
- `data_type`：資料格式類型（'I'、'F'、'D'、'A'）

**通道資訊：**
- `channel_count`：資料集中的通道數量
- `channels`：將通道編號對應到通道資訊的字典
- `pnn_labels`：PnN（短通道名稱）標籤列表
- `pns_labels`：PnS（描述性染色名稱）標籤列表
- `pnr_values`：每個通道的 PnR（範圍）值列表
- `fluoro_indices`：螢光通道的索引列表
- `scatter_indices`：散射通道的索引列表
- `time_index`：時間通道的索引（或 None）
- `null_channels`：空通道索引列表

**事件資料：**
- `event_count`：資料集中的事件（列）數量
- `events`：原始事件資料（作為位元組）

**元資料：**
- `text`：TEXT 區段鍵值對的字典
- `analysis`：ANALYSIS 區段鍵值對的字典（如果存在）

#### 方法

##### as_array()

```python
as_array(preprocess=True)
```

以 2 維 NumPy 陣列回傳事件資料。

**參數：**
- `preprocess`（bool）：套用增益、對數和時間縮放轉換（預設：True）

**回傳：**
- 形狀為 (event_count, channel_count) 的 NumPy ndarray

**範例：**
```python
flow_data = FlowData('sample.fcs')
events_array = flow_data.as_array()  # 預處理後的資料
raw_array = flow_data.as_array(preprocess=False)  # 原始資料
```

##### write_fcs()

```python
write_fcs(filename, metadata=None)
```

將 FlowData 實例匯出為新的 FCS 檔案。

**參數：**
- `filename`（str）：輸出檔案路徑
- `metadata`（dict）：要新增/更新的 TEXT 區段關鍵字的可選字典

**範例：**
```python
flow_data = FlowData('sample.fcs')
flow_data.write_fcs('output.fcs', metadata={'$SRC': '修改後的資料'})
```

**注意：** 以 FCS 3.1 格式匯出，使用單精度浮點數資料。

## 公用程式函式

### read_multiple_data_sets()

```python
read_multiple_data_sets(fcs_file,
                        ignore_offset_error=False,
                        ignore_offset_discrepancy=False,
                        use_header_offsets=False)
```

從包含多個資料集的 FCS 檔案讀取所有資料集。

**參數：**
- 與 FlowData 建構函式相同（除了 `nextdata_offset`）

**回傳：**
- FlowData 實例列表，每個資料集一個

**範例：**
```python
from flowio import read_multiple_data_sets

datasets = read_multiple_data_sets('multi_dataset.fcs')
print(f"發現 {len(datasets)} 個資料集")
for i, dataset in enumerate(datasets):
    print(f"資料集 {i}：{dataset.event_count} 個事件")
```

### create_fcs()

```python
create_fcs(filename,
           event_data,
           channel_names,
           opt_channel_names=None,
           metadata=None)
```

從事件資料建立新的 FCS 檔案。

**參數：**
- `filename`（str）：輸出檔案路徑
- `event_data`（ndarray）：2 維 NumPy 事件資料陣列（列=事件，欄=通道）
- `channel_names`（list）：PnN（短）通道名稱列表
- `opt_channel_names`（list）：PnS（描述性）通道名稱的可選列表
- `metadata`（dict）：TEXT 區段關鍵字的可選字典

**範例：**
```python
import numpy as np
from flowio import create_fcs

# 建立合成資料
events = np.random.rand(10000, 5)
channels = ['FSC-A', 'SSC-A', 'FL1-A', 'FL2-A', 'Time']
opt_channels = ['Forward Scatter', 'Side Scatter', 'FITC', 'PE', 'Time']

create_fcs('synthetic.fcs',
           events,
           channels,
           opt_channel_names=opt_channels,
           metadata={'$SRC': '合成資料'})
```

## 例外類別

### FlowIOWarning

用於非關鍵問題的通用警告類別。

### PnEWarning

建立 FCS 檔案時 PnE 值無效時發出的警告。

### FlowIOException

FlowIO 錯誤的基礎例外類別。

### FCSParsingError

解析 FCS 檔案時發生問題時引發。

### DataOffsetDiscrepancyError

當 HEADER 和 TEXT 區段為資料區段提供不同的位元組位移時引發。

**解決方法：** 建立 FlowData 實例時使用 `ignore_offset_discrepancy=True` 參數。

### MultipleDataSetsError

嘗試使用標準 FlowData 建構函式讀取包含多個資料集的檔案時引發。

**解決方案：** 改用 `read_multiple_data_sets()` 函式。

## FCS 檔案結構參考

FCS 檔案由四個區段組成：

1. **HEADER**：包含 FCS 版本和其他區段的位元組位置
2. **TEXT**：鍵值元資料對（分隔格式）
3. **DATA**：原始事件資料（二進位、浮點或 ASCII）
4. **ANALYSIS**（可選）：資料處理的結果

### 常見 TEXT 區段關鍵字

- `$BEGINDATA`、`$ENDDATA`：DATA 區段的位元組位移
- `$BEGINANALYSIS`、`$ENDANALYSIS`：ANALYSIS 區段的位元組位移
- `$BYTEORD`：位元組順序（1,2,3,4 為小端序；4,3,2,1 為大端序）
- `$DATATYPE`：資料類型（'I'=整數、'F'=浮點、'D'=雙精度、'A'=ASCII）
- `$MODE`：資料模式（'L'=列表模式，最常見）
- `$NEXTDATA`：到下一個資料集的位移（如果是單一資料集則為 0）
- `$PAR`：參數（通道）數量
- `$TOT`：事件總數
- `PnN`：參數 n 的短名稱
- `PnS`：參數 n 的描述性染色名稱
- `PnR`：參數 n 的範圍（最大值）
- `PnE`：參數 n 的放大指數（格式："a,b"，其中 value = a * 10^(b*x)）
- `PnG`：參數 n 的放大增益

## 通道類型

FlowIO 自動分類通道：

- **散射通道**：FSC（前向散射）、SSC（側向散射）
- **螢光通道**：FL1、FL2、FITC、PE 等
- **時間通道**：通常標記為「Time」

透過以下屬性存取索引：
- `flow_data.scatter_indices`
- `flow_data.fluoro_indices`
- `flow_data.time_index`

## 資料預處理

呼叫 `as_array(preprocess=True)` 時，FlowIO 套用：

1. **增益縮放**：乘以 PnG 值
2. **對數轉換**：如果存在，套用 PnE 指數轉換
3. **時間縮放**：將時間值轉換為適當的單位

要存取原始、未處理的資料：`as_array(preprocess=False)`

## 最佳實踐

1. **記憶體效率**：當只需要元資料時使用 `only_text=True`
2. **錯誤處理**：將檔案操作包裝在 try-except 區塊中處理 FCSParsingError
3. **多資料集檔案**：如果不確定資料集數量，始終使用 `read_multiple_data_sets()`
4. **位移問題**：如果遇到位移錯誤，嘗試 `ignore_offset_discrepancy=True`
5. **通道選擇**：使用 null_channel_list 在解析時排除不需要的通道

## 與 FlowKit 整合

如需進階流式細胞儀分析，包括補償、圈選和 GatingML 支援，請考慮將 FlowKit 函式庫與 FlowIO 一起使用。FlowKit 提供建構在 FlowIO 檔案解析功能之上的更高級抽象。

## 範例工作流程

### 基本檔案讀取

```python
from flowio import FlowData

# 讀取 FCS 檔案
flow = FlowData('experiment.fcs')

# 列印基本資訊
print(f"版本：{flow.version}")
print(f"事件數：{flow.event_count}")
print(f"通道數：{flow.channel_count}")
print(f"通道名稱：{flow.pnn_labels}")

# 取得事件資料
events = flow.as_array()
print(f"資料形狀：{events.shape}")
```

### 元資料擷取

```python
from flowio import FlowData

flow = FlowData('sample.fcs', only_text=True)

# 存取元資料
print(f"採集日期：{flow.text.get('$DATE', 'N/A')}")
print(f"儀器：{flow.text.get('$CYT', 'N/A')}")

# 通道資訊
for i, (pnn, pns) in enumerate(zip(flow.pnn_labels, flow.pns_labels)):
    print(f"通道 {i}：{pnn}（{pns}）")
```

### 建立新的 FCS 檔案

```python
import numpy as np
from flowio import create_fcs

# 產生或處理資料
data = np.random.rand(5000, 3) * 1000

# 定義通道
channels = ['FSC-A', 'SSC-A', 'FL1-A']
stains = ['Forward Scatter', 'Side Scatter', 'GFP']

# 建立 FCS 檔案
create_fcs('output.fcs',
           data,
           channels,
           opt_channel_names=stains,
           metadata={
               '$SRC': 'Python 腳本',
               '$DATE': '19-OCT-2025'
           })
```

### 處理多資料集檔案

```python
from flowio import read_multiple_data_sets

# 讀取所有資料集
datasets = read_multiple_data_sets('multi.fcs')

# 處理每個資料集
for i, dataset in enumerate(datasets):
    print(f"\n資料集 {i}：")
    print(f"  事件數：{dataset.event_count}")
    print(f"  通道：{dataset.pnn_labels}")

    # 取得資料陣列
    events = dataset.as_array()
    mean_values = events.mean(axis=0)
    print(f"  平均值：{mean_values}")
```

### 修改並重新匯出

```python
from flowio import FlowData

# 讀取原始檔案
flow = FlowData('original.fcs')

# 取得事件資料
events = flow.as_array(preprocess=False)

# 修改資料（範例：套用自訂轉換）
events[:, 0] = events[:, 0] * 1.5  # 縮放第一個通道

# 注意：目前 FlowIO 不支援直接修改事件資料
# 如需修改，請改用 create_fcs()：
from flowio import create_fcs

create_fcs('modified.fcs',
           events,
           flow.pnn_labels,
           opt_channel_names=flow.pns_labels,
           metadata=flow.text)
```
