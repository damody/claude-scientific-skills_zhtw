# Dask Futures

## 概述

Dask futures 擴展了 Python 的 `concurrent.futures` 介面，支援立即（非延遲）任務執行。與延遲運算（用於 DataFrames、Arrays 和 Bags）不同，futures 在運算可能隨時間演進或需要動態工作流程建構的情況下提供更多靈活性。

## 核心概念

Futures 代表即時任務執行：
- 任務在提交時立即執行（非延遲）
- 每個 future 代表一個遠端運算結果
- 自動追蹤 futures 之間的依賴關係
- 支援動態、演進的工作流程
- 直接控制任務排程和資料放置

## 主要功能

### 即時執行
- 任務在提交時立即執行
- 不需要明確的 `.compute()` 呼叫
- 使用 `.result()` 方法取得結果

### 自動依賴管理
當您提交具有 future 輸入的任務時，Dask 會自動處理依賴追蹤。一旦所有輸入 futures 完成，它們將被移動到單一 worker 以進行高效運算。

### 動態工作流程
根據中間結果建構運算：
- 根據先前結果提交新任務
- 條件執行路徑
- 具有變化結構的迭代演算法

## 何時使用 Futures

**使用 Futures 的情況**：
- 建構動態、演進的工作流程
- 需要立即任務執行（非延遲）
- 運算取決於執行時期條件
- 需要對任務放置的精細控制
- 實作自訂平行演算法
- 需要有狀態的運算（使用 actors）

**使用其他集合的情況**：
- 靜態、預定義的運算圖（使用 delayed、DataFrames、Arrays）
- 大型集合上的簡單資料平行（使用 Bags、DataFrames）
- 標準陣列/dataframe 操作就足夠

## 設定 Client

Futures 需要分散式 client：

```python
from dask.distributed import Client

# 本地叢集（在單機上）
client = Client()

# 或指定資源
client = Client(n_workers=4, threads_per_worker=2)

# 或連接到現有叢集
client = Client('scheduler-address:8786')
```

## 提交任務

### 基本 Submit
```python
from dask.distributed import Client

client = Client()

# 提交單一任務
def add(x, y):
    return x + y

future = client.submit(add, 1, 2)

# 取得結果
result = future.result()  # 阻塞直到完成
print(result)  # 3
```

### 多個任務
```python
# 提交多個獨立任務
futures = []
for i in range(10):
    future = client.submit(add, i, i)
    futures.append(future)

# 收集結果
results = client.gather(futures)  # 高效的平行收集
```

### Map 到多個輸入
```python
# 對多個輸入應用函數
def square(x):
    return x ** 2

# 批次提交任務
futures = client.map(square, range(100))

# 收集結果
results = client.gather(futures)
```

**注意**：每個任務有約 1ms 開銷，使 `map` 不適合數百萬個微小任務。對於大規模資料集，改用 Bags 或 DataFrames。

## 使用 Futures

### 檢查狀態
```python
future = client.submit(expensive_function, arg)

# 檢查是否完成
print(future.done())  # False 或 True

# 檢查狀態
print(future.status)  # 'pending'、'running'、'finished' 或 'error'
```

### 非阻塞結果取得
```python
# 非阻塞檢查
if future.done():
    result = future.result()
else:
    print("Still computing...")

# 或使用回調
def handle_result(future):
    print(f"Result: {future.result()}")

future.add_done_callback(handle_result)
```

### 錯誤處理
```python
def might_fail(x):
    if x < 0:
        raise ValueError("Negative value")
    return x ** 2

future = client.submit(might_fail, -5)

try:
    result = future.result()
except ValueError as e:
    print(f"Task failed: {e}")
```

## 任務依賴

### 自動依賴追蹤
```python
# 提交任務
future1 = client.submit(add, 1, 2)

# 使用 future 作為輸入（建立依賴）
future2 = client.submit(add, future1, 10)  # 依賴 future1

# 串連依賴
future3 = client.submit(add, future2, 100)  # 依賴 future2

# 取得最終結果
result = future3.result()  # 113
```

### 複雜依賴
```python
# 多個依賴
a = client.submit(func1, x)
b = client.submit(func2, y)
c = client.submit(func3, a, b)  # 依賴 a 和 b

result = c.result()
```

## 資料移動優化

### 分散資料
預先分散重要資料以避免重複傳輸：

```python
# 一次上傳資料到叢集
large_dataset = client.scatter(big_data)  # 回傳 future

# 在多個任務中使用分散的資料
futures = [client.submit(process, large_dataset, i) for i in range(100)]

# 每個任務使用相同的分散資料，無需重新傳輸
results = client.gather(futures)
```

### 高效收集
使用 `client.gather()` 進行並行結果收集：

```python
# 較佳：一次收集所有（平行）
results = client.gather(futures)

# 較差：順序取得結果
results = [f.result() for f in futures]
```

## Fire-and-Forget

對於不需要結果的副作用任務：

```python
from dask.distributed import fire_and_forget

def log_to_database(data):
    # 寫入資料庫，不需要回傳值
    database.write(data)

# 提交但不保留引用
future = client.submit(log_to_database, data)
fire_and_forget(future)

# 即使沒有活動的 future 引用，Dask 也不會放棄這個運算
```

## 效能特性

### 任務開銷
- 每個任務約 1ms 開銷
- 適合：數千個任務
- 不適合：數百萬個微小任務

### Worker 到 Worker 通訊
- 直接 worker 到 worker 資料傳輸
- 往返延遲：約 1ms
- 對任務依賴高效

### 記憶體管理
Dask 在本地追蹤活動的 futures。當 future 被您的本地 Python 會話垃圾回收時，Dask 將自由刪除該資料。

**保留引用**：
```python
# 保留引用以防止刪除
important_result = client.submit(expensive_calc, data)

# 多次使用結果
future1 = client.submit(process1, important_result)
future2 = client.submit(process2, important_result)
```

## 進階協調

### 分散式原語

**佇列**：
```python
from dask.distributed import Queue

queue = Queue()

def producer():
    for i in range(10):
        queue.put(i)

def consumer():
    results = []
    for _ in range(10):
        results.append(queue.get())
    return results

# 提交任務
client.submit(producer)
result_future = client.submit(consumer)
results = result_future.result()
```

**鎖**：
```python
from dask.distributed import Lock

lock = Lock()

def critical_section():
    with lock:
        # 一次只有一個任務執行這個
        shared_resource.update()
```

**事件**：
```python
from dask.distributed import Event

event = Event()

def waiter():
    event.wait()  # 阻塞直到事件被設定
    return "Event occurred"

def setter():
    time.sleep(5)
    event.set()

# 啟動兩個任務
wait_future = client.submit(waiter)
set_future = client.submit(setter)

result = wait_future.result()  # 等待 setter 完成
```

**變數**：
```python
from dask.distributed import Variable

var = Variable('my-var')

# 設定值
var.set(42)

# 從任務取得值
def reader():
    return var.get()

future = client.submit(reader)
print(future.result())  # 42
```

## Actors

對於有狀態、快速變化的工作流程，actors 支援約 1ms 的 worker 到 worker 往返延遲，同時繞過排程器協調。

### 建立 Actors
```python
from dask.distributed import Client

client = Client()

class Counter:
    def __init__(self):
        self.count = 0

    def increment(self):
        self.count += 1
        return self.count

    def get_count(self):
        return self.count

# 在 worker 上建立 actor
counter = client.submit(Counter, actor=True).result()

# 呼叫方法
future1 = counter.increment()
future2 = counter.increment()
result = counter.get_count().result()
print(result)  # 2
```

### Actor 使用情境
- 有狀態服務（資料庫、快取）
- 快速變化的狀態
- 複雜的協調模式
- 即時串流應用

## 常見模式

### Embarrassingly Parallel 任務
```python
from dask.distributed import Client

client = Client()

def process_item(item):
    # 獨立運算
    return expensive_computation(item)

# 平行處理多個項目
items = range(1000)
futures = client.map(process_item, items)

# 收集所有結果
results = client.gather(futures)
```

### 動態任務提交
```python
def recursive_compute(data, depth):
    if depth == 0:
        return process(data)

    # 分割和遞迴
    left, right = split(data)
    left_future = client.submit(recursive_compute, left, depth - 1)
    right_future = client.submit(recursive_compute, right, depth - 1)

    # 組合結果
    return combine(left_future.result(), right_future.result())

# 開始運算
result_future = client.submit(recursive_compute, initial_data, 5)
result = result_future.result()
```

### 參數掃描
```python
from itertools import product

def run_simulation(param1, param2, param3):
    # 使用參數執行模擬
    return simulate(param1, param2, param3)

# 生成參數組合
params = product(range(10), range(10), range(10))

# 提交所有組合
futures = [client.submit(run_simulation, p1, p2, p3) for p1, p2, p3 in params]

# 在完成時收集結果
from dask.distributed import as_completed

for future in as_completed(futures):
    result = future.result()
    process_result(result)
```

### 帶依賴的管道
```python
# 階段 1：載入資料
load_futures = [client.submit(load_data, file) for file in files]

# 階段 2：處理（依賴階段 1）
process_futures = [client.submit(process, f) for f in load_futures]

# 階段 3：聚合（依賴階段 2）
agg_future = client.submit(aggregate, process_futures)

# 取得最終結果
result = agg_future.result()
```

### 迭代演算法
```python
# 初始化
state = client.scatter(initial_state)

# 迭代
for iteration in range(num_iterations):
    # 根據當前狀態計算更新
    state = client.submit(update_function, state)

    # 檢查收斂
    converged = client.submit(check_convergence, state)
    if converged.result():
        break

# 取得最終狀態
final_state = state.result()
```

## 最佳實踐

### 1. 預先分散大型資料
```python
# 上傳一次，多次使用
large_data = client.scatter(big_dataset)
futures = [client.submit(process, large_data, i) for i in range(100)]
```

### 2. 使用 Gather 進行批次取得
```python
# 高效：平行收集
results = client.gather(futures)

# 低效：順序
results = [f.result() for f in futures]
```

### 3. 使用引用管理記憶體
```python
# 保留重要的 futures
important = client.submit(expensive_calc, data)

# 多次使用
f1 = client.submit(use_result, important)
f2 = client.submit(use_result, important)

# 完成後清理
del important
```

### 4. 適當處理錯誤
```python
futures = client.map(might_fail, inputs)

# 檢查錯誤
results = []
errors = []
for future in as_completed(futures):
    try:
        results.append(future.result())
    except Exception as e:
        errors.append(e)
```

### 5. 使用 as_completed 進行漸進處理
```python
from dask.distributed import as_completed

futures = client.map(process, items)

# 結果到達時處理
for future in as_completed(futures):
    result = future.result()
    handle_result(result)
```

## 除錯技巧

### 監控儀表板
查看 Dask 儀表板以了解：
- 任務進度
- Worker 使用率
- 記憶體使用
- 任務依賴

### 檢查任務狀態
```python
# 檢查 future
print(future.status)
print(future.done())

# 在錯誤時取得追蹤
try:
    future.result()
except Exception:
    print(future.traceback())
```

### 分析任務
```python
# 取得效能資料
client.profile(filename='profile.html')
```
