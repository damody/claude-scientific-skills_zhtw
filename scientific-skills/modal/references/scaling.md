# Modal 擴展

## 自動擴展

每個 Modal 函數對應一個自動擴展的容器池。Modal 的自動擴展器：
- 在沒有可用容量時啟動容器
- 在資源閒置時關閉容器
- 預設在沒有輸入要處理時擴展到零

自動擴展決策快速且頻繁地做出。

## 使用 `.map()` 平行執行

在不同輸入上重複平行執行函數：

```python
@app.function()
def evaluate_model(x):
    return x ** 2

@app.local_entrypoint()
def main():
    inputs = list(range(100))
    # 跨容器平行執行 100 個輸入
    for result in evaluate_model.map(inputs):
        print(result)
```

### 使用 `.starmap()` 處理多個參數

對於有多個參數的函數：

```python
@app.function()
def add(a, b):
    return a + b

@app.local_entrypoint()
def main():
    results = list(add.starmap([(1, 2), (3, 4)]))
    # [3, 7]
```

### 例外處理

```python
@app.function()
def may_fail(a):
    if a == 2:
        raise Exception("error")
    return a ** 2

@app.local_entrypoint()
def main():
    results = list(may_fail.map(
        range(3),
        return_exceptions=True,
        wrap_returned_exceptions=False
    ))
    # [0, 1, Exception('error')]
```

## 自動擴展配置

使用參數配置自動擴展器行為：

```python
@app.function(
    max_containers=100,      # 容器上限
    min_containers=2,        # 即使不活動也保持暖機
    buffer_containers=5,     # 活動時維持的緩衝
    scaledown_window=60,     # 縮減前的最大閒置時間（秒）
)
def my_function():
    ...
```

參數：
- **max_containers**：總容器的上限
- **min_containers**：即使不活動也保持暖機的最小數量
- **buffer_containers**：函數活動時的緩衝大小（額外輸入不需要排隊）
- **scaledown_window**：縮減前的最大閒置持續時間（秒）

權衡：
- 較大的暖機池/緩衝 → 更高成本、更低延遲
- 較長的縮減窗口 → 對不頻繁請求的流動較少

## 動態自動擴展器更新

在不重新部署的情況下更新自動擴展器設定：

```python
f = modal.Function.from_name("my-app", "f")
f.update_autoscaler(max_containers=100)
```

設定在下次部署時恢復為裝飾器配置，或被進一步更新覆蓋：

```python
f.update_autoscaler(min_containers=2, max_containers=10)
f.update_autoscaler(min_containers=4)  # max_containers=10 仍然有效
```

### 基於時間的擴展

根據時間調整暖機池：

```python
@app.function()
def inference_server():
    ...

@app.function(schedule=modal.Cron("0 6 * * *", timezone="America/New_York"))
def increase_warm_pool():
    inference_server.update_autoscaler(min_containers=4)

@app.function(schedule=modal.Cron("0 22 * * *", timezone="America/New_York"))
def decrease_warm_pool():
    inference_server.update_autoscaler(min_containers=0)
```

### 對於類別

為特定參數實例更新自動擴展器：

```python
MyClass = modal.Cls.from_name("my-app", "MyClass")
obj = MyClass(model_version="3.5")
obj.update_autoscaler(buffer_containers=2)  # type: ignore
```

## 輸入並行

使用 `@modal.concurrent` 在每個容器處理多個輸入：

```python
@app.function()
@modal.concurrent(max_inputs=100)
def my_function(input: str):
    # 容器可以處理最多 100 個並行輸入
    ...
```

適合 I/O 密集型工作負載：
- 資料庫查詢
- 外部 API 請求
- 遠端 Modal 函數呼叫

### 並行機制

**同步函數**：獨立執行緒（必須是執行緒安全的）

```python
@app.function()
@modal.concurrent(max_inputs=10)
def sync_function():
    time.sleep(1)  # 必須是執行緒安全的
```

**非同步函數**：獨立 asyncio 任務（不能阻塞事件迴圈）

```python
@app.function()
@modal.concurrent(max_inputs=10)
async def async_function():
    await asyncio.sleep(1)  # 不能阻塞事件迴圈
```

### 目標 vs 最大輸入

```python
@app.function()
@modal.concurrent(
    max_inputs=120,    # 硬性限制
    target_inputs=100  # 自動擴展器目標
)
def my_function(input: str):
    # 允許比目標高 20% 的突發
    ...
```

自動擴展器以 `target_inputs` 為目標，但容器可以在擴展期間突發到 `max_inputs`。

## 擴展限制

Modal 對每個函數強制限制：
- 2,000 個待處理輸入（尚未分配給容器）
- 25,000 個總輸入（執行中 + 待處理）

對於 `.spawn()` 非同步任務：最多 100 萬個待處理輸入。

超過限制返回 `Resource Exhausted` 錯誤 - 稍後重試。

每次 `.map()` 調用：最多 1,000 個並行輸入。

## 非同步用法

使用非同步 API 進行任意平行執行模式：

```python
@app.function()
async def async_task(x):
    await asyncio.sleep(1)
    return x * 2

@app.local_entrypoint()
async def main():
    tasks = [async_task.remote.aio(i) for i in range(100)]
    results = await asyncio.gather(*tasks)
```

## 常見陷阱

**錯誤**：使用 Python 的內建 map（順序執行）
```python
# 不要這樣做
results = map(evaluate_model, inputs)
```

**錯誤**：先呼叫函數
```python
# 不要這樣做
results = evaluate_model(inputs).map()
```

**正確**：在 Modal 函數物件上呼叫 .map()
```python
# 這樣做
results = evaluate_model.map(inputs)
```
