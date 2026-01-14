# Modal 函數

## 基本函數定義

使用 `@app.function()` 裝飾 Python 函數：

```python
import modal

app = modal.App(name="my-app")

@app.function()
def my_function():
    print("Hello from Modal!")
    return "result"
```

## 呼叫函數

### 遠端執行

呼叫 `.remote()` 在 Modal 上執行：

```python
@app.local_entrypoint()
def main():
    result = my_function.remote()
    print(result)
```

### 本地執行

呼叫 `.local()` 在本地執行（用於測試）：

```python
result = my_function.local()
```

## 函數參數

函數接受標準的 Python 參數：

```python
@app.function()
def process(x: int, y: str):
    return f"{y}: {x * 2}"

@app.local_entrypoint()
def main():
    result = process.remote(42, "answer")
```

## 部署

### 臨時應用程式

暫時執行：
```bash
modal run script.py
```

### 已部署的應用程式

持久部署：
```bash
modal deploy script.py
```

從其他程式碼存取已部署的函數：

```python
f = modal.Function.from_name("my-app", "my_function")
result = f.remote(args)
```

## 進入點

### 本地進入點

在本地機器上執行的程式碼：

```python
@app.local_entrypoint()
def main():
    result = my_function.remote()
    print(result)
```

### 遠端進入點

使用 `@app.function()` 而不使用 local_entrypoint - 完全在 Modal 上執行：

```python
@app.function()
def train_model():
    # 所有程式碼都在 Modal 中執行
    ...
```

使用以下命令調用：
```bash
modal run script.py::app.train_model
```

## 參數解析

具有原始類型參數的進入點會自動獲得 CLI 解析：

```python
@app.local_entrypoint()
def main(foo: int, bar: str):
    some_function.remote(foo, bar)
```

執行：
```bash
modal run script.py --foo 1 --bar "hello"
```

對於自訂解析，接受可變長度參數：

```python
import argparse

@app.function()
def train(*arglist):
    parser = argparse.ArgumentParser()
    parser.add_argument("--foo", type=int)
    args = parser.parse_args(args=arglist)
```

## 函數配置

常見參數：

```python
@app.function(
    image=my_image,           # 自訂環境
    gpu="A100",               # GPU 類型
    cpu=2.0,                  # CPU 核心
    memory=4096,              # 記憶體（MB）
    timeout=3600,             # 超時（秒）
    retries=3,                # 重試次數
    secrets=[my_secret],      # 環境密鑰
    volumes={"/data": vol},   # 持久化儲存
)
def my_function():
    ...
```

## 平行執行

### Map

在多個輸入上平行執行函數：

```python
@app.function()
def evaluate_model(x):
    return x ** 2

@app.local_entrypoint()
def main():
    inputs = list(range(100))
    for result in evaluate_model.map(inputs):
        print(result)
```

### Starmap

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
results = my_func.map(
    range(3),
    return_exceptions=True,
    wrap_returned_exceptions=False
)
# [0, 1, Exception('error')]
```

## 非同步函數

定義非同步函數：

```python
@app.function()
async def async_function(x: int):
    await asyncio.sleep(1)
    return x * 2

@app.local_entrypoint()
async def main():
    result = await async_function.remote.aio(42)
```

## 生成器函數

返回迭代器以串流結果：

```python
@app.function()
def generate_data():
    for i in range(10):
        yield i

@app.local_entrypoint()
def main():
    for value in generate_data.remote_gen():
        print(value)
```

## 產生函數

提交函數進行背景執行：

```python
@app.function()
def process_job(data):
    # 長時間執行的任務
    return result

@app.local_entrypoint()
def main():
    # 產生但不等待
    call = process_job.spawn(data)

    # 稍後取得結果
    result = call.get(timeout=60)
```

## 程式化執行

以程式化方式執行應用程式：

```python
def main():
    with modal.enable_output():
        with app.run():
            result = some_function.remote()
```

## 指定進入點

有多個函數時，指定要執行哪一個：

```python
@app.function()
def f():
    print("Function f")

@app.function()
def g():
    print("Function g")
```

執行特定函數：
```bash
modal run script.py::app.f
modal run script.py::app.g
```
