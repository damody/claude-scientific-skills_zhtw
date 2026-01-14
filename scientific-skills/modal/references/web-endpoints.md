# Web 端點

## 快速開始

使用單一裝飾器建立 web 端點：

```python
image = modal.Image.debian_slim().pip_install("fastapi[standard]")

@app.function(image=image)
@modal.fastapi_endpoint()
def hello():
    return "Hello world!"
```

## 開發和部署

### 使用 `modal serve` 開發

```bash
modal serve server.py
```

建立具有即時重新載入的臨時應用程式。端點的變更幾乎立即出現。

### 使用 `modal deploy` 部署

```bash
modal deploy server.py
```

建立具有穩定 URL 的持久端點。

## 簡單端點

### 查詢參數

```python
@app.function(image=image)
@modal.fastapi_endpoint()
def square(x: int):
    return {"square": x**2}
```

呼叫：
```bash
curl "https://workspace--app-square.modal.run?x=42"
```

### POST 請求

```python
@app.function(image=image)
@modal.fastapi_endpoint(method="POST")
def square(item: dict):
    return {"square": item['x']**2}
```

呼叫：
```bash
curl -X POST -H 'Content-Type: application/json' \
  --data '{"x": 42}' \
  https://workspace--app-square.modal.run
```

### Pydantic 模型

```python
from pydantic import BaseModel

class Item(BaseModel):
    name: str
    qty: int = 42

@app.function()
@modal.fastapi_endpoint(method="POST")
def process(item: Item):
    return {"processed": item.name, "quantity": item.qty}
```

## ASGI 應用程式（FastAPI、Starlette、FastHTML）

服務完整的 ASGI 應用程式：

```python
image = modal.Image.debian_slim().pip_install("fastapi[standard]")

@app.function(image=image)
@modal.concurrent(max_inputs=100)
@modal.asgi_app()
def fastapi_app():
    from fastapi import FastAPI

    web_app = FastAPI()

    @web_app.get("/")
    async def root():
        return {"message": "Hello"}

    @web_app.post("/echo")
    async def echo(request: Request):
        body = await request.json()
        return body

    return web_app
```

## WSGI 應用程式（Flask、Django）

服務同步 web 框架：

```python
image = modal.Image.debian_slim().pip_install("flask")

@app.function(image=image)
@modal.concurrent(max_inputs=100)
@modal.wsgi_app()
def flask_app():
    from flask import Flask, request

    web_app = Flask(__name__)

    @web_app.post("/echo")
    def echo():
        return request.json

    return web_app
```

## 非 ASGI Web 伺服器

對於具有自訂網路綁定的框架：

```python
@app.function()
@modal.concurrent(max_inputs=100)
@modal.web_server(8000)
def my_server():
    import subprocess
    # 必須綁定到 0.0.0.0，而非 127.0.0.1
    subprocess.Popen("python -m http.server -d / 8000", shell=True)
```

## 串流回應

使用 FastAPI 的 `StreamingResponse`：

```python
import time

def event_generator():
    for i in range(10):
        yield f"data: event {i}\n\n".encode()
        time.sleep(0.5)

@app.function(image=modal.Image.debian_slim().pip_install("fastapi[standard]"))
@modal.fastapi_endpoint()
def stream():
    from fastapi.responses import StreamingResponse
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"
    )
```

### 從 Modal 函數串流

```python
@app.function(gpu="any")
def process_gpu():
    for i in range(10):
        yield f"data: result {i}\n\n".encode()
        time.sleep(1)

@app.function(image=modal.Image.debian_slim().pip_install("fastapi[standard]"))
@modal.fastapi_endpoint()
def hook():
    from fastapi.responses import StreamingResponse
    return StreamingResponse(
        process_gpu.remote_gen(),
        media_type="text/event-stream"
    )
```

### 使用 .map()

```python
@app.function()
def process_segment(i):
    return f"segment {i}\n"

@app.function(image=modal.Image.debian_slim().pip_install("fastapi[standard]"))
@modal.fastapi_endpoint()
def stream_parallel():
    from fastapi.responses import StreamingResponse
    return StreamingResponse(
        process_segment.map(range(10)),
        media_type="text/plain"
    )
```

## WebSockets

`@web_server`、`@asgi_app` 和 `@wsgi_app` 都支援。每個連線維持單一函數呼叫。使用 `@modal.concurrent` 進行多個同時連線。

支援完整的 WebSocket 協定（RFC 6455）。每條訊息最多 2 MiB。

## 驗證

### 代理驗證權杖

透過 Modal 的一級驗證：

```python
@app.function()
@modal.fastapi_endpoint()
def protected():
    return "authenticated!"
```

在設定中使用權杖保護，在標頭中傳遞：
- `Modal-Key`
- `Modal-Secret`

### Bearer Token 驗證

```python
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

auth_scheme = HTTPBearer()

@app.function(secrets=[modal.Secret.from_name("auth-token")])
@modal.fastapi_endpoint()
async def protected(token: HTTPAuthorizationCredentials = Depends(auth_scheme)):
    import os
    if token.credentials != os.environ["AUTH_TOKEN"]:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )
    return "success!"
```

### 客戶端 IP 位址

```python
from fastapi import Request

@app.function()
@modal.fastapi_endpoint()
def get_ip(request: Request):
    return f"Your IP: {request.client.host}"
```

## Web 端點 URL

### 自動生成的 URL

格式：`https://<workspace>--<app>-<function>.modal.run`

帶環境後綴：`https://<workspace>-<suffix>--<app>-<function>.modal.run`

### 自訂標籤

```python
@app.function()
@modal.fastapi_endpoint(label="api")
def handler():
    ...
# URL: https://workspace--api.modal.run
```

### 程式化 URL 檢索

```python
@app.function()
@modal.fastapi_endpoint()
def my_endpoint():
    url = my_endpoint.get_web_url()
    return {"url": url}

# 從已部署的函數
f = modal.Function.from_name("app-name", "my_endpoint")
url = f.get_web_url()
```

### 自訂網域

Team 和 Enterprise 方案可用：

```python
@app.function()
@modal.fastapi_endpoint(custom_domains=["api.example.com"])
def hello(message: str):
    return {"message": f"hello {message}"}
```

多個網域：
```python
@modal.fastapi_endpoint(custom_domains=["api.example.com", "api.example.net"])
```

萬用字元網域：
```python
@modal.fastapi_endpoint(custom_domains=["*.example.com"])
```

TLS 憑證自動生成和更新。

## 效能

### 冷啟動

第一個請求可能會經歷冷啟動（幾秒鐘）。Modal 為後續請求保持容器活躍。

### 擴展

- 基於流量自動擴展
- 使用 `@modal.concurrent` 每個容器處理多個請求
- 超過並行限制時，啟動額外容器
- 達到最大容器數時請求排隊

### 速率限制

預設：每秒 200 個請求，5 秒突發倍數
- 超出返回 429 狀態碼
- 聯繫支援以增加限制

### 大小限制

- 請求本體：最多 4 GiB
- 回應本體：無限制
- WebSocket 訊息：最多 2 MiB
