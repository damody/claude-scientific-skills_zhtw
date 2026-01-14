# Modal 入門

## 註冊

在 https://modal.com 免費註冊並獲得每月 $30 的額度。

## 驗證

使用 Modal CLI 設定驗證：

```bash
modal token new
```

這會在 `~/.modal.toml` 中建立憑證。或者，設定環境變數：
- `MODAL_TOKEN_ID`
- `MODAL_TOKEN_SECRET`

## 基本概念

### Modal 是無伺服器的

Modal 是一個無伺服器平台 - 只需為使用的資源付費，並在幾秒內按需啟動容器。

### 核心元件

**App（應用程式）**：代表在 Modal 上執行的應用程式，將一個或多個函數分組以進行原子部署。

**Function（函數）**：作為獨立擴展的單位。沒有即時輸入時，不會執行容器（也不會產生費用）。

**Image（映像）**：程式碼執行的環境 - 安裝了相依性的容器快照。

## 第一個 Modal 應用程式

建立檔案 `hello_modal.py`：

```python
import modal

app = modal.App(name="hello-modal")

@app.function()
def hello():
    print("Hello from Modal!")
    return "success"

@app.local_entrypoint()
def main():
    hello.remote()
```

執行：
```bash
modal run hello_modal.py
```

## 執行應用程式

### 臨時應用程式（開發）

使用 `modal run` 暫時執行：
```bash
modal run script.py
```

腳本退出時應用程式停止。使用 `--detach` 在客戶端退出後保持執行。

### 已部署的應用程式（生產）

使用 `modal deploy` 持久部署：
```bash
modal deploy script.py
```

在 https://modal.com/apps 或使用以下命令查看已部署的應用程式：
```bash
modal app list
```

停止已部署的應用程式：
```bash
modal app stop app-name
```

## 主要功能

- **快速原型開發**：撰寫 Python，在幾秒內在 GPU 上執行
- **無伺服器 API**：使用裝飾器建立 web 端點
- **排程任務**：在雲端執行 cron 任務
- **GPU 推論**：存取 T4、L4、A10、A100、H100、H200、B200 GPU
- **分散式磁碟區**：ML 模型的持久化儲存
- **沙箱**：用於不受信任程式碼的安全容器
