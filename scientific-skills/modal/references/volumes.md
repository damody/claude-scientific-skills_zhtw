# Modal Volumes

## 概述

Modal Volumes 為 Modal 應用程式提供高效能分散式檔案系統。專為一次寫入、多次讀取的工作負載設計，如 ML 模型權重和分散式資料處理。

## 建立 Volumes

### 透過 CLI

```bash
modal volume create my-volume
```

對於 Volumes v2（測試版）：
```bash
modal volume create --version=2 my-volume
```

### 從程式碼

```python
vol = modal.Volume.from_name("my-volume", create_if_missing=True)

# 對於 v2
vol = modal.Volume.from_name("my-volume", create_if_missing=True, version=2)
```

## 使用 Volumes

透過掛載點附加到函數：

```python
vol = modal.Volume.from_name("my-volume")

@app.function(volumes={"/data": vol})
def run():
    with open("/data/xyz.txt", "w") as f:
        f.write("hello")
    vol.commit()  # 持久化變更
```

## 提交和重新載入

### 提交

將變更持久化到 Volume：

```python
@app.function(volumes={"/data": vol})
def write_data():
    with open("/data/file.txt", "w") as f:
        f.write("data")
    vol.commit()  # 使變更對其他容器可見
```

**背景提交**：Modal 每隔幾秒和容器關閉時自動提交 Volume 變更。

### 重新載入

從其他容器取得最新變更：

```python
@app.function(volumes={"/data": vol})
def read_data():
    vol.reload()  # 取得最新變更
    with open("/data/file.txt", "r") as f:
        content = f.read()
```

在容器建立時，掛載最新的 Volume 狀態。需要重新載入才能看到其他容器的後續提交。

## 上傳檔案

### 批次上傳（高效）

```python
vol = modal.Volume.from_name("my-volume")

with vol.batch_upload() as batch:
    batch.put_file("local-path.txt", "/remote-path.txt")
    batch.put_directory("/local/directory/", "/remote/directory")
    batch.put_file(io.BytesIO(b"some data"), "/foobar")
```

### 透過映像

```python
image = modal.Image.debian_slim().add_local_dir(
    local_path="/home/user/my_dir",
    remote_path="/app"
)

@app.function(image=image)
def process():
    # 檔案在 /app 可用
    ...
```

## 下載檔案

### 透過 CLI

```bash
modal volume get my-volume remote.txt local.txt
```

CLI 最大檔案大小：無限制
儀表板最大檔案大小：16 MB

### 透過 Python SDK

```python
vol = modal.Volume.from_name("my-volume")

for data in vol.read_file("path.txt"):
    print(data)
```

## Volume 效能

### Volumes v1

最適合：
- <50,000 個檔案（建議）
- <500,000 個檔案（硬性限制）
- 順序存取模式
- <5 個並行寫入者

### Volumes v2（測試版）

改進：
- 無限檔案數
- 數百個並行寫入者
- 隨機存取模式
- 大型檔案（最多 1 TiB）

目前 v2 限制：
- 最大檔案大小：1 TiB
- 每目錄最大檔案數：32,768
- 無限目錄深度

## 模型儲存

### 儲存模型權重

```python
volume = modal.Volume.from_name("model-weights", create_if_missing=True)
MODEL_DIR = "/models"

@app.function(volumes={MODEL_DIR: volume})
def train():
    model = train_model()
    save_model(f"{MODEL_DIR}/my_model.pt", model)
    volume.commit()
```

### 載入模型權重

```python
@app.function(volumes={MODEL_DIR: volume})
def inference(model_id: str):
    try:
        model = load_model(f"{MODEL_DIR}/{model_id}")
    except NotFound:
        volume.reload()  # 取得最新模型
        model = load_model(f"{MODEL_DIR}/{model_id}")
    return model.run(request)
```

## 模型檢查點

在長時間訓練任務期間儲存檢查點：

```python
volume = modal.Volume.from_name("checkpoints")
VOL_PATH = "/vol"

@app.function(
    gpu="A10G",
    timeout=2*60*60,  # 2 小時
    volumes={VOL_PATH: volume}
)
def finetune():
    from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(VOL_PATH / "model"),  # 檢查點儲存到 Volume
        save_steps=100,
        # ... 更多參數
    )

    trainer = Seq2SeqTrainer(model=model, args=training_args, ...)
    trainer.train()
```

背景提交確保即使訓練被中斷，檢查點也會持久化。

## CLI 命令

```bash
# 列出檔案
modal volume ls my-volume

# 上傳
modal volume put my-volume local.txt remote.txt

# 下載
modal volume get my-volume remote.txt local.txt

# 在 Volume 內複製
modal volume cp my-volume src.txt dst.txt

# 刪除
modal volume rm my-volume file.txt

# 列出所有 volumes
modal volume list

# 刪除 volume
modal volume delete my-volume
```

## 臨時 Volumes

建立會被垃圾收集的臨時 volumes：

```python
with modal.Volume.ephemeral() as vol:
    sb = modal.Sandbox.create(
        volumes={"/cache": vol},
        app=my_app,
    )
    # 使用 volume
    # 情境退出時自動清理
```

## 並行存取

### 並行讀取

多個容器可以同時讀取，沒有問題。

### 並行寫入

支援但：
- 避免同時修改相同檔案
- 最後寫入勝出（可能資料遺失）
- v1：限制約 5 個並行寫入者
- v2：支援數百個並行寫入者

## Volume 錯誤

### 「Volume Busy」

檔案開啟時無法重新載入：

```python
# 錯誤
f = open("/vol/data.txt", "r")
volume.reload()  # 錯誤：volume 忙碌
```

```python
# 正確
with open("/vol/data.txt", "r") as f:
    data = f.read()
# 重新載入前檔案已關閉
volume.reload()
```

### 「File Not Found」

記得使用掛載點：

```python
# 錯誤 - 檔案儲存到本地磁碟
with open("/xyz.txt", "w") as f:
    f.write("data")

# 正確 - 檔案儲存到 Volume
with open("/data/xyz.txt", "w") as f:
    f.write("data")
```

## 從 v1 升級到 v2

目前沒有自動遷移。手動步驟：

1. 建立新的 v2 Volume
2. 使用 `cp` 或 `rsync` 複製資料
3. 更新應用程式使用新 Volume

```bash
modal volume create --version=2 my-volume-v2
modal shell --volume my-volume --volume my-volume-v2

# 在 shell 中：
cp -rp /mnt/my-volume/. /mnt/my-volume-v2/.
sync /mnt/my-volume-v2
```

警告：已部署的應用程式透過 ID 參照 Volumes。建立新 Volume 後重新部署。
