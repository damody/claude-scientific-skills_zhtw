# CPU、記憶體和磁碟資源

## 預設資源

每個 Modal 容器有預設配置：
- **CPU**：0.125 核心
- **記憶體**：128 MiB

如果工作節點有可用資源，容器可以超出最低配置。

## CPU 核心

以浮點數請求 CPU 核心：

```python
@app.function(cpu=8.0)
def my_function():
    # 保證至少存取 8 個實體核心
    ...
```

值對應實體核心，而非 vCPU。

Modal 根據 CPU 配置設定多執行緒環境變數：
- `OPENBLAS_NUM_THREADS`
- `OMP_NUM_THREADS`
- `MKL_NUM_THREADS`

## 記憶體

以整數（兆位元組）請求記憶體：

```python
@app.function(memory=32768)
def my_function():
    # 保證至少存取 32 GiB RAM
    ...
```

## 資源限制

### CPU 限制

預設軟性 CPU 限制：請求 + 16 核心
- 預設請求：0.125 核心 → 預設限制：16.125 核心
- 超過限制時，主機會節流 CPU 使用

設定明確的 CPU 限制：

```python
cpu_request = 1.0
cpu_limit = 4.0

@app.function(cpu=(cpu_request, cpu_limit))
def f():
    ...
```

### 記憶體限制

設定硬性記憶體限制，在達到閾值時 OOM 終止容器：

```python
mem_request = 1024  # MB
mem_limit = 2048    # MB

@app.function(memory=(mem_request, mem_limit))
def f():
    # 超過 2048 MB 時容器被終止
    ...
```

用於及早發現記憶體洩漏。

### 磁碟限制

執行中的容器可存取許多 GB 的 SSD 磁碟，受以下限制：
1. 底層工作節點的 SSD 容量
2. 每容器磁碟配額（數百 GB）

達到限制會在磁碟寫入時導致 `OSError`。

使用 `ephemeral_disk` 請求更大的磁碟：

```python
@app.function(ephemeral_disk=10240)  # 10 GiB
def process_large_files():
    ...
```

最大磁碟大小：3.0 TiB（3,145,728 MiB）
預期用途：資料集處理

## 計費

根據較高者計費：保留量或實際用量。

磁碟請求以 20:1 的比率增加記憶體請求：
- 請求 500 GiB 磁碟 → 將記憶體請求增加到 25 GiB（如果還沒有更高的話）

## 最大請求

Modal 在建立函數時強制執行最大值。超過最大值的請求會被拒絕並顯示 `InvalidError`。

如果您需要更高的限制，請聯繫支援。

## 範例：資源配置

```python
@app.function(
    cpu=4.0,              # 4 個實體核心
    memory=16384,         # 16 GiB RAM
    ephemeral_disk=51200, # 50 GiB 磁碟
    timeout=3600,         # 1 小時超時
)
def process_data():
    # 處理大型檔案的繁重處理
    ...
```

## 監控資源使用

在 Modal 儀表板檢視資源使用：
- CPU 使用率
- 記憶體使用量
- 磁碟使用量
- GPU 指標（如適用）

透過 https://modal.com/apps 存取
