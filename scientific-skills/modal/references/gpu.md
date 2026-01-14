# Modal 上的 GPU 加速

## 快速開始

使用 `gpu` 參數在 GPU 上執行函數：

```python
import modal

image = modal.Image.debian_slim().pip_install("torch")
app = modal.App(image=image)

@app.function(gpu="A100")
def run():
    import torch
    assert torch.cuda.is_available()
```

## 可用的 GPU 類型

Modal 支援以下 GPU：

- `T4` - 入門級 GPU
- `L4` - 效能和成本平衡
- `A10` - 最多 4 個 GPU，總共 96 GB
- `A100` - 40GB 或 80GB 版本
- `A100-40GB` - 特定 40GB 版本
- `A100-80GB` - 特定 80GB 版本
- `L40S` - 48 GB，非常適合推論
- `H100` / `H100!` - 頂級 Hopper 架構
- `H200` - 改進的 Hopper，更多記憶體
- `B200` - 最新的 Blackwell 架構

價格請參閱 https://modal.com/pricing。

## GPU 數量

使用 `:n` 語法為每個容器請求多個 GPU：

```python
@app.function(gpu="H100:8")
def run_llama_405b():
    # 8 個 H100 GPU 可用
    ...
```

支援的數量：
- B200、H200、H100、A100、L4、T4、L40S：最多 8 個 GPU（最多 1,536 GB）
- A10：最多 4 個 GPU（最多 96 GB）

注意：請求 >2 個 GPU 可能導致較長的等待時間。

## GPU 選擇指南

**用於推論（推薦）**：從 L40S 開始
- 優異的成本/效能比
- 48 GB 記憶體
- 適合 LLaMA、Stable Diffusion 等

**用於訓練**：考慮 H100 或 A100
- 高運算吞吐量
- 大記憶體用於批次處理

**用於記憶體受限的任務**：H200 或 A100-80GB
- 更多記憶體容量
- 更適合大型模型

## B200 GPU

NVIDIA 的旗艦 Blackwell 晶片：

```python
@app.function(gpu="B200:8")
def run_deepseek():
    # 最強大的選項
    ...
```

## H200 和 H100 GPU

具有優秀軟體支援的 Hopper 架構 GPU：

```python
@app.function(gpu="H100")
def train():
    ...
```

### 自動升級到 H200

Modal 可能會免費將 `gpu="H100"` 升級到 H200。H200 提供：
- 141 GB 記憶體（對比 H100 的 80 GB）
- 4.8 TB/s 頻寬（對比 3.35 TB/s）

要避免自動升級（例如，用於基準測試）：
```python
@app.function(gpu="H100!")
def benchmark():
    ...
```

## A100 GPU

Ampere 架構，有 40GB 或 80GB 版本：

```python
# 可能會自動升級到 80GB
@app.function(gpu="A100")
def qwen_7b():
    ...

# 特定版本
@app.function(gpu="A100-40GB")
def model_40gb():
    ...

@app.function(gpu="A100-80GB")
def llama_70b():
    ...
```

## GPU 備援

指定多個 GPU 類型作為備援：

```python
@app.function(gpu=["H100", "A100-40GB:2"])
def run_on_80gb():
    # 先嘗試 H100，備援到 2x A100-40GB
    ...
```

Modal 遵循順序並分配最優先可用的 GPU。

## 多 GPU 訓練

Modal 支援單節點多 GPU 訓練。多節點訓練處於封閉測試階段。

### PyTorch 範例

對於會重新執行進入點的框架，使用子程序或特定策略：

```python
@app.function(gpu="A100:2")
def train():
    import subprocess
    import sys
    subprocess.run(
        ["python", "train.py"],
        stdout=sys.stdout,
        stderr=sys.stderr,
        check=True,
    )
```

對於 PyTorch Lightning，將策略設定為 `ddp_spawn` 或 `ddp_notebook`。

## 效能考量

**記憶體受限 vs 運算受限**：
- 使用小批次大小執行模型是記憶體受限的
- 較新的 GPU 的運算速度比記憶體存取更快
- 對於記憶體受限的工作負載，較新硬體的加速可能不值得成本

**最佳化**：
- 盡可能使用批次處理
- 在跳到 H100/B200 之前考慮 L40S
- 進行分析以識別瓶頸
