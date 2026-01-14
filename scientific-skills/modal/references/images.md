# Modal 映像

## 概述

Modal Images 定義程式碼執行的環境 - 安裝了相依性的容器。映像是從基礎映像開始透過方法鏈建構的。

## 基礎映像

從基礎映像開始並鏈接方法：

```python
image = (
    modal.Image.debian_slim(python_version="3.13")
    .apt_install("git")
    .uv_pip_install("torch<3")
    .env({"HALT_AND_CATCH_FIRE": "0"})
    .run_commands("git clone https://github.com/modal-labs/agi")
)
```

可用的基礎映像：
- `Image.debian_slim()` - 帶 Python 的 Debian Linux
- `Image.micromamba()` - 帶 Micromamba 套件管理器的基礎
- `Image.from_registry()` - 從 Docker Hub、ECR 等拉取
- `Image.from_dockerfile()` - 從現有 Dockerfile 建構

## 安裝 Python 套件

### 使用 uv（推薦）

使用 `.uv_pip_install()` 快速安裝套件：

```python
image = (
    modal.Image.debian_slim()
    .uv_pip_install("pandas==2.2.0", "numpy")
)
```

### 使用 pip

如果需要可以退回到標準 pip：

```python
image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install("pandas==2.2.0", "numpy")
)
```

為了可重現性，嚴格固定相依性版本（例如 `"torch==2.8.0"`）。

## 安裝系統套件

使用 apt 安裝 Linux 套件：

```python
image = modal.Image.debian_slim().apt_install("git", "curl")
```

## 設定環境變數

傳遞字典給 `.env()`：

```python
image = modal.Image.debian_slim().env({"PORT": "6443"})
```

## 執行 Shell 命令

在映像建構期間執行命令：

```python
image = (
    modal.Image.debian_slim()
    .apt_install("git")
    .run_commands("git clone https://github.com/modal-labs/gpu-glossary")
)
```

## 在建構時執行 Python 函數

下載模型權重或執行設定：

```python
def download_models():
    import diffusers
    model_name = "segmind/small-sd"
    pipe = diffusers.StableDiffusionPipeline.from_pretrained(model_name)

hf_cache = modal.Volume.from_name("hf-cache")

image = (
    modal.Image.debian_slim()
    .pip_install("diffusers[torch]", "transformers")
    .run_function(
        download_models,
        secrets=[modal.Secret.from_name("huggingface-secret")],
        volumes={"/root/.cache/huggingface": hf_cache},
    )
)
```

## 加入本地檔案

### 加入檔案或目錄

```python
image = modal.Image.debian_slim().add_local_dir(
    "/user/erikbern/.aws",
    remote_path="/root/.aws"
)
```

預設情況下，檔案在容器啟動時加入。使用 `copy=True` 將其包含在建構的映像中。

### 加入 Python 原始碼

加入可匯入的 Python 模組：

```python
image = modal.Image.debian_slim().add_local_python_source("local_module")

@app.function(image=image)
def f():
    import local_module
    local_module.do_stuff()
```

## 使用現有的容器映像

### 從公開 Registry

```python
sklearn_image = modal.Image.from_registry("huanjason/scikit-learn")

@app.function(image=sklearn_image)
def fit_knn():
    from sklearn.neighbors import KNeighborsClassifier
    ...
```

可以從 Docker Hub、Nvidia NGC、AWS ECR、GitHub ghcr.io 拉取。

### 從私有 Registry

使用 Modal Secrets 進行驗證：

**Docker Hub**：
```python
secret = modal.Secret.from_name("my-docker-secret")
image = modal.Image.from_registry(
    "private-repo/image:tag",
    secret=secret
)
```

**AWS ECR**：
```python
aws_secret = modal.Secret.from_name("my-aws-secret")
image = modal.Image.from_aws_ecr(
    "000000000000.dkr.ecr.us-east-1.amazonaws.com/my-private-registry:latest",
    secret=aws_secret,
)
```

### 從 Dockerfile

```python
image = modal.Image.from_dockerfile("Dockerfile")

@app.function(image=image)
def fit():
    import sklearn
    ...
```

匯入後仍可使用其他映像方法進行擴展。

## 使用 Micromamba

用於協調安裝 Python 和系統套件：

```python
numpyro_pymc_image = (
    modal.Image.micromamba()
    .micromamba_install("pymc==5.10.4", "numpyro==0.13.2", channels=["conda-forge"])
)
```

## 建構時的 GPU 支援

在 GPU 實例上執行建構步驟：

```python
image = (
    modal.Image.debian_slim()
    .pip_install("bitsandbytes", gpu="H100")
)
```

## 映像快取

映像按層快取。在某一層破壞快取會導致後續層級的連鎖重建。

將經常變更的層定義在最後以最大化快取重用。

### 強制重建

```python
image = (
    modal.Image.debian_slim()
    .apt_install("git")
    .pip_install("slack-sdk", force_build=True)
)
```

或設定環境變數：
```bash
MODAL_FORCE_BUILD=1 modal run ...
```

## 處理不同的本地/遠端套件

只在遠端可用的套件應在函數本體內匯入：

```python
@app.function(image=image)
def my_function():
    import pandas as pd  # 只在遠端匯入
    df = pd.DataFrame()
    ...
```

或使用 imports 情境管理器：

```python
pandas_image = modal.Image.debian_slim().pip_install("pandas")

with pandas_image.imports():
    import pandas as pd

@app.function(image=pandas_image)
def my_function():
    df = pd.DataFrame()
```

## 使用 eStargz 從 Registry 快速拉取

使用 eStargz 壓縮改善拉取效能：

```bash
docker buildx build --tag "<registry>/<namespace>/<repo>:<version>" \
  --output type=registry,compression=estargz,force-compression=true,oci-mediatypes=true \
  .
```

支援的 Registry：
- AWS ECR
- Docker Hub
- Google Artifact Registry
