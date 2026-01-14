# 科學運算的常見模式

## 機器學習模型推論

### 基本模型服務

```python
import modal

app = modal.App("ml-inference")

image = (
    modal.Image.debian_slim()
    .uv_pip_install("torch", "transformers")
)

@app.cls(
    image=image,
    gpu="L40S",
)
class Model:
    @modal.enter()
    def load_model(self):
        from transformers import AutoModel, AutoTokenizer
        self.model = AutoModel.from_pretrained("bert-base-uncased")
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    @modal.method()
    def predict(self, text: str):
        inputs = self.tokenizer(text, return_tensors="pt")
        outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).tolist()

@app.local_entrypoint()
def main():
    model = Model()
    result = model.predict.remote("Hello world")
    print(result)
```

### 使用 Volume 的模型服務

```python
volume = modal.Volume.from_name("models", create_if_missing=True)
MODEL_PATH = "/models"

@app.cls(
    image=image,
    gpu="A100",
    volumes={MODEL_PATH: volume}
)
class ModelServer:
    @modal.enter()
    def load(self):
        import torch
        self.model = torch.load(f"{MODEL_PATH}/model.pt")
        self.model.eval()

    @modal.method()
    def infer(self, data):
        import torch
        with torch.no_grad():
            return self.model(torch.tensor(data)).tolist()
```

## 批次處理

### 平行資料處理

```python
@app.function(
    image=modal.Image.debian_slim().uv_pip_install("pandas", "numpy"),
    cpu=2.0,
    memory=8192
)
def process_batch(batch_id: int):
    import pandas as pd

    # 載入批次
    df = pd.read_csv(f"s3://bucket/batch_{batch_id}.csv")

    # 處理
    result = df.apply(lambda row: complex_calculation(row), axis=1)

    # 儲存結果
    result.to_csv(f"s3://bucket/results_{batch_id}.csv")

    return batch_id

@app.local_entrypoint()
def main():
    # 平行處理 100 個批次
    results = list(process_batch.map(range(100)))
    print(f"Processed {len(results)} batches")
```

### 帶進度的批次處理

```python
@app.function()
def process_item(item_id: int):
    # 昂貴的處理
    result = compute_something(item_id)
    return result

@app.local_entrypoint()
def main():
    items = list(range(1000))

    print(f"Processing {len(items)} items...")
    results = []
    for i, result in enumerate(process_item.map(items)):
        results.append(result)
        if (i + 1) % 100 == 0:
            print(f"Completed {i + 1}/{len(items)}")

    print("All items processed!")
```

## 資料分析管線

### ETL 管線

```python
volume = modal.Volume.from_name("data-pipeline")
DATA_PATH = "/data"

@app.function(
    image=modal.Image.debian_slim().uv_pip_install("pandas", "polars"),
    volumes={DATA_PATH: volume},
    cpu=4.0,
    memory=16384
)
def extract_transform_load():
    import polars as pl

    # 擷取
    raw_data = pl.read_csv(f"{DATA_PATH}/raw/*.csv")

    # 轉換
    transformed = (
        raw_data
        .filter(pl.col("value") > 0)
        .group_by("category")
        .agg([
            pl.col("value").mean().alias("avg_value"),
            pl.col("value").sum().alias("total_value")
        ])
    )

    # 載入
    transformed.write_parquet(f"{DATA_PATH}/processed/data.parquet")
    volume.commit()

    return transformed.shape

@app.function(schedule=modal.Cron("0 2 * * *"))
def daily_pipeline():
    result = extract_transform_load.remote()
    print(f"Processed data shape: {result}")
```

## GPU 加速運算

### 分散式訓練

```python
@app.function(
    gpu="A100:2",
    image=modal.Image.debian_slim().uv_pip_install("torch", "accelerate"),
    timeout=7200,
)
def train_model():
    import torch
    from torch.nn.parallel import DataParallel

    # 載入資料
    train_loader = get_data_loader()

    # 初始化模型
    model = MyModel()
    model = DataParallel(model)
    model = model.cuda()

    # 訓練
    optimizer = torch.optim.Adam(model.parameters())
    for epoch in range(10):
        for batch in train_loader:
            loss = train_step(model, batch, optimizer)
            print(f"Epoch {epoch}, Loss: {loss}")

    return "Training complete"
```

### GPU 批次推論

```python
@app.function(
    gpu="L40S",
    image=modal.Image.debian_slim().uv_pip_install("torch", "transformers")
)
def batch_inference(texts: list[str]):
    from transformers import pipeline

    classifier = pipeline("sentiment-analysis", device=0)
    results = classifier(texts, batch_size=32)

    return results

@app.local_entrypoint()
def main():
    # 處理 10,000 個文本
    texts = load_texts()

    # 分成 100 個為一組
    chunks = [texts[i:i+100] for i in range(0, len(texts), 100)]

    # 在多個 GPU 上平行處理
    all_results = []
    for results in batch_inference.map(chunks):
        all_results.extend(results)

    print(f"Processed {len(all_results)} texts")
```

## 科學運算

### 分子動力學模擬

```python
@app.function(
    image=modal.Image.debian_slim().apt_install("openmpi-bin").uv_pip_install("mpi4py", "numpy"),
    cpu=16.0,
    memory=65536,
    timeout=7200,
)
def run_simulation(config: dict):
    import numpy as np

    # 初始化系統
    positions = initialize_positions(config["n_particles"])
    velocities = initialize_velocities(config["temperature"])

    # 執行 MD 步驟
    for step in range(config["n_steps"]):
        forces = compute_forces(positions)
        velocities += forces * config["dt"]
        positions += velocities * config["dt"]

        if step % 1000 == 0:
            energy = compute_energy(positions, velocities)
            print(f"Step {step}, Energy: {energy}")

    return positions, velocities
```

### 分散式蒙地卡羅

```python
@app.function(cpu=2.0)
def monte_carlo_trial(trial_id: int, n_samples: int):
    import random

    count = sum(1 for _ in range(n_samples)
                if random.random()**2 + random.random()**2 <= 1)

    return count

@app.local_entrypoint()
def estimate_pi():
    n_trials = 100
    n_samples_per_trial = 1_000_000

    # 平行執行試驗
    results = list(monte_carlo_trial.map(
        range(n_trials),
        [n_samples_per_trial] * n_trials
    ))

    total_count = sum(results)
    total_samples = n_trials * n_samples_per_trial

    pi_estimate = 4 * total_count / total_samples
    print(f"Estimated pi = {pi_estimate}")
```

## 使用 Volumes 的資料處理

### 圖像處理管線

```python
volume = modal.Volume.from_name("images")
IMAGE_PATH = "/images"

@app.function(
    image=modal.Image.debian_slim().uv_pip_install("Pillow", "numpy"),
    volumes={IMAGE_PATH: volume}
)
def process_image(filename: str):
    from PIL import Image
    import numpy as np

    # 載入圖像
    img = Image.open(f"{IMAGE_PATH}/raw/{filename}")

    # 處理
    img_array = np.array(img)
    processed = apply_filters(img_array)

    # 儲存
    result_img = Image.fromarray(processed)
    result_img.save(f"{IMAGE_PATH}/processed/{filename}")

    return filename

@app.function(volumes={IMAGE_PATH: volume})
def process_all_images():
    import os

    # 取得所有圖像
    filenames = os.listdir(f"{IMAGE_PATH}/raw")

    # 平行處理
    results = list(process_image.map(filenames))

    volume.commit()
    return f"Processed {len(results)} images"
```

## 科學運算的 Web API

```python
image = modal.Image.debian_slim().uv_pip_install("fastapi[standard]", "numpy", "scipy")

@app.function(image=image)
@modal.fastapi_endpoint(method="POST")
def compute_statistics(data: dict):
    import numpy as np
    from scipy import stats

    values = np.array(data["values"])

    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "skewness": float(stats.skew(values)),
        "kurtosis": float(stats.kurtosis(values))
    }
```

## 排程資料收集

```python
@app.function(
    schedule=modal.Cron("*/30 * * * *"),  # 每 30 分鐘
    secrets=[modal.Secret.from_name("api-keys")],
    volumes={"/data": modal.Volume.from_name("sensor-data")}
)
def collect_sensor_data():
    import requests
    import json
    from datetime import datetime

    # 從 API 取得
    response = requests.get(
        "https://api.example.com/sensors",
        headers={"Authorization": f"Bearer {os.environ['API_KEY']}"}
    )

    data = response.json()

    # 使用時間戳儲存
    timestamp = datetime.now().isoformat()
    with open(f"/data/{timestamp}.json", "w") as f:
        json.dump(data, f)

    volume.commit()

    return f"Collected {len(data)} sensor readings"
```

## 最佳實踐

### 使用類別處理有狀態的工作負載

```python
@app.cls(gpu="A100")
class ModelService:
    @modal.enter()
    def setup(self):
        # 載入一次，跨請求重用
        self.model = load_heavy_model()

    @modal.method()
    def predict(self, x):
        return self.model(x)
```

### 批量處理相似工作負載

```python
@app.function()
def process_many(items: list):
    # 比一次處理一個更有效率
    return [process(item) for item in items]
```

### 使用 Volumes 儲存大型資料集

```python
# 在 volumes 中儲存大型資料集，而非在映像中
volume = modal.Volume.from_name("dataset")

@app.function(volumes={"/data": volume})
def train():
    data = load_from_volume("/data/training.parquet")
    model = train_model(data)
```

### 在擴展到 GPU 之前先做分析

```python
# 先在 CPU 上測試
@app.function(cpu=4.0)
def test_pipeline():
    ...

# 然後如果需要再擴展到 GPU
@app.function(gpu="A100")
def gpu_pipeline():
    ...
```
