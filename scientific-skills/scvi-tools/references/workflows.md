# 常見工作流程與最佳實踐

本文檔涵蓋 scvi-tools 的常見工作流程、最佳實踐和進階使用模式。

## 標準分析工作流程

### 1. 數據載入與準備

```python
import scvi
import scanpy as sc
import numpy as np

# 載入數據（需要 AnnData 格式）
adata = sc.read_h5ad("data.h5ad")
# 或從其他格式載入
# adata = sc.read_10x_mtx("filtered_feature_bc_matrix/")
# adata = sc.read_csv("counts.csv")

# 基本 QC 指標
sc.pp.calculate_qc_metrics(adata, inplace=True)
adata.var['mt'] = adata.var_names.str.startswith('MT-')
sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], inplace=True)
```

### 2. 品質控制

```python
# 過濾細胞
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_cells(adata, max_genes=5000)

# 過濾基因
sc.pp.filter_genes(adata, min_cells=3)

# 按粒線體含量過濾
adata = adata[adata.obs['pct_counts_mt'] < 20, :]

# 移除雙細胞（可選，在訓練前）
sc.external.pp.scrublet(adata)
adata = adata[~adata.obs['predicted_doublet'], :]
```

### 3. scvi-tools 預處理

```python
# 重要：scvi-tools 需要原始計數
# 如果已經標準化，使用原始層或重新載入數據

# 如果原始計數不可用，儲存原始計數
if 'counts' not in adata.layers:
    adata.layers['counts'] = adata.X.copy()

# 特徵選擇（可選但建議）
sc.pp.highly_variable_genes(
    adata,
    n_top_genes=4000,
    subset=False,  # 保留所有基因，只標記 HVG
    batch_key="batch"  # 如果有多個批次
)

# 過濾到 HVG（可選）
# adata = adata[:, adata.var['highly_variable']]
```

### 4. 向 scvi-tools 註冊數據

```python
# 設定 AnnData 用於 scvi-tools
scvi.model.SCVI.setup_anndata(
    adata,
    layer="counts",  # 使用原始計數
    batch_key="batch",  # 技術批次
    categorical_covariate_keys=["donor", "condition"],
    continuous_covariate_keys=["percent_mito", "n_counts"]
)

# 檢查註冊
adata.uns['_scvi']['summary_stats']
```

### 5. 模型訓練

```python
# 創建模型
model = scvi.model.SCVI(
    adata,
    n_latent=30,  # 潛在維度
    n_layers=2,   # 網路深度
    n_hidden=128, # 隱藏層大小
    dropout_rate=0.1,
    gene_likelihood="zinb"  # 零膨脹負二項分佈
)

# 訓練模型
model.train(
    max_epochs=400,
    batch_size=128,
    train_size=0.9,
    early_stopping=True,
    check_val_every_n_epoch=10
)

# 查看訓練歷史
train_history = model.history["elbo_train"]
val_history = model.history["elbo_validation"]
```

### 6. 提取結果

```python
# 獲取潛在表示
latent = model.get_latent_representation()
adata.obsm["X_scVI"] = latent

# 獲取標準化表達
normalized = model.get_normalized_expression(
    adata,
    library_size=1e4,
    n_samples=25  # 蒙特卡羅樣本
)
adata.layers["scvi_normalized"] = normalized
```

### 7. 下游分析

```python
# 在 scVI 潛在空間上聚類
sc.pp.neighbors(adata, use_rep="X_scVI", n_neighbors=15)
sc.tl.umap(adata, min_dist=0.3)
sc.tl.leiden(adata, resolution=0.8, key_added="leiden")

# 視覺化
sc.pl.umap(adata, color=["leiden", "batch", "cell_type"])

# 差異表達
de_results = model.differential_expression(
    groupby="leiden",
    group1="0",
    group2="1",
    mode="change",
    delta=0.25
)
```

### 8. 模型持久化

```python
# 儲存模型
model_dir = "./scvi_model/"
model.save(model_dir, overwrite=True)

# 儲存帶結果的 AnnData
adata.write("analyzed_data.h5ad")

# 之後載入模型
model = scvi.model.SCVI.load(model_dir, adata=adata)
```

## 超參數調優

### 關鍵超參數

**架構**：
- `n_latent`：潛在空間維度（10-50）
  - 對於複雜、異質的數據集使用較大值
  - 對於簡單數據集或防止過擬合使用較小值
- `n_layers`：隱藏層數（1-3）
  - 更多層適用於複雜數據，但收益遞減
- `n_hidden`：每個隱藏層的節點數（64-256）
  - 隨數據集大小和複雜度調整

**訓練**：
- `max_epochs`：訓練迭代次數（200-500）
  - 使用早停防止過擬合
- `batch_size`：每批樣本數（64-256）
  - 大數據集使用較大值，記憶體有限使用較小值
- `lr`：學習率（預設 0.001，通常適用）

**模型特定**：
- `gene_likelihood`：分佈（「zinb」、「nb」、「poisson」）
  - 「zinb」用於零膨脹的稀疏數據
  - 「nb」用於較不稀疏的數據
- `dispersion`：基因或基因-批次特定
  - 「gene」用於簡單情況，「gene-batch」用於複雜批次效應

### 超參數搜索範例

```python
from scvi.model import SCVI

# 定義搜索空間
latent_dims = [10, 20, 30]
n_layers_options = [1, 2]

best_score = float('-inf')
best_params = None

for n_latent in latent_dims:
    for n_layers in n_layers_options:
        model = SCVI(
            adata,
            n_latent=n_latent,
            n_layers=n_layers
        )
        model.train(max_epochs=200)

        # 在驗證集上評估
        val_elbo = model.history["elbo_validation"][-1]

        if val_elbo > best_score:
            best_score = val_elbo
            best_params = {"n_latent": n_latent, "n_layers": n_layers}

print(f"最佳參數：{best_params}")
```

### 使用 Optuna 進行超參數優化

```python
import optuna

def objective(trial):
    n_latent = trial.suggest_int("n_latent", 10, 50)
    n_layers = trial.suggest_int("n_layers", 1, 3)
    n_hidden = trial.suggest_categorical("n_hidden", [64, 128, 256])

    model = scvi.model.SCVI(
        adata,
        n_latent=n_latent,
        n_layers=n_layers,
        n_hidden=n_hidden
    )

    model.train(max_epochs=200, early_stopping=True)
    return model.history["elbo_validation"][-1]

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)

print(f"最佳參數：{study.best_params}")
```

## GPU 加速

### 啟用 GPU 訓練

```python
# 自動 GPU 檢測
model = scvi.model.SCVI(adata)
model.train(accelerator="auto")  # 如果可用則使用 GPU

# 強制使用 GPU
model.train(accelerator="gpu")

# 多 GPU
model.train(accelerator="gpu", devices=2)

# 檢查是否正在使用 GPU
import torch
print(f"CUDA 可用：{torch.cuda.is_available()}")
print(f"GPU 數量：{torch.cuda.device_count()}")
```

### GPU 記憶體管理

```python
# 如果 OOM 則減少批次大小
model.train(batch_size=64)  # 而非預設的 128

# 混合精度訓練（節省記憶體）
model.train(precision=16)

# 清除運行之間的快取
import torch
torch.cuda.empty_cache()
```

## 批次整合策略

### 策略一：簡單批次鍵

```python
# 標準批次校正
scvi.model.SCVI.setup_anndata(adata, batch_key="batch")
model = scvi.model.SCVI(adata)
```

### 策略二：多個共變量

```python
# 校正多個技術因素
scvi.model.SCVI.setup_anndata(
    adata,
    batch_key="sequencing_batch",
    categorical_covariate_keys=["donor", "tissue"],
    continuous_covariate_keys=["percent_mito"]
)
```

### 策略三：層次批次

```python
# 當批次具有層次結構時
# 例如，研究內的樣本
adata.obs["batch_hierarchy"] = (
    adata.obs["study"].astype(str) + "_" +
    adata.obs["sample"].astype(str)
)

scvi.model.SCVI.setup_anndata(adata, batch_key="batch_hierarchy")
```

## 參考映射（scArches）

### 訓練參考模型

```python
# 在參考數據集上訓練
scvi.model.SCVI.setup_anndata(ref_adata, batch_key="batch")
ref_model = scvi.model.SCVI(ref_adata)
ref_model.train()

# 儲存參考
ref_model.save("reference_model")
```

### 將查詢映射到參考

```python
# 載入參考
ref_model = scvi.model.SCVI.load("reference_model", adata=ref_adata)

# 使用相同參數設定查詢
scvi.model.SCVI.setup_anndata(query_adata, batch_key="batch")

# 遷移學習
query_model = scvi.model.SCVI.load_query_data(
    query_adata,
    "reference_model"
)

# 在查詢上微調（可選）
query_model.train(max_epochs=200)

# 獲取查詢嵌入
query_latent = query_model.get_latent_representation()

# 使用 KNN 遷移標籤
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=15)
knn.fit(ref_model.get_latent_representation(), ref_adata.obs["cell_type"])
query_adata.obs["predicted_cell_type"] = knn.predict(query_latent)
```

## 模型精簡化

減少模型大小以加快推斷：

```python
# 訓練完整模型
model = scvi.model.SCVI(adata)
model.train()

# 精簡化用於部署
minified = model.minify_adata(adata)

# 儲存精簡版本
minified.write("minified_data.h5ad")
model.save("minified_model")

# 載入並使用（快很多）
mini_model = scvi.model.SCVI.load("minified_model", adata=minified)
```

## 記憶體高效數據載入

### 使用 AnnDataLoader

```python
from scvi.data import AnnDataLoader

# 對於非常大的數據集
dataloader = AnnDataLoader(
    adata,
    batch_size=128,
    shuffle=True,
    drop_last=False
)

# 自定義訓練迴圈（進階）
for batch in dataloader:
    # 處理批次
    pass
```

### 使用 Backed AnnData

```python
# 對於太大無法放入記憶體的數據
adata = sc.read_h5ad("huge_dataset.h5ad", backed='r')

# scvi-tools 支援 backed 模式
scvi.model.SCVI.setup_anndata(adata)
model = scvi.model.SCVI(adata)
model.train()
```

## 模型解釋

### 使用 SHAP 的特徵重要性

```python
import shap

# 獲取 SHAP 值用於可解釋性
explainer = shap.DeepExplainer(model.module, background_data)
shap_values = explainer.shap_values(test_data)

# 視覺化
shap.summary_plot(shap_values, feature_names=adata.var_names)
```

### 基因相關性分析

```python
# 獲取基因-基因相關性矩陣
correlation = model.get_feature_correlation_matrix(
    adata,
    transform_batch="batch1"
)

# 視覺化前幾個相關基因
import seaborn as sns
sns.heatmap(correlation[:50, :50], cmap="coolwarm")
```

## 常見問題排除

### 問題：訓練期間出現 NaN 損失

**原因**：
- 學習率太高
- 輸入未標準化（必須使用原始計數）
- 數據品質問題

**解決方案**：
```python
# 降低學習率
model.train(lr=0.0001)

# 檢查數據
assert adata.X.min() >= 0  # 無負值
assert np.isnan(adata.X).sum() == 0  # 無 NaN

# 使用更穩定的似然函數
model = scvi.model.SCVI(adata, gene_likelihood="nb")
```

### 問題：批次校正效果差

**解決方案**：
```python
# 增加批次效應建模
model = scvi.model.SCVI(
    adata,
    encode_covariates=True,  # 在編碼器中編碼批次
    deeply_inject_covariates=False
)

# 或嘗試相反設定
model = scvi.model.SCVI(adata, deeply_inject_covariates=True)

# 使用更多潛在維度
model = scvi.model.SCVI(adata, n_latent=50)
```

### 問題：模型沒有訓練（ELBO 沒有下降）

**解決方案**：
```python
# 增加學習率
model.train(lr=0.005)

# 增加網路容量
model = scvi.model.SCVI(adata, n_hidden=256, n_layers=2)

# 訓練更久
model.train(max_epochs=500)
```

### 問題：記憶體不足（OOM）

**解決方案**：
```python
# 減少批次大小
model.train(batch_size=64)

# 使用混合精度
model.train(precision=16)

# 減少模型大小
model = scvi.model.SCVI(adata, n_latent=10, n_hidden=64)

# 使用 backed AnnData
adata = sc.read_h5ad("data.h5ad", backed='r')
```

## 效能基準測試

```python
import time

# 計時訓練
start = time.time()
model.train(max_epochs=400)
training_time = time.time() - start
print(f"訓練時間：{training_time:.2f}s")

# 計時推斷
start = time.time()
latent = model.get_latent_representation()
inference_time = time.time() - start
print(f"推斷時間：{inference_time:.2f}s")

# 記憶體使用
import psutil
import os
process = psutil.Process(os.getpid())
memory_gb = process.memory_info().rss / 1024**3
print(f"記憶體使用：{memory_gb:.2f} GB")
```

## 最佳實踐總結

1. **始終使用原始計數**：在 scvi-tools 之前不要對數標準化
2. **特徵選擇**：使用高變異基因以提高效率
3. **批次校正**：註冊所有已知的技術共變量
4. **早停**：使用驗證集防止過擬合
5. **模型儲存**：始終儲存訓練好的模型
6. **GPU 使用**：對大數據集（>10k 細胞）使用 GPU
7. **超參數調優**：從預設值開始，必要時調整
8. **驗證**：視覺化檢查批次校正（按批次著色的 UMAP）
9. **文檔**：記錄預處理步驟
10. **可重複性**：設定隨機種子（`scvi.settings.seed = 0`）
