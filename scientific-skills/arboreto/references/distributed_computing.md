# 使用 Arboreto 進行分散式運算

Arboreto 利用 Dask 進行平行化運算，能夠從單機多核心處理高效擴展到多節點叢集環境進行 GRN 推斷。

## 運算架構

GRN 推斷本質上是可平行化的：
- 每個標靶基因的迴歸模型可以獨立訓練
- Arboreto 將運算表示為 Dask 任務圖
- 任務分散到可用的運算資源上

## 本機多核心處理（預設）

預設情況下，arboreto 使用本機機器上所有可用的 CPU 核心：

```python
from arboreto.algo import grnboost2

# 自動使用所有本機核心
network = grnboost2(expression_data=expression_matrix, tf_names=tf_names)
```

這對大多數使用案例已足夠，不需要額外配置。

## 自訂本機 Dask 客戶端

如需對本機資源進行細緻控制，可建立自訂 Dask 客戶端：

```python
from distributed import LocalCluster, Client
from arboreto.algo import grnboost2

if __name__ == '__main__':
    # 配置本機叢集
    local_cluster = LocalCluster(
        n_workers=10,              # 工作行程數量
        threads_per_worker=1,       # 每個工作者的執行緒數
        memory_limit='8GB'          # 每個工作者的記憶體限制
    )

    # 建立客戶端
    custom_client = Client(local_cluster)

    # 使用自訂客戶端執行推斷
    network = grnboost2(
        expression_data=expression_matrix,
        tf_names=tf_names,
        client_or_address=custom_client
    )

    # 清理
    custom_client.close()
    local_cluster.close()
```

### 自訂客戶端的好處
- **資源控制**：限制 CPU 和記憶體使用
- **多次執行**：為不同參數集重複使用同一客戶端
- **監控**：存取 Dask 儀表板以獲得效能洞察

## 使用同一客戶端進行多次推斷

為使用不同參數的多次推斷重複使用單一 Dask 客戶端：

```python
from distributed import LocalCluster, Client
from arboreto.algo import grnboost2

if __name__ == '__main__':
    # 初始化客戶端一次
    local_cluster = LocalCluster(n_workers=8, threads_per_worker=1)
    client = Client(local_cluster)

    # 執行多次推斷
    network_seed1 = grnboost2(
        expression_data=expression_matrix,
        tf_names=tf_names,
        client_or_address=client,
        seed=666
    )

    network_seed2 = grnboost2(
        expression_data=expression_matrix,
        tf_names=tf_names,
        client_or_address=client,
        seed=777
    )

    # 使用同一客戶端執行不同演算法
    from arboreto.algo import genie3
    network_genie3 = genie3(
        expression_data=expression_matrix,
        tf_names=tf_names,
        client_or_address=client
    )

    # 清理一次
    client.close()
    local_cluster.close()
```

## 分散式叢集運算

對於非常大的資料集，連接到叢集上執行的遠端 Dask 分散式排程器：

### 步驟 1：設定 Dask 排程器（在叢集主節點上）
```bash
dask-scheduler
# 輸出：Scheduler at tcp://10.118.224.134:8786
```

### 步驟 2：啟動 Dask 工作者（在叢集計算節點上）
```bash
dask-worker tcp://10.118.224.134:8786
```

### 步驟 3：從客戶端連接
```python
from distributed import Client
from arboreto.algo import grnboost2

if __name__ == '__main__':
    # 連接到遠端排程器
    scheduler_address = 'tcp://10.118.224.134:8786'
    cluster_client = Client(scheduler_address)

    # 在叢集上執行推斷
    network = grnboost2(
        expression_data=expression_matrix,
        tf_names=tf_names,
        client_or_address=cluster_client
    )

    cluster_client.close()
```

### 叢集配置最佳實務

**工作者配置**：
```bash
dask-worker tcp://scheduler:8786 \
    --nprocs 4 \              # 每個節點的行程數
    --nthreads 1 \            # 每個行程的執行緒數
    --memory-limit 16GB       # 每個行程的記憶體
```

**大規模推斷**：
- 使用更多中等記憶體的工作者，而非較少的大記憶體工作者
- 設定 `threads_per_worker=1` 以避免 scikit-learn 中的 GIL 競爭
- 監控記憶體使用以防止工作者被終止

## 監控和除錯

### Dask 儀表板

存取 Dask 儀表板進行即時監控：

```python
from distributed import Client

client = Client()  # 印出儀表板 URL
# 儀表板可在以下位址存取：http://localhost:8787/status
```

儀表板顯示：
- **任務進度**：已完成/待處理的任務數量
- **資源使用**：每個工作者的 CPU、記憶體
- **任務串流**：運算的即時視覺化
- **效能**：瓶頸識別

### 詳細輸出

啟用詳細日誌以追蹤推斷進度：

```python
network = grnboost2(
    expression_data=expression_matrix,
    tf_names=tf_names,
    verbose=True
)
```

## 效能優化技巧

### 1. 資料格式
- **盡可能使用 Pandas DataFrame**：對 Dask 操作更有效率
- **減少資料大小**：在推斷前過濾低變異基因

### 2. 工作者配置
- **CPU 密集型任務**：設定 `threads_per_worker=1`，增加 `n_workers`
- **記憶體密集型任務**：增加每個工作者的 `memory_limit`

### 3. 叢集設定
- **網路**：確保節點間有高頻寬、低延遲的網路
- **儲存**：對大型資料集使用共享檔案系統或物件儲存
- **排程**：分配專用節點以避免資源競爭

### 4. 轉錄因子過濾
- **限制 TF 清單**：提供特定的 TF 名稱可減少運算量
```python
# 完整搜尋（較慢）
network = grnboost2(expression_data=matrix)

# 過濾搜尋（較快）
network = grnboost2(expression_data=matrix, tf_names=known_tfs)
```

## 範例：大規模單細胞分析

在叢集上處理單細胞 RNA-seq 資料的完整工作流程：

```python
from distributed import Client
from arboreto.algo import grnboost2
import pandas as pd

if __name__ == '__main__':
    # 連接到叢集
    client = Client('tcp://cluster-scheduler:8786')

    # 載入大型單細胞資料集（50,000 細胞 x 20,000 基因）
    expression_data = pd.read_csv('scrnaseq_data.tsv', sep='\t')

    # 載入細胞類型特異性 TF
    tf_names = pd.read_csv('tf_list.txt', header=None)[0].tolist()

    # 執行分散式推斷
    network = grnboost2(
        expression_data=expression_data,
        tf_names=tf_names,
        client_or_address=client,
        verbose=True,
        seed=42
    )

    # 儲存結果
    network.to_csv('grn_results.tsv', sep='\t', index=False)

    client.close()
```

這種方法能夠分析在單機上不切實際的資料集。
