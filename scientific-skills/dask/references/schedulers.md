# Dask 排程器

## 概述

Dask 提供多種任務排程器，各適用於不同的工作負載。排程器決定任務如何執行：順序、平行執行緒、平行程序或分散到叢集。

## 排程器類型

### 單機排程器

#### 1. 本地執行緒（預設）

**描述**：執行緒排程器使用本地 `concurrent.futures.ThreadPoolExecutor` 執行運算。

**使用時機**：
- NumPy、Pandas、scikit-learn 中的數值運算
- 釋放 GIL（全域直譯器鎖）的函式庫
- 受益於共享記憶體存取的操作
- Dask Arrays 和 DataFrames 的預設

**特性**：
- 低開銷
- 執行緒間共享記憶體
- 最適合釋放 GIL 的操作
- 對純 Python 程式碼效果差（GIL 競爭）

**範例**：
```python
import dask.array as da

# 預設使用執行緒
x = da.random.random((10000, 10000), chunks=(1000, 1000))
result = x.mean().compute()  # 使用執行緒計算
```

**明確配置**：
```python
import dask

# 全域設定
dask.config.set(scheduler='threads')

# 或每次 compute 設定
result = x.mean().compute(scheduler='threads')
```

#### 2. 本地程序

**描述**：使用 `concurrent.futures.ProcessPoolExecutor` 的多程序排程器。

**使用時機**：
- 具有 GIL 競爭的純 Python 程式碼
- 文字處理和 Python 集合
- 受益於程序隔離的操作
- CPU 密集型 Python 程式碼

**特性**：
- 繞過 GIL 限制
- 程序間資料傳輸有成本
- 比執行緒開銷更高
- 適合具有小輸入/輸出的線性工作流程

**範例**：
```python
import dask.bag as db

# 適合 Python 物件處理
bag = db.read_text('data/*.txt')
result = bag.map(complex_python_function).compute(scheduler='processes')
```

**明確配置**：
```python
import dask

# 全域設定
dask.config.set(scheduler='processes')

# 或每次 compute 設定
result = computation.compute(scheduler='processes')
```

**限制**：
- 資料必須可序列化（pickle）
- 程序建立的開銷
- 資料複製的記憶體開銷

#### 3. 單執行緒（同步）

**描述**：單執行緒同步排程器在本地執行緒中執行所有運算，完全沒有平行化。

**使用時機**：
- 使用 pdb 除錯
- 使用標準 Python 工具進行效能分析
- 詳細理解錯誤
- 開發和測試

**特性**：
- 無平行化
- 易於除錯
- 無開銷
- 確定性執行

**範例**：
```python
import dask

# 啟用以進行除錯
dask.config.set(scheduler='synchronous')

# 現在可以使用 pdb
result = computation.compute()  # 在單執行緒中執行
```

**在 IPython 中除錯**：
```python
# 在 IPython/Jupyter 中
%pdb on

dask.config.set(scheduler='synchronous')
result = problematic_computation.compute()  # 錯誤時進入除錯器
```

### 分散式排程器

#### 4. 本地分散式

**描述**：儘管名稱如此，這個排程器使用分散式排程器基礎架構有效地在個人機器上執行。

**使用時機**：
- 需要診斷儀表板
- 非同步 API
- 比多程序更好的資料局部性處理
- 擴展到叢集前的開發
- 在單機上需要分散式功能

**特性**：
- 提供監控儀表板
- 更好的記憶體管理
- 比執行緒/程序更高的開銷
- 之後可以擴展到叢集

**範例**：
```python
from dask.distributed import Client
import dask.dataframe as dd

# 建立本地叢集
client = Client()  # 自動使用所有核心

# 使用分散式排程器
ddf = dd.read_csv('data.csv')
result = ddf.groupby('category').mean().compute()

# 查看儀表板
print(client.dashboard_link)

# 清理
client.close()
```

**配置選項**：
```python
# 控制資源
client = Client(
    n_workers=4,
    threads_per_worker=2,
    memory_limit='4GB'
)
```

#### 5. 叢集分散式

**描述**：用於使用分散式排程器跨多台機器擴展。

**使用時機**：
- 資料超過單機容量
- 需要超過一台機器的運算能力
- 生產部署
- 叢集運算環境（HPC、雲端）

**特性**：
- 擴展到數百台機器
- 需要叢集設定
- 網路通訊開銷
- 進階功能（自適應擴展、任務優先順序）

**使用 Dask-Jobqueue（HPC）的範例**：
```python
from dask_jobqueue import SLURMCluster
from dask.distributed import Client

# 在 HPC 上使用 SLURM 建立叢集
cluster = SLURMCluster(
    cores=24,
    memory='100GB',
    walltime='02:00:00',
    queue='regular'
)

# 擴展到 10 個作業
cluster.scale(jobs=10)

# 連接 client
client = Client(cluster)

# 執行運算
result = computation.compute()

client.close()
```

**Dask on Kubernetes 的範例**：
```python
from dask_kubernetes import KubeCluster
from dask.distributed import Client

cluster = KubeCluster()
cluster.scale(20)  # 20 個 workers

client = Client(cluster)
result = computation.compute()

client.close()
```

## 排程器配置

### 全域配置

```python
import dask

# 為會話全域設定排程器
dask.config.set(scheduler='threads')
dask.config.set(scheduler='processes')
dask.config.set(scheduler='synchronous')
```

### Context Manager

```python
import dask

# 暫時使用不同的排程器
with dask.config.set(scheduler='processes'):
    result = computation.compute()

# 回到預設排程器
result2 = computation2.compute()
```

### 每次 Compute

```python
# 每次 compute 呼叫指定排程器
result = computation.compute(scheduler='threads')
result = computation.compute(scheduler='processes')
result = computation.compute(scheduler='synchronous')
```

### 分散式 Client

```python
from dask.distributed import Client

# 使用 client 自動設定分散式排程器
client = Client()

# 所有運算使用分散式排程器
result = computation.compute()

client.close()
```

## 選擇正確的排程器

### 決策矩陣

| 工作負載類型 | 建議排程器 | 理由 |
|--------------|----------------------|-----------|
| NumPy/Pandas 操作 | 執行緒（預設） | 釋放 GIL，共享記憶體 |
| 純 Python 物件 | 程序 | 避免 GIL 競爭 |
| 文字/日誌處理 | 程序 | Python 密集型操作 |
| 除錯 | 同步 | 易於除錯，確定性 |
| 需要儀表板 | 本地分散式 | 監控和診斷 |
| 多機器 | 叢集分散式 | 超過單機容量 |
| 小資料，快速任務 | 執行緒 | 最低開銷 |
| 大資料，單機 | 本地分散式 | 更好的記憶體管理 |

### 效能考量

**執行緒**：
- 開銷：每個任務約 10 µs
- 最適合：數值操作
- 記憶體：共享
- GIL：受 GIL 影響

**程序**：
- 開銷：每個任務約 10 ms
- 最適合：Python 操作
- 記憶體：程序間複製
- GIL：不受影響

**同步**：
- 開銷：每個任務約 1 µs
- 最適合：除錯
- 記憶體：無平行化
- GIL：不相關

**分散式**：
- 開銷：每個任務約 1 ms
- 最適合：複雜工作流程、監控
- 記憶體：由排程器管理
- GIL：Workers 可以使用執行緒或程序

## 分散式排程器的執行緒配置

### 設定執行緒數量

```python
from dask.distributed import Client

# 控制執行緒/worker 配置
client = Client(
    n_workers=4,           # worker 程序數量
    threads_per_worker=2   # 每個 worker 程序的執行緒數
)
```

### 建議配置

**用於數值工作負載**：
- 目標約每個程序 4 個執行緒
- 平行性和開銷之間取得平衡
- 範例：8 核心 → 2 個 workers，每個 4 個執行緒

**用於 Python 工作負載**：
- 使用更多 workers，更少執行緒
- 範例：8 核心 → 8 個 workers，每個 1 個執行緒

### 環境變數

```bash
# 透過環境設定執行緒數量
export DASK_NUM_WORKERS=4
export DASK_THREADS_PER_WORKER=2

# 或透過設定檔
```

## 常見模式

### 從開發到生產

```python
# 開發：使用本地分散式進行測試
from dask.distributed import Client
client = Client(processes=False)  # 程序內以便除錯

# 生產：擴展到叢集
from dask.distributed import Client
client = Client('scheduler-address:8786')
```

### 混合工作負載

```python
import dask
import dask.dataframe as dd

# 對 DataFrame 操作使用執行緒
ddf = dd.read_parquet('data.parquet')
result1 = ddf.mean().compute(scheduler='threads')

# 對 Python 程式碼使用程序
import dask.bag as db
bag = db.read_text('logs/*.txt')
result2 = bag.map(parse_log).compute(scheduler='processes')
```

### 除錯工作流程

```python
import dask

# 步驟 1：使用同步排程器除錯
dask.config.set(scheduler='synchronous')
result = problematic_computation.compute()

# 步驟 2：使用執行緒測試
dask.config.set(scheduler='threads')
result = computation.compute()

# 步驟 3：使用分散式擴展
from dask.distributed import Client
client = Client()
result = computation.compute()
```

## 監控和診斷

### 儀表板存取（僅分散式）

```python
from dask.distributed import Client

client = Client()

# 取得儀表板 URL
print(client.dashboard_link)
# 在瀏覽器中開啟儀表板，顯示：
# - 任務進度
# - Worker 狀態
# - 記憶體使用
# - 任務串流
# - 資源使用率
```

### 效能分析

```python
# 分析運算
from dask.distributed import Client

client = Client()
result = computation.compute()

# 取得效能報告
client.profile(filename='profile.html')
```

### 資源監控

```python
# 檢查 worker 資訊
client.scheduler_info()

# 取得當前任務
client.who_has()

# 記憶體使用
client.run(lambda: psutil.virtual_memory().percent)
```

## 進階配置

### 自訂 Executors

```python
from concurrent.futures import ThreadPoolExecutor
import dask

# 使用自訂執行緒池
with ThreadPoolExecutor(max_workers=4) as executor:
    dask.config.set(pool=executor)
    result = computation.compute(scheduler='threads')
```

### 自適應擴展（分散式）

```python
from dask.distributed import Client

client = Client()

# 啟用自適應擴展
client.cluster.adapt(minimum=2, maximum=10)

# 叢集根據工作負載擴展
result = computation.compute()
```

### Worker 外掛

```python
from dask.distributed import Client, WorkerPlugin

class CustomPlugin(WorkerPlugin):
    def setup(self, worker):
        # 初始化 worker 特定資源
        worker.custom_resource = initialize_resource()

client = Client()
client.register_worker_plugin(CustomPlugin())
```

## 故障排除

### 執行緒效能緩慢
**問題**：純 Python 程式碼在執行緒排程器下很慢
**解決方案**：切換到程序或分散式排程器

### 程序記憶體錯誤
**問題**：資料太大無法在程序間 pickle/複製
**解決方案**：使用執行緒或分散式排程器

### 難以除錯
**問題**：無法在平行排程器中使用 pdb
**解決方案**：使用同步排程器進行除錯

### 任務開銷過高
**問題**：許多微小任務導致開銷
**解決方案**：使用執行緒排程器（最低開銷）或增加分塊大小
