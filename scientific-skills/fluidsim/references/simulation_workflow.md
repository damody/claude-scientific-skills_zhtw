# 模擬工作流程

## 標準工作流程

按照以下步驟執行 fluidsim 模擬：

### 1. 匯入求解器

```python
from fluidsim.solvers.ns2d.solver import Simul

# 或使用動態匯入
import fluidsim
Simul = fluidsim.import_simul_class_from_key("ns2d")
```

### 2. 建立預設參數

```python
params = Simul.create_default_params()
```

這會回傳一個包含所有模擬設定的層次結構 `Parameters` 物件。

### 3. 設定參數

根據需要修改參數。Parameters 物件透過對不存在的參數引發 `AttributeError` 來防止拼寫錯誤：

```python
# 域和解析度
params.oper.nx = 256  # x 方向的網格點
params.oper.ny = 256  # y 方向的網格點
params.oper.Lx = 2 * pi  # x 方向的域大小
params.oper.Ly = 2 * pi  # y 方向的域大小

# 物理參數
params.nu_2 = 1e-3  # 黏度（負拉普拉斯）

# 時間步進
params.time_stepping.t_end = 10.0  # 結束時間
params.time_stepping.deltat0 = 0.01  # 初始時間步長
params.time_stepping.USE_CFL = True  # 自適應時間步長

# 初始條件
params.init_fields.type = "noise"  # 或 "dipole"、"vortex" 等

# 輸出設定
params.output.periods_save.phys_fields = 1.0  # 每 1.0 時間單位儲存
params.output.periods_save.spectra = 0.5
params.output.periods_save.spatial_means = 0.1
```

### 4. 實例化模擬

```python
sim = Simul(params)
```

這會初始化：
- 運算子（FFT、微分運算子）
- 狀態變數（速度、渦度等）
- 輸出處理器
- 時間步進格式

### 5. 執行模擬

```python
sim.time_stepping.start()
```

模擬執行直到 `t_end` 或指定的迭代次數。

### 6. 在模擬期間/之後分析結果

```python
# 繪製物理場
sim.output.phys_fields.plot()
sim.output.phys_fields.plot("vorticity")
sim.output.phys_fields.plot("div")

# 繪製空間平均
sim.output.spatial_means.plot()

# 繪製頻譜
sim.output.spectra.plot1d()
sim.output.spectra.plot2d()
```

## 載入先前的模擬

### 快速載入（僅用於繪圖）

```python
from fluidsim import load_sim_for_plot

sim = load_sim_for_plot("path/to/simulation")
sim.output.phys_fields.plot()
sim.output.spatial_means.plot()
```

不進行完整狀態初始化的快速載入。用於後處理。

### 完整狀態載入（用於重新啟動）

```python
from fluidsim import load_state_phys_file

sim = load_state_phys_file("path/to/state_file.h5")
sim.time_stepping.start()  # 繼續模擬
```

載入完整狀態以繼續模擬。

## 重新啟動模擬

從已儲存的狀態重新啟動：

```python
params = Simul.create_default_params()
params.init_fields.type = "from_file"
params.init_fields.from_file.path = "path/to/state_file.h5"

# 可選擇修改延續的參數
params.time_stepping.t_end = 20.0  # 延長模擬

sim = Simul(params)
sim.time_stepping.start()
```

## 在叢集上執行

FluidSim 與叢集提交系統整合：

```python
from fluiddyn.clusters.legi import Calcul8 as Cluster

# 設定叢集工作
cluster = Cluster()
cluster.submit_script(
    "my_simulation.py",
    name_run="my_job",
    nb_nodes=4,
    nb_cores_per_node=24,
    walltime="24:00:00"
)
```

腳本應包含標準工作流程步驟（匯入、設定、執行）。

## 完整範例

```python
from fluidsim.solvers.ns2d.solver import Simul
from math import pi

# 建立和設定參數
params = Simul.create_default_params()
params.oper.nx = params.oper.ny = 256
params.oper.Lx = params.oper.Ly = 2 * pi
params.nu_2 = 1e-3
params.time_stepping.t_end = 10.0
params.init_fields.type = "dipole"
params.output.periods_save.phys_fields = 1.0

# 執行模擬
sim = Simul(params)
sim.time_stepping.start()

# 分析結果
sim.output.phys_fields.plot("vorticity")
sim.output.spatial_means.plot()
```
