# 進階功能

## 自訂強制

### 強制類型

FluidSim 支援多種強制機制，用於維持紊流或驅動特定動力學。

#### 時間相關隨機強制

最常用於持續紊流：

```python
params.forcing.enable = True
params.forcing.type = "tcrandom"
params.forcing.nkmin_forcing = 2  # 最小強制波數
params.forcing.nkmax_forcing = 5  # 最大強制波數
params.forcing.forcing_rate = 1.0  # 能量注入率
params.forcing.tcrandom_time_correlation = 1.0  # 相關時間
```

#### 比例強制

維持特定能量分布：

```python
params.forcing.type = "proportional"
params.forcing.forcing_rate = 1.0
```

#### 腳本中的自訂強制

在啟動腳本中直接定義強制：

```python
params.forcing.enable = True
params.forcing.type = "in_script"

sim = Simul(params)

# 定義自訂強制函式
def compute_forcing_fft(sim):
    """在傅立葉空間計算強制"""
    forcing_vx_fft = sim.oper.create_arrayK(value=0.)
    forcing_vy_fft = sim.oper.create_arrayK(value=0.)

    # 添加自訂強制邏輯
    # 範例：強制特定模態
    forcing_vx_fft[10, 10] = 1.0 + 0.5j

    return forcing_vx_fft, forcing_vy_fft

# 覆蓋強制方法
sim.forcing.forcing_maker.compute_forcing_fft = lambda: compute_forcing_fft(sim)

# 執行模擬
sim.time_stepping.start()
```

## 自訂初始條件

### 腳本內初始化

完全控制初始場：

```python
from math import pi
import numpy as np

params = Simul.create_default_params()
params.oper.nx = params.oper.ny = 256
params.oper.Lx = params.oper.Ly = 2 * pi

params.init_fields.type = "in_script"

sim = Simul(params)

# 取得座標陣列
X, Y = sim.oper.get_XY_loc()

# 定義速度場
vx = sim.state.state_phys.get_var("vx")
vy = sim.state.state_phys.get_var("vy")

# Taylor-Green 渦流
vx[:] = np.sin(X) * np.cos(Y)
vy[:] = -np.cos(X) * np.sin(Y)

# 在傅立葉空間初始化狀態
sim.state.statephys_from_statespect()

# 執行模擬
sim.time_stepping.start()
```

### 層初始化（分層流）

設定密度層：

```python
from fluidsim.solvers.ns2d.strat.solver import Simul

params = Simul.create_default_params()
params.N = 1.0  # 分層
params.init_fields.type = "in_script"

sim = Simul(params)

# 定義密度層
X, Y = sim.oper.get_XY_loc()
b = sim.state.state_phys.get_var("b")  # 浮力場

# 高斯密度異常
x0, y0 = pi, pi
sigma = 0.5
b[:] = np.exp(-((X - x0)**2 + (Y - y0)**2) / (2 * sigma**2))

sim.state.statephys_from_statespect()
sim.time_stepping.start()
```

## 使用 MPI 進行平行計算

### 執行 MPI 模擬

安裝 MPI 支援：
```bash
uv pip install "fluidsim[fft,mpi]"
```

使用 MPI 執行：
```bash
mpirun -np 8 python simulation_script.py
```

FluidSim 自動偵測 MPI 並分配計算。

### MPI 特定參數

```python
# 不需要特殊參數
# FluidSim 自動處理域分解

# 對於非常大的 3D 模擬
params.oper.nx = 512
params.oper.ny = 512
params.oper.nz = 512

# 執行：mpirun -np 64 python script.py
```

### MPI 輸出

輸出檔案由排名 0 的處理器寫入。分析腳本對於序列和 MPI 執行的運作方式相同。

## 參數研究

### 執行多個模擬

生成和執行多個參數組合的腳本：

```python
from fluidsim.solvers.ns2d.solver import Simul
import numpy as np

# 參數範圍
viscosities = [1e-3, 5e-4, 1e-4, 5e-5]
resolutions = [128, 256, 512]

for nu in viscosities:
    for nx in resolutions:
        params = Simul.create_default_params()

        # 設定模擬
        params.oper.nx = params.oper.ny = nx
        params.nu_2 = nu
        params.time_stepping.t_end = 10.0

        # 唯一輸出目錄
        params.output.sub_directory = f"nu{nu}_nx{nx}"

        # 執行模擬
        sim = Simul(params)
        sim.time_stepping.start()
```

### 叢集提交

提交多個工作到叢集：

```python
from fluiddyn.clusters.legi import Calcul8 as Cluster

cluster = Cluster()

for nu in viscosities:
    for nx in resolutions:
        script_content = f"""
from fluidsim.solvers.ns2d.solver import Simul

params = Simul.create_default_params()
params.oper.nx = params.oper.ny = {nx}
params.nu_2 = {nu}
params.time_stepping.t_end = 10.0
params.output.sub_directory = "nu{nu}_nx{nx}"

sim = Simul(params)
sim.time_stepping.start()
"""

        with open(f"job_nu{nu}_nx{nx}.py", "w") as f:
            f.write(script_content)

        cluster.submit_script(
            f"job_nu{nu}_nx{nx}.py",
            name_run=f"sim_nu{nu}_nx{nx}",
            nb_nodes=1,
            nb_cores_per_node=24,
            walltime="12:00:00"
        )
```

### 分析參數研究

```python
import os
import pandas as pd
from fluidsim import load_sim_for_plot
import matplotlib.pyplot as plt

results = []

# 從所有模擬收集資料
for sim_dir in os.listdir("simulations"):
    sim_path = f"simulations/{sim_dir}"
    if not os.path.isdir(sim_path):
        continue

    try:
        sim = load_sim_for_plot(sim_path)

        # 提取參數
        nu = sim.params.nu_2
        nx = sim.params.oper.nx

        # 提取結果
        df = sim.output.spatial_means.load()
        final_energy = df["E"].iloc[-1]
        mean_energy = df["E"].mean()

        results.append({
            "nu": nu,
            "nx": nx,
            "final_energy": final_energy,
            "mean_energy": mean_energy
        })
    except Exception as e:
        print(f"載入 {sim_dir} 時發生錯誤：{e}")

# 分析結果
results_df = pd.DataFrame(results)

# 繪製結果
plt.figure(figsize=(10, 6))
for nx in results_df["nx"].unique():
    subset = results_df[results_df["nx"] == nx]
    plt.plot(subset["nu"], subset["mean_energy"],
             marker="o", label=f"nx={nx}")

plt.xlabel("黏度")
plt.ylabel("平均能量")
plt.xscale("log")
plt.legend()
plt.savefig("parametric_study_results.png")
```

## 自訂求解器

### 擴展現有求解器

透過繼承現有求解器建立新求解器：

```python
from fluidsim.solvers.ns2d.solver import Simul as SimulNS2D
from fluidsim.base.setofvariables import SetOfVariables

class SimulCustom(SimulNS2D):
    """具有額外物理的自訂求解器"""

    @staticmethod
    def _complete_params_with_default(params):
        """添加自訂參數"""
        SimulNS2D._complete_params_with_default(params)
        params._set_child("custom", {"param1": 0.0})

    def __init__(self, params):
        super().__init__(params)
        # 自訂初始化

    def tendencies_nonlin(self, state_spect=None):
        """覆蓋以添加自訂趨勢"""
        tendencies = super().tendencies_nonlin(state_spect)

        # 添加自訂項
        # tendencies.vx_fft += custom_term_vx
        # tendencies.vy_fft += custom_term_vy

        return tendencies
```

使用自訂求解器：
```python
params = SimulCustom.create_default_params()
# 設定參數...
sim = SimulCustom(params)
sim.time_stepping.start()
```

## 線上視覺化

在模擬期間顯示場：

```python
params.output.ONLINE_PLOT_OK = True
params.output.periods_plot.phys_fields = 1.0  # 每 1.0 時間單位繪製
params.output.phys_fields.field_to_plot = "vorticity"

sim = Simul(params)
sim.time_stepping.start()
```

圖表在執行期間即時顯示。

## 檢查點和重新啟動

### 自動檢查點

```python
params.output.periods_save.phys_fields = 1.0  # 每 1.0 時間單位儲存
```

場在模擬期間自動儲存。

### 手動檢查點

```python
# 在模擬期間
sim.output.phys_fields.save()
```

### 從檢查點重新啟動

```python
params = Simul.create_default_params()
params.init_fields.type = "from_file"
params.init_fields.from_file.path = "simulation_dir/state_phys_t5.000.h5"
params.time_stepping.t_end = 20.0  # 延長模擬

sim = Simul(params)
sim.time_stepping.start()
```

## 記憶體和效能最佳化

### 減少記憶體使用

```python
# 停用不必要的輸出
params.output.periods_save.spectra = 0  # 停用頻譜儲存
params.output.periods_save.spect_energy_budg = 0  # 停用能量預算

# 減少空間場儲存
params.output.periods_save.phys_fields = 10.0  # 較少頻率儲存
```

### 最佳化 FFT 效能

```python
import os

# 選擇 FFT 函式庫
os.environ["FLUIDSIM_TYPE_FFT2D"] = "fft2d.with_fftw"
os.environ["FLUIDSIM_TYPE_FFT3D"] = "fft3d.with_fftw"

# 或者如果可用，使用 MKL
# os.environ["FLUIDSIM_TYPE_FFT2D"] = "fft2d.with_mkl"
```

### 時間步長最佳化

```python
# 使用自適應時間步進
params.time_stepping.USE_CFL = True
params.time_stepping.CFL = 0.8  # 稍大的 CFL 以加快執行

# 使用高效時間格式
params.time_stepping.type_time_scheme = "RK4"  # 四階 Runge-Kutta
```
