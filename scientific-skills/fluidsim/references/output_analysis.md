# 輸出與分析

## 輸出類型

FluidSim 在模擬期間自動儲存多種類型的輸出。

### 物理場

**檔案格式**：HDF5（`.h5`）

**位置**：`simulation_dir/state_phys_t*.h5`

**內容**：特定時間的速度、渦度和其他物理空間場

**存取**：
```python
sim.output.phys_fields.plot()
sim.output.phys_fields.plot("vorticity")
sim.output.phys_fields.plot("vx")
sim.output.phys_fields.plot("div")  # 檢查散度

# 手動儲存
sim.output.phys_fields.save()

# 取得資料
vorticity = sim.state.state_phys.get_var("rot")
```

### 空間平均

**檔案格式**：文字檔（`.txt`）

**位置**：`simulation_dir/spatial_means.txt`

**內容**：體積平均量隨時間的變化（能量、渦度等）

**存取**：
```python
sim.output.spatial_means.plot()

# 從檔案載入
from fluidsim import load_sim_for_plot
sim = load_sim_for_plot("simulation_dir")
sim.output.spatial_means.load()
spatial_means_data = sim.output.spatial_means
```

### 頻譜

**檔案格式**：HDF5（`.h5`）

**位置**：`simulation_dir/spectra_*.h5`

**內容**：能量和渦度頻譜隨波數的變化

**存取**：
```python
sim.output.spectra.plot1d()  # 1D 頻譜
sim.output.spectra.plot2d()  # 2D 頻譜

# 載入頻譜資料
spectra = sim.output.spectra.load2d_mean()
```

### 頻譜能量預算

**檔案格式**：HDF5（`.h5`）

**位置**：`simulation_dir/spect_energy_budg_*.h5`

**內容**：尺度間的能量傳遞

**存取**：
```python
sim.output.spect_energy_budg.plot()
```

## 後處理

### 載入模擬進行分析

#### 快速載入（僅讀取）

```python
from fluidsim import load_sim_for_plot

sim = load_sim_for_plot("simulation_dir")

# 存取所有輸出類型
sim.output.phys_fields.plot()
sim.output.spatial_means.plot()
sim.output.spectra.plot1d()
```

用於快速視覺化和分析。不會初始化完整的模擬狀態。

#### 完整狀態載入

```python
from fluidsim import load_state_phys_file

sim = load_state_phys_file("simulation_dir/state_phys_t10.000.h5")

# 可以繼續模擬
sim.time_stepping.start()
```

### 視覺化工具

#### 內建繪圖

FluidSim 透過 matplotlib 提供基本繪圖：

```python
# 物理場
sim.output.phys_fields.plot("vorticity")
sim.output.phys_fields.animate("vorticity")

# 時間序列
sim.output.spatial_means.plot()

# 頻譜
sim.output.spectra.plot1d()
```

#### 進階視覺化

用於發表品質或 3D 視覺化：

**ParaView**：直接開啟 `.h5` 檔案
```bash
paraview simulation_dir/state_phys_t*.h5
```

**VisIt**：類似 ParaView，適用於大型資料集

**自訂 Python**：
```python
import h5py
import matplotlib.pyplot as plt

# 手動載入場
with h5py.File("state_phys_t10.000.h5", "r") as f:
    vx = f["state_phys"]["vx"][:]
    vy = f["state_phys"]["vy"][:]

# 自訂繪圖
plt.contourf(vx)
plt.show()
```

## 分析範例

### 能量演化

```python
from fluidsim import load_sim_for_plot
import matplotlib.pyplot as plt

sim = load_sim_for_plot("simulation_dir")
df = sim.output.spatial_means.load()

plt.figure()
plt.plot(df["t"], df["E"], label="動能")
plt.xlabel("時間")
plt.ylabel("能量")
plt.legend()
plt.show()
```

### 頻譜分析

```python
sim = load_sim_for_plot("simulation_dir")

# 繪製能量頻譜
sim.output.spectra.plot1d(tmin=5.0, tmax=10.0)  # 在時間範圍內平均

# 取得頻譜資料
k, E_k = sim.output.spectra.load1d_mean(tmin=5.0, tmax=10.0)

# 檢查冪律
import numpy as np
log_k = np.log(k)
log_E = np.log(E_k)
# 在慣性區域擬合冪律
```

### 參數研究分析

當使用不同參數執行多個模擬時：

```python
import os
import pandas as pd
from fluidsim import load_sim_for_plot

# 從多個模擬收集結果
results = []
for sim_dir in os.listdir("simulations"):
    if not os.path.isdir(f"simulations/{sim_dir}"):
        continue

    sim = load_sim_for_plot(f"simulations/{sim_dir}")

    # 提取關鍵指標
    df = sim.output.spatial_means.load()
    final_energy = df["E"].iloc[-1]

    # 取得參數
    nu = sim.params.nu_2

    results.append({
        "nu": nu,
        "final_energy": final_energy,
        "sim_dir": sim_dir
    })

# 分析結果
results_df = pd.DataFrame(results)
results_df.plot(x="nu", y="final_energy", logx=True)
```

### 場操作

```python
sim = load_sim_for_plot("simulation_dir")

# 載入特定時間
sim.output.phys_fields.set_of_phys_files.update_times()
times = sim.output.phys_fields.set_of_phys_files.times

# 在特定時間載入場
field_file = sim.output.phys_fields.get_field_to_plot(time=5.0)
vorticity = field_file.get_var("rot")

# 計算衍生量
import numpy as np
vorticity_rms = np.sqrt(np.mean(vorticity**2))
vorticity_max = np.max(np.abs(vorticity))
```

## 輸出目錄結構

```
simulation_dir/
├── params_simul.xml         # 模擬參數
├── stdout.txt               # 標準輸出日誌
├── state_phys_t*.h5         # 不同時間的物理場
├── spatial_means.txt        # 空間平均的時間序列
├── spectra_*.h5            # 頻譜資料
├── spect_energy_budg_*.h5  # 能量預算資料
└── info_solver.txt         # 求解器資訊
```

## 效能監控

```python
# 在模擬期間檢查進度
sim.output.print_stdout.complete_timestep()

# 模擬後檢視效能
sim.output.print_stdout.plot_deltat()  # 繪製時間步長演化
sim.output.print_stdout.plot_clock_times()  # 繪製計算時間
```

## 資料匯出

將 fluidsim 輸出轉換為其他格式：

```python
import h5py
import numpy as np

# 匯出為 numpy 陣列
with h5py.File("state_phys_t10.000.h5", "r") as f:
    vx = f["state_phys"]["vx"][:]
    np.save("vx.npy", vx)

# 匯出為 CSV
df = sim.output.spatial_means.load()
df.to_csv("spatial_means.csv", index=False)
```
