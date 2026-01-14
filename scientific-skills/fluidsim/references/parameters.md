# 參數設定

## Parameters 物件

`Parameters` 物件是層次結構的，並組織成邏輯群組。使用點記法存取：

```python
params = Simul.create_default_params()
params.group.subgroup.parameter = value
```

## 主要參數群組

### 運算子（`params.oper`）

定義域和解析度：

```python
params.oper.nx = 256  # x 方向的網格點數
params.oper.ny = 256  # y 方向的網格點數
params.oper.nz = 128  # z 方向的網格點數（僅 3D）

params.oper.Lx = 2 * pi  # x 方向的域長度
params.oper.Ly = 2 * pi  # y 方向的域長度
params.oper.Lz = pi      # z 方向的域長度（僅 3D）

params.oper.coef_dealiasing = 2./3.  # 去混疊截止（預設 2/3）
```

**解析度指南**：使用 2 的冪次以獲得最佳 FFT 效能（128、256、512、1024 等）

### 物理參數

#### 黏度

```python
params.nu_2 = 1e-3  # 拉普拉斯黏度（負拉普拉斯）
params.nu_4 = 0     # 超黏度（可選）
params.nu_8 = 0     # 超超黏度（非常高波數阻尼）
```

高階黏度（`nu_4`、`nu_8`）阻尼高波數而不影響大尺度。

#### 分層（分層求解器）

```python
params.N = 1.0  # Brunt-Väisälä 頻率（浮力頻率）
```

#### 旋轉（淺水）

```python
params.f = 1.0  # 科里奧利參數
params.c2 = 10.0  # 相速度平方（重力波速度）
```

### 時間步進（`params.time_stepping`）

```python
params.time_stepping.t_end = 10.0  # 模擬結束時間
params.time_stepping.it_end = 100  # 或最大迭代次數

params.time_stepping.deltat0 = 0.01  # 初始時間步長
params.time_stepping.USE_CFL = True  # 自適應 CFL 基礎時間步長
params.time_stepping.CFL = 0.5  # CFL 數（如果 USE_CFL=True）

params.time_stepping.type_time_scheme = "RK4"  # 或 "RK2"、"Euler"
```

**建議**：使用 `USE_CFL=True` 搭配 `CFL=0.5` 進行自適應時間步進。

### 初始場（`params.init_fields`）

```python
params.init_fields.type = "noise"  # 初始化方法
```

**可用類型**：
- `"noise"`：隨機噪聲
- `"dipole"`：渦流偶極
- `"vortex"`：單渦流
- `"taylor_green"`：Taylor-Green 渦流
- `"from_file"`：從檔案載入
- `"in_script"`：在腳本中定義

#### 從檔案

```python
params.init_fields.type = "from_file"
params.init_fields.from_file.path = "path/to/state_file.h5"
```

#### 在腳本中

```python
params.init_fields.type = "in_script"

# 建立 sim 後定義初始化
sim = Simul(params)

# 存取狀態場
vx = sim.state.state_phys.get_var("vx")
vy = sim.state.state_phys.get_var("vy")

# 設定場
X, Y = sim.oper.get_XY_loc()
vx[:] = np.sin(X) * np.cos(Y)
vy[:] = -np.cos(X) * np.sin(Y)

# 執行模擬
sim.time_stepping.start()
```

### 輸出設定（`params.output`）

#### 輸出目錄

```python
params.output.sub_directory = "my_simulation"
```

目錄建立在 `$FLUIDSIM_PATH` 或目前目錄內。

#### 儲存週期

```python
params.output.periods_save.phys_fields = 1.0  # 每 1.0 時間單位儲存場
params.output.periods_save.spectra = 0.5      # 儲存頻譜
params.output.periods_save.spatial_means = 0.1  # 儲存空間平均
params.output.periods_save.spect_energy_budg = 0.5  # 頻譜能量預算
```

設為 `0` 以停用特定輸出類型。

#### 列印控制

```python
params.output.periods_print.print_stdout = 0.5  # 每 0.5 時間單位列印狀態
```

#### 線上繪圖

```python
params.output.periods_plot.phys_fields = 2.0  # 每 2.0 時間單位繪圖

# 必須同時啟用輸出模組
params.output.ONLINE_PLOT_OK = True
params.output.phys_fields.field_to_plot = "vorticity"  # 或 "vx"、"vy" 等
```

### 強制（`params.forcing`）

添加強制項以維持能量：

```python
params.forcing.enable = True
params.forcing.type = "tcrandom"  # 時間相關隨機強制

# 強制參數
params.forcing.nkmax_forcing = 5  # 最大強制波數
params.forcing.nkmin_forcing = 2  # 最小強制波數
params.forcing.forcing_rate = 1.0  # 能量注入率
```

**常見強制類型**：
- `"tcrandom"`：時間相關隨機強制
- `"proportional"`：比例強制（維持特定頻譜）
- `"in_script"`：在腳本中定義的自訂強制

## 參數安全性

Parameters 物件在存取不存在的參數時會引發 `AttributeError`：

```python
params.nu_2 = 1e-3  # 正確
params.nu2 = 1e-3   # 錯誤：AttributeError
```

這防止了在文字基礎設定檔中會被靜默忽略的拼寫錯誤。

## 檢視所有參數

```python
# 列印所有參數
params._print_as_xml()

# 取得為字典
param_dict = params._make_dict()
```

## 儲存參數設定

參數會自動與模擬輸出一起儲存：

```python
params._save_as_xml("simulation_params.xml")
params._save_as_json("simulation_params.json")
```
