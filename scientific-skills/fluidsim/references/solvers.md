# FluidSim 求解器

FluidSim 為不同的流體動力學方程式提供多種求解器。所有求解器都在週期性域上使用基於 FFT 的擬譜方法。

## 可用求解器

### 2D 不可壓縮 Navier-Stokes

**求解器鍵值**：`ns2d`

**匯入**：
```python
from fluidsim.solvers.ns2d.solver import Simul
# 或動態匯入
Simul = fluidsim.import_simul_class_from_key("ns2d")
```

**用途**：2D 紊流研究、渦流動力學、基礎流體流動模擬

**主要特徵**：能量和渦度級聯、渦度動力學

### 3D 不可壓縮 Navier-Stokes

**求解器鍵值**：`ns3d`

**匯入**：
```python
from fluidsim.solvers.ns3d.solver import Simul
```

**用途**：3D 紊流、真實流體流動模擬、高解析度 DNS

**主要特徵**：完整 3D 紊流動力學、平行計算支援

### 分層流（2D/3D）

**求解器鍵值**：`ns2d.strat`、`ns3d.strat`

**匯入**：
```python
from fluidsim.solvers.ns2d.strat.solver import Simul  # 2D
from fluidsim.solvers.ns3d.strat.solver import Simul  # 3D
```

**用途**：海洋和大氣流、密度驅動流

**主要特徵**：Boussinesq 近似、浮力效應、恆定 Brunt-Väisälä 頻率

**參數**：透過 `params.N`（Brunt-Väisälä 頻率）設定分層

### 淺水方程式

**求解器鍵值**：`sw1l`（單層）

**匯入**：
```python
from fluidsim.solvers.sw1l.solver import Simul
```

**用途**：地球物理流、海嘯模擬、旋轉流

**主要特徵**：旋轉座標系支援、地轉平衡

**參數**：透過 `params.f`（科里奧利參數）設定旋轉

### Föppl-von Kármán 方程式

**求解器鍵值**：`fvk`（彈性板方程式）

**匯入**：
```python
from fluidsim.solvers.fvk.solver import Simul
```

**用途**：彈性板動力學、流固交互作用研究

## 求解器選擇指南

根據物理問題選擇求解器：

1. **2D 紊流、快速測試**：使用 `ns2d`
2. **3D 流場、真實模擬**：使用 `ns3d`
3. **密度分層流**：使用 `ns2d.strat` 或 `ns3d.strat`
4. **地球物理流、旋轉系統**：使用 `sw1l`
5. **彈性板**：使用 `fvk`

## 修改版本

許多求解器有具有額外物理的修改版本：
- 強制項
- 不同邊界條件
- 額外純量場

檢查 `fluidsim.solvers` 模組以獲取完整列表。
