# Cirq 中的模擬

本指南涵蓋量子電路模擬，包括精確和雜訊模擬、參數掃描和量子虛擬機（QVM）。

## 精確模擬

### 基本模擬

```python
import cirq
import numpy as np

# 建立電路
q0, q1 = cirq.LineQubit.range(2)
circuit = cirq.Circuit(
    cirq.H(q0),
    cirq.CNOT(q0, q1),
    cirq.measure(q0, q1, key='result')
)

# 模擬
simulator = cirq.Simulator()
result = simulator.run(circuit, repetitions=1000)

# 取得測量結果
print(result.histogram(key='result'))
```

### 狀態向量模擬

```python
# 不帶測量的模擬以取得最終態
simulator = cirq.Simulator()
result = simulator.simulate(circuit_without_measurement)

# 存取狀態向量
state_vector = result.final_state_vector
print(f"狀態向量：{state_vector}")

# 取得振幅
print(f"|00⟩ 的振幅：{state_vector[0]}")
print(f"|11⟩ 的振幅：{state_vector[3]}")
```

### 密度矩陣模擬

```python
# 對混合態使用密度矩陣模擬器
simulator = cirq.DensityMatrixSimulator()
result = simulator.simulate(circuit)

# 存取密度矩陣
density_matrix = result.final_density_matrix
print(f"密度矩陣形狀：{density_matrix.shape}")
```

### 逐步模擬

```python
# 逐 moment 模擬
simulator = cirq.Simulator()
for step in simulator.simulate_moment_steps(circuit):
    print(f"moment {step.moment} 後的狀態：{step.state_vector()}")
```

## 取樣和測量

### 執行多次取樣

```python
# 執行電路多次
result = simulator.run(circuit, repetitions=10000)

# 存取測量計數
counts = result.histogram(key='result')
print(f"測量計數：{counts}")

# 取得原始測量資料
measurements = result.measurements['result']
print(f"形狀：{measurements.shape}")  # (repetitions, num_qubits)
```

### 期望值

```python
# 測量可觀測量期望值
from cirq import PauliString

observable = PauliString({q0: cirq.Z, q1: cirq.Z})
result = simulator.simulate_expectation_values(
    circuit,
    observables=[observable]
)
print(f"⟨ZZ⟩ = {result[0]}")
```

## 參數掃描

### 參數掃描

```python
import sympy

# 建立參數化電路
theta = sympy.Symbol('theta')
q = cirq.LineQubit(0)
circuit = cirq.Circuit(
    cirq.ry(theta)(q),
    cirq.measure(q, key='m')
)

# 定義參數掃描
sweep = cirq.Linspace(key='theta', start=0, stop=2*np.pi, length=50)

# 執行掃描
simulator = cirq.Simulator()
results = simulator.run_sweep(circuit, params=sweep, repetitions=1000)

# 處理結果
for params, result in zip(sweep, results):
    theta_val = params['theta']
    counts = result.histogram(key='m')
    print(f"θ={theta_val:.2f}: {counts}")
```

### 多參數

```python
# 掃描多個參數
theta = sympy.Symbol('theta')
phi = sympy.Symbol('phi')

circuit = cirq.Circuit(
    cirq.ry(theta)(q0),
    cirq.rz(phi)(q1)
)

# 乘積掃描（所有組合）
sweep = cirq.Product(
    cirq.Linspace('theta', 0, np.pi, 10),
    cirq.Linspace('phi', 0, 2*np.pi, 10)
)

results = simulator.run_sweep(circuit, params=sweep, repetitions=100)
```

### Zip 掃描（配對參數）

```python
# 一起掃描參數
sweep = cirq.Zip(
    cirq.Linspace('theta', 0, np.pi, 20),
    cirq.Linspace('phi', 0, 2*np.pi, 20)
)

results = simulator.run_sweep(circuit, params=sweep, repetitions=100)
```

## 雜訊模擬

### 加入雜訊通道

```python
# 建立雜訊電路
noisy_circuit = circuit.with_noise(cirq.depolarize(p=0.01))

# 模擬雜訊電路
simulator = cirq.DensityMatrixSimulator()
result = simulator.run(noisy_circuit, repetitions=1000)
```

### 自訂雜訊模型

```python
# 對不同閘應用不同雜訊
noise_model = cirq.NoiseModel.from_noise_model_like(
    cirq.ConstantQubitNoiseModel(cirq.depolarize(0.01))
)

# 使用雜訊模型模擬
result = cirq.DensityMatrixSimulator(noise=noise_model).run(
    circuit, repetitions=1000
)
```

詳細的雜訊建模請參閱 `noise.md`。

## 狀態直方圖

### 視覺化結果

```python
import matplotlib.pyplot as plt

# 取得直方圖
result = simulator.run(circuit, repetitions=1000)
counts = result.histogram(key='result')

# 繪圖
plt.bar(counts.keys(), counts.values())
plt.xlabel('狀態')
plt.ylabel('計數')
plt.title('測量結果')
plt.show()
```

### 狀態機率分佈

```python
# 取得狀態向量
result = simulator.simulate(circuit_without_measurement)
state_vector = result.final_state_vector

# 計算機率
probabilities = np.abs(state_vector) ** 2

# 繪圖
plt.bar(range(len(probabilities)), probabilities)
plt.xlabel('基態索引')
plt.ylabel('機率')
plt.show()
```

## 量子虛擬機（QVM）

QVM 使用裝置特定的約束和雜訊模擬真實量子硬體。

### 使用虛擬裝置

```python
# 使用虛擬 Google 裝置
import cirq_google

# 取得虛擬裝置
device = cirq_google.Sycamore

# 在裝置上建立電路
qubits = device.metadata.qubit_set
circuit = cirq.Circuit(device=device)

# 加入符合裝置約束的操作
circuit.append(cirq.CZ(qubits[0], qubits[1]))

# 根據裝置驗證電路
device.validate_circuit(circuit)
```

### 雜訊虛擬硬體

```python
# 使用裝置雜訊模擬
processor = cirq_google.get_engine().get_processor('weber')
noise_props = processor.get_device_specification()

# 建立真實雜訊模擬器
noisy_sim = cirq.DensityMatrixSimulator(
    noise=cirq_google.NoiseModelFromGoogleNoiseProperties(noise_props)
)

result = noisy_sim.run(circuit, repetitions=1000)
```

## 進階模擬技術

### 自訂初始態

```python
# 從自訂狀態開始
initial_state = np.array([1, 0, 0, 1]) / np.sqrt(2)  # |00⟩ + |11⟩

simulator = cirq.Simulator()
result = simulator.simulate(circuit, initial_state=initial_state)
```

### 部分跡

```python
# 對子系統取跡
result = simulator.simulate(circuit)
full_state = result.final_state_vector

# 計算第一個量子位元的約化密度矩陣
from cirq import partial_trace
reduced_dm = partial_trace(result.final_density_matrix, keep_indices=[0])
```

### 中間狀態存取

```python
# 取得特定 moment 的狀態
simulator = cirq.Simulator()
for i, step in enumerate(simulator.simulate_moment_steps(circuit)):
    if i == 5:  # 第 5 個 moment 之後
        state = step.state_vector()
        print(f"moment 5 後的狀態：{state}")
        break
```

## 模擬效能

### 優化大型模擬

1. **對純態使用狀態向量**：比密度矩陣更快
2. **盡可能避免密度矩陣**：指數級更昂貴
3. **批次參數掃描**：比個別執行更有效率
4. **使用適當的重複次數**：平衡準確性與計算時間

```python
# 有效率：單一掃描
results = simulator.run_sweep(circuit, params=sweep, repetitions=100)

# 低效率：多個個別執行
results = [simulator.run(circuit, param_resolver=p, repetitions=100)
           for p in sweep]
```

### 記憶體考量

```python
# 對大型系統，監控狀態向量大小
n_qubits = 20
state_size = 2**n_qubits * 16  # 位元組（complex128）
print(f"狀態向量大小：{state_size / 1e9:.2f} GB")
```

## 穩定器模擬

對於只有 Clifford 閘的電路，使用有效率的穩定器模擬：

```python
# Clifford 電路（H、S、CNOT）
circuit = cirq.Circuit(
    cirq.H(q0),
    cirq.S(q1),
    cirq.CNOT(q0, q1)
)

# 使用穩定器模擬器（指數級更快）
simulator = cirq.CliffordSimulator()
result = simulator.run(circuit, repetitions=1000)
```

## 最佳實務

1. **選擇適當的模擬器**：對純態使用 Simulator，對混合態使用 DensityMatrixSimulator
2. **使用參數掃描**：比執行個別電路更有效率
3. **驗證電路**：長時間模擬前檢查電路有效性
4. **監控資源使用**：追蹤大規模模擬的記憶體
5. **使用穩定器模擬**：當電路只包含 Clifford 閘時
6. **儲存中間結果**：用於長時間參數掃描或優化執行
