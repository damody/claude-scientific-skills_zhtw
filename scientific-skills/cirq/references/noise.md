# 雜訊建模和緩解

本指南涵蓋 Cirq 中的雜訊模型、雜訊模擬、特性分析和錯誤緩解。

## 雜訊通道

### 去極化雜訊

```python
import cirq
import numpy as np

# 單量子位元去極化通道
depol_channel = cirq.depolarize(p=0.01)

# 應用到量子位元
q = cirq.LineQubit(0)
noisy_op = depol_channel(q)

# 加入電路
circuit = cirq.Circuit(
    cirq.H(q),
    depol_channel(q),
    cirq.measure(q, key='m')
)
```

### 振幅阻尼

```python
# 振幅阻尼（T1 衰減）
gamma = 0.1
amp_damp = cirq.amplitude_damp(gamma)

# 在閘之後應用
circuit = cirq.Circuit(
    cirq.X(q),
    amp_damp(q)
)
```

### 相位阻尼

```python
# 相位阻尼（T2 退相干）
gamma = 0.1
phase_damp = cirq.phase_damp(gamma)

circuit = cirq.Circuit(
    cirq.H(q),
    phase_damp(q)
)
```

### 位元翻轉雜訊

```python
# 位元翻轉通道
bit_flip_prob = 0.01
bit_flip = cirq.bit_flip(bit_flip_prob)

circuit = cirq.Circuit(
    cirq.H(q),
    bit_flip(q)
)
```

### 相位翻轉雜訊

```python
# 相位翻轉通道
phase_flip_prob = 0.01
phase_flip = cirq.phase_flip(phase_flip_prob)

circuit = cirq.Circuit(
    cirq.H(q),
    phase_flip(q)
)
```

### 廣義振幅阻尼

```python
# 廣義振幅阻尼
p = 0.1  # 阻尼機率
gamma = 0.2  # 激發機率
gen_amp_damp = cirq.generalized_amplitude_damp(p=p, gamma=gamma)
```

### 重置通道

```python
# 重置到 |0⟩ 或 |1⟩
reset_to_zero = cirq.reset(q)

# 重置表現為測量後條件翻轉
circuit = cirq.Circuit(
    cirq.H(q),
    reset_to_zero
)
```

## 雜訊模型

### 常數雜訊模型

```python
# 對所有量子位元應用相同雜訊
noise = cirq.ConstantQubitNoiseModel(
    qubit_noise_gate=cirq.depolarize(0.01)
)

# 使用雜訊模擬
simulator = cirq.DensityMatrixSimulator(noise=noise)
result = simulator.run(circuit, repetitions=1000)
```

### 閘特定雜訊

```python
class CustomNoiseModel(cirq.NoiseModel):
    """對不同閘類型應用不同雜訊。"""

    def noisy_operation(self, op):
        # 單量子位元閘：去極化雜訊
        if len(op.qubits) == 1:
            return [op, cirq.depolarize(0.001)(op.qubits[0])]

        # 雙量子位元閘：更高的去極化雜訊
        elif len(op.qubits) == 2:
            return [
                op,
                cirq.depolarize(0.01)(op.qubits[0]),
                cirq.depolarize(0.01)(op.qubits[1])
            ]

        return op

# 使用自訂雜訊模型
noise_model = CustomNoiseModel()
simulator = cirq.DensityMatrixSimulator(noise=noise_model)
```

### 量子位元特定雜訊

```python
class QubitSpecificNoise(cirq.NoiseModel):
    """對不同量子位元應用不同雜訊。"""

    def __init__(self, qubit_noise_map):
        self.qubit_noise_map = qubit_noise_map

    def noisy_operation(self, op):
        noise_ops = [op]
        for qubit in op.qubits:
            if qubit in self.qubit_noise_map:
                noise = self.qubit_noise_map[qubit]
                noise_ops.append(noise(qubit))
        return noise_ops

# 定義每個量子位元的雜訊
q0, q1, q2 = cirq.LineQubit.range(3)
noise_map = {
    q0: cirq.depolarize(0.001),
    q1: cirq.depolarize(0.005),
    q2: cirq.depolarize(0.002)
}

noise_model = QubitSpecificNoise(noise_map)
```

### 熱雜訊

```python
class ThermalNoise(cirq.NoiseModel):
    """熱弛豫雜訊。"""

    def __init__(self, T1, T2, gate_time):
        self.T1 = T1  # 振幅阻尼時間
        self.T2 = T2  # 退相干時間
        self.gate_time = gate_time

    def noisy_operation(self, op):
        # 計算機率
        p_amp = 1 - np.exp(-self.gate_time / self.T1)
        p_phase = 1 - np.exp(-self.gate_time / self.T2)

        noise_ops = [op]
        for qubit in op.qubits:
            noise_ops.append(cirq.amplitude_damp(p_amp)(qubit))
            noise_ops.append(cirq.phase_damp(p_phase)(qubit))

        return noise_ops

# 典型超導量子位元參數
T1 = 50e-6  # 50 μs
T2 = 30e-6  # 30 μs
gate_time = 25e-9  # 25 ns

noise_model = ThermalNoise(T1, T2, gate_time)
```

## 將雜訊加入電路

### with_noise 方法

```python
# 對所有操作加入雜訊
noisy_circuit = circuit.with_noise(cirq.depolarize(p=0.01))

# 模擬雜訊電路
simulator = cirq.DensityMatrixSimulator()
result = simulator.run(noisy_circuit, repetitions=1000)
```

### insert_into_circuit 方法

```python
# 手動雜訊插入
def add_noise_to_circuit(circuit, noise_model):
    noisy_moments = []
    for moment in circuit:
        ops = []
        for op in moment:
            ops.extend(noise_model.noisy_operation(op))
        noisy_moments.append(cirq.Moment(ops))
    return cirq.Circuit(noisy_moments)
```

## 讀取雜訊

### 測量錯誤模型

```python
class ReadoutNoiseModel(cirq.NoiseModel):
    """建模讀取/測量錯誤。"""

    def __init__(self, p0_given_1, p1_given_0):
        # p0_given_1：狀態為 1 時測量到 0 的機率
        # p1_given_0：狀態為 0 時測量到 1 的機率
        self.p0_given_1 = p0_given_1
        self.p1_given_0 = p1_given_0

    def noisy_operation(self, op):
        if isinstance(op.gate, cirq.MeasurementGate):
            # 在測量前應用位元翻轉
            noise_ops = []
            for qubit in op.qubits:
                # 平均讀取錯誤
                p_error = (self.p0_given_1 + self.p1_given_0) / 2
                noise_ops.append(cirq.bit_flip(p_error)(qubit))
            noise_ops.append(op)
            return noise_ops
        return op

# 典型讀取錯誤
readout_noise = ReadoutNoiseModel(p0_given_1=0.02, p1_given_0=0.01)
```

## 雜訊特性分析

### 隨機基準測試

```python
import cirq

def generate_rb_circuit(qubits, depth):
    """生成隨機基準測試電路。"""
    # 隨機 Clifford 閘
    clifford_gates = [cirq.X, cirq.Y, cirq.Z, cirq.H, cirq.S]

    circuit = cirq.Circuit()
    for _ in range(depth):
        for qubit in qubits:
            gate = np.random.choice(clifford_gates)
            circuit.append(gate(qubit))

    # 加入逆操作以返回初始態（理想情況下）
    # （簡化版 - 正確的 RB 需要追蹤完整序列）

    circuit.append(cirq.measure(*qubits, key='result'))
    return circuit

# 執行 RB 實驗
def run_rb_experiment(qubits, depths, repetitions=1000):
    """在不同深度執行隨機基準測試。"""
    simulator = cirq.DensityMatrixSimulator(
        noise=cirq.ConstantQubitNoiseModel(cirq.depolarize(0.01))
    )

    survival_probs = []
    for depth in depths:
        circuits = [generate_rb_circuit(qubits, depth) for _ in range(20)]

        total_survival = 0
        for circuit in circuits:
            result = simulator.run(circuit, repetitions=repetitions)
            # 計算存活機率（返回到 |0⟩）
            counts = result.histogram(key='result')
            survival = counts.get(0, 0) / repetitions
            total_survival += survival

        avg_survival = total_survival / len(circuits)
        survival_probs.append(avg_survival)

    return survival_probs

# 擬合以提取錯誤率
# p_survival = A * p^depth + B
# 每閘錯誤 ≈ (1 - p) / 2
```

### 交叉熵基準測試（XEB）

```python
def xeb_fidelity(circuit, simulator, ideal_probs, repetitions=10000):
    """計算 XEB 保真度。"""

    # 執行雜訊模擬
    result = simulator.run(circuit, repetitions=repetitions)
    measured_probs = result.histogram(key='result')

    # 正規化
    for key in measured_probs:
        measured_probs[key] /= repetitions

    # 計算交叉熵
    cross_entropy = 0
    for bitstring, prob in measured_probs.items():
        if bitstring in ideal_probs:
            cross_entropy += prob * np.log2(ideal_probs[bitstring])

    # 轉換為保真度
    n_qubits = len(circuit.all_qubits())
    fidelity = (2**n_qubits * cross_entropy + 1) / (2**n_qubits - 1)

    return fidelity
```

## 雜訊視覺化

### 熱圖視覺化

```python
import matplotlib.pyplot as plt

def plot_noise_heatmap(device, noise_metric):
    """繪製 2D 網格裝置上的雜訊特性。"""

    # 取得裝置量子位元（假設為 GridQubit）
    qubits = sorted(device.metadata.qubit_set)
    rows = max(q.row for q in qubits) + 1
    cols = max(q.col for q in qubits) + 1

    # 建立熱圖資料
    heatmap = np.full((rows, cols), np.nan)

    for qubit in qubits:
        if isinstance(qubit, cirq.GridQubit):
            value = noise_metric.get(qubit, 0)
            heatmap[qubit.row, qubit.col] = value

    # 繪圖
    plt.figure(figsize=(10, 8))
    plt.imshow(heatmap, cmap='RdYlGn_r', interpolation='nearest')
    plt.colorbar(label='錯誤率')
    plt.title('量子位元錯誤率')
    plt.xlabel('列')
    plt.ylabel('行')
    plt.show()

# 範例用法
noise_metric = {q: np.random.random() * 0.01 for q in device.metadata.qubit_set}
plot_noise_heatmap(device, noise_metric)
```

### 閘保真度視覺化

```python
def plot_gate_fidelities(calibration_data):
    """繪製單量子位元和雙量子位元閘保真度。"""

    sq_fidelities = []
    tq_fidelities = []

    for qubit, metrics in calibration_data.items():
        if 'single_qubit_rb_fidelity' in metrics:
            sq_fidelities.append(metrics['single_qubit_rb_fidelity'])
        if 'two_qubit_rb_fidelity' in metrics:
            tq_fidelities.append(metrics['two_qubit_rb_fidelity'])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.hist(sq_fidelities, bins=20)
    ax1.set_xlabel('單量子位元閘保真度')
    ax1.set_ylabel('計數')
    ax1.set_title('單量子位元閘保真度')

    ax2.hist(tq_fidelities, bins=20)
    ax2.set_xlabel('雙量子位元閘保真度')
    ax2.set_ylabel('計數')
    ax2.set_title('雙量子位元閘保真度')

    plt.tight_layout()
    plt.show()
```

## 錯誤緩解技術

### 零雜訊外推

```python
def zero_noise_extrapolation(circuit, noise_levels, simulator):
    """外推到零雜訊極限。"""

    expectation_values = []

    for noise_level in noise_levels:
        # 縮放雜訊
        noisy_circuit = circuit.with_noise(
            cirq.depolarize(p=noise_level)
        )

        # 測量期望值
        result = simulator.simulate(noisy_circuit)
        # ... 計算期望值
        exp_val = calculate_expectation(result)
        expectation_values.append(exp_val)

    # 外推到零雜訊
    from scipy.optimize import curve_fit

    def exponential_fit(x, a, b, c):
        return a * np.exp(-b * x) + c

    popt, _ = curve_fit(exponential_fit, noise_levels, expectation_values)
    zero_noise_value = popt[2]

    return zero_noise_value
```

### 機率錯誤消除

```python
def quasi_probability_decomposition(noisy_gate, ideal_gate, noise_model):
    """將雜訊閘分解為準機率分佈。"""

    # 將雜訊閘分解為：N = ideal + error
    # 反轉：ideal = (N - error) / (1 - error_rate)

    # 這會建立準機率分佈
    # （部分機率可能為負）

    # 實作取決於具體的雜訊模型
    pass
```

### 讀取錯誤緩解

```python
def mitigate_readout_errors(results, confusion_matrix):
    """使用混淆矩陣應用讀取錯誤緩解。"""

    # 反轉混淆矩陣
    inv_confusion = np.linalg.inv(confusion_matrix)

    # 取得測量計數
    counts = results.histogram(key='result')

    # 轉換為機率向量
    total_counts = sum(counts.values())
    measured_probs = np.array([counts.get(i, 0) / total_counts
                               for i in range(len(confusion_matrix))])

    # 應用逆矩陣
    corrected_probs = inv_confusion @ measured_probs

    # 轉換回計數
    corrected_counts = {i: int(p * total_counts)
                       for i, p in enumerate(corrected_probs) if p > 0}

    return corrected_counts
```

## 基於硬體的雜訊模型

### 從 Google 校準

```python
import cirq_google

# 取得校準資料
processor = cirq_google.get_engine().get_processor('weber')
noise_props = processor.get_device_specification()

# 從校準建立雜訊模型
noise_model = cirq_google.NoiseModelFromGoogleNoiseProperties(noise_props)

# 使用真實雜訊模擬
simulator = cirq.DensityMatrixSimulator(noise=noise_model)
result = simulator.run(circuit, repetitions=1000)
```

## 最佳實務

1. **使用密度矩陣模擬器進行雜訊模擬**：狀態向量模擬器無法建模混合態
2. **將雜訊模型與硬體匹配**：可用時使用校準資料
3. **包含所有錯誤來源**：閘錯誤、退相干、讀取錯誤
4. **先特性分析再緩解**：緩解前先了解雜訊
5. **考慮錯誤傳播**：雜訊會隨電路深度累積
6. **使用適當的基準測試**：RB 用於閘錯誤，XEB 用於完整電路保真度
7. **視覺化雜訊模式**：識別有問題的量子位元和閘
8. **應用針對性緩解**：專注於主要錯誤來源
9. **驗證緩解效果**：確認緩解改善了結果
10. **保持電路淺層**：最小化雜訊累積
