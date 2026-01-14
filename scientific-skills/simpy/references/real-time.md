# SimPy 即時模擬

本指南涵蓋 SimPy 中的即時模擬功能，其中模擬時間與實際時間同步。

## 概述

即時模擬將模擬時間與實際時間同步。這對以下情況很有用：

- **硬體迴路（HIL）**測試
- **人機互動**模擬
- **演算法行為分析**在即時限制下
- **系統整合**測試
- **展示**目的

## RealtimeEnvironment

將標準 `Environment` 替換為 `simpy.rt.RealtimeEnvironment` 以啟用即時同步。

### 基本使用

```python
import simpy.rt

def process(env):
    while True:
        print(f'Tick at {env.now}')
        yield env.timeout(1)

# Real-time environment with 1:1 time mapping
env = simpy.rt.RealtimeEnvironment(factor=1.0)
env.process(process(env))
env.run(until=5)
```

### 建構函式參數

```python
simpy.rt.RealtimeEnvironment(
    initial_time=0,      # Starting simulation time
    factor=1.0,          # Real time per simulation time unit
    strict=True          # Raise errors on timing violations
)
```

## 使用 Factor 進行時間縮放

`factor` 參數控制模擬時間如何對應到實際時間。

### Factor 範例

```python
import simpy.rt
import time

def timed_process(env, label):
    start = time.time()
    print(f'{label}: Starting at {env.now}')
    yield env.timeout(2)
    elapsed = time.time() - start
    print(f'{label}: Completed at {env.now} (real time: {elapsed:.2f}s)')

# Factor = 1.0: 1 simulation time unit = 1 second
print('Factor = 1.0 (2 sim units = 2 seconds)')
env = simpy.rt.RealtimeEnvironment(factor=1.0)
env.process(timed_process(env, 'Normal speed'))
env.run()

# Factor = 0.5: 1 simulation time unit = 0.5 seconds
print('\nFactor = 0.5 (2 sim units = 1 second)')
env = simpy.rt.RealtimeEnvironment(factor=0.5)
env.process(timed_process(env, 'Double speed'))
env.run()

# Factor = 2.0: 1 simulation time unit = 2 seconds
print('\nFactor = 2.0 (2 sim units = 4 seconds)')
env = simpy.rt.RealtimeEnvironment(factor=2.0)
env.process(timed_process(env, 'Half speed'))
env.run()
```

**Factor 解釋：**
- `factor=1.0` → 1 個模擬時間單位需要 1 實際秒
- `factor=0.1` → 1 個模擬時間單位需要 0.1 實際秒（10 倍速）
- `factor=60` → 1 個模擬時間單位需要 60 實際秒（1 分鐘）

## 嚴格模式

### strict=True（預設）

如果計算超過分配的實時預算，則拋出 `RuntimeError`。

```python
import simpy.rt
import time

def heavy_computation(env):
    print(f'Starting computation at {env.now}')
    yield env.timeout(1)

    # Simulate heavy computation (exceeds 1 second budget)
    time.sleep(1.5)

    print(f'Computation done at {env.now}')

env = simpy.rt.RealtimeEnvironment(factor=1.0, strict=True)
env.process(heavy_computation(env))

try:
    env.run()
except RuntimeError as e:
    print(f'Error: {e}')
```

### strict=False

允許模擬執行比預期慢而不會崩潰。

```python
import simpy.rt
import time

def heavy_computation(env):
    print(f'Starting at {env.now}')
    yield env.timeout(1)

    # Heavy computation
    time.sleep(1.5)

    print(f'Done at {env.now}')

env = simpy.rt.RealtimeEnvironment(factor=1.0, strict=False)
env.process(heavy_computation(env))
env.run()

print('Simulation completed (slower than real-time)')
```

**在以下情況使用 strict=False：**
- 開發和除錯
- 計算時間不可預測
- 可接受比目標速率慢
- 分析最壞情況行為

## 硬體迴路範例

```python
import simpy.rt

class HardwareInterface:
    """Simulated hardware interface."""

    def __init__(self):
        self.sensor_value = 0

    def read_sensor(self):
        """Simulate reading from hardware sensor."""
        import random
        self.sensor_value = random.uniform(20.0, 30.0)
        return self.sensor_value

    def write_actuator(self, value):
        """Simulate writing to hardware actuator."""
        print(f'Actuator set to {value:.2f}')

def control_loop(env, hardware, setpoint):
    """Simple control loop running in real-time."""
    while True:
        # Read sensor
        sensor_value = hardware.read_sensor()
        print(f'[{env.now}] Sensor: {sensor_value:.2f}°C')

        # Simple proportional control
        error = setpoint - sensor_value
        control_output = error * 0.1

        # Write actuator
        hardware.write_actuator(control_output)

        # Control loop runs every 0.5 seconds
        yield env.timeout(0.5)

# Real-time environment: 1 sim unit = 1 second
env = simpy.rt.RealtimeEnvironment(factor=1.0, strict=False)
hardware = HardwareInterface()
setpoint = 25.0

env.process(control_loop(env, hardware, setpoint))
env.run(until=5)
```

## 人機互動範例

```python
import simpy.rt

def interactive_process(env):
    """Process that waits for simulated user input."""
    print('Simulation started. Events will occur in real-time.')

    yield env.timeout(2)
    print(f'[{env.now}] Event 1: System startup')

    yield env.timeout(3)
    print(f'[{env.now}] Event 2: Initialization complete')

    yield env.timeout(2)
    print(f'[{env.now}] Event 3: Ready for operation')

# Real-time environment for human-paced demonstration
env = simpy.rt.RealtimeEnvironment(factor=1.0)
env.process(interactive_process(env))
env.run()
```

## 監控即時效能

```python
import simpy.rt
import time

class RealTimeMonitor:
    def __init__(self):
        self.step_times = []
        self.drift_values = []

    def record_step(self, sim_time, real_time, expected_real_time):
        self.step_times.append(sim_time)
        drift = real_time - expected_real_time
        self.drift_values.append(drift)

    def report(self):
        if self.drift_values:
            avg_drift = sum(self.drift_values) / len(self.drift_values)
            max_drift = max(abs(d) for d in self.drift_values)
            print(f'\nReal-time performance:')
            print(f'Average drift: {avg_drift*1000:.2f} ms')
            print(f'Maximum drift: {max_drift*1000:.2f} ms')

def monitored_process(env, monitor, start_time, factor):
    for i in range(5):
        step_start = time.time()
        yield env.timeout(1)

        real_elapsed = time.time() - start_time
        expected_elapsed = env.now * factor
        monitor.record_step(env.now, real_elapsed, expected_elapsed)

        print(f'Sim time: {env.now}, Real time: {real_elapsed:.2f}s, ' +
              f'Expected: {expected_elapsed:.2f}s')

start = time.time()
factor = 1.0
env = simpy.rt.RealtimeEnvironment(factor=factor, strict=False)
monitor = RealTimeMonitor()

env.process(monitored_process(env, monitor, start, factor))
env.run()
monitor.report()
```

## 混合即時和快速模擬

```python
import simpy.rt

def background_simulation(env):
    """Fast background simulation."""
    for i in range(100):
        yield env.timeout(0.01)
    print(f'Background simulation completed at {env.now}')

def real_time_display(env):
    """Real-time display updates."""
    for i in range(5):
        print(f'Display update at {env.now}')
        yield env.timeout(1)

# Note: This is conceptual - SimPy doesn't directly support mixed modes
# Consider running separate simulations or using strict=False
env = simpy.rt.RealtimeEnvironment(factor=1.0, strict=False)
env.process(background_simulation(env))
env.process(real_time_display(env))
env.run()
```

## 將標準轉換為即時

將標準模擬轉換為即時很簡單：

```python
import simpy
import simpy.rt

def process(env):
    print(f'Event at {env.now}')
    yield env.timeout(1)
    print(f'Event at {env.now}')
    yield env.timeout(1)
    print(f'Event at {env.now}')

# Standard simulation (runs instantly)
print('Standard simulation:')
env = simpy.Environment()
env.process(process(env))
env.run()

# Real-time simulation (2 real seconds)
print('\nReal-time simulation:')
env_rt = simpy.rt.RealtimeEnvironment(factor=1.0)
env_rt.process(process(env_rt))
env_rt.run()
```

## 最佳實踐

1. **Factor 選擇**：根據硬體/人類限制選擇 factor
   - 人機互動：`factor=1.0`（1:1 時間對應）
   - 快速硬體：`factor=0.01`（100 倍速）
   - 慢速流程：`factor=60`（1 個模擬單位 = 1 分鐘）

2. **嚴格模式使用**：
   - 使用 `strict=True` 進行時序驗證
   - 使用 `strict=False` 進行開發和可變工作負載

3. **計算預算**：確保流程邏輯執行速度快於超時持續時間

4. **錯誤處理**：將即時執行包裝在 try-except 中以處理時序違規

5. **測試策略**：
   - 使用標準 Environment 開發（快速迭代）
   - 使用 RealtimeEnvironment 測試（驗證）
   - 使用適當的 factor 和 strict 設定進行部署

6. **效能監控**：追蹤模擬和實際時間之間的漂移

7. **優雅降級**：當時序保證不重要時使用 `strict=False`

## 常見模式

### 週期性即時任務

```python
import simpy.rt

def periodic_task(env, name, period, duration):
    """Task that runs periodically in real-time."""
    while True:
        start = env.now
        print(f'{name}: Starting at {start}')

        # Simulate work
        yield env.timeout(duration)

        print(f'{name}: Completed at {env.now}')

        # Wait for next period
        elapsed = env.now - start
        wait_time = period - elapsed
        if wait_time > 0:
            yield env.timeout(wait_time)

env = simpy.rt.RealtimeEnvironment(factor=1.0)
env.process(periodic_task(env, 'Task', period=2.0, duration=0.5))
env.run(until=6)
```

### 同步多設備控制

```python
import simpy.rt

def device_controller(env, device_id, update_rate):
    """Control loop for individual device."""
    while True:
        print(f'Device {device_id}: Update at {env.now}')
        yield env.timeout(update_rate)

# All devices synchronized to real-time
env = simpy.rt.RealtimeEnvironment(factor=1.0)

# Different update rates for different devices
env.process(device_controller(env, 'A', 1.0))
env.process(device_controller(env, 'B', 0.5))
env.process(device_controller(env, 'C', 2.0))

env.run(until=5)
```

## 限制

1. **效能**：即時模擬增加開銷；不適合高頻事件
2. **同步**：單執行緒；所有流程共享同一時間基準
3. **精度**：受限於 Python 的時間解析度和系統排程
4. **嚴格模式**：對於計算密集型流程可能經常拋出錯誤
5. **平台相依**：時序準確度因作業系統而異
