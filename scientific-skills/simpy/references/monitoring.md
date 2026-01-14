# SimPy 監控與資料收集

本指南涵蓋在 SimPy 中收集資料和監控模擬行為的技術。

## 監控策略

在實施監控之前，定義三件事：

1. **監控什麼**：流程、資源、事件或系統狀態
2. **何時監控**：在變更時、定期間隔或特定事件時
3. **如何儲存資料**：列表、檔案、資料庫或即時輸出

## 1. 流程監控

### 狀態變數追蹤

透過在變數變更時記錄來追蹤流程狀態。

```python
import simpy

def customer(env, name, service_time, log):
    arrival_time = env.now
    log.append(('arrival', name, arrival_time))

    yield env.timeout(service_time)

    departure_time = env.now
    log.append(('departure', name, departure_time))

    wait_time = departure_time - arrival_time
    log.append(('wait_time', name, wait_time))

env = simpy.Environment()
log = []

env.process(customer(env, 'Customer 1', 5, log))
env.process(customer(env, 'Customer 2', 3, log))
env.run()

print('Simulation log:')
for entry in log:
    print(entry)
```

### 時間序列資料收集

```python
import simpy

def system_monitor(env, system_state, data_log, interval):
    while True:
        data_log.append((env.now, system_state['queue_length'], system_state['utilization']))
        yield env.timeout(interval)

def process(env, system_state):
    while True:
        system_state['queue_length'] += 1
        yield env.timeout(2)
        system_state['queue_length'] -= 1
        system_state['utilization'] = system_state['queue_length'] / 10
        yield env.timeout(3)

env = simpy.Environment()
system_state = {'queue_length': 0, 'utilization': 0.0}
data_log = []

env.process(system_monitor(env, system_state, data_log, interval=1))
env.process(process(env, system_state))
env.run(until=20)

print('Time series data:')
for time, queue, util in data_log:
    print(f'Time {time}: Queue={queue}, Utilization={util:.2f}')
```

### 多變數追蹤

```python
import simpy

class SimulationData:
    def __init__(self):
        self.timestamps = []
        self.queue_lengths = []
        self.processing_times = []
        self.utilizations = []

    def record(self, timestamp, queue_length, processing_time, utilization):
        self.timestamps.append(timestamp)
        self.queue_lengths.append(queue_length)
        self.processing_times.append(processing_time)
        self.utilizations.append(utilization)

def monitored_process(env, data):
    queue_length = 0
    processing_time = 0
    utilization = 0.0

    for i in range(5):
        queue_length = i % 3
        processing_time = 2 + i
        utilization = queue_length / 10

        data.record(env.now, queue_length, processing_time, utilization)
        yield env.timeout(2)

env = simpy.Environment()
data = SimulationData()
env.process(monitored_process(env, data))
env.run()

print(f'Collected {len(data.timestamps)} data points')
```

## 2. 資源監控

### 猴子補丁資源

對資源方法進行補丁以攔截和記錄操作。

```python
import simpy

def patch_resource(resource, data_log):
    """Patch a resource to log all requests and releases."""

    # Save original methods
    original_request = resource.request
    original_release = resource.release

    # Create wrapper for request
    def logged_request(*args, **kwargs):
        req = original_request(*args, **kwargs)
        data_log.append(('request', resource._env.now, len(resource.queue)))
        return req

    # Create wrapper for release
    def logged_release(*args, **kwargs):
        result = original_release(*args, **kwargs)
        data_log.append(('release', resource._env.now, len(resource.queue)))
        return result

    # Replace methods
    resource.request = logged_request
    resource.release = logged_release

def user(env, name, resource):
    with resource.request() as req:
        yield req
        print(f'{name} using resource at {env.now}')
        yield env.timeout(3)
        print(f'{name} releasing resource at {env.now}')

env = simpy.Environment()
resource = simpy.Resource(env, capacity=1)
log = []

patch_resource(resource, log)

env.process(user(env, 'User 1', resource))
env.process(user(env, 'User 2', resource))
env.run()

print('\nResource log:')
for entry in log:
    print(entry)
```

### 資源子類別化

建立具有內建監控的自訂資源類別。

```python
import simpy

class MonitoredResource(simpy.Resource):
    def __init__(self, env, capacity):
        super().__init__(env, capacity)
        self.data = []
        self.utilization_data = []

    def request(self, *args, **kwargs):
        req = super().request(*args, **kwargs)
        queue_length = len(self.queue)
        utilization = self.count / self.capacity
        self.data.append(('request', self._env.now, queue_length, utilization))
        self.utilization_data.append((self._env.now, utilization))
        return req

    def release(self, *args, **kwargs):
        result = super().release(*args, **kwargs)
        queue_length = len(self.queue)
        utilization = self.count / self.capacity
        self.data.append(('release', self._env.now, queue_length, utilization))
        self.utilization_data.append((self._env.now, utilization))
        return result

    def average_utilization(self):
        if not self.utilization_data:
            return 0.0
        return sum(u for _, u in self.utilization_data) / len(self.utilization_data)

def user(env, name, resource):
    with resource.request() as req:
        yield req
        print(f'{name} using resource at {env.now}')
        yield env.timeout(2)

env = simpy.Environment()
resource = MonitoredResource(env, capacity=2)

for i in range(5):
    env.process(user(env, f'User {i+1}', resource))

env.run()

print(f'\nAverage utilization: {resource.average_utilization():.2%}')
print(f'Total operations: {len(resource.data)}')
```

### 容器水位監控

```python
import simpy

class MonitoredContainer(simpy.Container):
    def __init__(self, env, capacity, init=0):
        super().__init__(env, capacity, init)
        self.level_data = [(0, init)]

    def put(self, amount):
        result = super().put(amount)
        self.level_data.append((self._env.now, self.level))
        return result

    def get(self, amount):
        result = super().get(amount)
        self.level_data.append((self._env.now, self.level))
        return result

def producer(env, container, amount, interval):
    while True:
        yield env.timeout(interval)
        yield container.put(amount)
        print(f'Produced {amount}. Level: {container.level} at {env.now}')

def consumer(env, container, amount, interval):
    while True:
        yield env.timeout(interval)
        yield container.get(amount)
        print(f'Consumed {amount}. Level: {container.level} at {env.now}')

env = simpy.Environment()
container = MonitoredContainer(env, capacity=100, init=50)

env.process(producer(env, container, 20, 3))
env.process(consumer(env, container, 15, 4))
env.run(until=20)

print('\nLevel history:')
for time, level in container.level_data:
    print(f'Time {time}: Level={level}')
```

## 3. 事件追蹤

### 環境步驟監控

透過對環境的步驟函式進行補丁來監控所有事件。

```python
import simpy

def trace(env, callback):
    """Trace all events processed by the environment."""

    def _trace_step():
        # Get next event before it's processed
        if env._queue:
            time, priority, event_id, event = env._queue[0]
            callback(time, priority, event_id, event)

        # Call original step
        return original_step()

    original_step = env.step
    env.step = _trace_step

def event_callback(time, priority, event_id, event):
    print(f'Event: time={time}, priority={priority}, id={event_id}, type={type(event).__name__}')

def process(env, name):
    print(f'{name}: Starting at {env.now}')
    yield env.timeout(5)
    print(f'{name}: Done at {env.now}')

env = simpy.Environment()
trace(env, event_callback)

env.process(process(env, 'Process 1'))
env.process(process(env, 'Process 2'))
env.run()
```

### 事件排程監控

追蹤事件何時被排程。

```python
import simpy

class MonitoredEnvironment(simpy.Environment):
    def __init__(self):
        super().__init__()
        self.scheduled_events = []

    def schedule(self, event, priority=simpy.core.NORMAL, delay=0):
        super().schedule(event, priority, delay)
        scheduled_time = self.now + delay
        self.scheduled_events.append((scheduled_time, priority, type(event).__name__))

def process(env, name, delay):
    print(f'{name}: Scheduling timeout for {delay} at {env.now}')
    yield env.timeout(delay)
    print(f'{name}: Resumed at {env.now}')

env = MonitoredEnvironment()
env.process(process(env, 'Process 1', 5))
env.process(process(env, 'Process 2', 3))
env.run()

print('\nScheduled events:')
for time, priority, event_type in env.scheduled_events:
    print(f'Time {time}, Priority {priority}, Type {event_type}')
```

## 4. 統計監控

### 佇列統計

```python
import simpy

class QueueStatistics:
    def __init__(self):
        self.arrival_times = []
        self.departure_times = []
        self.queue_lengths = []
        self.wait_times = []

    def record_arrival(self, time, queue_length):
        self.arrival_times.append(time)
        self.queue_lengths.append(queue_length)

    def record_departure(self, arrival_time, departure_time):
        self.departure_times.append(departure_time)
        self.wait_times.append(departure_time - arrival_time)

    def average_wait_time(self):
        return sum(self.wait_times) / len(self.wait_times) if self.wait_times else 0

    def average_queue_length(self):
        return sum(self.queue_lengths) / len(self.queue_lengths) if self.queue_lengths else 0

def customer(env, resource, stats):
    arrival_time = env.now
    stats.record_arrival(arrival_time, len(resource.queue))

    with resource.request() as req:
        yield req
        departure_time = env.now
        stats.record_departure(arrival_time, departure_time)
        yield env.timeout(2)

env = simpy.Environment()
resource = simpy.Resource(env, capacity=1)
stats = QueueStatistics()

for i in range(5):
    env.process(customer(env, resource, stats))

env.run()

print(f'Average wait time: {stats.average_wait_time():.2f}')
print(f'Average queue length: {stats.average_queue_length():.2f}')
```

## 5. 資料匯出

### CSV 匯出

```python
import simpy
import csv

def export_to_csv(data, filename):
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Time', 'Metric', 'Value'])
        writer.writerows(data)

def monitored_simulation(env, data_log):
    for i in range(10):
        data_log.append((env.now, 'queue_length', i % 3))
        data_log.append((env.now, 'utilization', (i % 3) / 10))
        yield env.timeout(1)

env = simpy.Environment()
data = []
env.process(monitored_simulation(env, data))
env.run()

export_to_csv(data, 'simulation_data.csv')
print('Data exported to simulation_data.csv')
```

### 即時繪圖（需要 matplotlib）

```python
import simpy
import matplotlib.pyplot as plt

class RealTimePlotter:
    def __init__(self):
        self.times = []
        self.values = []

    def update(self, time, value):
        self.times.append(time)
        self.values.append(value)

    def plot(self, title='Simulation Results'):
        plt.figure(figsize=(10, 6))
        plt.plot(self.times, self.values)
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.title(title)
        plt.grid(True)
        plt.show()

def monitored_process(env, plotter):
    value = 0
    for i in range(20):
        value = value * 0.9 + (i % 5)
        plotter.update(env.now, value)
        yield env.timeout(1)

env = simpy.Environment()
plotter = RealTimePlotter()
env.process(monitored_process(env, plotter))
env.run()

plotter.plot('Process Value Over Time')
```

## 最佳實踐

1. **最小化開銷**：只監控必要的內容；過度記錄可能會減慢模擬速度

2. **結構化資料**：對複雜資料點使用類別或具名元組

3. **時間戳記**：監控資料始終包含時間戳記

4. **聚合**：對於長時間模擬，聚合資料而不是儲存每個事件

5. **延遲評估**：考慮收集原始資料並在模擬後計算統計

6. **記憶體管理**：對於非常長的模擬，定期將資料刷新到磁碟

7. **驗證**：確認監控程式碼不會影響模擬行為

8. **關注點分離**：將監控程式碼與模擬邏輯分開

9. **可重用元件**：建立可在各模擬中重用的通用監控類別
