# SimPy 流程互動

本指南涵蓋 SimPy 模擬中流程互動和同步的機制。

## 互動機制概述

SimPy 提供三種主要的流程互動方式：

1. **基於事件的暫停/重新啟動** - 用於信號傳遞的共享事件
2. **等待流程終止** - yield 流程物件
3. **中斷** - 強制恢復暫停的流程

## 1. 基於事件的暫停和重新啟動

流程可以共享事件以協調其執行。

### 基本信號模式

```python
import simpy

def controller(env, signal_event):
    print(f'Controller: Preparing at {env.now}')
    yield env.timeout(5)
    print(f'Controller: Sending signal at {env.now}')
    signal_event.succeed()

def worker(env, signal_event):
    print(f'Worker: Waiting for signal at {env.now}')
    yield signal_event
    print(f'Worker: Received signal, starting work at {env.now}')
    yield env.timeout(3)
    print(f'Worker: Work complete at {env.now}')

env = simpy.Environment()
signal = env.event()
env.process(controller(env, signal))
env.process(worker(env, signal))
env.run()
```

**使用情境：**
- 協調操作的啟動信號
- 完成通知
- 廣播狀態變更

### 多個等待者

多個流程可以等待同一個信號事件。

```python
import simpy

def broadcaster(env, signal):
    yield env.timeout(5)
    print(f'Broadcasting signal at {env.now}')
    signal.succeed(value='Go!')

def listener(env, name, signal):
    print(f'{name}: Waiting at {env.now}')
    msg = yield signal
    print(f'{name}: Received "{msg}" at {env.now}')
    yield env.timeout(2)
    print(f'{name}: Done at {env.now}')

env = simpy.Environment()
broadcast_signal = env.event()

env.process(broadcaster(env, broadcast_signal))
for i in range(3):
    env.process(listener(env, f'Listener {i+1}', broadcast_signal))

env.run()
```

### 屏障同步

```python
import simpy

class Barrier:
    def __init__(self, env, n):
        self.env = env
        self.n = n
        self.count = 0
        self.event = env.event()

    def wait(self):
        self.count += 1
        if self.count >= self.n:
            self.event.succeed()
        return self.event

def worker(env, barrier, name, work_time):
    print(f'{name}: Working at {env.now}')
    yield env.timeout(work_time)
    print(f'{name}: Reached barrier at {env.now}')
    yield barrier.wait()
    print(f'{name}: Passed barrier at {env.now}')

env = simpy.Environment()
barrier = Barrier(env, 3)

env.process(worker(env, barrier, 'Worker A', 3))
env.process(worker(env, barrier, 'Worker B', 5))
env.process(worker(env, barrier, 'Worker C', 7))

env.run()
```

## 2. 等待流程終止

流程本身就是事件，所以您可以 yield 它們以等待完成。

### 順序流程執行

```python
import simpy

def task(env, name, duration):
    print(f'{name}: Starting at {env.now}')
    yield env.timeout(duration)
    print(f'{name}: Completed at {env.now}')
    return f'{name} result'

def sequential_coordinator(env):
    # Execute tasks sequentially
    result1 = yield env.process(task(env, 'Task 1', 5))
    print(f'Coordinator: {result1}')

    result2 = yield env.process(task(env, 'Task 2', 3))
    print(f'Coordinator: {result2}')

    result3 = yield env.process(task(env, 'Task 3', 4))
    print(f'Coordinator: {result3}')

env = simpy.Environment()
env.process(sequential_coordinator(env))
env.run()
```

### 平行流程執行

```python
import simpy

def task(env, name, duration):
    print(f'{name}: Starting at {env.now}')
    yield env.timeout(duration)
    print(f'{name}: Completed at {env.now}')
    return f'{name} result'

def parallel_coordinator(env):
    # Start all tasks
    task1 = env.process(task(env, 'Task 1', 5))
    task2 = env.process(task(env, 'Task 2', 3))
    task3 = env.process(task(env, 'Task 3', 4))

    # Wait for all to complete
    results = yield task1 & task2 & task3
    print(f'All tasks completed at {env.now}')
    print(f'Task 1 result: {task1.value}')
    print(f'Task 2 result: {task2.value}')
    print(f'Task 3 result: {task3.value}')

env = simpy.Environment()
env.process(parallel_coordinator(env))
env.run()
```

### 先完成模式

```python
import simpy

def server(env, name, processing_time):
    print(f'{name}: Starting request at {env.now}')
    yield env.timeout(processing_time)
    print(f'{name}: Completed at {env.now}')
    return name

def load_balancer(env):
    # Send request to multiple servers
    server1 = env.process(server(env, 'Server 1', 5))
    server2 = env.process(server(env, 'Server 2', 3))
    server3 = env.process(server(env, 'Server 3', 7))

    # Wait for first to respond
    result = yield server1 | server2 | server3

    # Get the winner
    winner = list(result.values())[0]
    print(f'Load balancer: {winner} responded first at {env.now}')

env = simpy.Environment()
env.process(load_balancer(env))
env.run()
```

## 3. 流程中斷

流程可以使用 `process.interrupt()` 來中斷，這會拋出 `Interrupt` 例外。

### 基本中斷

```python
import simpy

def worker(env):
    try:
        print(f'Worker: Starting long task at {env.now}')
        yield env.timeout(10)
        print(f'Worker: Task completed at {env.now}')
    except simpy.Interrupt as interrupt:
        print(f'Worker: Interrupted at {env.now}')
        print(f'Interrupt cause: {interrupt.cause}')

def interrupter(env, target_process):
    yield env.timeout(5)
    print(f'Interrupter: Interrupting worker at {env.now}')
    target_process.interrupt(cause='Higher priority task')

env = simpy.Environment()
worker_process = env.process(worker(env))
env.process(interrupter(env, worker_process))
env.run()
```

### 可恢復中斷

流程可以在中斷後重新 yield 同一事件以繼續等待。

```python
import simpy

def resumable_worker(env):
    work_left = 10

    while work_left > 0:
        try:
            print(f'Worker: Working ({work_left} units left) at {env.now}')
            start = env.now
            yield env.timeout(work_left)
            work_left = 0
            print(f'Worker: Completed at {env.now}')
        except simpy.Interrupt:
            work_left -= (env.now - start)
            print(f'Worker: Interrupted! {work_left} units left at {env.now}')

def interrupter(env, worker_proc):
    yield env.timeout(3)
    worker_proc.interrupt()
    yield env.timeout(2)
    worker_proc.interrupt()

env = simpy.Environment()
worker_proc = env.process(resumable_worker(env))
env.process(interrupter(env, worker_proc))
env.run()
```

### 帶自訂原因的中斷

```python
import simpy

def machine(env, name):
    while True:
        try:
            print(f'{name}: Operating at {env.now}')
            yield env.timeout(5)
        except simpy.Interrupt as interrupt:
            if interrupt.cause == 'maintenance':
                print(f'{name}: Maintenance required at {env.now}')
                yield env.timeout(2)
                print(f'{name}: Maintenance complete at {env.now}')
            elif interrupt.cause == 'emergency':
                print(f'{name}: Emergency stop at {env.now}')
                break

def maintenance_scheduler(env, machine_proc):
    yield env.timeout(7)
    machine_proc.interrupt(cause='maintenance')
    yield env.timeout(10)
    machine_proc.interrupt(cause='emergency')

env = simpy.Environment()
machine_proc = env.process(machine(env, 'Machine 1'))
env.process(maintenance_scheduler(env, machine_proc))
env.run()
```

### 帶中斷的搶占式資源

```python
import simpy

def user(env, name, resource, priority, duration):
    with resource.request(priority=priority) as req:
        try:
            yield req
            print(f'{name} (priority {priority}): Got resource at {env.now}')
            yield env.timeout(duration)
            print(f'{name}: Done at {env.now}')
        except simpy.Interrupt:
            print(f'{name}: Preempted at {env.now}')

env = simpy.Environment()
resource = simpy.PreemptiveResource(env, capacity=1)

env.process(user(env, 'Low priority user', resource, priority=10, duration=10))
env.process(user(env, 'High priority user', resource, priority=1, duration=5))
env.run()
```

## 進階模式

### 帶信號的生產者-消費者

```python
import simpy

class Buffer:
    def __init__(self, env, capacity):
        self.env = env
        self.capacity = capacity
        self.items = []
        self.item_available = env.event()

    def put(self, item):
        if len(self.items) < self.capacity:
            self.items.append(item)
            if not self.item_available.triggered:
                self.item_available.succeed()
            return True
        return False

    def get(self):
        if self.items:
            return self.items.pop(0)
        return None

def producer(env, buffer):
    item_id = 0
    while True:
        yield env.timeout(2)
        item = f'Item {item_id}'
        if buffer.put(item):
            print(f'Producer: Added {item} at {env.now}')
            item_id += 1

def consumer(env, buffer):
    while True:
        if buffer.items:
            item = buffer.get()
            print(f'Consumer: Retrieved {item} at {env.now}')
            yield env.timeout(3)
        else:
            print(f'Consumer: Waiting for items at {env.now}')
            yield buffer.item_available
            buffer.item_available = env.event()

env = simpy.Environment()
buffer = Buffer(env, capacity=5)
env.process(producer(env, buffer))
env.process(consumer(env, buffer))
env.run(until=20)
```

### 握手協議

```python
import simpy

def sender(env, request_event, acknowledge_event):
    for i in range(3):
        print(f'Sender: Sending request {i} at {env.now}')
        request_event.succeed(value=f'Request {i}')
        yield acknowledge_event
        print(f'Sender: Received acknowledgment at {env.now}')

        # Reset events for next iteration
        request_event = env.event()
        acknowledge_event = env.event()
        yield env.timeout(1)

def receiver(env, request_event, acknowledge_event):
    for i in range(3):
        request = yield request_event
        print(f'Receiver: Got {request} at {env.now}')
        yield env.timeout(2)  # Process request
        acknowledge_event.succeed()
        print(f'Receiver: Sent acknowledgment at {env.now}')

        # Reset for next iteration
        request_event = env.event()
        acknowledge_event = env.event()

env = simpy.Environment()
request = env.event()
ack = env.event()
env.process(sender(env, request, ack))
env.process(receiver(env, request, ack))
env.run()
```

## 最佳實踐

1. **選擇正確的機制**：
   - 使用事件進行信號和廣播
   - 使用流程 yield 進行順序/平行工作流程
   - 使用中斷進行搶占和緊急處理

2. **例外處理**：始終將容易中斷的程式碼包裝在 try-except 區塊中

3. **事件生命週期**：記住事件只能觸發一次；為重複信號建立新事件

4. **流程參考**：如果需要稍後中斷流程，請儲存流程物件

5. **原因資訊**：使用中斷原因來傳達中斷發生的原因

6. **可恢復模式**：追蹤進度以在中斷後啟用恢復

7. **避免死鎖**：確保任何時候至少有一個流程可以繼續進行
