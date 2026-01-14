# SimPy 事件系統

本指南涵蓋 SimPy 中的事件系統，這是離散事件模擬的基礎。

## 事件基礎

事件是控制模擬流程的核心機制。流程 yield 事件並在這些事件被觸發時恢復。

### 事件生命週期

事件經過三個狀態：

1. **未觸發** - 作為記憶體物件的初始狀態
2. **已觸發** - 已排入事件佇列；`triggered` 屬性為 `True`
3. **已處理** - 從佇列移除並執行回調；`processed` 屬性為 `True`

```python
import simpy

env = simpy.Environment()

# Create an event
event = env.event()
print(f'Triggered: {event.triggered}, Processed: {event.processed}')  # Both False

# Trigger the event
event.succeed(value='Event result')
print(f'Triggered: {event.triggered}, Processed: {event.processed}')  # True, False

# Run to process the event
env.run()
print(f'Triggered: {event.triggered}, Processed: {event.processed}')  # True, True
print(f'Value: {event.value}')  # 'Event result'
```

## 核心事件類型

### Timeout

控制模擬中的時間進程。最常見的事件類型。

```python
import simpy

def process(env):
    print(f'Starting at {env.now}')
    yield env.timeout(5)
    print(f'Resumed at {env.now}')

    # Timeout with value
    result = yield env.timeout(3, value='Done')
    print(f'Result: {result} at {env.now}')

env = simpy.Environment()
env.process(process(env))
env.run()
```

**使用方式：**
- `env.timeout(delay)` - 等待指定時間
- `env.timeout(delay, value=val)` - 等待並返回值

### Process 事件

流程本身就是事件，允許流程等待其他流程完成。

```python
import simpy

def worker(env, name, duration):
    print(f'{name} starting at {env.now}')
    yield env.timeout(duration)
    print(f'{name} finished at {env.now}')
    return f'{name} result'

def coordinator(env):
    # Start worker processes
    worker1 = env.process(worker(env, 'Worker 1', 5))
    worker2 = env.process(worker(env, 'Worker 2', 3))

    # Wait for worker1 to complete
    result = yield worker1
    print(f'Coordinator received: {result}')

    # Wait for worker2
    result = yield worker2
    print(f'Coordinator received: {result}')

env = simpy.Environment()
env.process(coordinator(env))
env.run()
```

### Event

可以手動觸發的通用事件。

```python
import simpy

def waiter(env, event):
    print(f'Waiting for event at {env.now}')
    value = yield event
    print(f'Event received with value: {value} at {env.now}')

def triggerer(env, event):
    yield env.timeout(5)
    print(f'Triggering event at {env.now}')
    event.succeed(value='Hello!')

env = simpy.Environment()
event = env.event()
env.process(waiter(env, event))
env.process(triggerer(env, event))
env.run()
```

## 複合事件

### AllOf - 等待多個事件

當所有指定事件都發生時觸發。

```python
import simpy

def process(env):
    # Start multiple tasks
    task1 = env.timeout(3, value='Task 1 done')
    task2 = env.timeout(5, value='Task 2 done')
    task3 = env.timeout(4, value='Task 3 done')

    # Wait for all to complete
    results = yield simpy.AllOf(env, [task1, task2, task3])
    print(f'All tasks completed at {env.now}')
    print(f'Results: {results}')

    # Alternative syntax using & operator
    task4 = env.timeout(2)
    task5 = env.timeout(3)
    yield task4 & task5
    print(f'Tasks 4 and 5 completed at {env.now}')

env = simpy.Environment()
env.process(process(env))
env.run()
```

**返回值：** 將事件對應到其值的字典

**使用情境：**
- 平行任務完成
- 屏障同步
- 等待多個資源

### AnyOf - 等待任一事件

當至少一個指定事件發生時觸發。

```python
import simpy

def process(env):
    # Start multiple tasks with different durations
    fast_task = env.timeout(2, value='Fast')
    slow_task = env.timeout(10, value='Slow')

    # Wait for first to complete
    results = yield simpy.AnyOf(env, [fast_task, slow_task])
    print(f'First task completed at {env.now}')
    print(f'Results: {results}')

    # Alternative syntax using | operator
    task1 = env.timeout(5)
    task2 = env.timeout(3)
    yield task1 | task2
    print(f'One of the tasks completed at {env.now}')

env = simpy.Environment()
env.process(process(env))
env.run()
```

**返回值：** 包含已完成事件及其值的字典

**使用情境：**
- 競爭條件
- 超時機制
- 先回應者情境

## 事件觸發方法

事件可以透過三種方式觸發：

### succeed(value=None)

將事件標記為成功。

```python
event = env.event()
event.succeed(value='Success!')
```

### fail(exception)

將事件標記為失敗並帶有例外。

```python
def process(env):
    event = env.event()
    event.fail(ValueError('Something went wrong'))

    try:
        yield event
    except ValueError as e:
        print(f'Caught exception: {e}')

env = simpy.Environment()
env.process(process(env))
env.run()
```

### trigger(event)

複製另一個事件的結果。

```python
event1 = env.event()
event1.succeed(value='Original')

event2 = env.event()
event2.trigger(event1)  # event2 now has same outcome as event1
```

## 回調

附加函式以在事件觸發時執行。

```python
import simpy

def callback(event):
    print(f'Callback executed! Event value: {event.value}')

def process(env):
    event = env.timeout(5, value='Done')
    event.callbacks.append(callback)
    yield event

env = simpy.Environment()
env.process(process(env))
env.run()
```

**注意：** 從流程 yield 事件會自動將流程的恢復方法添加為回調。

## 事件共享

多個流程可以等待同一個事件。

```python
import simpy

def waiter(env, name, event):
    print(f'{name} waiting at {env.now}')
    value = yield event
    print(f'{name} resumed with {value} at {env.now}')

def trigger_event(env, event):
    yield env.timeout(5)
    event.succeed(value='Go!')

env = simpy.Environment()
shared_event = env.event()

env.process(waiter(env, 'Process 1', shared_event))
env.process(waiter(env, 'Process 2', shared_event))
env.process(waiter(env, 'Process 3', shared_event))
env.process(trigger_event(env, shared_event))

env.run()
```

**使用情境：**
- 廣播信號
- 屏障同步
- 協調流程恢復

## 進階事件模式

### 帶取消的超時

```python
import simpy

def process_with_timeout(env):
    work = env.timeout(10, value='Work complete')
    timeout = env.timeout(5, value='Timeout!')

    # Race between work and timeout
    result = yield work | timeout

    if work in result:
        print(f'Work completed: {result[work]}')
    else:
        print(f'Timed out: {result[timeout]}')

env = simpy.Environment()
env.process(process_with_timeout(env))
env.run()
```

### 事件鏈接

```python
import simpy

def event_chain(env):
    # Create chain of dependent events
    event1 = env.event()
    event2 = env.event()
    event3 = env.event()

    def trigger_sequence(env):
        yield env.timeout(2)
        event1.succeed(value='Step 1')
        yield env.timeout(2)
        event2.succeed(value='Step 2')
        yield env.timeout(2)
        event3.succeed(value='Step 3')

    env.process(trigger_sequence(env))

    # Wait for sequence
    val1 = yield event1
    print(f'{val1} at {env.now}')
    val2 = yield event2
    print(f'{val2} at {env.now}')
    val3 = yield event3
    print(f'{val3} at {env.now}')

env = simpy.Environment()
env.process(event_chain(env))
env.run()
```

### 條件事件

```python
import simpy

def conditional_process(env):
    temperature = 20

    if temperature > 25:
        yield env.timeout(5)  # Cooling required
        print('System cooled')
    else:
        yield env.timeout(1)  # No cooling needed
        print('Temperature acceptable')

env = simpy.Environment()
env.process(conditional_process(env))
env.run()
```

## 最佳實踐

1. **始終 yield 事件**：流程必須 yield 事件以暫停執行
2. **不要多次觸發事件**：事件只能觸發一次
3. **處理失敗**：在 yield 可能失敗的事件時使用 try-except
4. **使用複合事件實現平行性**：對並行操作使用 AllOf/AnyOf
5. **使用共享事件進行廣播**：多個流程可以 yield 同一個事件
6. **使用事件值傳遞資料**：使用事件值在流程之間傳遞結果
