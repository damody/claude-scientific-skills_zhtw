# SimPy 共享資源

本指南涵蓋 SimPy 中用於建模擁塞點和資源配置的所有資源類型。

## 資源類型概述

SimPy 提供三大類共享資源：

1. **Resources** - 有限容量資源（例如加油機、伺服器）
2. **Containers** - 同質批量材料（例如燃料槽、筒倉）
3. **Stores** - Python 物件儲存（例如項目佇列、倉庫）

## 1. Resources

建模一次只能由有限數量的流程使用的資源。

### Resource（基本）

基本資源是具有指定容量的信號量。

```python
import simpy

env = simpy.Environment()
resource = simpy.Resource(env, capacity=2)

def process(env, resource, name):
    with resource.request() as req:
        yield req
        print(f'{name} has the resource at {env.now}')
        yield env.timeout(5)
        print(f'{name} releases the resource at {env.now}')

env.process(process(env, resource, 'Process 1'))
env.process(process(env, resource, 'Process 2'))
env.process(process(env, resource, 'Process 3'))
env.run()
```

**關鍵屬性：**
- `capacity` - 最大並行使用者數量（預設：1）
- `count` - 當前使用者數量
- `queue` - 排隊請求的列表

### PriorityResource

擴展基本資源，增加優先順序層級（數字越小 = 優先順序越高）。

```python
import simpy

env = simpy.Environment()
resource = simpy.PriorityResource(env, capacity=1)

def process(env, resource, name, priority):
    with resource.request(priority=priority) as req:
        yield req
        print(f'{name} (priority {priority}) has the resource at {env.now}')
        yield env.timeout(5)

env.process(process(env, resource, 'Low priority', priority=10))
env.process(process(env, resource, 'High priority', priority=1))
env.run()
```

**使用情境：**
- 緊急服務（救護車優先於一般車輛）
- VIP 客戶佇列
- 帶優先順序的任務排程

### PreemptiveResource

允許高優先順序請求中斷低優先順序使用者。

```python
import simpy

env = simpy.Environment()
resource = simpy.PreemptiveResource(env, capacity=1)

def process(env, resource, name, priority):
    with resource.request(priority=priority) as req:
        try:
            yield req
            print(f'{name} acquired resource at {env.now}')
            yield env.timeout(10)
            print(f'{name} finished at {env.now}')
        except simpy.Interrupt:
            print(f'{name} was preempted at {env.now}')

env.process(process(env, resource, 'Low priority', priority=10))
env.process(process(env, resource, 'High priority', priority=1))
env.run()
```

**使用情境：**
- 作業系統 CPU 排程
- 急診室分流
- 網路封包優先順序

## 2. Containers

建模同質批量材料（連續或離散）的生產和消費。

```python
import simpy

env = simpy.Environment()
container = simpy.Container(env, capacity=100, init=50)

def producer(env, container):
    while True:
        yield env.timeout(5)
        yield container.put(20)
        print(f'Produced 20. Level: {container.level}')

def consumer(env, container):
    while True:
        yield env.timeout(7)
        yield container.get(15)
        print(f'Consumed 15. Level: {container.level}')

env.process(producer(env, container))
env.process(consumer(env, container))
env.run(until=50)
```

**關鍵屬性：**
- `capacity` - 最大數量（預設：float('inf')）
- `level` - 當前數量
- `init` - 初始數量（預設：0）

**操作：**
- `put(amount)` - 添加到容器（滿時阻塞）
- `get(amount)` - 從容器移除（不足時阻塞）

**使用情境：**
- 加油站燃料槽
- 製造業中的緩衝儲存
- 水庫
- 電池電量水準

## 3. Stores

建模 Python 物件的生產和消費。

### Store（基本）

通用 FIFO 物件儲存。

```python
import simpy

env = simpy.Environment()
store = simpy.Store(env, capacity=2)

def producer(env, store):
    for i in range(5):
        yield env.timeout(2)
        item = f'Item {i}'
        yield store.put(item)
        print(f'Produced {item} at {env.now}')

def consumer(env, store):
    while True:
        yield env.timeout(3)
        item = yield store.get()
        print(f'Consumed {item} at {env.now}')

env.process(producer(env, store))
env.process(consumer(env, store))
env.run()
```

**關鍵屬性：**
- `capacity` - 最大項目數量（預設：float('inf')）
- `items` - 已儲存項目的列表

**操作：**
- `put(item)` - 將項目添加到儲存庫（滿時阻塞）
- `get()` - 移除並返回項目（空時阻塞）

### FilterStore

允許根據過濾函式檢索特定物件。

```python
import simpy

env = simpy.Environment()
store = simpy.FilterStore(env, capacity=10)

def producer(env, store):
    for color in ['red', 'blue', 'green', 'red', 'blue']:
        yield env.timeout(1)
        yield store.put({'color': color, 'time': env.now})
        print(f'Produced {color} item at {env.now}')

def consumer(env, store, color):
    while True:
        yield env.timeout(2)
        item = yield store.get(lambda x: x['color'] == color)
        print(f'{color} consumer got item from {item["time"]} at {env.now}')

env.process(producer(env, store))
env.process(consumer(env, store, 'red'))
env.process(consumer(env, store, 'blue'))
env.run(until=15)
```

**使用情境：**
- 倉庫項目揀選（特定 SKU）
- 具有技能匹配的工作佇列
- 按目的地的封包路由

### PriorityStore

項目按優先順序檢索（最低優先）。

```python
import simpy

class PriorityItem:
    def __init__(self, priority, data):
        self.priority = priority
        self.data = data

    def __lt__(self, other):
        return self.priority < other.priority

env = simpy.Environment()
store = simpy.PriorityStore(env, capacity=10)

def producer(env, store):
    items = [(10, 'Low'), (1, 'High'), (5, 'Medium')]
    for priority, name in items:
        yield env.timeout(1)
        yield store.put(PriorityItem(priority, name))
        print(f'Produced {name} priority item')

def consumer(env, store):
    while True:
        yield env.timeout(5)
        item = yield store.get()
        print(f'Retrieved {item.data} priority item')

env.process(producer(env, store))
env.process(consumer(env, store))
env.run()
```

**使用情境：**
- 任務排程
- 列印工作佇列
- 訊息優先順序

## 選擇正確的資源類型

| 情境 | 資源類型 |
|------|---------|
| 有限伺服器/機器 | Resource |
| 基於優先順序的佇列 | PriorityResource |
| 搶占式排程 | PreemptiveResource |
| 燃料、水、批量材料 | Container |
| 通用項目佇列（FIFO） | Store |
| 選擇性項目檢索 | FilterStore |
| 優先順序排序的項目 | PriorityStore |

## 最佳實踐

1. **容量規劃**：根據系統限制設定實際容量
2. **請求模式**：使用上下文管理器（`with resource.request()`）以自動清理
3. **錯誤處理**：將搶占式資源包裝在 try-except 中以處理 Interrupt
4. **監控**：追蹤佇列長度和利用率（參見 monitoring.md）
5. **效能**：FilterStore 和 PriorityStore 的檢索時間為 O(n)；對於大型儲存庫請謹慎使用
