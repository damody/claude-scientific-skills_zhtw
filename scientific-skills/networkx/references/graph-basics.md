# NetworkX 圖基礎

## 圖類型

NetworkX 支援四種主要圖類別：

### Graph（無向圖）
```python
import networkx as nx
G = nx.Graph()
```
- 節點間具有單邊的無向圖
- 不允許平行邊
- 邊是雙向的

### DiGraph（有向圖）
```python
G = nx.DiGraph()
```
- 具有單向連接的有向圖
- 邊的方向很重要：(u, v) ≠ (v, u)
- 用於建模有向關係

### MultiGraph（無向多邊圖）
```python
G = nx.MultiGraph()
```
- 允許相同節點對之間有多條邊
- 用於建模多重關係

### MultiDiGraph（有向多邊圖）
```python
G = nx.MultiDiGraph()
```
- 節點間具有多條邊的有向圖
- 結合 DiGraph 和 MultiGraph 的特性

## 建立和添加節點

### 單節點添加
```python
G.add_node(1)
G.add_node("protein_A")
G.add_node((x, y))  # 節點可以是任何可雜湊類型
```

### 批次節點添加
```python
G.add_nodes_from([2, 3, 4])
G.add_nodes_from(range(100, 110))
```

### 帶屬性的節點
```python
G.add_node(1, time='5pm', color='red')
G.add_nodes_from([
    (4, {"color": "red"}),
    (5, {"color": "blue", "weight": 1.5})
])
```

### 重要節點屬性
- 節點可以是任何可雜湊的 Python 物件：字串、元組、數字、自訂物件
- 節點屬性以鍵值對形式儲存
- 使用有意義的節點識別碼以提高清晰度

## 建立和添加邊

### 單邊添加
```python
G.add_edge(1, 2)
G.add_edge('gene_A', 'gene_B')
```

### 批次邊添加
```python
G.add_edges_from([(1, 2), (1, 3), (2, 4)])
G.add_edges_from(edge_list)
```

### 帶屬性的邊
```python
G.add_edge(1, 2, weight=4.7, relation='interacts')
G.add_edges_from([
    (1, 2, {'weight': 4.7}),
    (2, 3, {'weight': 8.2, 'color': 'blue'})
])
```

### 從帶屬性的邊列表添加
```python
# 從 pandas DataFrame
import pandas as pd
df = pd.DataFrame({'source': [1, 2], 'target': [2, 3], 'weight': [4.7, 8.2]})
G = nx.from_pandas_edgelist(df, 'source', 'target', edge_attr='weight')
```

## 檢查圖結構

### 基本屬性
```python
# 取得集合
G.nodes              # 所有節點的 NodeView
G.edges              # 所有邊的 EdgeView
G.adj                # 鄰居關係的 AdjacencyView

# 計算元素數量
G.number_of_nodes()  # 總節點數
G.number_of_edges()  # 總邊數
len(G)              # 節點數（簡寫）

# 度資訊
G.degree()          # 所有節點度的 DegreeView
G.degree(1)         # 特定節點的度
list(G.degree())    # (節點, 度) 對的列表
```

### 檢查存在性
```python
# 檢查節點是否存在
1 in G              # 返回 True/False
G.has_node(1)

# 檢查邊是否存在
G.has_edge(1, 2)
```

### 存取鄰居
```python
# 取得節點 1 的鄰居
list(G.neighbors(1))
list(G[1])          # 字典式存取

# 對於有向圖
list(G.predecessors(1))  # 入邊
list(G.successors(1))    # 出邊
```

### 迭代元素
```python
# 迭代節點
for node in G.nodes:
    print(node, G.nodes[node])  # 存取節點屬性

# 迭代邊
for u, v in G.edges:
    print(u, v, G[u][v])  # 存取邊屬性

# 帶屬性迭代
for node, attrs in G.nodes(data=True):
    print(node, attrs)

for u, v, attrs in G.edges(data=True):
    print(u, v, attrs)
```

## 修改圖

### 移除元素
```python
# 移除單個節點（同時移除相關邊）
G.remove_node(1)

# 移除多個節點
G.remove_nodes_from([1, 2, 3])

# 移除邊
G.remove_edge(1, 2)
G.remove_edges_from([(1, 2), (2, 3)])
```

### 清除圖
```python
G.clear()           # 移除所有節點和邊
G.clear_edges()     # 只移除邊，保留節點
```

## 屬性和元資料

### 圖級屬性
```python
G.graph['name'] = 'Social Network'
G.graph['date'] = '2025-01-15'
print(G.graph)
```

### 節點屬性
```python
# 建立時設定
G.add_node(1, time='5pm', weight=0.5)

# 建立後設定
G.nodes[1]['time'] = '6pm'
nx.set_node_attributes(G, {1: 'red', 2: 'blue'}, 'color')

# 取得屬性
G.nodes[1]
G.nodes[1]['time']
nx.get_node_attributes(G, 'color')
```

### 邊屬性
```python
# 建立時設定
G.add_edge(1, 2, weight=4.7, color='red')

# 建立後設定
G[1][2]['weight'] = 5.0
nx.set_edge_attributes(G, {(1, 2): 10.5}, 'weight')

# 取得屬性
G[1][2]
G[1][2]['weight']
G.edges[1, 2]
nx.get_edge_attributes(G, 'weight')
```

## 子圖和視圖

### 子圖建立
```python
# 從節點列表建立子圖
nodes_subset = [1, 2, 3, 4]
H = G.subgraph(nodes_subset)  # 返回視圖（參照原圖）

# 建立獨立副本
H = G.subgraph(nodes_subset).copy()

# 邊誘導子圖
edge_subset = [(1, 2), (2, 3)]
H = G.edge_subgraph(edge_subset)
```

### 圖視圖
```python
# 反向視圖（對於有向圖）
G_reversed = G.reverse()

# 有向/無向間轉換
G_undirected = G.to_undirected()
G_directed = G.to_directed()
```

## 圖資訊和診斷

### 基本資訊
```python
print(nx.info(G))   # 圖結構摘要

# 密度（實際邊數與可能邊數的比率）
nx.density(G)

# 檢查圖是否有向
G.is_directed()

# 檢查圖是否為多重圖
G.is_multigraph()
```

### 連通性檢查
```python
# 對於無向圖
nx.is_connected(G)
nx.number_connected_components(G)

# 對於有向圖
nx.is_strongly_connected(G)
nx.is_weakly_connected(G)
```

## 重要考量

### 浮點精度
一旦圖包含浮點數，由於精度限制，所有結果本質上是近似的。小的算術誤差可能影響演算法結果，特別是在最小/最大計算中。

### 記憶體考量
每次腳本啟動時，圖資料必須載入記憶體。對於大型資料集，這可能導致效能問題。考慮：
- 使用高效的資料格式（Python 物件使用 pickle）
- 只載入必要的子圖
- 對非常大的網路使用圖資料庫

### 節點和邊移除行為
當移除節點時，所有與該節點相關的邊也會自動移除。
