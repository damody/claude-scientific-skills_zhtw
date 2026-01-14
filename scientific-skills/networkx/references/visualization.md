# NetworkX 圖視覺化

## 使用 Matplotlib 的基本繪圖

### 簡單視覺化
```python
import networkx as nx
import matplotlib.pyplot as plt

# 建立和繪製圖
G = nx.karate_club_graph()
nx.draw(G)
plt.show()

# 儲存到檔案
nx.draw(G)
plt.savefig('graph.png', dpi=300, bbox_inches='tight')
plt.close()
```

### 帶標籤繪製
```python
# 繪製帶節點標籤
nx.draw(G, with_labels=True)
plt.show()

# 自訂標籤
labels = {i: f"Node {i}" for i in G.nodes()}
nx.draw(G, labels=labels, with_labels=True)
plt.show()
```

## 佈局演算法

### 彈簧佈局（力導向）
```python
# Fruchterman-Reingold 力導向演算法
pos = nx.spring_layout(G, seed=42)
nx.draw(G, pos=pos, with_labels=True)
plt.show()

# 帶參數
pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)
```

### 圓形佈局
```python
# 將節點排列成圓形
pos = nx.circular_layout(G)
nx.draw(G, pos=pos, with_labels=True)
plt.show()
```

### 隨機佈局
```python
# 隨機定位
pos = nx.random_layout(G, seed=42)
nx.draw(G, pos=pos, with_labels=True)
plt.show()
```

### 殼形佈局
```python
# 同心圓
pos = nx.shell_layout(G)
nx.draw(G, pos=pos, with_labels=True)
plt.show()

# 自訂殼
shells = [[0, 1, 2], [3, 4, 5, 6], [7, 8, 9]]
pos = nx.shell_layout(G, nlist=shells)
```

### 譜佈局
```python
# 使用圖拉普拉斯算子的特徵向量
pos = nx.spectral_layout(G)
nx.draw(G, pos=pos, with_labels=True)
plt.show()
```

### Kamada-Kawai 佈局
```python
# 基於能量的佈局
pos = nx.kamada_kawai_layout(G)
nx.draw(G, pos=pos, with_labels=True)
plt.show()
```

### 平面佈局
```python
# 僅適用於平面圖
if nx.is_planar(G):
    pos = nx.planar_layout(G)
    nx.draw(G, pos=pos, with_labels=True)
    plt.show()
```

### 樹佈局
```python
# 對於樹形圖
if nx.is_tree(G):
    pos = nx.nx_agraph.graphviz_layout(G, prog='dot')
    nx.draw(G, pos=pos, with_labels=True)
    plt.show()
```

## 自訂節點外觀

### 節點顏色
```python
# 單一顏色
nx.draw(G, node_color='red')

# 每個節點不同顏色
node_colors = ['red' if G.degree(n) > 5 else 'blue' for n in G.nodes()]
nx.draw(G, node_color=node_colors)

# 按屬性著色
colors = [G.nodes[n].get('value', 0) for n in G.nodes()]
nx.draw(G, node_color=colors, cmap=plt.cm.viridis)
plt.colorbar()
plt.show()
```

### 節點大小
```python
# 按度數調整大小
node_sizes = [100 * G.degree(n) for n in G.nodes()]
nx.draw(G, node_size=node_sizes)

# 按中心性調整大小
centrality = nx.degree_centrality(G)
node_sizes = [3000 * centrality[n] for n in G.nodes()]
nx.draw(G, node_size=node_sizes)
```

### 節點形狀
```python
# 分別繪製不同形狀的節點
pos = nx.spring_layout(G)

# 圓形節點
nx.draw_networkx_nodes(G, pos, nodelist=[0, 1, 2],
                       node_shape='o', node_color='red')

# 方形節點
nx.draw_networkx_nodes(G, pos, nodelist=[3, 4, 5],
                       node_shape='s', node_color='blue')

nx.draw_networkx_edges(G, pos)
nx.draw_networkx_labels(G, pos)
plt.show()
```

### 節點邊框
```python
nx.draw(G, pos=pos,
        node_color='lightblue',
        edgecolors='black',  # 節點邊框顏色
        linewidths=2)        # 節點邊框寬度
plt.show()
```

## 自訂邊外觀

### 邊顏色
```python
# 單一顏色
nx.draw(G, edge_color='gray')

# 每條邊不同顏色
edge_colors = ['red' if G[u][v].get('weight', 1) > 0.5 else 'blue'
               for u, v in G.edges()]
nx.draw(G, edge_color=edge_colors)

# 按權重著色
edges = G.edges()
weights = [G[u][v].get('weight', 1) for u, v in edges]
nx.draw(G, edge_color=weights, edge_cmap=plt.cm.Reds)
```

### 邊寬度
```python
# 按權重調整寬度
edge_widths = [3 * G[u][v].get('weight', 1) for u, v in G.edges()]
nx.draw(G, width=edge_widths)

# 按介數調整寬度
edge_betweenness = nx.edge_betweenness_centrality(G)
edge_widths = [5 * edge_betweenness[(u, v)] for u, v in G.edges()]
nx.draw(G, width=edge_widths)
```

### 邊樣式
```python
# 虛線邊
nx.draw(G, style='dashed')

# 每條邊不同樣式
pos = nx.spring_layout(G)
strong_edges = [(u, v) for u, v in G.edges() if G[u][v].get('weight', 0) > 0.5]
weak_edges = [(u, v) for u, v in G.edges() if G[u][v].get('weight', 0) <= 0.5]

nx.draw_networkx_nodes(G, pos)
nx.draw_networkx_edges(G, pos, edgelist=strong_edges, style='solid', width=2)
nx.draw_networkx_edges(G, pos, edgelist=weak_edges, style='dashed', width=1)
plt.show()
```

### 有向圖（箭頭）
```python
# 繪製帶箭頭的有向圖
G_directed = nx.DiGraph([(1, 2), (2, 3), (3, 1)])
pos = nx.spring_layout(G_directed)

nx.draw(G_directed, pos=pos, with_labels=True,
        arrows=True,
        arrowsize=20,
        arrowstyle='->',
        connectionstyle='arc3,rad=0.1')
plt.show()
```

## 標籤和註釋

### 節點標籤
```python
pos = nx.spring_layout(G)

# 自訂標籤
labels = {n: f"N{n}" for n in G.nodes()}
nx.draw_networkx_labels(G, pos, labels=labels, font_size=12, font_color='white')

# 字體自訂
nx.draw_networkx_labels(G, pos,
                       font_size=10,
                       font_family='serif',
                       font_weight='bold')
```

### 邊標籤
```python
pos = nx.spring_layout(G)
nx.draw_networkx_nodes(G, pos)
nx.draw_networkx_edges(G, pos)

# 從屬性取得邊標籤
edge_labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
plt.show()

# 自訂邊標籤
edge_labels = {(u, v): f"{u}-{v}" for u, v in G.edges()}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
```

## 進階繪圖技術

### 組合繪圖函數
```python
# 分離組件以完全控制
pos = nx.spring_layout(G, seed=42)

# 繪製邊
nx.draw_networkx_edges(G, pos, alpha=0.3, width=1)

# 繪製節點
nx.draw_networkx_nodes(G, pos,
                       node_color='lightblue',
                       node_size=500,
                       edgecolors='black')

# 繪製標籤
nx.draw_networkx_labels(G, pos, font_size=10)

# 移除軸
plt.axis('off')
plt.tight_layout()
plt.show()
```

### 子圖高亮
```python
pos = nx.spring_layout(G)

# 識別要高亮的子圖
subgraph_nodes = [1, 2, 3, 4]
subgraph = G.subgraph(subgraph_nodes)

# 繪製主圖
nx.draw_networkx_nodes(G, pos, node_color='lightgray', node_size=300)
nx.draw_networkx_edges(G, pos, alpha=0.2)

# 高亮子圖
nx.draw_networkx_nodes(subgraph, pos, node_color='red', node_size=500)
nx.draw_networkx_edges(subgraph, pos, edge_color='red', width=2)

nx.draw_networkx_labels(G, pos)
plt.axis('off')
plt.show()
```

### 社群著色
```python
from networkx.algorithms import community

# 檢測社群
communities = community.greedy_modularity_communities(G)

# 指派顏色
color_map = {}
colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange']
for i, comm in enumerate(communities):
    for node in comm:
        color_map[node] = colors[i % len(colors)]

node_colors = [color_map[n] for n in G.nodes()]

pos = nx.spring_layout(G)
nx.draw(G, pos=pos, node_color=node_colors, with_labels=True)
plt.show()
```

## 建立出版品質圖形

### 高解析度匯出
```python
plt.figure(figsize=(12, 8))
pos = nx.spring_layout(G, seed=42)

nx.draw(G, pos=pos,
        node_color='lightblue',
        node_size=500,
        edge_color='gray',
        width=1,
        with_labels=True,
        font_size=10)

plt.title('Graph Visualization', fontsize=16)
plt.axis('off')
plt.tight_layout()
plt.savefig('publication_graph.png', dpi=300, bbox_inches='tight')
plt.savefig('publication_graph.pdf', bbox_inches='tight')  # 向量格式
plt.close()
```

### 多面板圖形
```python
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# 不同佈局
layouts = [nx.circular_layout(G), nx.spring_layout(G), nx.spectral_layout(G)]
titles = ['Circular', 'Spring', 'Spectral']

for ax, pos, title in zip(axes, layouts, titles):
    nx.draw(G, pos=pos, ax=ax, with_labels=True, node_color='lightblue')
    ax.set_title(title)
    ax.axis('off')

plt.tight_layout()
plt.savefig('layouts_comparison.png', dpi=300)
plt.close()
```

## 互動式視覺化函式庫

### Plotly（互動式）
```python
import plotly.graph_objects as go

# 建立位置
pos = nx.spring_layout(G)

# 邊追蹤
edge_x = []
edge_y = []
for edge in G.edges():
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    edge_x.extend([x0, x1, None])
    edge_y.extend([y0, y1, None])

edge_trace = go.Scatter(
    x=edge_x, y=edge_y,
    line=dict(width=0.5, color='#888'),
    hoverinfo='none',
    mode='lines')

# 節點追蹤
node_x = [pos[node][0] for node in G.nodes()]
node_y = [pos[node][1] for node in G.nodes()]

node_trace = go.Scatter(
    x=node_x, y=node_y,
    mode='markers',
    hoverinfo='text',
    marker=dict(
        showscale=True,
        colorscale='YlGnBu',
        size=10,
        colorbar=dict(thickness=15, title='Node Connections'),
        line_width=2))

# 按度數著色
node_adjacencies = [len(list(G.neighbors(node))) for node in G.nodes()]
node_trace.marker.color = node_adjacencies

fig = go.Figure(data=[edge_trace, node_trace],
                layout=go.Layout(
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=0, l=0, r=0, t=0)))

fig.show()
```

### PyVis（互動式 HTML）
```python
from pyvis.network import Network

# 建立網路
net = Network(notebook=True, height='750px', width='100%')

# 從 NetworkX 添加節點和邊
net.from_nx(G)

# 自訂
net.show_buttons(filter_=['physics'])

# 儲存
net.show('graph.html')
```

### Graphviz（透過 pydot）
```python
# 需要 graphviz 和 pydot
from networkx.drawing.nx_pydot import graphviz_layout

pos = graphviz_layout(G, prog='neato')  # neato、dot、fdp、sfdp、circo、twopi
nx.draw(G, pos=pos, with_labels=True)
plt.show()

# 匯出到 graphviz
nx.drawing.nx_pydot.write_dot(G, 'graph.dot')
```

## 二部圖視覺化

### 雙集佈局
```python
from networkx.algorithms import bipartite

# 建立二部圖
B = nx.Graph()
B.add_nodes_from([1, 2, 3, 4], bipartite=0)
B.add_nodes_from(['a', 'b', 'c', 'd', 'e'], bipartite=1)
B.add_edges_from([(1, 'a'), (1, 'b'), (2, 'b'), (2, 'c'), (3, 'd'), (4, 'e')])

# 雙欄佈局
pos = {}
top_nodes = [n for n, d in B.nodes(data=True) if d['bipartite'] == 0]
bottom_nodes = [n for n, d in B.nodes(data=True) if d['bipartite'] == 1]

pos.update({node: (0, i) for i, node in enumerate(top_nodes)})
pos.update({node: (1, i) for i, node in enumerate(bottom_nodes)})

nx.draw(B, pos=pos, with_labels=True,
        node_color=['lightblue' if B.nodes[n]['bipartite'] == 0 else 'lightgreen'
                   for n in B.nodes()])
plt.show()
```

## 3D 視覺化

### 3D 網路圖
```python
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 3D 彈簧佈局
pos = nx.spring_layout(G, dim=3, seed=42)

# 提取座標
node_xyz = np.array([pos[v] for v in G.nodes()])
edge_xyz = np.array([(pos[u], pos[v]) for u, v in G.edges()])

# 建立圖形
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 繪製邊
for vizedge in edge_xyz:
    ax.plot(*vizedge.T, color='gray', alpha=0.5)

# 繪製節點
ax.scatter(*node_xyz.T, s=100, c='lightblue', edgecolors='black')

# 標籤
for i, (x, y, z) in enumerate(node_xyz):
    ax.text(x, y, z, str(i))

ax.set_axis_off()
plt.show()
```

## 最佳實踐

### 效能
- 對於大型圖（>1000 節點），使用較簡單的佈局（圓形、隨機）
- 使用 `alpha` 參數讓密集邊更可見
- 對於非常大的網路，考慮降採樣或顯示子圖

### 美學
- 使用一致的配色方案
- 有意義地縮放節點大小（例如，按度數或重要性）
- 保持標籤可讀（調整字體大小和位置）
- 有效使用空白（調整圖形大小）

### 可重現性
- 始終為佈局設定隨機種子：`nx.spring_layout(G, seed=42)`
- 儲存佈局位置以在多個圖表中保持一致性
- 在圖例或說明中記錄顏色/大小映射

### 檔案格式
- PNG 用於點陣圖像（網頁、簡報）
- PDF 用於向量圖形（出版物、可縮放）
- SVG 用於網頁和互動式應用
- HTML 用於互動式視覺化
