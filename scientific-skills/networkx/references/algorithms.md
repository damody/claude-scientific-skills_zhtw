# NetworkX 圖演算法

## 最短路徑

### 單源最短路徑
```python
# Dijkstra 演算法（加權圖）
path = nx.shortest_path(G, source=1, target=5, weight='weight')
length = nx.shortest_path_length(G, source=1, target=5, weight='weight')

# 從源點出發的所有最短路徑
paths = nx.single_source_shortest_path(G, source=1)
lengths = nx.single_source_shortest_path_length(G, source=1)

# Bellman-Ford（處理負權重）
path = nx.bellman_ford_path(G, source=1, target=5, weight='weight')
```

### 全對最短路徑
```python
# 所有節點對（返回迭代器）
for source, paths in nx.all_pairs_shortest_path(G):
    print(f"From {source}: {paths}")

# Floyd-Warshall 演算法
lengths = dict(nx.all_pairs_shortest_path_length(G))
```

### 專門最短路徑演算法
```python
# A* 演算法（帶啟發函數）
def heuristic(u, v):
    # 自訂啟發函數
    return abs(u - v)

path = nx.astar_path(G, source=1, target=5, heuristic=heuristic, weight='weight')

# 平均最短路徑長度
avg_length = nx.average_shortest_path_length(G)
```

## 連通性

### 連通分量（無向）
```python
# 檢查是否連通
is_connected = nx.is_connected(G)

# 分量數量
num_components = nx.number_connected_components(G)

# 取得所有分量（返回集合的迭代器）
components = list(nx.connected_components(G))
largest_component = max(components, key=len)

# 取得包含特定節點的分量
component = nx.node_connected_component(G, node=1)
```

### 強/弱連通（有向）
```python
# 強連通（相互可達）
is_strongly_connected = nx.is_strongly_connected(G)
strong_components = list(nx.strongly_connected_components(G))
largest_scc = max(strong_components, key=len)

# 弱連通（忽略方向）
is_weakly_connected = nx.is_weakly_connected(G)
weak_components = list(nx.weakly_connected_components(G))

# 凝縮圖（強連通分量的 DAG）
condensed = nx.condensation(G)
```

### 割集和連通性
```python
# 最小節點/邊割集
min_node_cut = nx.minimum_node_cut(G, s=1, t=5)
min_edge_cut = nx.minimum_edge_cut(G, s=1, t=5)

# 節點/邊連通性
node_connectivity = nx.node_connectivity(G)
edge_connectivity = nx.edge_connectivity(G)
```

## 中心性度量

### 度中心性
```python
# 每個節點連接的節點比例
degree_cent = nx.degree_centrality(G)

# 對於有向圖
in_degree_cent = nx.in_degree_centrality(G)
out_degree_cent = nx.out_degree_centrality(G)
```

### 介數中心性
```python
# 經過節點的最短路徑比例
betweenness = nx.betweenness_centrality(G, weight='weight')

# 邊介數
edge_betweenness = nx.edge_betweenness_centrality(G, weight='weight')

# 大型圖的近似值
approx_betweenness = nx.betweenness_centrality(G, k=100)  # 採樣 100 個節點
```

### 接近中心性
```python
# 平均最短路徑長度的倒數
closeness = nx.closeness_centrality(G)

# 對於非連通圖
closeness = nx.closeness_centrality(G, wf_improved=True)
```

### 特徵向量中心性
```python
# 基於與高中心性節點連接的中心性
eigenvector = nx.eigenvector_centrality(G, max_iter=1000)

# Katz 中心性（帶衰減因子的變體）
katz = nx.katz_centrality(G, alpha=0.1, beta=1.0)
```

### PageRank
```python
# Google 的 PageRank 演算法
pagerank = nx.pagerank(G, alpha=0.85)

# 個人化 PageRank
personalization = {node: 1.0 if node in [1, 2] else 0.0 for node in G}
ppr = nx.pagerank(G, personalization=personalization)
```

## 聚類

### 聚類係數
```python
# 每個節點的聚類係數
clustering = nx.clustering(G)

# 平均聚類係數
avg_clustering = nx.average_clustering(G)

# 加權聚類
weighted_clustering = nx.clustering(G, weight='weight')
```

### 傳遞性
```python
# 整體聚類（三角形與三元組的比率）
transitivity = nx.transitivity(G)
```

### 三角形
```python
# 每個節點的三角形計數
triangles = nx.triangles(G)

# 三角形總數
total_triangles = sum(triangles.values()) // 3
```

## 社群檢測

### 基於模組度
```python
from networkx.algorithms import community

# 貪婪模組度最大化
communities = community.greedy_modularity_communities(G)

# 計算模組度
modularity = community.modularity(G, communities)
```

### 標籤傳播
```python
# 快速社群檢測
communities = community.label_propagation_communities(G)
```

### Girvan-Newman
```python
# 透過邊介數的階層式社群檢測
comp = community.girvan_newman(G)
limited = itertools.takewhile(lambda c: len(c) <= 10, comp)
for communities in limited:
    print(tuple(sorted(c) for c in communities))
```

## 匹配和覆蓋

### 最大匹配
```python
# 最大基數匹配
matching = nx.max_weight_matching(G)

# 檢查匹配是否有效
is_matching = nx.is_matching(G, matching)
is_perfect = nx.is_perfect_matching(G, matching)
```

### 最小頂點/邊覆蓋
```python
# 覆蓋所有邊的最小節點集
min_vertex_cover = nx.approximation.min_weighted_vertex_cover(G)

# 最小邊支配集
min_edge_dom = nx.approximation.min_edge_dominating_set(G)
```

## 樹演算法

### 最小生成樹
```python
# Kruskal 或 Prim 演算法
mst = nx.minimum_spanning_tree(G, weight='weight')

# 最大生成樹
mst_max = nx.maximum_spanning_tree(G, weight='weight')

# 列舉所有生成樹
all_spanning = nx.all_spanning_trees(G)
```

### 樹屬性
```python
# 檢查圖是否為樹
is_tree = nx.is_tree(G)
is_forest = nx.is_forest(G)

# 對於有向圖
is_arborescence = nx.is_arborescence(G)
```

## 流和容量

### 最大流
```python
# 最大流值
flow_value = nx.maximum_flow_value(G, s=1, t=5, capacity='capacity')

# 帶流字典的最大流
flow_value, flow_dict = nx.maximum_flow(G, s=1, t=5, capacity='capacity')

# 最小割
cut_value, partition = nx.minimum_cut(G, s=1, t=5, capacity='capacity')
```

### 成本流
```python
# 最小成本流
flow_dict = nx.min_cost_flow(G, demand='demand', capacity='capacity', weight='weight')
cost = nx.cost_of_flow(G, flow_dict, weight='weight')
```

## 迴圈

### 尋找迴圈
```python
# 簡單迴圈（對於有向圖）
cycles = list(nx.simple_cycles(G))

# 迴圈基底（對於無向圖）
basis = nx.cycle_basis(G)

# 檢查是否無環
is_dag = nx.is_directed_acyclic_graph(G)
```

### 拓撲排序
```python
# 僅適用於 DAG
try:
    topo_order = list(nx.topological_sort(G))
except nx.NetworkXError:
    print("Graph has cycles")

# 所有拓撲排序
all_topo = nx.all_topological_sorts(G)
```

## 團

### 尋找團
```python
# 所有極大團
cliques = list(nx.find_cliques(G))

# 最大團（NP 完全，近似）
max_clique = nx.approximation.max_clique(G)

# 團數
clique_number = nx.graph_clique_number(G)

# 包含每個節點的極大團數量
clique_counts = nx.node_clique_number(G)
```

## 圖著色

### 節點著色
```python
# 貪婪著色
coloring = nx.greedy_color(G, strategy='largest_first')

# 不同策略：'largest_first'、'smallest_last'、'random_sequential'
coloring = nx.greedy_color(G, strategy='smallest_last')
```

## 同構

### 圖同構
```python
# 檢查圖是否同構
is_isomorphic = nx.is_isomorphic(G1, G2)

# 取得同構映射
from networkx.algorithms import isomorphism
GM = isomorphism.GraphMatcher(G1, G2)
if GM.is_isomorphic():
    mapping = GM.mapping
```

### 子圖同構
```python
# 檢查 G1 是否為 G2 的子圖同構
is_subgraph_iso = nx.is_isomorphic(G1, G2.subgraph(nodes))
```

## 遍歷演算法

### 深度優先搜索（DFS）
```python
# DFS 邊
dfs_edges = list(nx.dfs_edges(G, source=1))

# DFS 樹
dfs_tree = nx.dfs_tree(G, source=1)

# DFS 前驅
dfs_pred = nx.dfs_predecessors(G, source=1)

# 前序和後序
preorder = list(nx.dfs_preorder_nodes(G, source=1))
postorder = list(nx.dfs_postorder_nodes(G, source=1))
```

### 廣度優先搜索（BFS）
```python
# BFS 邊
bfs_edges = list(nx.bfs_edges(G, source=1))

# BFS 樹
bfs_tree = nx.bfs_tree(G, source=1)

# BFS 前驅和後繼
bfs_pred = nx.bfs_predecessors(G, source=1)
bfs_succ = nx.bfs_successors(G, source=1)
```

## 效率考量

### 演算法複雜度
- 許多演算法有控制計算時間的參數
- 對於大型圖，考慮近似演算法
- 使用 `k` 參數在中心性計算中採樣節點
- 為迭代演算法設定 `max_iter`

### 記憶體使用
- 基於迭代器的函數（例如 `nx.simple_cycles()`）節省記憶體
- 只在必要時轉換為列表
- 對大型結果集使用生成器

### 數值精度
當使用帶有浮點數的加權演算法時，結果是近似的。考慮：
- 盡可能使用整數權重
- 設定適當的容差參數
- 注意迭代演算法中累積的捨入誤差
